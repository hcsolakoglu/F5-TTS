"""Single-GPU CUDA validation for global masked-mean training.

These tests are skipped on non-CUDA hosts. They deliberately use the real
Trainer window/scaling helpers and CFM compile entrypoint while keeping the
model tiny enough for a developer RTX-class GPU.
"""

from __future__ import annotations

import contextlib

import pytest
import torch
from accelerate.utils import DistributedType, send_to_device
from torch import nn

from f5_tts.model.cfm import CFM
from f5_tts.model.trainer import Trainer


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


class _CudaAccelerator:
    def __init__(self, *, gradient_accumulation_steps):
        self.device = torch.device("cuda", 0)
        self.num_processes = 1
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.distributed_type = DistributedType.NO
        self.sync_gradients = True
        self.reduced_dtypes = []

    def reduce(self, value, reduction):
        assert reduction == "sum"
        self.reduced_dtypes.append(value.dtype)
        return value

    def no_sync(self, model):
        return contextlib.nullcontext()

    def backward(self, loss):
        (loss / self.gradient_accumulation_steps).backward()


class _FixedCudaMaskSampler:
    def __init__(self, masks, *, num_channels):
        self.masks = iter(masks)
        self.num_channels = num_channels
        self.returned = []

    def sample_training_mask(self, lens, seq_len):
        mask = next(self.masks).to(device="cuda")
        assert mask.shape == (lens.shape[0], seq_len)
        self.returned.append(mask)
        return mask


def _linear_model(channels):
    model = nn.Linear(channels, channels, bias=True, device="cuda")
    with torch.no_grad():
        values = torch.arange(channels * channels, device="cuda", dtype=torch.float32).reshape(channels, channels)
        model.weight.copy_((values - values.mean()) / (channels * 3))
        model.bias.copy_(torch.linspace(-0.1, 0.1, channels, device="cuda"))
    return model


def _cuda_batches(mask_counts, *, channels):
    batches = []
    masks = []
    for index, count in enumerate(mask_counts):
        seq_len = max(count + 2, 3)
        raw = torch.arange(seq_len * channels, dtype=torch.float32).reshape(seq_len, channels)
        x = ((raw + index * 3) % 17 - 8) / 7
        target = 0.25 * x.flip(-1) - 0.13 * x + (index + 1) * 0.03
        mask = torch.zeros((1, seq_len), dtype=torch.bool)
        mask[0, :count] = True
        batches.append(
            {
                "mel": x.transpose(0, 1).unsqueeze(0),
                "mel_lengths": torch.tensor([seq_len]),
                "target": target.unsqueeze(0),
                "text": ["cuda"],
            }
        )
        masks.append(mask)
    return batches, masks


def _cuda_candidate_and_reference(mask_counts, *, gradient_accumulation_steps, autocast_dtype=None):
    channels = 4
    batches, masks = _cuda_batches(mask_counts, channels=channels)
    candidate = _linear_model(channels)
    reference = _linear_model(channels)
    reference.load_state_dict(candidate.state_dict())
    candidate_optimizer = torch.optim.SGD(candidate.parameters(), lr=0.03, momentum=0.8)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03, momentum=0.8)
    accelerator = _CudaAccelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    sampler = _FixedCudaMaskSampler(masks, num_channels=channels)
    trainer = Trainer.__new__(Trainer)
    trainer.grad_accumulation_steps = gradient_accumulation_steps
    trainer.accelerator = accelerator
    trainer.model = candidate
    trainer._unwrapped_model = sampler
    optimizer_steps = 0
    yielded_denominators = []
    yielded_masks = []

    autocast = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else contextlib.nullcontext()
    )
    for batch, mask, loss_scale, denominator, is_boundary in trainer._iter_global_masked_mean_batches(batches):
        # The final production iterator yields host batches; tolerate the
        # pre-fix GPU-yielding form so this test also diagnoses its residency.
        if batch["mel"].device.type != "cuda":
            batch = send_to_device(batch, accelerator.device)
        yielded_denominators.append(denominator)
        yielded_masks.append(mask.detach().cpu())
        with trainer._accumulation_context(is_boundary), autocast:
            x = batch["mel"].permute(0, 2, 1)
            prediction = candidate(x)
            residual = prediction.float() - batch["target"].float()
            loss_sum = (residual.square() * mask[..., None]).sum()
            accelerator.backward(loss_sum * loss_scale)
            if accelerator.sync_gradients:
                candidate_optimizer.step()
                candidate_optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

    reference_loss_sum = torch.zeros((), device="cuda", dtype=torch.float32)
    denominator_value = 0
    with autocast:
        for batch, mask in zip(batches, masks):
            batch = send_to_device(batch, accelerator.device)
            prediction = reference(batch["mel"].permute(0, 2, 1))
            residual = prediction.float() - batch["target"].float()
            reference_loss_sum = reference_loss_sum + (residual.square() * mask.to(device="cuda")[..., None]).sum()
            denominator_value += int(mask.sum()) * channels
    (reference_loss_sum / denominator_value).backward()
    reference_optimizer.step()
    torch.cuda.synchronize()

    return {
        "candidate": candidate,
        "reference": reference,
        "optimizer_steps": optimizer_steps,
        "denominators": yielded_denominators,
        "yielded_masks": yielded_masks,
        "input_masks": masks,
        "reduced_dtypes": accelerator.reduced_dtypes,
    }


@pytest.mark.parametrize(
    "mask_counts,gradient_accumulation_steps",
    [
        ([1, 7, 2], 3),
        ([0, 5], 4),
    ],
    ids=["full-window", "partial-window"],
)
def test_cuda_fp32_full_and_partial_updates_match_global_reference(mask_counts, gradient_accumulation_steps):
    result = _cuda_candidate_and_reference(
        mask_counts,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    torch.testing.assert_close(result["candidate"].weight, result["reference"].weight, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(result["candidate"].bias, result["reference"].bias, atol=2e-6, rtol=2e-6)
    assert result["optimizer_steps"] == 1
    assert all(
        denominator.dtype == torch.int64 and denominator.device.type == "cuda" for denominator in result["denominators"]
    )
    assert result["reduced_dtypes"] == [torch.int64]
    for yielded, expected in zip(result["yielded_masks"], result["input_masks"]):
        assert torch.equal(yielded, expected)


@pytest.mark.skipif(not torch.cuda.is_bf16_supported(), reason="CUDA BF16 unavailable")
@pytest.mark.parametrize(
    "mask_counts,gradient_accumulation_steps",
    [
        ([1, 7, 2], 3),
        ([0, 5], 4),
    ],
    ids=["full-window", "partial-window"],
)
def test_cuda_bf16_full_and_partial_updates_match_global_reference(mask_counts, gradient_accumulation_steps):
    result = _cuda_candidate_and_reference(
        mask_counts,
        gradient_accumulation_steps=gradient_accumulation_steps,
        autocast_dtype=torch.bfloat16,
    )

    torch.testing.assert_close(result["candidate"].weight, result["reference"].weight, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(result["candidate"].bias, result["reference"].bias, atol=3e-5, rtol=3e-5)
    assert result["optimizer_steps"] == 1


class _DummyMelSpec(nn.Module):
    n_mel_channels = 2
    target_sample_rate = 24_000
    hop_length = 256

    def forward(self, wav):  # pragma: no cover - validation passes mel tensors
        raise AssertionError("audio frontend must not run")


class _CompiledTinyTransformer(nn.Module):
    dim = 2

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2)

    def forward(self, *, x, **kwargs):
        return self.projection(x)


def _compiled_cfm():
    return CFM(
        _CompiledTinyTransformer(),
        num_channels=2,
        mel_spec_module=_DummyMelSpec(),
        audio_drop_prob=0,
        cond_drop_prob=0,
    ).cuda()


def _fixed_core_args(seed, mask):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    batch, seq_len = mask.shape
    x1 = torch.randn((batch, seq_len, 2), device="cuda", generator=generator)
    text = torch.zeros((batch, 2), device="cuda", dtype=torch.long)
    valid_mask = torch.ones((batch, seq_len), device="cuda", dtype=torch.bool)
    x0 = torch.randn(x1.shape, device="cuda", generator=generator)
    time = torch.rand((batch,), device="cuda", generator=generator)
    return x1, text, valid_mask, mask, x0, time, False, False


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_cuda_inductor_compiled_core_uses_production_window_scale_before_backward():
    masks = [
        torch.tensor([[True, False, True, False, False, False]], device="cuda"),
        torch.tensor([[False, True, True, True, False, False]], device="cuda"),
    ]
    compiled = _compiled_cfm()
    eager = _compiled_cfm()
    eager.load_state_dict(compiled.state_dict())
    compiled.compile_training_core(
        backend="inductor",
        fullgraph=True,
        dynamic=True,
        runtime_fallback=False,
    )
    compiled_optimizer = torch.optim.SGD(compiled.parameters(), lr=0.02)
    eager_optimizer = torch.optim.SGD(eager.parameters(), lr=0.02)
    accelerator = _CudaAccelerator(gradient_accumulation_steps=2)
    trainer = Trainer.__new__(Trainer)
    trainer.grad_accumulation_steps = 2
    trainer.accelerator = accelerator
    trainer.model = compiled
    trainer._unwrapped_model = _FixedCudaMaskSampler(
        [mask.cpu() for mask in masks],
        num_channels=2,
    )
    host_batches = [
        {
            "mel": torch.zeros((1, 2, 6)),
            "mel_lengths": torch.tensor([6]),
            "text": ["compile"],
        }
        for _ in masks
    ]
    entries = list(trainer._iter_global_masked_mean_batches(host_batches))
    denominator = int(sum(int(mask.sum()) * 2 for mask in masks))
    eager_total = torch.zeros((), device="cuda")

    for index, (_, yielded_mask, loss_scale, yielded_denominator, is_boundary) in enumerate(entries):
        args = _fixed_core_args(500 + index, yielded_mask)
        _, compiled_sum, _, _, _ = compiled._run_loss_core_components(*args)
        with trainer._accumulation_context(is_boundary):
            accelerator.backward(compiled_sum * loss_scale)
            if accelerator.sync_gradients:
                compiled_optimizer.step()
        _, eager_sum, _, _, _ = eager._forward_loss_core_components(*args)
        eager_total = eager_total + eager_sum
        assert yielded_denominator.dtype == torch.int64
        assert int(yielded_denominator) == denominator

    (eager_total / denominator).backward()
    eager_optimizer.step()
    torch.cuda.synchronize()

    for compiled_parameter, eager_parameter in zip(compiled.parameters(), eager.parameters()):
        torch.testing.assert_close(compiled_parameter, eager_parameter, atol=2e-5, rtol=2e-5)
    assert compiled.training_compile_state["enabled"] is True
    assert compiled.training_compile_state["fallback_active"] is False


def test_cuda_window_peak_residency_is_bounded_to_one_transferred_mel_batch():
    channels = 32
    frames = 300_000
    accumulation_steps = 3
    mel_bytes = channels * frames * torch.tensor([], dtype=torch.float32).element_size()
    host_batches = [
        {
            "mel": torch.zeros((1, channels, frames), dtype=torch.float32),
            "mel_lengths": torch.tensor([frames]),
            "text": ["memory"],
        }
        for _ in range(accumulation_steps)
    ]
    masks = [torch.ones((1, frames), dtype=torch.bool) for _ in range(accumulation_steps)]
    accelerator = _CudaAccelerator(gradient_accumulation_steps=accumulation_steps)
    trainer = Trainer.__new__(Trainer)
    trainer.grad_accumulation_steps = accumulation_steps
    trainer.accelerator = accelerator
    trainer.model = nn.Identity().cuda()
    trainer._unwrapped_model = _FixedCudaMaskSampler(masks, num_channels=channels)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for batch, _, _, _, _ in trainer._iter_global_masked_mean_batches(host_batches):
        if batch["mel"].device.type != "cuda":
            batch = send_to_device(batch, accelerator.device)
        assert batch["mel"].device.type == "cuda"
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - baseline
    print(f"cuda_peak_delta_bytes={peak_delta} mel_batch_bytes={mel_bytes} peak_batches={peak_delta / mel_bytes:.3f}")

    # Masks/scalars add less than 1 MiB. A 25% allowance still rejects the
    # approximately 2x peak caused by allocating the next device batch while
    # the caller retains the previous one.
    assert peak_delta <= int(mel_bytes * 1.25), (
        f"peak CUDA allocation grew by {peak_delta} bytes for a {mel_bytes}-byte mel batch; "
        "more than one transferred batch was simultaneously resident"
    )
