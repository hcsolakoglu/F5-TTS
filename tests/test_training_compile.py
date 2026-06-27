import copy
import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
import torch._dynamo as dynamo
import yaml

from f5_tts.model import CFM, DiT, UNetT
from f5_tts.train.finetune_cli import parse_args


ROOT = Path(__file__).resolve().parents[1]
PreparedArgs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]

CUDA_INDUCTOR_EQUIVALENCE_KWARGS = [
    pytest.param({"fullgraph": False, "dynamic": None}, id="default_autodynamic"),
    pytest.param({"fullgraph": True, "dynamic": None}, id="fullgraph_autodynamic"),
    pytest.param({"fullgraph": True, "dynamic": True}, id="fullgraph_dynamic"),
]


def _build_model(
    *,
    audio_drop_prob=0.0,
    cond_drop_prob=0.0,
    vocab_size=32,
    # Deterministic tests must pin dropout=0.0 so eager-vs-compiled numerical
    # equality does not silently rely on eval() disabling nn.Dropout; this also
    # keeps assertions valid if a future change routes the loss core through
    # train() mode. Production dropout defaults are untouched.
    dropout=0.0,
    # Real production configs exercise compile-sensitive paths the tiny default
    # does not: ConvNeXt text stem, partial rotary heads, and unmasked text pad.
    conv_layers=0,
    pe_attn_head=None,
    text_mask_padding=True,
    average_upsampling=False,
):
    model = CFM(
        transformer=DiT(
            dim=32,
            depth=1,
            heads=2,
            dim_head=16,
            mel_dim=8,
            text_num_embeds=vocab_size,
            text_dim=16,
            dropout=dropout,
            conv_layers=conv_layers,
            pe_attn_head=pe_attn_head,
            text_mask_padding=text_mask_padding,
            text_embedding_average_upsampling=average_upsampling,
        ),
        mel_spec_kwargs={"n_mel_channels": 8},
        audio_drop_prob=audio_drop_prob,
        cond_drop_prob=cond_drop_prob,
    ).cpu()
    model.eval()
    return model


def _build_real_config_model(**kwargs):
    """Small model that still exercises the production DiT config path."""
    return _build_model(
        conv_layers=4,
        pe_attn_head=1,
        text_mask_padding=False,
        **kwargs,
    )


def _build_unett_model():
    model = CFM(
        transformer=UNetT(
            dim=32,
            depth=2,
            heads=2,
            dim_head=16,
            mel_dim=8,
            text_num_embeds=32,
            text_dim=16,
            dropout=0.0,
            conv_layers=0,
        ),
        mel_spec_kwargs={"n_mel_channels": 8},
        audio_drop_prob=0.0,
        cond_drop_prob=0.0,
    ).cpu()
    model.eval()
    return model


class _ZeroTransformer(torch.nn.Module):
    """Minimal CFM-compatible transformer for loss dtype/overflow tests."""

    dim = 2

    def forward(self, *, x, **_kwargs):
        return torch.zeros_like(x)


class _LearnedConstantTransformer(torch.nn.Module):
    """CFM-compatible transformer with one trainable scalar for AMP gradient checks."""

    dim = 2

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, *, x, **_kwargs):
        return self.bias.to(device=x.device, dtype=x.dtype) * torch.ones_like(x)


def _sample_batch(batch_size=2, frames=12, text_len=7, vocab_size=32, lens=None):
    torch.manual_seed(1234)
    mel = torch.randn(batch_size, frames, 8)
    text = torch.randint(0, vocab_size, (batch_size, text_len))
    if lens is None:
        lens_tensor = torch.tensor([frames] * batch_size, dtype=torch.long)
    else:
        lens_tensor = torch.tensor(lens, dtype=torch.long)
    for index, valid_len in enumerate(lens_tensor.tolist()):
        if valid_len < frames:
            mel[index, valid_len:] = 0.0
    return mel, text, lens_tensor


def _assert_close(actual, expected, name, *, atol=1e-5, rtol=1e-5):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        diff = (actual - expected).abs().max().item()
        raise AssertionError(f"{name} mismatch: max_diff={diff}")


def test_forward_api_remains_tuple_and_state_dict_stays_clean():
    model = _build_model()
    mel, text, lens = _sample_batch()

    out = model(mel, text=text, lens=lens)

    assert isinstance(out, tuple)
    assert len(out) == 3
    loss, cond, pred = out
    assert loss.ndim == 0
    assert cond.shape == mel.shape
    assert pred.shape == mel.shape

    state_dict_keys_before = set(model.state_dict().keys())
    model.compile_training_core(backend="eager", fullgraph=False, dynamic=None)
    state_dict_keys_after = set(model.state_dict().keys())

    assert state_dict_keys_after == state_dict_keys_before
    assert not any("_orig_mod" in key or "compile" in key or "compiled" in key for key in state_dict_keys_after)


def test_compiled_loss_core_matches_eager_loss_outputs_and_gradients():
    eager_model = _build_model()
    compiled_model = copy.deepcopy(eager_model)
    mel, text, lens = _sample_batch(batch_size=3, frames=12, text_len=7, lens=[12, 8, 5])

    prepared_args = cast(PreparedArgs, eager_model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))

    eager_loss, eager_cond, eager_pred = eager_model._forward_loss_core(*prepared_args)
    eager_loss.backward()
    eager_grads = [
        param.grad.detach().clone() if param.grad is not None else None for param in eager_model.parameters()
    ]

    compiled_model.compile_training_core(backend="eager", fullgraph=False, dynamic=None)
    compiled_args = cast(
        PreparedArgs,
        tuple(arg.detach().clone() if torch.is_tensor(arg) else arg for arg in prepared_args),
    )
    compiled_loss, compiled_cond, compiled_pred = compiled_model._run_loss_core(*compiled_args)
    compiled_loss.backward()
    compiled_grads = [
        param.grad.detach().clone() if param.grad is not None else None for param in compiled_model.parameters()
    ]

    _assert_close(compiled_loss.detach(), eager_loss.detach(), "loss")
    _assert_close(compiled_cond.detach(), eager_cond.detach(), "cond")
    _assert_close(compiled_pred.detach(), eager_pred.detach(), "pred")
    assert torch.count_nonzero(compiled_cond[1, 8:]) == 0
    assert torch.count_nonzero(compiled_cond[2, 5:]) == 0
    for index, (compiled_grad, eager_grad) in enumerate(zip(compiled_grads, eager_grads, strict=True)):
        if compiled_grad is None or eager_grad is None:
            assert compiled_grad is eager_grad, f"gradient None mismatch at parameter {index}"
        else:
            _assert_close(compiled_grad, eager_grad, f"grad_{index}", atol=1e-4, rtol=1e-4)


def test_loss_core_components_preserve_public_forward_contract():
    model = _build_model()
    mel, text, lens = _sample_batch(batch_size=2, frames=12, text_len=7, lens=[12, 8])
    prepared_args = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))

    loss, loss_sum, denom, cond, pred = model._forward_loss_core_components(*prepared_args)
    public_loss, public_cond, public_pred = model._forward_loss_core(*prepared_args)

    assert loss.ndim == 0
    assert loss_sum.ndim == 0
    assert denom.ndim == 0
    assert denom.item() > 0
    _assert_close(loss.detach(), (loss_sum / denom).detach(), "component_loss")
    _assert_close(public_loss.detach(), loss.detach(), "public_loss")
    _assert_close(public_cond.detach(), cond.detach(), "public_cond")
    _assert_close(public_pred.detach(), pred.detach(), "public_pred")

    model.compile_training_core(backend="eager", fullgraph=False, dynamic=None)
    compiled_loss, compiled_loss_sum, compiled_denom, compiled_cond, compiled_pred = model._run_loss_core_components(
        *prepared_args
    )
    _assert_close(compiled_loss.detach(), loss.detach(), "compiled_component_loss")
    _assert_close(compiled_loss_sum.detach(), loss_sum.detach(), "compiled_component_loss_sum")
    _assert_close(compiled_denom.detach(), denom.detach(), "compiled_component_denom")
    _assert_close(compiled_cond.detach(), cond.detach(), "compiled_component_cond")
    _assert_close(compiled_pred.detach(), pred.detach(), "compiled_component_pred")


def test_loss_core_accumulates_mse_in_fp32_for_half_precision_inputs():
    model = CFM(
        transformer=_ZeroTransformer(),
        mel_spec_kwargs={"n_mel_channels": 2},
        audio_drop_prob=0.0,
        cond_drop_prob=0.0,
    )
    x1 = torch.full((1, 4, 2), 400.0, dtype=torch.float16)
    x0 = torch.zeros_like(x1)
    text = torch.zeros((1, 1), dtype=torch.long)
    mask = torch.ones((1, 4), dtype=torch.bool)
    rand_span_mask = torch.ones((1, 4), dtype=torch.bool)
    time = torch.zeros((1,), dtype=torch.float16)

    loss, loss_sum, denom, _, pred = model._forward_loss_core_components(
        x1, text, mask, rand_span_mask, x0, time, False, False
    )

    assert pred.dtype == torch.float16
    assert loss.dtype == torch.float32
    assert loss_sum.dtype == torch.float32
    assert denom.dtype == torch.float32
    assert torch.isfinite(loss_sum)
    assert torch.isfinite(loss)
    assert loss_sum.item() == pytest.approx(1_280_000.0)
    assert denom.item() == pytest.approx(8.0)
    assert loss.item() == pytest.approx(160_000.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fp16 AMP GradScaler probe")
def test_loss_sum_fp16_amp_gradscaler_stays_finite_for_large_errors():
    device = torch.device("cuda")
    model = CFM(
        transformer=_LearnedConstantTransformer(),
        mel_spec_kwargs={"n_mel_channels": 2},
        audio_drop_prob=0.0,
        cond_drop_prob=0.0,
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    # Keep the scale at 1.0 to isolate the fp32 loss-sum safety property. The
    # default GradScaler scale can legitimately overflow very large gradients
    # before it backs off, which is separate from loss_sum overflowing in fp16.
    scaler = getattr(torch.amp, "GradScaler")("cuda", init_scale=1.0)

    x1 = torch.full((1, 4, 2), 400.0, device=device, dtype=torch.float16)
    x0 = torch.zeros_like(x1)
    text = torch.zeros((1, 1), device=device, dtype=torch.long)
    mask = torch.ones((1, 4), device=device, dtype=torch.bool)
    rand_span_mask = torch.ones((1, 4), device=device, dtype=torch.bool)
    time = torch.zeros((1,), device=device, dtype=torch.float16)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        loss, loss_sum, denom, _, _ = model._forward_loss_core_components(
            x1, text, mask, rand_span_mask, x0, time, False, False
        )

    assert loss.dtype == torch.float32
    assert loss_sum.dtype == torch.float32
    assert denom.dtype == torch.float32
    assert torch.isfinite(loss_sum)
    assert torch.isfinite(loss)
    scaler.scale(loss_sum).backward()
    scaler.unscale_(optimizer)

    bias = cast(Any, model.transformer).bias
    assert bias.grad is not None
    assert torch.isfinite(bias.grad)


def _toy_microbatch(n_frames: int, n_masked: int, dim: int = 4):
    x = torch.randn(1, n_frames, dim)
    target = torch.randn(1, n_frames, dim)
    mask = torch.zeros(1, n_frames, dtype=torch.bool)
    mask[:, :n_masked] = True
    return x, target, mask


def _toy_loss_components(model: torch.nn.Module, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    pred = model(x)
    err = torch.nn.functional.mse_loss(pred, target, reduction="none")
    loss_mask = mask[..., None].to(err.dtype)
    loss_sum = (err * loss_mask).sum()
    denom = (loss_mask.sum() * err.shape[-1]).clamp(min=1.0)
    return loss_sum, denom


class _FakeReduceAccelerator:
    def __init__(self, *, num_processes: int, reduced_denom: torch.Tensor | None = None):
        self.num_processes = num_processes
        self._reduced_denom = reduced_denom

    def reduce(self, tensor: torch.Tensor, reduction: str = "sum"):
        assert reduction == "sum"
        return tensor if self._reduced_denom is None else self._reduced_denom.to(tensor)


def test_loss_sum_gradient_scaling_matches_global_masked_mean_for_accumulation():
    from f5_tts.model.trainer import Trainer

    torch.manual_seed(0)
    accumulation_steps = 2
    current = torch.nn.Linear(4, 4, bias=False)
    correct = copy.deepcopy(current)
    microbatches = [_toy_microbatch(10, 8), _toy_microbatch(4, 3)]

    local_denom = torch.zeros(())
    for x, target, mask in microbatches:
        loss_sum, denom = _toy_loss_components(current, x, target, mask)
        local_denom = local_denom + denom.detach()
        # Mirrors Accelerator.backward(loss_sum), which divides by accumulation_steps.
        (loss_sum / accumulation_steps).backward()

    trainer = Trainer.__new__(Trainer)
    trainer.model = cast(Any, current)
    trainer.grad_accumulation_steps = accumulation_steps
    trainer.accelerator = cast(Any, _FakeReduceAccelerator(num_processes=1))
    global_denom = trainer._global_loss_denom(local_denom)
    trainer._scale_gradients_by_loss_denom(global_denom)

    total_loss_sum = torch.zeros(())
    total_denom = torch.zeros(())
    for x, target, mask in microbatches:
        loss_sum, denom = _toy_loss_components(correct, x, target, mask)
        total_loss_sum = total_loss_sum + loss_sum
        total_denom = total_denom + denom.detach()
    (total_loss_sum / total_denom).backward()

    for current_param, correct_param in zip(current.parameters(), correct.parameters(), strict=True):
        assert current_param.grad is not None
        assert correct_param.grad is not None
        _assert_close(current_param.grad, correct_param.grad, "accumulated_grad", atol=1e-6, rtol=1e-6)


def test_loss_sum_gradient_scaling_accounts_for_ddp_gradient_average():
    from f5_tts.model.trainer import Trainer

    torch.manual_seed(1)
    rank0 = torch.nn.Linear(4, 4, bias=False)
    rank1 = copy.deepcopy(rank0)
    current = copy.deepcopy(rank0)
    correct = copy.deepcopy(rank0)
    rank_batches = [_toy_microbatch(10, 8), _toy_microbatch(4, 3)]

    rank_denoms = []
    rank_grads = []
    for rank_model, batch in zip((rank0, rank1), rank_batches, strict=True):
        loss_sum, denom = _toy_loss_components(rank_model, *batch)
        rank_denoms.append(denom.detach())
        loss_sum.backward()
        rank_grads.append([param.grad.detach().clone() for param in rank_model.parameters()])

    # Simulate the gradient buffer after DDP has averaged raw loss_sum gradients.
    for param_index, current_param in enumerate(current.parameters()):
        current_param.grad = (rank_grads[0][param_index] + rank_grads[1][param_index]) / 2

    global_denom = rank_denoms[0] + rank_denoms[1]
    trainer = Trainer.__new__(Trainer)
    trainer.model = cast(Any, current)
    trainer.grad_accumulation_steps = 1
    trainer.accelerator = cast(Any, _FakeReduceAccelerator(num_processes=2, reduced_denom=global_denom))
    reduced_denom = trainer._global_loss_denom(rank_denoms[0])
    trainer._scale_gradients_by_loss_denom(reduced_denom)

    total_loss_sum = torch.zeros(())
    for batch in rank_batches:
        loss_sum, _ = _toy_loss_components(correct, *batch)
        total_loss_sum = total_loss_sum + loss_sum
    (total_loss_sum / global_denom).backward()

    for current_param, correct_param in zip(current.parameters(), correct.parameters(), strict=True):
        assert current_param.grad is not None
        assert correct_param.grad is not None
        _assert_close(current_param.grad, correct_param.grad, "ddp_scaled_grad", atol=1e-6, rtol=1e-6)


def test_compiled_loss_core_handles_cfg_branches_and_empty_mask():
    model = _build_model()
    model.compile_training_core(backend="eager", fullgraph=False, dynamic=None)
    mel, text, lens = _sample_batch()
    x1, text_tensor, mask, rand_span_mask, x0, time, _, _ = model._prepare_training_inputs(
        mel.clone(), text.clone(), lens.clone()
    )

    for drop_audio_cond, drop_text in ((False, False), (True, False), (True, True)):
        loss, cond, pred = model._run_loss_core(
            x1, text_tensor, mask, rand_span_mask, x0, time, drop_audio_cond, drop_text
        )
        assert torch.isfinite(loss)
        assert cond.shape == mel.shape
        assert pred.shape == mel.shape

    rand_span_mask.zero_()
    loss, _, _ = model._forward_loss_core(x1, text_tensor, mask, rand_span_mask, x0, time, False, False)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_fullgraph_compile_handles_ragged_lens_without_text_embedding_graph_break():
    model = _build_model()
    mel, text, lens = _sample_batch(batch_size=2, frames=12, text_len=7, lens=[12, 8])
    prepared_args = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))

    model.compile_training_core(backend="eager", fullgraph=True, dynamic=None)
    loss, cond, pred = model._run_loss_core(*prepared_args)
    loss.backward()

    dynamo.reset()
    explanation = dynamo.explain(model._forward_loss_core)(*prepared_args)

    assert torch.isfinite(loss)
    assert cond.shape == mel.shape
    assert pred.shape == mel.shape
    assert explanation.graph_break_count == 0


def test_real_config_compiled_loss_core_matches_eager():
    """CPU parity for production DiT knobs that the tiny default model misses."""
    eager_model = _build_real_config_model()
    compiled_model = copy.deepcopy(eager_model)
    mel, text, lens = _sample_batch(batch_size=2, frames=12, text_len=7, lens=[12, 8])

    prepared_args = cast(PreparedArgs, eager_model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    eager_loss, eager_cond, eager_pred = eager_model._forward_loss_core(*prepared_args)
    eager_loss.backward()
    eager_grads = [
        param.grad.detach().clone() if param.grad is not None else None for param in eager_model.parameters()
    ]

    compiled_model.compile_training_core(backend="eager", fullgraph=True, dynamic=None)
    compiled_args = cast(
        PreparedArgs,
        tuple(arg.detach().clone() if torch.is_tensor(arg) else arg for arg in prepared_args),
    )
    compiled_loss, compiled_cond, compiled_pred = compiled_model._run_loss_core(*compiled_args)
    compiled_loss.backward()
    compiled_grads = [
        param.grad.detach().clone() if param.grad is not None else None for param in compiled_model.parameters()
    ]

    _assert_close(compiled_loss.detach(), eager_loss.detach(), "real_config_loss")
    _assert_close(compiled_cond.detach(), eager_cond.detach(), "real_config_cond")
    _assert_close(compiled_pred.detach(), eager_pred.detach(), "real_config_pred")
    for index, (compiled_grad, eager_grad) in enumerate(zip(compiled_grads, eager_grads, strict=True)):
        if compiled_grad is None or eager_grad is None:
            assert compiled_grad is eager_grad, f"real_config gradient None mismatch at parameter {index}"
        else:
            _assert_close(compiled_grad, eager_grad, f"real_config_grad_{index}", atol=1e-4, rtol=1e-4)


def test_unett_compiled_loss_core_matches_eager():
    """CPU parity for E2TTS/UNetT, whose text embedding path differs from DiT."""
    eager_model = _build_unett_model()
    compiled_model = copy.deepcopy(eager_model)
    mel, text, lens = _sample_batch(batch_size=2, frames=12, text_len=7, lens=[12, 8])

    prepared_args = cast(PreparedArgs, eager_model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    eager_loss, eager_cond, eager_pred = eager_model._forward_loss_core(*prepared_args)
    eager_loss.backward()
    eager_grads = [
        param.grad.detach().clone() if param.grad is not None else None for param in eager_model.parameters()
    ]

    compiled_model.compile_training_core(backend="eager", fullgraph=True, dynamic=None)
    compiled_args = cast(
        PreparedArgs,
        tuple(arg.detach().clone() if torch.is_tensor(arg) else arg for arg in prepared_args),
    )
    compiled_loss, compiled_cond, compiled_pred = compiled_model._run_loss_core(*compiled_args)
    compiled_loss.backward()
    compiled_grads = [
        param.grad.detach().clone() if param.grad is not None else None for param in compiled_model.parameters()
    ]

    _assert_close(compiled_loss.detach(), eager_loss.detach(), "unett_loss")
    _assert_close(compiled_cond.detach(), eager_cond.detach(), "unett_cond")
    _assert_close(compiled_pred.detach(), eager_pred.detach(), "unett_pred")
    for index, (compiled_grad, eager_grad) in enumerate(zip(compiled_grads, eager_grads, strict=True)):
        if compiled_grad is None or eager_grad is None:
            assert compiled_grad is eager_grad, f"unett gradient None mismatch at parameter {index}"
        else:
            _assert_close(compiled_grad, eager_grad, f"unett_grad_{index}", atol=1e-4, rtol=1e-4)


def test_fullgraph_real_config_no_graph_break():
    """Graph-break regression for conv_layers=4, pe_attn_head=1, ragged lens."""
    model = _build_real_config_model()
    mel, text, lens = _sample_batch(batch_size=2, frames=12, text_len=7, lens=[12, 8])
    prepared_args = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))

    model.compile_training_core(backend="eager", fullgraph=True, dynamic=None)
    loss, cond, pred = model._run_loss_core(*prepared_args)
    loss.backward()

    dynamo.reset()
    explanation = dynamo.explain(model._forward_loss_core)(*prepared_args)

    assert torch.isfinite(loss)
    assert cond.shape == mel.shape
    assert pred.shape == mel.shape
    assert explanation.graph_break_count == 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for GPU real-config torch.compile smoke test"
)
def test_cuda_inductor_real_config_smoke():
    device = torch.device("cuda")
    model = _build_real_config_model().to(device)
    mel, text, _ = _sample_batch(batch_size=2, frames=12, text_len=7)
    lens = torch.tensor([12, 8], dtype=torch.long)
    mel = mel.to(device)
    text = text.to(device)
    lens = lens.to(device)

    model.compile_training_core(backend="inductor", fullgraph=True, dynamic=None)
    loss, cond, pred = model(mel, text=text, lens=lens)
    loss.backward()
    torch.cuda.synchronize(device)

    assert torch.isfinite(loss)
    assert cond.shape == mel.shape
    assert pred.shape == mel.shape
    assert model.training_compile_state == {"enabled": True, "fallback_active": False, "error": None}


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for GPU real-config inductor equivalence test"
)
@pytest.mark.parametrize("compile_kwargs", CUDA_INDUCTOR_EQUIVALENCE_KWARGS)
def test_cuda_inductor_real_config_matches_eager_across_compile_knobs(compile_kwargs):
    """CUDA inductor vs eager numerical parity for real-config knobs and compile knobs."""
    device = torch.device("cuda")
    eager_model = _build_real_config_model().to(device)
    compiled_model = copy.deepcopy(eager_model)
    mel, text, lens = _sample_batch(batch_size=2, frames=12, text_len=7, lens=[12, 8])
    mel = mel.to(device)
    text = text.to(device)
    lens = lens.to(device)

    prepared_args = cast(PreparedArgs, eager_model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    eager_loss, eager_cond, eager_pred = eager_model._forward_loss_core(*prepared_args)
    eager_loss.backward()
    eager_grads = [
        param.grad.detach().clone() if param.grad is not None else None for param in eager_model.parameters()
    ]

    compiled_model.compile_training_core(backend="inductor", **compile_kwargs)
    compiled_args = cast(
        PreparedArgs,
        tuple(arg.detach().clone() if torch.is_tensor(arg) else arg for arg in prepared_args),
    )
    compiled_loss, compiled_cond, compiled_pred = compiled_model._run_loss_core(*compiled_args)
    compiled_loss.backward()
    torch.cuda.synchronize(device)
    compiled_grads = [
        param.grad.detach().clone() if param.grad is not None else None for param in compiled_model.parameters()
    ]

    _assert_close(compiled_loss.detach(), eager_loss.detach(), "cuda_real_config_loss", atol=1e-4, rtol=1e-4)
    _assert_close(compiled_cond.detach(), eager_cond.detach(), "cuda_real_config_cond", atol=1e-4, rtol=1e-4)
    _assert_close(compiled_pred.detach(), eager_pred.detach(), "cuda_real_config_pred", atol=1e-3, rtol=1e-3)
    for index, (compiled_grad, eager_grad) in enumerate(zip(compiled_grads, eager_grads, strict=True)):
        if compiled_grad is None or eager_grad is None:
            assert compiled_grad is eager_grad, f"cuda real_config gradient None mismatch at parameter {index}"
        else:
            _assert_close(compiled_grad, eager_grad, f"cuda_real_config_grad_{index}", atol=1e-3, rtol=1e-3)
    assert compiled_model.training_compile_state == {"enabled": True, "fallback_active": False, "error": None}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU torch.compile smoke test")
def test_cuda_inductor_training_loss_core_smoke():
    device = torch.device("cuda")
    model = _build_model().to(device)
    mel, text, _ = _sample_batch(batch_size=2, frames=12, text_len=7)
    lens = torch.tensor([12, 8], dtype=torch.long)
    mel = mel.to(device)
    text = text.to(device)
    lens = lens.to(device)

    model.compile_training_core(backend="inductor", fullgraph=True, dynamic=None)
    loss, cond, pred = model(mel, text=text, lens=lens)
    loss.backward()
    torch.cuda.synchronize(device)

    assert torch.isfinite(loss)
    assert cond.shape == mel.shape
    assert pred.shape == mel.shape
    assert model.training_compile_state == {"enabled": True, "fallback_active": False, "error": None}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU inductor equivalence test")
@pytest.mark.parametrize("compile_kwargs", CUDA_INDUCTOR_EQUIVALENCE_KWARGS)
def test_cuda_inductor_matches_eager_loss_outputs_and_gradients_across_compile_knobs(compile_kwargs):
    device = torch.device("cuda")
    eager_model = _build_model().to(device)
    compiled_model = copy.deepcopy(eager_model)
    mel, text, lens = _sample_batch(batch_size=2, frames=12, text_len=7, lens=[12, 8])
    mel = mel.to(device)
    text = text.to(device)
    lens = lens.to(device)

    prepared_args = cast(PreparedArgs, eager_model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    eager_loss, eager_cond, eager_pred = eager_model._forward_loss_core(*prepared_args)
    eager_loss.backward()
    eager_grads = [
        param.grad.detach().clone() if param.grad is not None else None for param in eager_model.parameters()
    ]

    compiled_model.compile_training_core(backend="inductor", **compile_kwargs)
    compiled_args = cast(
        PreparedArgs,
        tuple(arg.detach().clone() if torch.is_tensor(arg) else arg for arg in prepared_args),
    )
    compiled_loss, compiled_cond, compiled_pred = compiled_model._run_loss_core(*compiled_args)
    compiled_loss.backward()
    torch.cuda.synchronize(device)
    compiled_grads = [
        param.grad.detach().clone() if param.grad is not None else None for param in compiled_model.parameters()
    ]

    _assert_close(compiled_loss.detach(), eager_loss.detach(), "cuda_loss", atol=1e-4, rtol=1e-4)
    _assert_close(compiled_cond.detach(), eager_cond.detach(), "cuda_cond", atol=1e-4, rtol=1e-4)
    _assert_close(compiled_pred.detach(), eager_pred.detach(), "cuda_pred", atol=1e-3, rtol=1e-3)
    for index, (compiled_grad, eager_grad) in enumerate(zip(compiled_grads, eager_grads, strict=True)):
        if compiled_grad is None or eager_grad is None:
            assert compiled_grad is eager_grad, f"cuda gradient None mismatch at parameter {index}"
        else:
            _assert_close(compiled_grad, eager_grad, f"cuda_grad_{index}", atol=1e-3, rtol=1e-3)
    assert compiled_model.training_compile_state == {"enabled": True, "fallback_active": False, "error": None}


def test_runtime_fallback_can_be_enabled_or_disabled():
    model = _build_model()
    mel, text, lens = _sample_batch()
    prepared_args = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))

    def raise_compile_error(*_args):
        raise RuntimeError("synthetic compile failure")

    object.__setattr__(model, "_compiled_loss_core", raise_compile_error)
    object.__setattr__(model, "_compile_runtime_fallback", True)
    loss, _, _ = model._run_loss_core(*prepared_args)
    assert torch.isfinite(loss)
    assert model.training_compile_state["enabled"] is False
    assert model.training_compile_state["fallback_active"] is True
    assert "synthetic compile failure" in model.training_compile_state["error"]

    model = _build_model()
    prepared_args = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    object.__setattr__(model, "_compiled_loss_core", raise_compile_error)
    object.__setattr__(model, "_compile_runtime_fallback", False)
    with pytest.raises(RuntimeError, match="synthetic compile failure"):
        model._run_loss_core(*prepared_args)


def test_cuda_oom_from_compiled_core_is_not_swallowed_into_eager_fallback():
    model = _build_model()
    mel, text, lens = _sample_batch()
    prepared_args = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))

    def raise_oom(*_args):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    object.__setattr__(model, "_compiled_loss_core", raise_oom)
    object.__setattr__(model, "_compile_runtime_fallback", True)

    with pytest.raises(RuntimeError, match="ran out of GPU memory") as exc_info:
        model._run_loss_core(*prepared_args)

    assert isinstance(exc_info.value.__cause__, torch.cuda.OutOfMemoryError)
    assert model.training_compile_state["enabled"] is True
    assert model.training_compile_state["fallback_active"] is False


def test_message_based_oom_runtimeerror_also_skips_fallback():
    model = _build_model()
    mel, text, lens = _sample_batch()
    prepared_args = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))

    def raise_oom_msg(*_args):
        raise RuntimeError("cuda runtime error: out of memory")

    object.__setattr__(model, "_compiled_loss_core", raise_oom_msg)
    object.__setattr__(model, "_compile_runtime_fallback", True)

    with pytest.raises(RuntimeError, match="ran out of GPU memory"):
        model._run_loss_core(*prepared_args)
    assert model.training_compile_state["fallback_active"] is False


def test_oom_still_raises_when_runtime_fallback_disabled():
    model = _build_model()
    mel, text, lens = _sample_batch()
    prepared_args = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))

    def raise_oom(*_args):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    object.__setattr__(model, "_compiled_loss_core", raise_oom)
    object.__setattr__(model, "_compile_runtime_fallback", False)

    with pytest.raises(RuntimeError, match="ran out of GPU memory"):
        model._run_loss_core(*prepared_args)


def test_non_oom_compile_failure_still_falls_back_when_enabled():
    model = _build_model()
    mel, text, lens = _sample_batch()
    prepared_args = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))

    def raise_compile_error(*_args):
        raise RuntimeError("synthetic compile failure")

    object.__setattr__(model, "_compiled_loss_core", raise_compile_error)
    object.__setattr__(model, "_compile_runtime_fallback", True)
    loss, _, _ = model._run_loss_core(*prepared_args)
    assert torch.isfinite(loss)
    assert model.training_compile_state["fallback_active"] is True
    assert "synthetic compile failure" in model.training_compile_state["error"]


def test_trainer_materialises_loss_scalar_at_most_once_per_step():
    import inspect

    from f5_tts.model.trainer import Trainer

    source = inspect.getsource(Trainer.train)
    assert "loss_scalar = loss.item()" in source
    assert "loss=loss.item()" not in source
    assert '"loss": loss.item()' not in source
    assert 'add_scalar("loss", loss.item()' not in source


def test_trainer_uses_persistent_workers_only_when_workers_are_enabled(monkeypatch):
    from f5_tts.model import trainer as trainer_module
    from f5_tts.model.trainer import Trainer

    class StopAfterDataLoader(RuntimeError):
        pass

    class DummyAccelerator:
        even_batches = True

    class DummyBatchSampler:
        def __init__(self, *_args, **_kwargs):
            pass

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return object()

    calls = []

    def fake_dataloader(*_args, **kwargs):
        calls.append(kwargs)
        raise StopAfterDataLoader

    monkeypatch.setattr(trainer_module, "DataLoader", fake_dataloader)
    monkeypatch.setattr(trainer_module, "DynamicBatchSampler", DummyBatchSampler)

    trainer = object.__new__(Trainer)
    trainer.log_samples = False
    trainer.batch_size_per_gpu = 2
    trainer.max_samples = 2
    cast(Any, trainer).accelerator = DummyAccelerator()
    dataset = DummyDataset()

    trainer.batch_size_type = "sample"
    with pytest.raises(StopAfterDataLoader):
        trainer.train(dataset, num_workers=0)
    assert calls.pop()["persistent_workers"] is False

    trainer.batch_size_type = "frame"
    with pytest.raises(StopAfterDataLoader):
        trainer.train(dataset, num_workers=0)
    assert calls.pop()["persistent_workers"] is False

    trainer.batch_size_type = "sample"
    with pytest.raises(StopAfterDataLoader):
        trainer.train(dataset, num_workers=2)
    assert calls.pop()["persistent_workers"] is True


def test_duration_predictor_scalar_is_logged_lazily_with_main_metrics():
    import inspect

    from f5_tts.model.trainer import Trainer

    source = inspect.getsource(Trainer.train)
    assert '{"duration loss": dur_loss.item()}' not in source
    assert "duration_loss_scalar = None" in source
    assert "duration_loss_scalar = duration_loss.item()" in source
    assert 'metrics["duration loss"] = duration_loss_scalar' in source


def test_cli_compile_flags_parse():
    with patch.object(
        sys,
        "argv",
        [
            "prog",
            "--compile_enabled",
            "--compile_backend",
            "eager",
            "--compile_mode",
            "reduce-overhead",
            "--compile_fullgraph",
            "--compile_dynamic",
            "true",
            "--compile_no_fallback",
        ],
    ):
        args = parse_args()

    assert args.compile_enabled is True
    assert args.compile_backend == "eager"
    assert args.compile_mode == "reduce-overhead"
    assert args.compile_fullgraph is True
    assert args.compile_dynamic == "true"
    assert args.compile_no_fallback is True


def test_all_training_configs_define_default_off_compile_block_without_metrics():
    for config_path in sorted((ROOT / "src/f5_tts/configs").glob("*.yaml")):
        config = yaml.safe_load(config_path.read_text())
        assert "compile" in config, config_path.name
        assert "metrics" not in config, config_path.name
        assert config["compile"] == {
            "enabled": False,
            "backend": "inductor",
            "mode": None,
            "fullgraph": False,
            "dynamic": None,
            "fallback_to_eager": True,
        }


def test_compile_guard_rejects_average_upsampling_with_clear_value_error():
    model = _build_model(average_upsampling=True)
    assert cast(Any, model.transformer.text_embed).average_upsampling is True

    with pytest.raises(ValueError, match="text_embedding_average_upsampling"):
        model.compile_training_core(backend="eager", fullgraph=False, dynamic=None)

    assert model.training_compile_state == {
        "enabled": False,
        "fallback_active": False,
        "error": None,
    }


def test_compile_guard_average_upsampling_eager_forward_still_works():
    model = _build_model(average_upsampling=True)
    mel, text, lens = _sample_batch(batch_size=2, frames=12, text_len=7, lens=[12, 8])

    loss, cond, pred = model(mel, text=text, lens=lens)

    assert torch.isfinite(loss)
    assert cond.shape == mel.shape
    assert pred.shape == mel.shape
    assert model.training_compile_state["enabled"] is False


def test_compile_guard_default_off_path_still_compiles():
    model = _build_model(average_upsampling=False)
    assert cast(Any, model.transformer.text_embed).average_upsampling is False

    model.compile_training_core(backend="eager", fullgraph=False, dynamic=None)
    mel, text, lens = _sample_batch()
    loss, _, _ = model(mel, text=text, lens=lens)

    assert torch.isfinite(loss)
    assert model.training_compile_state["enabled"] is True


def test_compile_guard_trainer_fallback_to_eager_true_falls_back():
    from f5_tts.model.trainer import Trainer

    model = _build_model(average_upsampling=True)
    trainer = Trainer.__new__(Trainer)
    trainer.compile_enabled = True
    trainer.compile_backend = "eager"
    trainer.compile_mode = None
    trainer.compile_fullgraph = False
    trainer.compile_dynamic = None
    trainer.compile_fallback_to_eager = True
    trainer.compile_active = False
    trainer.compile_fallback_active = False
    trainer._unwrapped_model = model

    class _FakeAccel:
        num_processes = 1
        is_main_process = True

    trainer.accelerator = cast(Any, _FakeAccel())
    trainer._configure_compile()

    assert trainer.compile_active is False
    assert trainer.compile_fallback_active is True
    assert model.training_compile_state["enabled"] is False


def test_compile_guard_trainer_fallback_to_eager_false_raises():
    from f5_tts.model.trainer import Trainer

    model = _build_model(average_upsampling=True)
    trainer = Trainer.__new__(Trainer)
    trainer.compile_enabled = True
    trainer.compile_backend = "eager"
    trainer.compile_mode = None
    trainer.compile_fullgraph = False
    trainer.compile_dynamic = None
    trainer.compile_fallback_to_eager = False
    trainer.compile_active = False
    trainer.compile_fallback_active = False
    trainer._unwrapped_model = model

    class _FakeAccel:
        num_processes = 1
        is_main_process = True

    trainer.accelerator = cast(Any, _FakeAccel())
    with pytest.raises(ValueError, match="text_embedding_average_upsampling"):
        trainer._configure_compile()

    assert trainer.compile_active is False
    assert trainer.compile_fallback_active is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for inductor guard smoke")
def test_compile_guard_blocks_cuda_inductor_for_average_upsampling():
    device = torch.device("cuda")
    model = _build_model(average_upsampling=True).to(device)

    with pytest.raises(ValueError, match="text_embedding_average_upsampling"):
        model.compile_training_core(backend="inductor", fullgraph=True, dynamic=None)

    mel, text, lens = _sample_batch(batch_size=2, frames=12, text_len=7, lens=[12, 8])
    mel = mel.to(device)
    text = text.to(device)
    lens = lens.to(device)
    loss, _, _ = model(mel, text=text, lens=lens)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Review-agent follow-up tests: opt-in global masked-mean, AdamW fused device,
# DiT SymInt casts.
# ---------------------------------------------------------------------------


def test_trainer_global_masked_mean_defaults_false_and_gates_backward_path():
    """Default preserves old per-microbatch mean backward; opt-in enables loss_sum path."""
    import inspect

    from f5_tts.model.trainer import Trainer

    sig = inspect.signature(Trainer.__init__)
    assert sig.parameters["global_masked_mean"].default is False

    source = inspect.getsource(Trainer.train)
    # Opt-in path backprops loss_sum and rescales by global denominator.
    assert "self.accelerator.backward(loss_sum)" in source
    assert "self._scale_gradients_by_loss_denom(global_loss_denom)" in source
    # Default path backprops the per-microbatch mean loss, not loss_sum.
    assert "self.accelerator.backward(loss)" in source
    # The branch is gated on the flag, not unconditional.
    assert "if self.global_masked_mean:" in source


def test_default_loss_path_gradient_matches_average_of_means_not_global_mean():
    """With global_masked_mean=False, gradients equal backprop of per-microbatch means
    (historical average-of-means), which differs from the global masked-mean when
    masked-frame denominators differ across microbatches."""
    torch.manual_seed(7)
    dim = 4
    microbatches = [_toy_microbatch(10, 8, dim=dim), _toy_microbatch(4, 3, dim=dim)]

    # Old/historical behaviour: gradient accumulation sums per-microbatch mean gradients
    # (average-of-means). The common 1/G factor is omitted; it does not affect the inequality.
    old = torch.nn.Linear(dim, dim, bias=False)
    for x, target, mask in microbatches:
        loss_sum, denom = _toy_loss_components(old, x, target, mask)
        (loss_sum / denom).backward()

    # Global masked-mean behaviour: grad(total_loss_sum / total_denom).
    glob = copy.deepcopy(old)
    for p in glob.parameters():
        p.grad = None
    total_loss_sum = torch.zeros(())
    total_denom = torch.zeros(())
    for x, target, mask in microbatches:
        loss_sum, denom = _toy_loss_components(glob, x, target, mask)
        total_loss_sum = total_loss_sum + loss_sum
        total_denom = total_denom + denom.detach()
    (total_loss_sum / total_denom).backward()

    # The two objectives differ because denominators differ (8*dim vs 3*dim).
    for old_p, glob_p in zip(old.parameters(), glob.parameters(), strict=True):
        assert not torch.allclose(old_p.grad, glob_p.grad, atol=1e-5), (
            "average-of-means and global masked-mean gradients should differ when "
            "masked-frame denominators differ across microbatches"
        )


def test_adamw_fused_uses_accelerator_device_not_global_cuda():
    """fused must be driven by the actual accelerator device, not torch.cuda.is_available()."""
    import inspect

    from f5_tts.model.trainer import Trainer

    source = inspect.getsource(Trainer.__init__)
    assert 'self.accelerator.device.type == "cuda"' in source
    assert "fused=torch.cuda.is_available()" not in source


def test_adamw_fused_false_when_accelerator_cpu_on_cuda_host():
    """On a CUDA host with a CPU accelerator, AdamW must not request the fused kernel."""
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA-capable host to prove the CPU-path divergence")

    from accelerate import Accelerator

    # On a CUDA host the old code (torch.cuda.is_available()) would set fused=True;
    # the patched logic keys off the actual accelerator device and must yield False.
    cpu_accel = Accelerator(cpu=True)
    use_fused = cpu_accel.device.type == "cuda"
    assert use_fused is False
    assert torch.cuda.is_available() is True  # proves the two checks diverge here
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=use_fused)
    assert opt.param_groups[0].get("fused", False) is False


def test_dit_text_embed_keeps_symint_in_non_tensor_path():
    """The non-tensor seq_len path must not call int() (which specializes dynamic graphs)."""
    import inspect

    from f5_tts.model.backbones.dit import TextEmbedding

    src = inspect.getsource(TextEmbedding.forward)
    assert "int(seq_len)" not in src
    # The non-tensor branch keeps the value as-is (Python int in eager, SymInt under compile).
    assert "max_seq_len = seq_len" in src


def test_dit_text_embed_non_tensor_seq_len_still_masks_correctly():
    """Behavioral check: int seq_len path produces correct valid-position masking (eager)."""
    from f5_tts.model.backbones.dit import TextEmbedding

    embed = TextEmbedding(text_num_embeds=32, text_dim=8, mask_padding=False)
    text = torch.randint(1, 32, (2, 20))
    # Per-sample valid lengths 7 and 4; seq_len is a plain Python int (max mel frames).
    out = embed(text, seq_len=7, drop_text=False, valid_seq_lens=torch.tensor([7, 4]))
    assert out.shape == (2, 7, 8)
    # Sample 0 is fully valid (len 7); sample 1 valid only up to position 4.
    assert torch.any(out[0, 4:7] != 0)
    assert torch.all(out[1, 4:7] == 0)
    assert torch.any(out[0, :4] != 0)


def test_cli_global_masked_mean_flag_parses():
    with patch.object(sys, "argv", ["prog", "--global_masked_mean"]):
        args = parse_args()
    assert args.global_masked_mean is True

    with patch.object(sys, "argv", ["prog"]):
        args = parse_args()
    assert args.global_masked_mean is False


def test_all_training_configs_define_global_masked_mean_default_false():
    for config_path in sorted((ROOT / "src/f5_tts/configs").glob("*.yaml")):
        config = yaml.safe_load(config_path.read_text())
        assert "optim" in config, config_path.name
        assert config["optim"].get("global_masked_mean") is False, config_path.name
