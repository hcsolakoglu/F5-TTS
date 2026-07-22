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
PreparedArgs = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    bool | torch.Tensor,
    bool | torch.Tensor,
]

CUDA_INDUCTOR_EQUIVALENCE_KWARGS = [
    pytest.param({"fullgraph": False, "dynamic": None}, id="default_autodynamic"),
    pytest.param({"fullgraph": True, "dynamic": None}, id="fullgraph_autodynamic"),
    pytest.param({"fullgraph": True, "dynamic": True}, id="fullgraph_dynamic"),
]


def _synthetic_compiler_error(message="synthetic compile failure"):
    """Build a realistic torch.compile backend failure.

    The runtime fallback only catches compiler-raised exception types. Simulating a
    compile failure with a bare RuntimeError would test a code path that cannot occur
    in production and would hide the fact that ordinary model errors must propagate
    (see test_model_error_is_not_swallowed_by_compile_fallback).
    """
    from torch._dynamo.exc import BackendCompilerFailed

    return BackendCompilerFailed(lambda: None, RuntimeError(message), None)


def _randomize_zero_init_(model, *, seed=0, std=0.05):
    """Move a freshly built DiT out of its identity-initialized state.

    DiT.initialize_weights zero-initializes the AdaLN gates (`block.attn_norm.linear`)
    and the output projection. That is correct for training -- the residual branches start
    as no-ops -- but it makes an as-built model useless for eager-vs-compiled comparison:
    `pred` is identically zero and only proj_out receives a gradient (2 of 29 parameters on
    the tiny test model), so a completely broken attention or feed-forward inside the
    compiled region still compares equal. Every parity test must run through this first,
    and then assert non-vacuity with _assert_parity_is_meaningful.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.abs().sum() == 0:
                noise = torch.empty_like(parameter).normal_(0.0, std, generator=generator)
                parameter.copy_(noise)
    return model


def _assert_parity_is_meaningful(pred, model, *, min_grad_fraction=0.5):
    """Fail if a parity comparison could pass on a broken implementation.

    Guards against the zero-initialization trap above and against silently comparing
    all-zero tensors: the prediction must be non-trivial and most parameters must carry
    gradient, otherwise the surrounding assertions prove nothing about the compiled region.
    """
    assert torch.count_nonzero(pred) > 0, "prediction is identically zero; parity assertions would be vacuous"
    parameters = list(model.parameters())
    with_grad = sum(1 for p in parameters if p.grad is not None and torch.count_nonzero(p.grad) > 0)
    fraction = with_grad / max(len(parameters), 1)
    assert fraction >= min_grad_fraction, (
        f"only {with_grad}/{len(parameters)} parameters received a nonzero gradient "
        f"({fraction:.0%} < {min_grad_fraction:.0%}); the compiled region is barely exercised"
    )


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


def _build_unett_model(*, audio_drop_prob=0.0, cond_drop_prob=0.0):
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
        audio_drop_prob=audio_drop_prob,
        cond_drop_prob=cond_drop_prob,
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


class _PrecomputedMelDataset(torch.utils.data.Dataset):
    """Tiny real Trainer dataset: variable mel/text lengths, no external downloads."""

    def __init__(self):
        generator = torch.Generator().manual_seed(2026)
        self.lengths = [8, 12, 16, 20]
        self.texts = ["a", "abcdef", "hi", "long text"]
        self.mels = [torch.randn(8, frames, generator=generator) for frames in self.lengths]

    def __len__(self):
        return len(self.mels)

    def get_frame_len(self, index):
        return self.lengths[index]

    def __getitem__(self, index):
        return {"mel_spec": self.mels[index], "text": self.texts[index]}


class _SilentProgress:
    def __init__(self, iterable, *args, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def update(self, *args, **kwargs):
        pass

    def set_postfix(self, *args, **kwargs):
        pass


def _assert_clean_checkpoint_state_dict(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model_state_dict"]
    assert state_dict
    assert not any("_orig_mod" in key or "compile" in key or "compiled" in key for key in state_dict)


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
    eager_model = _randomize_zero_init_(_build_model())
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

    _assert_parity_is_meaningful(eager_pred, eager_model)
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


def test_regional_dit_blocks_matches_eager_loss_outputs_and_gradients():
    eager_model = _randomize_zero_init_(_build_model())
    compiled_model = copy.deepcopy(eager_model)
    # dit_blocks compile is training-only by design, so both models must be in train mode
    # or the compiled callable is bypassed and this compares eager against eager.
    # _build_model pins dropout=0.0, so train mode stays deterministic.
    eager_model.train()
    compiled_model.train()
    mel, text, lens = _sample_batch(batch_size=3, frames=12, text_len=7, lens=[12, 8, 5])
    transformer = cast(Any, compiled_model.transformer)
    for block in transformer.transformer_blocks:
        assert "forward" not in block.__dict__

    prepared_args = cast(PreparedArgs, eager_model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))

    eager_loss, eager_cond, eager_pred = eager_model._forward_loss_core(*prepared_args)
    eager_loss.backward()
    eager_grads = [
        param.grad.detach().clone() if param.grad is not None else None for param in eager_model.parameters()
    ]

    compiled = compiled_model.compile_training_core(target="dit_blocks", backend="eager", fullgraph=False, dynamic=None)
    compiled_args = cast(
        PreparedArgs,
        tuple(arg.detach().clone() if torch.is_tensor(arg) else arg for arg in prepared_args),
    )
    compiled_loss, compiled_cond, compiled_pred = compiled_model._run_loss_core(*compiled_args)
    compiled_loss.backward()
    compiled_grads = [
        param.grad.detach().clone() if param.grad is not None else None for param in compiled_model.parameters()
    ]

    _assert_parity_is_meaningful(eager_pred, eager_model)
    assert isinstance(compiled, tuple)
    compiled_forwards = cast(tuple[Any, ...], compiled)
    assert len(compiled_forwards) == len(transformer.transformer_blocks)
    for block in transformer.transformer_blocks:
        assert "forward" in block.__dict__
    assert compiled_model.training_compile_state == {
        "enabled": True,
        "target": "dit_blocks",
        "fallback_active": False,
        "error": None,
    }
    assert not any("_orig_mod" in key or "compile" in key or "compiled" in key for key in compiled_model.state_dict())
    _assert_close(compiled_loss.detach(), eager_loss.detach(), "dit_blocks_loss")
    _assert_close(compiled_cond.detach(), eager_cond.detach(), "dit_blocks_cond")
    _assert_close(compiled_pred.detach(), eager_pred.detach(), "dit_blocks_pred")
    for index, (compiled_grad, eager_grad) in enumerate(zip(compiled_grads, eager_grads, strict=True)):
        if compiled_grad is None or eager_grad is None:
            assert compiled_grad is eager_grad, f"dit_blocks gradient None mismatch at parameter {index}"
        else:
            _assert_close(compiled_grad, eager_grad, f"dit_blocks_grad_{index}", atol=1e-4, rtol=1e-4)

    compiled_model.clear_training_compile()
    assert compiled_model.training_compile_state == {
        "enabled": False,
        "target": None,
        "fallback_active": False,
        "error": None,
    }
    for block in transformer.transformer_blocks:
        assert "forward" not in block.__dict__


def test_regional_dit_blocks_runtime_fallback_restores_eager_blocks():
    model = _build_model()
    mel, text, lens = _sample_batch()
    prepared_args = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    transformer = cast(Any, model.transformer)
    block = transformer.transformer_blocks[0]

    assert "forward" not in block.__dict__
    model.compile_training_core(target="dit_blocks", backend="eager", fullgraph=False, dynamic=None)
    assert "forward" in block.__dict__

    def raise_compile_error(*_args, **_kwargs):
        raise _synthetic_compiler_error("synthetic dit block compile failure")

    block.forward = raise_compile_error
    loss, _, _ = model._run_loss_core(*prepared_args)

    assert torch.isfinite(loss)
    assert model.training_compile_state["enabled"] is False
    assert model.training_compile_state["fallback_active"] is True
    assert "synthetic dit block compile failure" in model.training_compile_state["error"]
    assert "forward" not in block.__dict__


def _count_compiled_block_calls(monkeypatch):
    """Wrap torch.compile so tests can assert the compiled callable actually ran.

    Asserting on `training_compile_state` only proves setup succeeded; it cannot detect a
    dispatch that silently bypasses the compiled block.
    """
    calls = {"n": 0}
    real_compile = torch.compile

    def counting_compile(fn, **kwargs):
        inner = real_compile(fn, **kwargs)

        def wrapper(*args, **inner_kwargs):
            calls["n"] += 1
            return inner(*args, **inner_kwargs)

        return wrapper

    monkeypatch.setattr(torch, "compile", counting_compile)
    return calls


def test_regional_dit_blocks_compile_is_training_only(monkeypatch):
    """Inference must not run through the compiled training blocks.

    `CFM.sample` shares `DiTBlock.forward` with training, and Trainer calls it whenever
    log_samples=True. Inference shapes (cfg_infer doubles the batch, the ODE solver sweeps
    durations) would otherwise consume the same per-code-object recompile budget as
    training -- default 8 -- and push new training shapes back to eager while the trainer
    still reports compile as active.
    """
    calls = _count_compiled_block_calls(monkeypatch)
    model = _build_model()
    mel, text, lens = _sample_batch()
    prepared_args = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    model.compile_training_core(target="dit_blocks", backend="eager", fullgraph=False, dynamic=None)

    model.eval()
    model._run_loss_core(*prepared_args)
    assert calls["n"] == 0, "eval-mode forward must bypass the compiled training blocks"

    model.train()
    loss, _, _ = model._run_loss_core(*prepared_args)
    assert torch.isfinite(loss)
    assert calls["n"] >= len(cast(Any, model.transformer).transformer_blocks), (
        "train-mode forward must dispatch through every compiled block"
    )

    # and compile state is unaffected by the mode switching
    assert model.training_compile_state["enabled"] is True
    assert model.training_compile_state["fallback_active"] is False


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


def test_tensor_aware_training_cfg_flags_are_tensorized_for_dit_and_unett():
    mel, text, lens = _sample_batch(batch_size=2, frames=12, text_len=7, lens=[12, 8])

    dit_model = _build_model(audio_drop_prob=1.0, cond_drop_prob=0.0)
    dit_prepared = cast(PreparedArgs, dit_model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    drop_audio_cond, drop_text = dit_prepared[6], dit_prepared[7]
    assert torch.is_tensor(drop_audio_cond)
    assert torch.is_tensor(drop_text)
    assert drop_audio_cond.shape == torch.Size([])
    assert drop_text.shape == torch.Size([])
    assert drop_audio_cond.dtype is torch.bool
    assert drop_text.dtype is torch.bool
    assert drop_audio_cond.item() is True
    assert drop_text.item() is False

    unett_model = _build_unett_model(audio_drop_prob=1.0, cond_drop_prob=0.0)
    unett_prepared = cast(PreparedArgs, unett_model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    unett_drop_audio_cond, unett_drop_text = unett_prepared[6], unett_prepared[7]
    assert torch.is_tensor(unett_drop_audio_cond)
    assert torch.is_tensor(unett_drop_text)
    assert unett_drop_audio_cond.shape == torch.Size([])
    assert unett_drop_text.shape == torch.Size([])
    assert unett_drop_audio_cond.dtype is torch.bool
    assert unett_drop_text.dtype is torch.bool
    assert unett_drop_audio_cond.item() is True
    assert unett_drop_text.item() is False


def test_branchless_tensor_cfg_flags_match_bool_loss_outputs_and_gradients():
    bool_model = _build_model()
    tensor_model = copy.deepcopy(bool_model)
    mel, text, lens = _sample_batch(batch_size=3, frames=12, text_len=7, lens=[12, 8, 5])

    prepared_args = cast(PreparedArgs, bool_model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    core_args = prepared_args[:6]

    bool_loss, bool_cond, bool_pred = bool_model._forward_loss_core(*core_args, True, True)
    bool_loss.backward()
    bool_grads = [param.grad.detach().clone() if param.grad is not None else None for param in bool_model.parameters()]

    tensor_core_args = tuple(arg.detach().clone() if torch.is_tensor(arg) else arg for arg in core_args)
    tensor_flag = torch.tensor(True, device=tensor_core_args[0].device)
    tensor_loss, tensor_cond, tensor_pred = tensor_model._forward_loss_core(*tensor_core_args, tensor_flag, tensor_flag)
    tensor_loss.backward()
    tensor_grads = [
        param.grad.detach().clone() if param.grad is not None else None for param in tensor_model.parameters()
    ]

    _assert_close(tensor_loss.detach(), bool_loss.detach(), "branchless_loss")
    _assert_close(tensor_cond.detach(), bool_cond.detach(), "branchless_cond")
    _assert_close(tensor_pred.detach(), bool_pred.detach(), "branchless_pred")
    for index, (tensor_grad, bool_grad) in enumerate(zip(tensor_grads, bool_grads, strict=True)):
        if tensor_grad is None or bool_grad is None:
            assert tensor_grad is bool_grad, f"branchless gradient None mismatch at parameter {index}"
        else:
            _assert_close(tensor_grad, bool_grad, f"branchless_grad_{index}", atol=1e-4, rtol=1e-4)


def test_unett_branchless_tensor_cfg_flags_match_bool_loss_outputs_and_gradients():
    bool_model = _build_unett_model()
    tensor_model = copy.deepcopy(bool_model)
    mel, text, lens = _sample_batch(batch_size=3, frames=12, text_len=7, lens=[12, 8, 5])

    prepared_args = cast(PreparedArgs, bool_model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    core_args = prepared_args[:6]

    bool_loss, bool_cond, bool_pred = bool_model._forward_loss_core(*core_args, True, True)
    bool_loss.backward()
    bool_grads = [param.grad.detach().clone() if param.grad is not None else None for param in bool_model.parameters()]

    tensor_core_args = tuple(arg.detach().clone() if torch.is_tensor(arg) else arg for arg in core_args)
    tensor_flag = torch.tensor(True, device=tensor_core_args[0].device)
    tensor_loss, tensor_cond, tensor_pred = tensor_model._forward_loss_core(*tensor_core_args, tensor_flag, tensor_flag)
    tensor_loss.backward()
    tensor_grads = [
        param.grad.detach().clone() if param.grad is not None else None for param in tensor_model.parameters()
    ]

    _assert_close(tensor_loss.detach(), bool_loss.detach(), "unett_branchless_loss")
    _assert_close(tensor_cond.detach(), bool_cond.detach(), "unett_branchless_cond")
    _assert_close(tensor_pred.detach(), bool_pred.detach(), "unett_branchless_pred")
    for index, (tensor_grad, bool_grad) in enumerate(zip(tensor_grads, bool_grads, strict=True)):
        if tensor_grad is None or bool_grad is None:
            assert tensor_grad is bool_grad, f"unett branchless gradient None mismatch at parameter {index}"
        else:
            _assert_close(tensor_grad, bool_grad, f"unett_branchless_grad_{index}", atol=1e-4, rtol=1e-4)


def test_unett_fullgraph_compile_accepts_tensor_cfg_flags():
    model = _build_unett_model()
    mel, text, lens = _sample_batch(batch_size=2, frames=12, text_len=7, lens=[12, 8])
    prepared_args = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    core_args = prepared_args[:6]
    tensor_flag = torch.tensor(True, device=core_args[0].device)
    tensor_args = (*core_args, tensor_flag, tensor_flag)

    model.compile_training_core(backend="eager", fullgraph=True, dynamic=None, runtime_fallback=False)
    loss, cond, pred = model._run_loss_core(*tensor_args)
    loss.backward()

    dynamo.reset()
    explanation = dynamo.explain(model._forward_loss_core)(*tensor_args)

    assert torch.isfinite(loss)
    assert cond.shape == mel.shape
    assert pred.shape == mel.shape
    assert explanation.graph_break_count == 0


def test_unett_compiled_loss_core_handles_tensor_cfg_flags_without_fallback():
    """UNetT tensor CFG path must compile all CFG combos without fallback or graph breaks."""
    model = _build_unett_model()
    mel, text, lens = _sample_batch(batch_size=2, frames=12, text_len=7, lens=[12, 8])
    prepared_args = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    core_args = prepared_args[:6]

    model.compile_training_core(backend="eager", fullgraph=True, dynamic=True)

    cfg_combos = ((False, False), (True, False), (True, True))
    for drop_audio_cond, drop_text in cfg_combos:
        drop_audio_cond_tensor = torch.tensor(drop_audio_cond, device=core_args[0].device)
        drop_text_tensor = torch.tensor(drop_text, device=core_args[0].device)
        loss, cond, pred = model._run_loss_core(*core_args, drop_audio_cond_tensor, drop_text_tensor)
        assert torch.isfinite(loss), f"UNetT compiled loss not finite for combo {(drop_audio_cond, drop_text)}"
        assert cond.shape == mel.shape
        assert pred.shape == mel.shape

    assert not model._compile_fallback_active

    dynamo.reset()
    drop_audio_cond_tensor = torch.tensor(True, device=core_args[0].device)
    drop_text_tensor = torch.tensor(True, device=core_args[0].device)
    explanation = dynamo.explain(model._forward_loss_core)(*core_args, drop_audio_cond_tensor, drop_text_tensor)
    assert explanation.graph_break_count == 0

    dynamo.reset()
    torch._dynamo.utils.counters.clear()
    for drop_audio_cond, drop_text in cfg_combos:
        model._run_loss_core(*core_args, drop_audio_cond, drop_text)
    bool_unique_graphs = int(torch._dynamo.utils.counters.get("stats", {}).get("unique_graphs", 0))

    dynamo.reset()
    torch._dynamo.utils.counters.clear()
    model.clear_training_compile()
    model.compile_training_core(backend="eager", fullgraph=True, dynamic=True)
    for drop_audio_cond, drop_text in cfg_combos:
        drop_audio_cond_tensor = torch.tensor(drop_audio_cond, device=core_args[0].device)
        drop_text_tensor = torch.tensor(drop_text, device=core_args[0].device)
        model._run_loss_core(*core_args, drop_audio_cond_tensor, drop_text_tensor)
    tensor_unique_graphs = int(torch._dynamo.utils.counters.get("stats", {}).get("unique_graphs", 0))

    # Without this the whole assertion passes vacuously as 0 <= 0 whenever the Dynamo
    # counters are unavailable, renamed, or simply never populated.
    assert bool_unique_graphs > 0, "Dynamo captured no graphs; the comparison below proves nothing"
    assert tensor_unique_graphs > 0, "Dynamo captured no graphs for the tensor-flag path"
    assert tensor_unique_graphs <= bool_unique_graphs, (
        f"UNetT tensor CFG unique graphs ({tensor_unique_graphs}) should be <= bool ({bool_unique_graphs})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the inductor dit_blocks test")
def test_cuda_inductor_dit_blocks_matches_eager_with_variable_shapes():
    """The shipped default target for F5 configs, through the backend it actually uses.

    Every other dit_blocks test compiles with backend="eager", and every Inductor/CUDA test
    compiles the *other* target (cfm_loss_core), so an Inductor-only defect in regional
    block compilation -- in the patched callable, the compiled backward, or dynamic-shape
    handling inside DiTBlock -- would not be caught anywhere. F5TTS_Base, F5TTS_Small,
    F5TTS_v1_Base and F5TTS_v1_Small all ship `target: dit_blocks`.
    """
    device = torch.device("cuda")
    dynamo.reset()
    eager_model = _randomize_zero_init_(_build_real_config_model()).to(device)
    compiled_model = copy.deepcopy(eager_model)
    eager_model.train()
    compiled_model.train()
    compiled_model.compile_training_core(target="dit_blocks", backend="inductor", fullgraph=False, dynamic=None)

    # more than one shape, so the compiled region must survive a dynamic-shape recompile
    for frames, text_len, lens in ((16, 9, [16, 11]), (24, 13, [24, 19])):
        mel, text, lens_tensor = _sample_batch(batch_size=2, frames=frames, text_len=text_len, lens=lens)
        prepared = cast(
            PreparedArgs,
            eager_model._prepare_training_inputs(mel.to(device), text.to(device), lens_tensor.to(device)),
        )
        eager_loss, _cond, eager_pred = eager_model._forward_loss_core(*prepared)
        eager_loss.backward()
        _assert_parity_is_meaningful(eager_pred, eager_model)

        compiled_args = cast(PreparedArgs, tuple(a.detach().clone() if torch.is_tensor(a) else a for a in prepared))
        compiled_loss, _cond2, compiled_pred = compiled_model._run_loss_core(*compiled_args)
        compiled_loss.backward()

        _assert_close(compiled_loss.detach(), eager_loss.detach(), "inductor_blocks_loss", atol=1e-4, rtol=1e-4)
        _assert_close(compiled_pred.detach(), eager_pred.detach(), "inductor_blocks_pred", atol=1e-3, rtol=1e-3)

        for name, eager_param in eager_model.named_parameters():
            compiled_param = dict(compiled_model.named_parameters())[name]
            if eager_param.grad is None or compiled_param.grad is None:
                assert eager_param.grad is None and compiled_param.grad is None, name
                continue
            _assert_close(compiled_param.grad, eager_param.grad, f"inductor_blocks_grad_{name}", atol=1e-3, rtol=1e-3)

        eager_model.zero_grad(set_to_none=True)
        compiled_model.zero_grad(set_to_none=True)

    assert compiled_model.training_compile_state["fallback_active"] is False, (
        "inductor regional compile silently fell back to eager"
    )


def test_unett_bool_inference_cache_behavior_still_works():
    """Existing UNetT bool-flag inference + cache path must remain functional."""
    model = _build_unett_model()
    model.eval()
    mel, text, _ = _sample_batch(batch_size=2, frames=12, text_len=7, lens=[12, 8])
    x = mel
    cond = mel.clone()
    time = torch.rand((mel.shape[0],))

    out_cond = model.transformer(
        x=x,
        cond=cond,
        text=text,
        time=time,
        mask=None,
        drop_audio_cond=False,
        drop_text=False,
        cache=True,
    )
    out_uncond = model.transformer(
        x=x,
        cond=cond,
        text=text,
        time=time,
        mask=None,
        drop_audio_cond=True,
        drop_text=True,
        cache=True,
    )

    assert out_cond.shape == mel.shape
    assert out_uncond.shape == mel.shape
    assert not torch.allclose(out_cond, out_uncond, atol=1e-6)

    out_cfg = model.transformer(
        x=x,
        cond=cond,
        text=text,
        time=time,
        mask=None,
        drop_audio_cond=False,
        drop_text=False,
        cfg_infer=True,
        cache=True,
    )
    assert out_cfg.shape == (mel.shape[0] * 2, mel.shape[1], mel.shape[2])

    keys = set(model.transformer.state_dict().keys())
    assert "text_embed.text_embed.weight" in keys
    assert "input_embed.proj.weight" in keys
    assert "proj_out.weight" in keys


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
    eager_model = _randomize_zero_init_(_build_real_config_model())
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

    _assert_parity_is_meaningful(eager_pred, eager_model)
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
    eager_model = _randomize_zero_init_(_build_unett_model())
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

    _assert_parity_is_meaningful(eager_pred, eager_model)
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
    assert model.training_compile_state == {
        "enabled": True,
        "target": "cfm_loss_core",
        "fallback_active": False,
        "error": None,
    }


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for GPU real-config inductor equivalence test"
)
@pytest.mark.parametrize("compile_kwargs", CUDA_INDUCTOR_EQUIVALENCE_KWARGS)
def test_cuda_inductor_real_config_matches_eager_across_compile_knobs(compile_kwargs):
    """CUDA inductor vs eager numerical parity for real-config knobs and compile knobs."""
    device = torch.device("cuda")
    eager_model = _randomize_zero_init_(_build_real_config_model()).to(device)
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

    _assert_parity_is_meaningful(eager_pred, eager_model)
    _assert_close(compiled_loss.detach(), eager_loss.detach(), "cuda_real_config_loss", atol=1e-4, rtol=1e-4)
    _assert_close(compiled_cond.detach(), eager_cond.detach(), "cuda_real_config_cond", atol=1e-4, rtol=1e-4)
    _assert_close(compiled_pred.detach(), eager_pred.detach(), "cuda_real_config_pred", atol=1e-3, rtol=1e-3)
    for index, (compiled_grad, eager_grad) in enumerate(zip(compiled_grads, eager_grads, strict=True)):
        if compiled_grad is None or eager_grad is None:
            assert compiled_grad is eager_grad, f"cuda real_config gradient None mismatch at parameter {index}"
        else:
            _assert_close(compiled_grad, eager_grad, f"cuda_real_config_grad_{index}", atol=1e-3, rtol=1e-3)
    assert compiled_model.training_compile_state == {
        "enabled": True,
        "target": "cfm_loss_core",
        "fallback_active": False,
        "error": None,
    }


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
    assert model.training_compile_state == {
        "enabled": True,
        "target": "cfm_loss_core",
        "fallback_active": False,
        "error": None,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU inductor equivalence test")
@pytest.mark.parametrize("compile_kwargs", CUDA_INDUCTOR_EQUIVALENCE_KWARGS)
def test_cuda_inductor_matches_eager_loss_outputs_and_gradients_across_compile_knobs(compile_kwargs):
    device = torch.device("cuda")
    eager_model = _randomize_zero_init_(_build_model()).to(device)
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

    _assert_parity_is_meaningful(eager_pred, eager_model)
    _assert_close(compiled_loss.detach(), eager_loss.detach(), "cuda_loss", atol=1e-4, rtol=1e-4)
    _assert_close(compiled_cond.detach(), eager_cond.detach(), "cuda_cond", atol=1e-4, rtol=1e-4)
    _assert_close(compiled_pred.detach(), eager_pred.detach(), "cuda_pred", atol=1e-3, rtol=1e-3)
    for index, (compiled_grad, eager_grad) in enumerate(zip(compiled_grads, eager_grads, strict=True)):
        if compiled_grad is None or eager_grad is None:
            assert compiled_grad is eager_grad, f"cuda gradient None mismatch at parameter {index}"
        else:
            _assert_close(compiled_grad, eager_grad, f"cuda_grad_{index}", atol=1e-3, rtol=1e-3)
    assert compiled_model.training_compile_state == {
        "enabled": True,
        "target": "cfm_loss_core",
        "fallback_active": False,
        "error": None,
    }


def test_runtime_fallback_can_be_enabled_or_disabled():
    model = _build_model()
    mel, text, lens = _sample_batch()
    prepared_args = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))

    def raise_compile_error(*_args):
        raise _synthetic_compiler_error()

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


def test_model_error_is_not_swallowed_by_compile_fallback():
    """A genuine model bug must propagate, not be disguised as a compile failure.

    The fallback used to catch bare `Exception`, so a shape mismatch, a bad vocabulary
    index or a failed assertion inside the transformer would silently disable compile,
    report a misleading "compile failed" state, and then re-run the *same* bad input
    eagerly -- producing a second, more confusing traceback and, for any transformer with
    mutable state, applying that state twice.
    """
    model = _build_model()
    mel, text, lens = _sample_batch()
    prepared_args = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))

    calls = []

    def raise_model_error(*_args):
        calls.append(1)
        raise RuntimeError("index out of range in self")

    object.__setattr__(model, "_compiled_loss_core", raise_model_error)
    object.__setattr__(model, "_compile_runtime_fallback", True)

    with pytest.raises(RuntimeError, match="index out of range in self"):
        model._run_loss_core(*prepared_args)

    # the failing callable ran exactly once: no eager retry of a possibly-partial forward
    assert calls == [1]
    # and compile state is untouched, so the error is not misreported as a compile problem
    assert model.training_compile_state["fallback_active"] is False
    assert model.training_compile_state["error"] is None


def test_host_out_of_memory_compiler_error_is_not_treated_as_cuda_capacity_oom():
    """A compiler-side 'out of memory' must fall back, not hard-fail as a GPU OOM.

    `_is_cuda_oom` used to match any RuntimeError containing "out of memory", so a Triton
    autotune worker reporting *host* OOM was rewritten as a GPU capacity error and the
    requested eager fallback was skipped.
    """
    model = _build_model()
    mel, text, lens = _sample_batch()
    prepared_args = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))

    def raise_host_oom(*_args):
        raise _synthetic_compiler_error("Triton compile worker exited: host out of memory")

    object.__setattr__(model, "_compiled_loss_core", raise_host_oom)
    object.__setattr__(model, "_compile_runtime_fallback", True)

    loss, _, _ = model._run_loss_core(*prepared_args)
    assert torch.isfinite(loss)
    assert model.training_compile_state["fallback_active"] is True


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
        raise _synthetic_compiler_error()

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

    # NOTE: this bypasses Trainer.__init__ and hand-sets only the attributes train()
    # happens to touch, so it breaks whenever Trainer gains a field. Kept because it is
    # the cheapest way to intercept DataLoader construction, but the defaults below must
    # be updated alongside Trainer.__init__.
    trainer = object.__new__(Trainer)
    trainer.log_samples = False
    trainer.batch_size_per_gpu = 2
    trainer.max_samples = 2
    trainer.max_padded_frames = 0
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
    with patch.object(sys, "argv", ["prog"]):
        args = parse_args()
    assert args.compile_target == "auto"

    with patch.object(
        sys,
        "argv",
        [
            "prog",
            "--compile_enabled",
            "--compile_backend",
            "eager",
            "--compile_target",
            "dit_blocks",
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
    assert args.compile_target == "dit_blocks"
    assert args.compile_mode == "reduce-overhead"
    assert args.compile_fullgraph is True
    assert args.compile_dynamic == "true"
    assert args.compile_no_fallback is True


def test_all_training_configs_define_default_off_compile_block_without_metrics():
    for config_path in sorted((ROOT / "src/f5_tts/configs").glob("*.yaml")):
        config = yaml.safe_load(config_path.read_text())
        expected_target = "cfm_loss_core" if config["model"]["backbone"] == "UNetT" else "dit_blocks"
        assert "compile" in config, config_path.name
        assert "metrics" not in config, config_path.name
        assert config["compile"] == {
            "enabled": False,
            "target": expected_target,
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
        "target": None,
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


def test_dit_blocks_compile_target_allows_average_upsampling_outside_compiled_region():
    model = _build_model(average_upsampling=True)
    mel, text, lens = _sample_batch(batch_size=2, frames=12, text_len=7, lens=[12, 8])

    model.compile_training_core(target="dit_blocks", backend="eager", fullgraph=False, dynamic=None)
    loss, cond, pred = model(mel, text=text, lens=lens)

    assert torch.isfinite(loss)
    assert cond.shape == mel.shape
    assert pred.shape == mel.shape
    assert model.training_compile_state["enabled"] is True
    assert model.training_compile_state["target"] == "dit_blocks"


def test_real_trainer_frame_dataset_dit_blocks_compile_runs_variable_shape_epoch(tmp_path, monkeypatch):
    """Regression for compile bugs missed by direct CFM calls.

    This uses the actual Trainer.train DataLoader path with DynamicBatchSampler and
    collate_fn, variable mel/text lengths, string tokenization, optimizer/scheduler,
    EMA checkpoint save, and target=dit_blocks compile setup after Accelerator.prepare.
    """
    from f5_tts.model import dataset as dataset_module
    from f5_tts.model import trainer as trainer_module
    from f5_tts.model.trainer import Trainer

    monkeypatch.setattr(trainer_module, "tqdm", _SilentProgress)
    monkeypatch.setattr(dataset_module, "tqdm", _SilentProgress)
    torch.manual_seed(2026)

    train_dataset = _PrecomputedMelDataset()
    model = _build_model(
        audio_drop_prob=1.0,
        cond_drop_prob=1.0,
        vocab_size=256,
    )
    trainer = Trainer(
        model,
        epochs=1,
        learning_rate=1e-4,
        num_warmup_updates=1,
        save_per_updates=10**9,
        keep_last_n_checkpoints=0,
        checkpoint_path=str(tmp_path),
        batch_size_per_gpu=20,
        batch_size_type="frame",
        max_samples=2,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        logger=None,
        log_samples=False,
        last_per_updates=10**9,
        compile_enabled=True,
        compile_backend="eager",
        compile_target="dit_blocks",
        compile_fullgraph=False,
        compile_dynamic=None,
        compile_fallback_to_eager=False,
    )

    module = cast(Any, trainer._unwrapped_model)
    original_run_components = module._run_loss_core_components
    seen_core_calls = []

    def recording_run_components(*loss_args):
        x1, text, _mask, _rand_span_mask, _x0, _time, drop_audio_cond, drop_text = loss_args
        seen_core_calls.append(
            {
                "mel_shape": tuple(x1.shape),
                "text_shape": tuple(text.shape),
                "drop_audio_cond": bool(drop_audio_cond),
                "drop_text": bool(drop_text),
            }
        )
        return original_run_components(*loss_args)

    object.__setattr__(module, "_run_loss_core_components", recording_run_components)

    trainer.train(train_dataset, num_workers=0, resumable_with_seed=123)

    assert trainer.compile_active is True
    assert trainer.compile_fallback_active is False
    assert module.training_compile_state == {
        "enabled": True,
        "target": "dit_blocks",
        "fallback_active": False,
        "error": None,
    }
    assert len(seen_core_calls) == 3
    assert len({call["mel_shape"] for call in seen_core_calls}) > 1
    assert len({call["text_shape"] for call in seen_core_calls}) > 1
    assert all(call["drop_audio_cond"] and call["drop_text"] for call in seen_core_calls)
    _assert_clean_checkpoint_state_dict(tmp_path / "model_last.pt")


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


def test_adamw_fused_policy_preserves_upstream_on_cpu_and_cuda():
    """CPU must keep upstream's fused kernel; only genuinely unsupported devices opt out.

    Upstream requests `fused=True` unconditionally. Narrowing that to CUDA-only silently
    downgraded CPU training to the unfused kernel, which changes optimizer numerics and
    the serialized optimizer state even with compile disabled -- a backward-compatibility
    break in a code path nobody opted into. Only MPS, which has no fused AdamW, may differ.
    """
    from f5_tts.model.trainer import FUSED_ADAMW_DEVICE_TYPES

    assert "cpu" in FUSED_ADAMW_DEVICE_TYPES, "CPU fused AdamW is upstream behaviour"
    assert "cuda" in FUSED_ADAMW_DEVICE_TYPES
    assert "mps" not in FUSED_ADAMW_DEVICE_TYPES, "MPS has no fused AdamW kernel"


def test_adamw_fused_is_actually_supported_on_cpu():
    """Behavioural counterpart: the fused CPU kernel this policy relies on must exist.

    If a future torch drops fused CPU AdamW, this fails loudly instead of letting the
    trainer raise at optimizer construction for every CPU user.
    """
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    assert opt.param_groups[0].get("fused", False) is True

    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    opt.step()  # must not raise
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_trainer_requests_fused_adamw_for_its_accelerator_device():
    """The trainer must derive `fused` from the accelerator device, not global CUDA state.

    Asserted behaviourally against the constructed optimizer rather than by grepping the
    source, and without constructing a second Accelerator (AcceleratorState is a process
    global; building one with a different device makes the test order-dependent and it
    fails whenever an earlier test has already initialised CUDA).
    """
    from f5_tts.model.trainer import FUSED_ADAMW_DEVICE_TYPES, Trainer

    model = _build_model()
    trainer = Trainer(
        model,
        epochs=1,
        learning_rate=1e-4,
        num_warmup_updates=1,
        save_per_updates=10**9,
        keep_last_n_checkpoints=0,
        logger=None,
        log_samples=False,
    )
    expected = trainer.accelerator.device.type in FUSED_ADAMW_DEVICE_TYPES
    assert trainer.optimizer.param_groups[0].get("fused", False) is expected


class _FrameLenDataset:
    """Minimal dataset exposing only what DynamicBatchSampler consumes."""

    def __init__(self, frame_lens):
        self.frame_lens = list(frame_lens)

    def get_frame_len(self, index):
        return float(self.frame_lens[index])

    def __len__(self):
        return len(self.frame_lens)


class _IndexSampler:
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


def _build_batches(frame_lens, threshold, *, max_samples=0, max_padded_frames=0):
    from f5_tts.model.dataset import DynamicBatchSampler

    dataset = _FrameLenDataset(frame_lens)
    sampler = DynamicBatchSampler(
        _IndexSampler(dataset),
        threshold,
        max_samples=max_samples,
        random_seed=None,
        drop_residual=False,
        max_padded_frames=max_padded_frames,
    )
    return sampler.batches, dataset


def test_max_padded_frames_defaults_to_upstream_batch_composition():
    """Default (0) must not change a single batch; this is the backward-compat contract."""
    frame_lens = [94, 120, 300, 301, 500, 900, 2100, 2200, 2800, 2810]
    baseline, _ = _build_batches(frame_lens, 3000, max_samples=8)
    guarded, _ = _build_batches(frame_lens, 3000, max_samples=8, max_padded_frames=0)
    assert guarded == baseline


def test_max_padded_frames_bounds_the_padded_rectangle_on_bimodal_data():
    """The adversarial case the guard exists for.

    A frame-sum budget lets a batch of short utterances be closed out by a much longer
    one, so the allocated rectangle len(batch)*max_len can far exceed the requested
    budget. Measured worst case on a bimodal corpus at N=100k is 1.82x. With the guard set,
    no batch may exceed it.
    """
    import math

    rng = torch.Generator().manual_seed(11)
    short = (torch.rand(400, generator=rng) * 140 + 94).tolist()
    long = (torch.rand(400, generator=rng) * 560 + 2250).tolist()
    frame_lens = short + long

    threshold = 4000
    unguarded, dataset = _build_batches(frame_lens, threshold, max_samples=64)
    worst = max(len(b) * math.ceil(max(dataset.get_frame_len(i) for i in b)) for b in unguarded)
    assert worst > threshold, "expected the frame-sum budget to overshoot the padded rectangle here"

    guarded, dataset = _build_batches(frame_lens, threshold, max_samples=64, max_padded_frames=threshold)
    for batch in guarded:
        padded = len(batch) * math.ceil(max(dataset.get_frame_len(i) for i in batch))
        assert padded <= threshold, f"padded rectangle {padded} exceeds cap {threshold}"


def test_max_padded_frames_keeps_every_usable_sample():
    """The guard may repartition batches but must not silently drop fitting samples."""
    frame_lens = [94, 120, 300, 301, 500, 900, 1100, 1200]
    baseline, _ = _build_batches(frame_lens, 3000, max_samples=8)
    guarded, _ = _build_batches(frame_lens, 3000, max_samples=8, max_padded_frames=2400)

    assert sorted(i for b in baseline for i in b) == sorted(i for b in guarded for i in b)


def _upstream_reference_loss(pred, flow, rand_span_mask):
    """The exact pre-compile upstream reduction, transcribed from SWivid/F5-TTS 2ae2c9b.

    loss = F.mse_loss(pred, flow, reduction="none")
    loss = loss[rand_span_mask]
    return loss.mean(), cond, pred
    """
    loss = torch.nn.functional.mse_loss(pred, flow, reduction="none")
    return loss[rand_span_mask].mean()


def test_default_path_loss_is_bitwise_identical_to_upstream():
    """With compile disabled, the loss must be bit-for-bit upstream's -- not merely close.

    The compile-friendly reduction `(loss * mask).sum() / denom` is algebraically equal to
    upstream's `loss[mask].mean()` but reassociates the summation, so it differs by ~1 ULP
    on roughly 40% of inputs. Training is chaotic: a 1-ULP loss difference compounds into a
    visibly different trajectory, so users who never enabled compile would silently stop
    reproducing their previous runs. This test fails if the default path ever adopts the
    compiled reduction.
    """
    model = _build_model()
    mismatches = 0
    for seed in range(25):
        torch.manual_seed(seed)
        mel = torch.randn(3, 37, 8)
        text = torch.randint(0, 32, (3, 11))
        lens = torch.tensor([37, 29, 33])
        prepared = cast(PreparedArgs, model._prepare_training_inputs(mel, text, lens))
        x1, _text, _mask, rand_span_mask, x0, _time, _dac, _dt = prepared

        loss, _cond, pred = model._run_loss_core(*prepared)
        reference = _upstream_reference_loss(pred, x1 - x0, rand_span_mask)

        assert loss.dtype == reference.dtype
        if loss.item() != reference.item():
            mismatches += 1
    assert mismatches == 0, f"default loss diverged from upstream on {mismatches}/25 seeds"


def test_compiled_reduction_is_close_but_not_required_to_be_bitwise_equal():
    """The compiled reduction may differ by ~1 ULP; it must stay within tight tolerance.

    Documents the deliberate asymmetry: exactness is owed to users who did NOT opt in,
    while the opt-in compiled path only owes numerical equivalence.
    """
    model = _build_model()
    torch.manual_seed(0)
    mel = torch.randn(3, 37, 8)
    text = torch.randint(0, 32, (3, 11))
    lens = torch.tensor([37, 29, 33])
    prepared = cast(PreparedArgs, model._prepare_training_inputs(mel, text, lens))
    x1, _text, _mask, rand_span_mask, x0, _time, _dac, _dt = prepared

    upstream_loss, _cond, pred = model._run_loss_core(*prepared)
    components_loss, _sum, _denom, _cond2, _pred2 = model._forward_loss_core_components(*prepared)

    assert torch.allclose(upstream_loss, components_loss, rtol=1e-6, atol=1e-7)
    assert components_loss.dtype == torch.float32  # fp32 accumulation guard is preserved
    del pred, rand_span_mask


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
