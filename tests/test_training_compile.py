import copy
import io
import sys
import types
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
import torch._dynamo as dynamo
import yaml

import f5_tts.model.cfm as cfm_module
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


def test_unselected_nonfinite_noise_does_not_poison_loss(monkeypatch):
    # Nonfinites outside the masked span (here: corrupted transformer output at
    # unselected positions) must not reach the loss. Masked-multiply reduction
    # propagates them (0 * NaN == NaN); the torch.where selection keeps the
    # boolean-indexing semantics of the historical eager reduction while staying
    # compile-friendly.
    model = _build_model()
    mel, text, lens = _sample_batch()
    span = torch.zeros(mel.shape[:2], dtype=torch.bool)
    span[:, : max(1, span.shape[1] // 2)] = True

    def _span_of_two_lengths(seq_len, frac_lengths, length=None):
        del seq_len, frac_lengths, length
        return span

    original_transformer_forward = model.transformer.forward

    def _pred_with_corrupted_unselected(*args, **kwargs):
        pred = original_transformer_forward(*args, **kwargs)
        return pred.masked_fill(~span[..., None], float("nan"))

    monkeypatch.setattr(cfm_module, "mask_from_frac_lengths", _span_of_two_lengths)
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.zeros_like(value))
    monkeypatch.setattr(model.transformer, "forward", _pred_with_corrupted_unselected)

    # Route through the compiled components core: with compile off, _run_loss_core
    # dispatches to the upstream-exact boolean-indexing path, which trivially
    # excludes unselected values and would not exercise the reduction under test.
    model.compile_training_core(backend="eager", fullgraph=False, dynamic=None)

    loss, cond, pred = model(mel, text=text, lens=lens)

    assert torch.isfinite(loss)
    assert torch.isfinite(pred[span]).all()


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
    assert transformer._compiled_dit_block_forwards is None

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
    # Compiled callables live on the owning DiT, not on block.forward; modules stay
    # unpatched so deepcopy/pickle produce eager modules.
    assert transformer._compiled_dit_block_forwards is not None
    assert len(transformer._compiled_dit_block_forwards) == len(transformer.transformer_blocks)
    for block in transformer.transformer_blocks:
        assert "forward" not in block.__dict__
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
    assert transformer._compiled_dit_block_forwards is None


def test_regional_dit_blocks_runtime_fallback_restores_eager_blocks():
    model = _build_model()
    mel, text, lens = _sample_batch()
    prepared_args = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    transformer = cast(Any, model.transformer)
    block = transformer.transformer_blocks[0]

    assert transformer._compiled_dit_block_forwards is None
    model.compile_training_core(target="dit_blocks", backend="eager", fullgraph=False, dynamic=None)
    assert transformer._compiled_dit_block_forwards is not None
    # Modules are never patched; the compiled callable is dispatched from the owner loop.
    assert "forward" not in block.__dict__

    def raise_compile_error(*_args, **_kwargs):
        raise _synthetic_compiler_error("synthetic dit block compile failure")

    # Inject a failing callable into the compiled slot to simulate a runtime compile
    # failure dispatched through the owner loop (not through block.forward).
    compiled = transformer._compiled_dit_block_forwards
    object.__setattr__(
        transformer,
        "_compiled_dit_block_forwards",
        (raise_compile_error, *compiled[1:]),
    )
    model.train()
    loss, _, _ = model._run_loss_core(*prepared_args)

    assert torch.isfinite(loss)
    assert model.training_compile_state["enabled"] is False
    assert model.training_compile_state["fallback_active"] is True
    assert "synthetic dit block compile failure" in model.training_compile_state["error"]
    assert transformer._compiled_dit_block_forwards is None


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


def test_compiled_dit_deepcopy_runs_eager_and_does_not_share_source_closure(monkeypatch):
    """A deep-copied compiled DiT must execute its own parameters eagerly.

    Regression for the monkey-patch design where the installed closure captured the source
    block, so a copied model silently ran the source model's weights. Compiled state is
    stripped on deepcopy; the copy dispatches eager and its outputs depend only on its own
    parameters.
    """
    model = _build_model()
    mel, text, lens = _sample_batch()
    model.compile_training_core(target="dit_blocks", backend="eager", fullgraph=False, dynamic=None)
    model.train()

    copied = copy.deepcopy(model)
    # Compile state is stripped: the copy is eager and pickleable.
    assert cast(Any, copied.transformer)._compiled_dit_block_forwards is None
    assert copied.training_compile_state["enabled"] is False

    copied.train()
    prepared = cast(PreparedArgs, copied._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    loss, _, _ = copied._run_loss_core(*prepared)
    assert torch.isfinite(loss)

    # Mutating the source model's parameters must not change the copy's output: the copy
    # runs its own parameters, not a shared closure over the source block.
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
    loss_after, _, _ = copied._run_loss_core(*prepared)
    assert torch.equal(loss.detach(), loss_after.detach()), (
        "deep-copied model must not share the source model's parameter closure"
    )


def test_compiled_dit_model_is_pickleable_via_torch_save():
    """A compiled DiT must survive torch.save/torch.load and deserialize eager.

    Regression for the local-closure pickle failure: ``torch.save(model, ...)`` raised
    ``AttributeError: Can't pickle local object`` while the monkey-patch was active.
    __getstate__ now strips compiled callables; the loaded model runs eager.
    """
    model = _build_model()
    mel, text, lens = _sample_batch()
    model.compile_training_core(target="dit_blocks", backend="eager", fullgraph=False, dynamic=None)
    model.train()

    buf = io.BytesIO()
    torch.save(model, buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=False)

    assert cast(Any, loaded.transformer)._compiled_dit_block_forwards is None
    assert loaded.training_compile_state["enabled"] is False
    loaded.train()
    prepared = cast(PreparedArgs, loaded._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    loss, _, _ = loaded._run_loss_core(*prepared)
    assert torch.isfinite(loss)


def test_compiled_cfm_deepcopy_resets_to_eager_without_sharing_source_closure():
    """A copied full-loss-core model must not retain a callable bound to the source CFM."""
    model = _build_model()
    model.compile_training_core(target="cfm_loss_core", backend="eager", fullgraph=False, dynamic=None)
    source_compiled = model.__dict__["_compiled_loss_core"]

    copied = copy.deepcopy(model)

    assert copied is not model
    assert copied.training_compile_state["enabled"] is False
    assert copied.__dict__["_compiled_loss_core"] is None
    assert model.__dict__["_compiled_loss_core"] is source_compiled


def test_compiled_cfm_model_is_pickleable_via_torch_save_and_loads_eager():
    """Whole-model serialization must strip the non-pickleable compiled CFM wrapper."""
    model = _build_model()
    state_dict_before = copy.deepcopy(model.state_dict())
    model.compile_training_core(target="cfm_loss_core", backend="eager", fullgraph=False, dynamic=None)

    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)

    assert loaded.training_compile_state["enabled"] is False
    assert loaded.__dict__["_compiled_loss_core"] is None
    assert loaded.state_dict().keys() == state_dict_before.keys()
    for key, value in state_dict_before.items():
        assert torch.equal(loaded.state_dict()[key], value), key


def test_compile_state_serialization_does_not_require_nn_module_getstate(monkeypatch):
    """CFM and DiT serialization must remain compatible with the torch 2.0 parent contract."""
    model = _build_model()
    transformer = cast(Any, model.transformer)
    compiled_call_sentinels = {}
    modules_and_compile_attrs = (
        (model, model._CFM_COMPILE_ONLY_ATTRS),
        (transformer, transformer._DIT_COMPILE_ONLY_ATTRS),
    )

    for module, compile_attrs in modules_and_compile_attrs:
        compiled_call_sentinel = object()
        compiled_call_sentinels[module] = compiled_call_sentinel
        object.__setattr__(module, "_compiled_call_impl", compiled_call_sentinel)
        for attr in compile_attrs:
            object.__setattr__(module, attr, object())

    # torch 2.0's nn.Module has no __getstate__. Remove the newer parent method to
    # exercise that supported legacy contract without changing the installed torch.
    monkeypatch.delattr(torch.nn.Module, "__getstate__", raising=False)

    for module, compile_attrs in modules_and_compile_attrs:
        state = module.__getstate__()
        assert "_compiled_call_impl" not in state
        assert all(attr not in state for attr in compile_attrs)

        # __getstate__ must filter a copy, not mutate the live module.
        assert module.__dict__["_compiled_call_impl"] is compiled_call_sentinels[module]
        assert all(attr in module.__dict__ for attr in compile_attrs)
        assert state["_modules"] is module.__dict__["_modules"]

    # Exercise the EMA construction mechanism itself: deepcopy must also succeed
    # while all ordinary child modules use torch 2.0's parent serialization contract.
    copied = copy.deepcopy(model)
    copied_transformer = cast(Any, copied.transformer)
    assert "_compiled_call_impl" not in copied.__dict__
    assert "_compiled_call_impl" not in copied_transformer.__dict__
    assert copied.training_compile_state["enabled"] is False
    assert copied_transformer.training_compile_state["enabled"] is False


def test_clear_training_compile_after_deepcopy_leaves_forward_callable():
    """clear_training_compile() on a deep-copied compiled DiT must not install a sentinel.

    Regression for the identity-based sentinel restore: deepcopy produced a distinct
    ``object()`` sentinel that failed the ``is _NO_INSTANCE_FORWARD`` check, so clear
    installed a bare ``object`` as ``block.forward`` -> ``TypeError`` on next call. With
    the monkey-patch removed, clear is a no-op on blocks and forward stays callable.
    """
    model = _build_model()
    model.compile_training_core(target="dit_blocks", backend="eager", fullgraph=False, dynamic=None)
    copied = copy.deepcopy(model)

    copied.clear_training_compile()
    transformer = cast(Any, copied.transformer)
    assert transformer._compiled_dit_block_forwards is None
    for block in transformer.transformer_blocks:
        assert isinstance(block.forward, types.MethodType), "block.forward must remain a bound method"

    mel, text, lens = _sample_batch()
    copied.train()
    prepared = cast(PreparedArgs, copied._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    loss, _, _ = copied._run_loss_core(*prepared)
    assert torch.isfinite(loss)


def test_compiled_dit_state_dict_keys_unchanged_after_compile_and_deepcopy():
    """Compile state must not leak into state_dict keys (no _orig_mod / compile keys)."""
    model = _build_model()
    keys_before = set(model.state_dict().keys())
    model.compile_training_core(target="dit_blocks", backend="eager", fullgraph=False, dynamic=None)
    keys_after_compile = set(model.state_dict().keys())
    keys_after_deepcopy = set(copy.deepcopy(model).state_dict().keys())

    assert keys_before == keys_after_compile == keys_after_deepcopy
    assert not any("_orig_mod" in k or "compile" in k or "compiled" in k for k in keys_after_compile)


def test_regional_dit_blocks_dispatch_uses_compiled_callable_in_train_mode(monkeypatch):
    """The owner-loop dispatch must call the compiled callable, not block.forward, in train mode.

    Guards against a regression where the dispatch bypasses the compiled callable (e.g. by
    checking the wrong attribute or falling through to _forward_block_range in train mode).
    """
    calls = _count_compiled_block_calls(monkeypatch)
    model = _build_model()
    mel, text, lens = _sample_batch()
    prepared = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))

    model.compile_training_core(target="dit_blocks", backend="eager", fullgraph=False, dynamic=None)
    model.train()
    model._run_loss_core(*prepared)
    transformer = cast(Any, model.transformer)
    assert calls["n"] >= len(transformer.transformer_blocks)

    # A second distinct shape must also dispatch through the compiled callable.
    mel2, text2, lens2 = _sample_batch(batch_size=2, frames=16, text_len=9, lens=[16, 10])
    prepared2 = cast(PreparedArgs, model._prepare_training_inputs(mel2.clone(), text2.clone(), lens2.clone()))
    before = calls["n"]
    model._run_loss_core(*prepared2)
    assert calls["n"] > before, "a new training shape must still dispatch through the compiled callable"


def _attach_recording_hooks(block):
    """Register forward pre/post and full backward hooks that record every firing.

    Returns a dict of lists so a test can compare hook fire counts and captured
    tensors between an eager and a compiled run. Hooks must be attached *before*
    ``torch.compile`` so the compiled ``_call_impl`` graph traces them in.
    """
    records: dict[str, list[Any]] = {"fwd_pre": [], "fwd": [], "bwd": []}

    def fwd_pre_hook(module, args):
        records["fwd_pre"].append(len(args))

    def fwd_hook(module, args, output):
        records["fwd"].append(output.detach().clone())

    def bwd_hook(module, grad_input, grad_output):
        records["bwd"].append(tuple(g.detach().clone() if g is not None else None for g in grad_output))

    block.register_forward_pre_hook(fwd_pre_hook)
    block.register_forward_hook(fwd_hook)
    block.register_full_backward_hook(bwd_hook)
    return records


def test_regional_dit_blocks_forward_and_backward_hooks_fire_as_eager():
    """Forward pre/post hooks and full backward hooks must fire exactly as eager under regional compile.

    Regression for the bare-``block.forward`` compile design: calling a compiled
    ``block.forward`` directly from the owner loop bypasses ``nn.Module._call_impl``, so
    none of the module hooks registered on the block fired. Compiling ``_call_impl``
    instead (nn.Module's hook-dispatch path, the same mechanism ``nn.Module.compile``
    uses) keeps the hook machinery inside the compiled graph. This test proves the
    compiled dispatch fires forward pre-hooks, forward hooks, and full backward hooks
    with the same counts and numerically equal captured tensors as the eager path.
    """
    eager_model = _randomize_zero_init_(_build_model())
    compiled_model = copy.deepcopy(eager_model)
    eager_model.train()
    compiled_model.train()

    eager_block = cast(Any, eager_model.transformer).transformer_blocks[0]
    compiled_block = cast(Any, compiled_model.transformer).transformer_blocks[0]
    eager_records = _attach_recording_hooks(eager_block)
    compiled_records = _attach_recording_hooks(compiled_block)
    # Attach before compile so the compiled _call_impl graph traces the hooks in.
    compiled_model.compile_training_core(target="dit_blocks", backend="eager", fullgraph=False, dynamic=None)

    mel, text, lens = _sample_batch(batch_size=3, frames=12, text_len=7, lens=[12, 8, 5])
    prepared = cast(PreparedArgs, eager_model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    prepared_compiled = cast(
        PreparedArgs,
        tuple(arg.detach().clone() if torch.is_tensor(arg) else arg for arg in prepared),
    )

    eager_loss, _, eager_pred = eager_model._run_loss_core(*prepared)
    eager_loss.backward()
    compiled_loss, _, compiled_pred = compiled_model._run_loss_core(*prepared_compiled)
    compiled_loss.backward()

    _assert_parity_is_meaningful(eager_pred, eager_model)
    # Hook fire counts match exactly: one forward pre, one forward, one full backward.
    assert len(eager_records["fwd_pre"]) == len(compiled_records["fwd_pre"]) == 1
    assert len(eager_records["fwd"]) == len(compiled_records["fwd"]) == 1
    assert len(eager_records["bwd"]) == len(compiled_records["bwd"]) == 1
    # Captured forward outputs and backward grad_outputs are numerically equal.
    _assert_close(compiled_records["fwd"][0], eager_records["fwd"][0], "compiled_block_hook_forward_output")
    _assert_close(compiled_records["bwd"][0][0], eager_records["bwd"][0][0], "compiled_block_hook_grad_output")
    _assert_close(compiled_loss.detach(), eager_loss.detach(), "hook_parity_loss")

    # Inference must still bypass the compiled path: with the model in eval mode, the
    # training-gated dispatch falls through to eager ``block(...)`` and the compiled
    # callable is never invoked, so hooks fire through the eager path only once more.
    compiled_model.eval()
    before_fwd = len(compiled_records["fwd"])
    with torch.no_grad():
        compiled_model._run_loss_core(*prepared_compiled)
    assert len(compiled_records["fwd"]) == before_fwd + 1, "eval-mode forward must still fire hooks via the eager path"
    assert len(compiled_records["bwd"]) == 1, "no-grad eval forward must not add a backward hook firing"


def test_regional_dit_blocks_hook_preservation_keeps_blocks_deepcopy_and_pickle_safe():
    """Compiling ``_call_impl`` must not leave unpickleable state on the blocks.

    The design never sets ``block._compiled_call_impl`` (which would make eval/inference
    ``block(...)`` accidentally enter the compiled path); the compiled callables live
    only in the DiT-owned ``_compiled_dit_block_forwards`` tuple, stripped by
    ``__getstate__``. This asserts the blocks stay free of compile state so deepcopy and
    ``torch.save``/``load`` deserialize to eager modules that still support hooks.
    """
    model = _build_model()
    model.compile_training_core(target="dit_blocks", backend="eager", fullgraph=False, dynamic=None)
    transformer = cast(Any, model.transformer)
    # The hook-preservation design compiles _call_impl without setting _compiled_call_impl
    # on the blocks, so blocks carry no compiled closure and stay pickleable.
    for block in transformer.transformer_blocks:
        assert block._compiled_call_impl is None
        assert "forward" not in block.__dict__

    # deepcopy strips the DiT-owned compiled tuple; the copy is eager and hooks fire eager.
    copied = copy.deepcopy(model)
    copied_transformer = cast(Any, copied.transformer)
    assert copied_transformer._compiled_dit_block_forwards is None
    for block in copied_transformer.transformer_blocks:
        assert block._compiled_call_impl is None
    copied_records = _attach_recording_hooks(copied_transformer.transformer_blocks[0])
    copied.train()
    mel, text, lens = _sample_batch()
    prepared = cast(PreparedArgs, copied._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    loss, _, _ = copied._run_loss_core(*prepared)
    loss.backward()
    assert len(copied_records["fwd"]) == 1 and len(copied_records["bwd"]) == 1, (
        "deep-copied eager model must fire hooks"
    )

    # torch.save/load deserializes eager; hooks attached post-load fire eager.
    buf = io.BytesIO()
    torch.save(model, buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=False)
    loaded_transformer = cast(Any, loaded.transformer)
    assert loaded_transformer._compiled_dit_block_forwards is None
    for block in loaded_transformer.transformer_blocks:
        assert block._compiled_call_impl is None
    loaded_records = _attach_recording_hooks(loaded_transformer.transformer_blocks[0])
    loaded.train()
    prepared_loaded = cast(PreparedArgs, loaded._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    loaded_loss, _, _ = loaded._run_loss_core(*prepared_loaded)
    loaded_loss.backward()
    assert torch.isfinite(loaded_loss)
    assert len(loaded_records["fwd"]) == 1 and len(loaded_records["bwd"]) == 1, "loaded eager model must fire hooks"


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


def test_sync_compile_setup_ddp_uses_accelerator_reduce_max():
    """_sync_compile_setup_ddp must use accelerator.reduce('max'), not torch.distributed.

    This verifies the collective goes through Accelerate's dispatch layer (handles
    DeepSpeed/FSDP process groups) and that the return value correctly reflects whether
    any rank failed. Uses a fake accelerator to avoid real process groups.
    """
    from f5_tts.model.trainer import Trainer

    class _FakeReduceAcceleratorMax:
        def __init__(self, *, num_processes, reduced_flag):
            self.num_processes = num_processes
            self._reduced_flag = reduced_flag
            self.reduce_calls: list[str] = []
            self.device = torch.device("cpu")

        def reduce(self, tensor, reduction="sum"):
            self.reduce_calls.append(reduction)
            return self._reduced_flag.to(tensor.device) if self._reduced_flag is not None else tensor

        @property
        def is_main_process(self):
            return True

    # Case 1: this rank failed (fallback mode), collective returns max=1.0 → any_failed=True.
    fake = _FakeReduceAcceleratorMax(num_processes=2, reduced_flag=torch.tensor(1.0))
    trainer = Trainer.__new__(Trainer)
    trainer.accelerator = cast(Any, fake)
    trainer.compile_fallback_active = True
    trainer.compile_active = False
    trainer._unwrapped_model = cast(Any, type("M", (), {"clear_training_compile": lambda self: None})())
    any_failed = trainer._sync_compile_setup_ddp()
    assert any_failed is True
    assert fake.reduce_calls == ["max"], f"expected reduce('max'), got {fake.reduce_calls}"

    # Case 2: no rank failed, collective returns max=0.0 → any_failed=False.
    fake = _FakeReduceAcceleratorMax(num_processes=2, reduced_flag=torch.tensor(0.0))
    trainer = Trainer.__new__(Trainer)
    trainer.accelerator = cast(Any, fake)
    trainer.compile_fallback_active = False
    trainer.compile_active = True
    any_failed = trainer._sync_compile_setup_ddp()
    assert any_failed is False
    assert fake.reduce_calls == ["max"]

    # Case 3: strict-mode failure via local_failed (compile_fallback_active stays False).
    fake = _FakeReduceAcceleratorMax(num_processes=2, reduced_flag=torch.tensor(1.0))
    trainer = Trainer.__new__(Trainer)
    trainer.accelerator = cast(Any, fake)
    trainer.compile_fallback_active = False
    trainer.compile_active = False
    any_failed = trainer._sync_compile_setup_ddp(local_failed=True)
    assert any_failed is True
    assert fake.reduce_calls == ["max"]

    # Case 4: single-process returns local flag directly, no collective.
    fake = _FakeReduceAcceleratorMax(num_processes=1, reduced_flag=None)
    trainer = Trainer.__new__(Trainer)
    trainer.accelerator = cast(Any, fake)
    trainer.compile_fallback_active = True
    any_failed = trainer._sync_compile_setup_ddp()
    assert any_failed is True
    assert fake.reduce_calls == [], "single-process must not call reduce"


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
    """When a compile target is active, CFG flags become 0-D bool tensors (branchless)."""
    mel, text, lens = _sample_batch(batch_size=2, frames=12, text_len=7, lens=[12, 8])

    dit_model = _build_model(audio_drop_prob=1.0, cond_drop_prob=0.0)
    dit_model.compile_training_core(backend="eager", fullgraph=False, dynamic=None)
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
    unett_model.compile_training_core(backend="eager", fullgraph=False, dynamic=None)
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


def test_cfg_flags_are_python_bools_on_never_compiled_default_path():
    """The default compile-disabled path must keep Python bool CFG flags (upstream parity).

    Tensorizing flags unconditionally added per-step host->device transfers and
    full-size zeros_like/where work that upstream's `if drop_text:` branches skipped on
    no-drop steps. The default path is documented as byte-identical to upstream, so it
    must use plain bools unless a compile target is actually active.
    """
    mel, text, lens = _sample_batch(batch_size=2, frames=12, text_len=7, lens=[12, 8])

    dit_model = _build_model(audio_drop_prob=1.0, cond_drop_prob=0.0)
    dit_prepared = cast(PreparedArgs, dit_model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    drop_audio_cond, drop_text = dit_prepared[6], dit_prepared[7]
    assert isinstance(drop_audio_cond, bool) and not torch.is_tensor(drop_audio_cond)
    assert isinstance(drop_text, bool) and not torch.is_tensor(drop_text)
    assert drop_audio_cond is True
    assert drop_text is False

    unett_model = _build_unett_model(audio_drop_prob=1.0, cond_drop_prob=0.0)
    unett_prepared = cast(PreparedArgs, unett_model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    unett_drop_audio_cond, unett_drop_text = unett_prepared[6], unett_prepared[7]
    assert isinstance(unett_drop_audio_cond, bool) and not torch.is_tensor(unett_drop_audio_cond)
    assert isinstance(unett_drop_text, bool) and not torch.is_tensor(unett_drop_text)
    assert unett_drop_audio_cond is True
    assert unett_drop_text is False


def test_cfg_flags_revert_to_bools_after_runtime_compile_fallback():
    """After clear_training_compile() (runtime fallback), flags must revert to bools.

    A compile failure that triggers eager fallback clears compile state, so
    _training_compile_enabled() returns False and flags become Python bools again --
    the fallback path must not keep paying the tensorization overhead.
    """
    mel, text, lens = _sample_batch(batch_size=2, frames=12, text_len=7, lens=[12, 8])

    model = _build_model(audio_drop_prob=1.0, cond_drop_prob=0.0)
    model.compile_training_core(backend="eager", fullgraph=False, dynamic=None)
    prepared = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    assert torch.is_tensor(prepared[6]) and torch.is_tensor(prepared[7]), "flags must be tensors while compiled"

    model.clear_training_compile()
    prepared = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    drop_audio_cond, drop_text = prepared[6], prepared[7]
    assert isinstance(drop_audio_cond, bool) and not torch.is_tensor(drop_audio_cond)
    assert isinstance(drop_text, bool) and not torch.is_tensor(drop_text)


def test_cfg_flags_are_tensors_for_dit_blocks_regional_compile_target():
    """target='dit_blocks' (transformer-level compile) must also tensorize CFG flags.

    _training_compile_enabled() covers both cfm_loss_core (_compiled_loss_core set) and
    dit_blocks (transformer training_compile_state enabled); the gate must not miss the
    regional target.
    """
    mel, text, lens = _sample_batch(batch_size=2, frames=12, text_len=7, lens=[12, 8])

    model = _build_model(audio_drop_prob=1.0, cond_drop_prob=0.0)
    model.transformer.compile_training_target("dit_blocks", backend="eager", fullgraph=False, dynamic=None)
    prepared = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))
    drop_audio_cond, drop_text = prepared[6], prepared[7]
    assert torch.is_tensor(drop_audio_cond) and drop_audio_cond.dtype is torch.bool
    assert torch.is_tensor(drop_text) and drop_text.dtype is torch.bool


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


def test_compiled_dropout_uses_its_own_rng_unless_fallback_random_is_set():
    """Document the one place compiled training is NOT numerically equal to eager.

    Inductor lowers nn.Dropout to its own Philox RNG, so with dropout > 0 -- which every
    shipped config has (0.1) -- the compiled loss differs from eager by far more than
    floating-point noise. This is expected torch.compile behaviour and statistically
    harmless (a different random mask is still a valid mask), but it means the suite's
    other parity tests, which all pin dropout=0.0, do not establish equality for the
    production configuration. Users who need eager-identical dropout can pass
    `options={"fallback_random": True}` through to torch.compile.

    This test exists so the claim stays honest and so a future PyTorch change to either
    behaviour is caught rather than silently altering training.
    """
    import pytest as _pytest

    def build():
        torch.manual_seed(7)
        model = CFM(
            transformer=DiT(
                dim=32, depth=1, heads=2, dim_head=16, mel_dim=8, text_num_embeds=32, text_dim=16, dropout=0.3
            ),
            mel_spec_kwargs={"n_mel_channels": 8},
            audio_drop_prob=0.0,
            cond_drop_prob=0.0,
        ).cpu()
        # dropout only changes the output if the weights are not zero-initialized
        for parameter in model.parameters():
            if parameter.dim() > 1:
                torch.nn.init.normal_(parameter, std=0.05)
        model.train()
        return model

    def run(**compile_kwargs):
        dynamo.reset()
        model = build()
        torch.manual_seed(3)
        prepared = cast(
            PreparedArgs,
            model._prepare_training_inputs(
                torch.randn(2, 40, 8), torch.randint(0, 32, (2, 11)), torch.tensor([40, 31])
            ),
        )
        torch.manual_seed(999)
        eager = model._forward_loss_core_components(*prepared)[0].item()
        model.compile_training_core(target="cfm_loss_core", runtime_fallback=False, **compile_kwargs)
        torch.manual_seed(999)
        compiled = model._run_loss_core_components(*prepared)[0].item()
        return eager, compiled

    try:
        eager, compiled = run(backend="inductor")
    except Exception as exc:  # inductor needs a working compiler toolchain
        _pytest.skip(f"inductor unavailable: {exc}")

    assert abs(eager - compiled) > 1e-5, (
        "compiled dropout unexpectedly matched eager; if PyTorch changed this, the "
        "documentation and the fallback_random guidance below need updating"
    )

    eager_fr, compiled_fr = run(backend="inductor", options={"fallback_random": True})
    assert abs(eager_fr - compiled_fr) < 1e-5, "fallback_random=True must restore eager RNG parity for dropout"


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
    """A real CUDA OOM must keep its original type/message so standard handlers match.

    The OOM is re-raised unchanged (bare ``raise``) with a compile-context note that does
    not alter ``str(exc)``; eager fallback must stay disabled for OOM.
    """
    model = _build_model()
    mel, text, lens = _sample_batch()
    prepared_args = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))

    def raise_oom(*_args):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    object.__setattr__(model, "_compiled_loss_core", raise_oom)
    object.__setattr__(model, "_compile_runtime_fallback", True)

    with pytest.raises(torch.cuda.OutOfMemoryError, match="CUDA out of memory") as exc_info:
        model._run_loss_core(*prepared_args)

    # Original type and message preserved; no synthetic rewrite.
    assert "CUDA out of memory" in str(exc_info.value)
    # Python 3.11+ supports exception notes; Python 3.10 is still supported by the
    # project, so absence of BaseException.add_note there must remain a clean no-op.
    notes = getattr(exc_info.value, "__notes__", None) or []
    if hasattr(exc_info.value, "add_note"):
        assert any("not falling back to eager" in note for note in notes)
    else:
        assert notes == []
    # Bare raise sets __context__ (not __cause__); there is no chained synthetic error.
    assert exc_info.value.__cause__ is None
    assert model.training_compile_state["enabled"] is True
    assert model.training_compile_state["fallback_active"] is False


def test_message_based_oom_runtimeerror_also_skips_fallback():
    """A message-based CUDA OOM RuntimeError keeps its original message and skips fallback."""
    model = _build_model()
    mel, text, lens = _sample_batch()
    prepared_args = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))

    def raise_oom_msg(*_args):
        raise RuntimeError("cuda runtime error: out of memory")

    object.__setattr__(model, "_compiled_loss_core", raise_oom_msg)
    object.__setattr__(model, "_compile_runtime_fallback", True)

    with pytest.raises(RuntimeError, match="out of memory") as exc_info:
        model._run_loss_core(*prepared_args)
    # Original message preserved (no 'ran out of GPU memory' rewrite).
    assert "cuda runtime error: out of memory" in str(exc_info.value)
    assert "ran out of GPU memory" not in str(exc_info.value)
    assert model.training_compile_state["fallback_active"] is False


def test_oom_still_raises_when_runtime_fallback_disabled():
    """With runtime fallback disabled, a CUDA OOM still raises the original exception."""
    model = _build_model()
    mel, text, lens = _sample_batch()
    prepared_args = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))

    def raise_oom(*_args):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    object.__setattr__(model, "_compiled_loss_core", raise_oom)
    object.__setattr__(model, "_compile_runtime_fallback", False)

    with pytest.raises(torch.cuda.OutOfMemoryError, match="CUDA out of memory"):
        model._run_loss_core(*prepared_args)
    assert model.training_compile_state["fallback_active"] is False


def test_dit_blocks_eager_region_oom_is_not_mislabeled_as_compiled_loss_core_oom():
    """Under target='dit_blocks' the CFM loss core runs eager; an OOM there must keep its
    original type/message and must NOT be relabeled as a 'compiled CFM loss core' OOM.

    With _compiled_loss_core is None (regional target), _run_loss_core_components takes the
    eager _forward_loss_core_components path (cfm.py); an OOM raised from that eager region
    must propagate unchanged.
    """
    model = _build_model()
    mel, text, lens = _sample_batch()
    prepared_args = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))

    # Simulate a dit_blocks setup: no compiled loss-core callable, but compile "enabled"
    # via the transformer state so _training_compile_enabled() is True and the eager
    # loss-core path is selected inside _run_loss_core_components.
    object.__setattr__(model, "_compiled_loss_core", None)
    object.__setattr__(model, "_compile_target", "dit_blocks")
    object.__setattr__(model, "_compile_runtime_fallback", True)
    object.__setattr__(model.transformer, "_dit_compile_target", "dit_blocks")

    original_components = model._forward_loss_core_components

    def raise_oom_eager(*_args):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 4.00 GiB")

    object.__setattr__(model, "_forward_loss_core_components", raise_oom_eager)
    try:
        with pytest.raises(torch.cuda.OutOfMemoryError, match="CUDA out of memory") as exc_info:
            model._run_loss_core(*prepared_args)
    finally:
        object.__setattr__(model, "_forward_loss_core_components", original_components)

    # The eager-region OOM must not be relabeled with a compiled-loss-core message.
    assert "ran out of GPU memory" not in str(exc_info.value)
    assert "CFM loss core" not in str(exc_info.value)
    assert model.training_compile_state["fallback_active"] is False


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


def test_dit_blocks_compile_target_allows_average_upsampling_outside_compiled_region(monkeypatch):
    """dit_blocks compile must actually invoke the compiled blocks for average-upsampling.

    The average-upsampling text-embedding path lives outside the compiled DiT-block region,
    so target='dit_blocks' must accept it. This test must run in train mode and assert the
    compiled callable is dispatched -- otherwise it only proves the eager eval path works
    (the dispatch keys on Module.training and _build_model returns an eval-mode model).
    """
    calls = _count_compiled_block_calls(monkeypatch)
    model = _build_model(average_upsampling=True)
    mel, text, lens = _sample_batch(batch_size=2, frames=12, text_len=7, lens=[12, 8])

    model.compile_training_core(target="dit_blocks", backend="eager", fullgraph=False, dynamic=None)
    model.train()
    loss, cond, pred = model(mel, text=text, lens=lens)

    assert torch.isfinite(loss)
    assert cond.shape == mel.shape
    assert pred.shape == mel.shape
    assert model.training_compile_state["enabled"] is True
    assert model.training_compile_state["target"] == "dit_blocks"
    transformer = cast(Any, model.transformer)
    assert calls["n"] >= len(transformer.transformer_blocks), (
        "train-mode forward must dispatch through every compiled block"
    )


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


class _TwoSampleDataset(torch.utils.data.Dataset):
    """Four samples with different mel lengths so masked-frame denoms differ."""

    def __init__(self, sample_count=4):
        gen = torch.Generator().manual_seed(2026)
        self.lengths = [8, 12, 16, 20]
        self.texts = ["ab", "cde", "abcdef", "long text"]
        self.mels = [torch.randn(8, f, generator=gen) for f in self.lengths]
        if not 1 <= sample_count <= len(self.mels):
            raise ValueError(f"sample_count must be in [1, {len(self.mels)}]")
        self.lengths = self.lengths[:sample_count]
        self.texts = self.texts[:sample_count]
        self.mels = self.mels[:sample_count]

    def __len__(self):
        return len(self.mels)

    def get_frame_len(self, index):
        return self.lengths[index]

    def __getitem__(self, index):
        return {"mel_spec": self.mels[index], "text": self.texts[index]}


def _run_trainer_one_update(global_masked_mean, tmp_path, monkeypatch, *, sample_count=4):
    """Run a real Trainer.train loop for two updates; return wiring + gradient evidence.

    Uses 4 samples with batch_size=1 and grad_accumulation_steps=2 → 2 updates, which
    avoids the LinearLR ZeroDivisionError that occurs with total_updates == warmup_updates.
    Gradients are captured at each sync_gradients (before zero_grad clears them).
    """
    from f5_tts.model import trainer as trainer_module
    from f5_tts.model.trainer import Trainer

    monkeypatch.setattr(trainer_module, "tqdm", _SilentProgress)

    torch.manual_seed(2026)
    model = _build_model(vocab_size=256)
    dataset = _TwoSampleDataset(sample_count=sample_count)

    backward_vals: list[float] = []
    scale_calls: list[float] = []
    component_vals: list[tuple[float, float, float]] = []
    captured_grads: list[list[torch.Tensor | None]] = []

    trainer = Trainer(
        model,
        epochs=1,
        learning_rate=0.0,
        num_warmup_updates=1,
        save_per_updates=10**9,
        keep_last_n_checkpoints=0,
        checkpoint_path=str(tmp_path / f"ckpt_{global_masked_mean}"),
        batch_size_per_gpu=1,
        batch_size_type="sample",
        grad_accumulation_steps=2,
        max_grad_norm=0.0,
        logger=None,
        log_samples=False,
        last_per_updates=10**9,
        compile_enabled=False,
        global_masked_mean=global_masked_mean,
    )

    # Record what backward receives.
    original_backward = trainer.accelerator.backward

    def recording_backward(loss, **kwargs):
        backward_vals.append(float(loss.detach().item()))
        return original_backward(loss, **kwargs)

    trainer.accelerator.backward = recording_backward

    # Record loss components (only called on the return_loss_components=True path).
    module = cast(Any, trainer._unwrapped_model)
    original_run = module._run_loss_core_components

    def recording_run(*args):
        result = original_run(*args)
        component_vals.append(
            (float(result[0].detach().item()), float(result[1].detach().item()), float(result[2].detach().item()))
        )
        return result

    object.__setattr__(module, "_run_loss_core_components", recording_run)

    # Record scaler calls.
    original_scale = trainer._scale_gradients_by_loss_denom

    def recording_scale(global_loss_denom):
        scale_calls.append(float(global_loss_denom.item()))
        return original_scale(global_loss_denom)

    trainer._scale_gradients_by_loss_denom = recording_scale

    # Capture grads before zero_grad clears them (only at sync_gradients).
    original_zero_grad = trainer.optimizer.zero_grad

    def capturing_zero_grad(*args, **kwargs):
        if trainer.accelerator.sync_gradients:
            captured_grads.append(
                [p.grad.detach().clone() if p.grad is not None else None for p in trainer._unwrapped_model.parameters()]
            )
        return original_zero_grad(*args, **kwargs)

    trainer.optimizer.zero_grad = capturing_zero_grad

    # Seed global RNG identically for both runs so forward passes match.
    torch.manual_seed(42)
    trainer.train(dataset, num_workers=0, resumable_with_seed=123)

    return backward_vals, scale_calls, component_vals, captured_grads


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


def _upstream_reference_loss(pred, flow, rand_span_mask):
    """Return the exact masked-mean reduction used by upstream commit 2ae2c9b."""
    loss = torch.nn.functional.mse_loss(pred, flow, reduction="none")
    return loss[rand_span_mask].mean()


def _base_commit_forward(model, mel, text, lens):
    """Transcribe the CFM forward *orchestration* from base commit 2ae2c9b (compile-disabled).

    Pinned to SWivid/F5-TTS 2ae2c9b ``src/f5_tts/model/cfm.py`` lines 231-302.

    Scope -- what this does and does not prove:
      * It transcribes the CFM-level orchestration only: stochastic input preparation
        (frac_lengths -> rand_span_mask, x0/time sampling, cond construction, bool CFG
        drop decisions) and the ``loss[rand_span_mask].mean()`` reduction.
      * It calls ``model.transformer`` -- the *current* DiT/UNetT implementation -- not a
        frozen base-commit transformer. Both this oracle and the current default path
        share the same transformer, so the transformer tree (TextEmbedding,
        InputEmbedding, DiTBlock, attention, projections) is a held constant, not a
        variable under test. This oracle therefore cannot detect a divergence between
        the current transformer and the base-commit transformer; it only proves the
        CFM orchestration wrapping it is unchanged.
      * Full-tree base parity (TextEmbedding/InputEmbedding/block internals) is out of
        scope here and is not claimed. Vendoring a frozen base-commit transformer would
        add that coverage but at large maintenance/drift cost for a regression surface
        already covered by the dedicated backbone tests in this file.

    Differences from the current ``_prepare_training_inputs`` + ``_run_loss_core`` path
    that this transcription preserves:
      * Bool CFG drop flags (base) vs 0-D bool tensors (current DiT path) -- but only
        when a compile target is active; the compile-disabled default path this oracle
        compares against also uses bool flags, so the flag dtype is not a variable here.
      * Inline stochastic preparation with no compile-friendly split.
      * ``loss[rand_span_mask].mean()`` reduction (no masked-multiply reassociation).

    No git-history runtime dependence: the body is straight-line code kept in lockstep
    with the named commit. If the base forward ever changes, update this transcription
    deliberately and record the new pinned SHA.
    """
    from random import random

    import torch.nn.functional as F

    from f5_tts.model.utils import exists, lens_to_mask, list_str_to_idx, list_str_to_tensor, mask_from_frac_lengths

    inp = mel  # test passes mel directly; base raw-wave branch omitted (not exercised)
    batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, model.device

    if isinstance(text, list):
        if exists(model.vocab_char_map):
            text = list_str_to_idx(text, model.vocab_char_map).to(device)
        else:
            text = list_str_to_tensor(text).to(device)
        assert text.shape[0] == batch

    if not exists(lens):
        lens = torch.full((batch,), seq_len, device=device)
    mask = lens_to_mask(lens, length=seq_len)

    frac_lengths = torch.zeros((batch,), device=model.device).float().uniform_(*model.frac_lengths_mask)
    rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)
    if exists(mask):
        rand_span_mask &= mask

    x1 = inp
    x0 = torch.randn_like(x1)
    time = torch.rand((batch,), dtype=dtype, device=model.device)

    t = time.unsqueeze(-1).unsqueeze(-1)
    phi = (1 - t) * x0 + t * x1
    flow = x1 - x0

    cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

    drop_audio_cond = random() < model.audio_drop_prob
    if random() < model.cond_drop_prob:
        drop_audio_cond = True
        drop_text = True
    else:
        drop_text = False

    pred = model.transformer(
        x=phi,
        cond=cond,
        text=text,
        time=time,
        drop_audio_cond=drop_audio_cond,
        drop_text=drop_text,
        mask=mask,
    )

    loss = F.mse_loss(pred, flow, reduction="none")
    loss = loss[rand_span_mask]
    return loss.mean(), cond, pred, rand_span_mask


def test_default_path_matches_base_commit_forward_end_to_end():
    """The compile-disabled default path's CFM orchestration must match base 2ae2c9b.

    The previous parity test compared the new path's prediction against itself (it fed the
    same ``pred`` into a reduction helper), so changes to the stochastic input preparation
    or the loss reduction could alter outputs while the test stayed green. This replacement
    runs a transcribed base-commit CFM orchestration and the current default path
    (``_prepare_training_inputs`` + ``_run_loss_core``) from identical model state and RNG
    state, then asserts bitwise equality of predictions, conditioning, loss, and the span
    mask end-to-end.

    Scope: both paths call the *same* ``model.transformer``, so this proves the CFM-level
    orchestration (stochastic preparation, cond construction, reduction form) is unchanged
    -- not full-tree base parity. A divergence in TextEmbedding/InputEmbedding/block
    internals would change both paths equally and stay green here; that regression surface
    is covered by the dedicated backbone tests (e.g. ``test_dit_text_embed_*``,
    ``test_branchless_tensor_cfg_flags_match_bool_loss_outputs_and_gradients``) rather than
    by this orchestration oracle.

    With ``audio_drop_prob=cond_drop_prob=0`` the Python ``random()`` drop decisions are
    deterministic, and the compile-disabled default path keeps bool CFG flags, so the two
    paths exercise identical flag dtypes -- the only remaining variable is the orchestration
    split, which is exactly what this test verifies.
    """
    model = _build_model()
    for seed in range(8):
        torch.manual_seed(seed)
        mel = torch.randn(3, 37, 8)
        text = torch.randint(0, 32, (3, 11))
        lens = torch.tensor([37, 29, 33])

        # Base-commit forward from a cloned model so weights are identical to the new path.
        torch.manual_seed(seed)
        base_model = copy.deepcopy(model)
        base_loss, base_cond, base_pred, base_mask = _base_commit_forward(base_model, mel, text, lens)

        # Current default path (compile-disabled) from the same seed/state.
        torch.manual_seed(seed)
        prepared = cast(PreparedArgs, model._prepare_training_inputs(mel, text, lens))
        new_loss, new_cond, new_pred = model._run_loss_core(*prepared)
        _, _, _, new_mask, _, _, _, _ = prepared

        assert torch.equal(new_mask, base_mask), f"seed {seed}: rand_span_mask diverged"
        assert torch.equal(new_pred, base_pred), f"seed {seed}: pred diverged"
        assert torch.equal(new_cond, base_cond), f"seed {seed}: cond diverged"
        assert torch.equal(new_loss, base_loss), f"seed {seed}: loss diverged"


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


def test_dit_text_embed_keeps_one_dynamic_graph_across_sequence_lengths():
    """TextEmbedding must preserve symbolic sequence lengths instead of specializing."""
    from f5_tts.model.backbones.dit import TextEmbedding

    torch._dynamo.reset()
    captured_graphs = []

    def counting_backend(graph_module, _example_inputs):
        captured_graphs.append(graph_module)
        return graph_module.forward

    embed = TextEmbedding(text_num_embeds=32, text_dim=8, mask_padding=False)

    def run(text, valid_seq_lens):
        return embed(
            text,
            seq_len=text.shape[1],
            drop_text=False,
            valid_seq_lens=valid_seq_lens,
        )

    compiled = torch.compile(run, backend=counting_backend, fullgraph=True, dynamic=True)
    # Keep batch size fixed so this assertion isolates sequence-length symbolism.
    # PyTorch may legitimately specialize batch=1 separately from batch>1 even with
    # dynamic=True; that is not sequence-length graph churn.
    for batch, seq_len in ((2, 7), (2, 11), (2, 15)):
        text = torch.randint(1, 32, (batch, seq_len))
        valid_seq_lens = torch.full((batch,), seq_len, dtype=torch.long)
        output = compiled(text, valid_seq_lens)
        assert output.shape == (batch, seq_len, 8)

    assert len(captured_graphs) == 1, (
        f"symbolic batch/sequence lengths should reuse one graph, captured {len(captured_graphs)}"
    )


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


def test_post_fallback_default_path_keeps_fp32_components_reduction():
    """After a runtime compile fallback, the default (non-components) loss path must keep
    the fp32 masked-multiply reduction, not revert to the upstream input-dtype reduction.

    The upstream-exact path computes ``F.mse_loss(pred, flow)`` in the input dtype; an fp16
    AMP run that fell back mid-training would switch to fp16 squared error and could
    overflow. The post-fallback path reuses ``_forward_loss_core_components`` (fp32), which
    is numerically safe and matches the reduction the run used while compiled.
    """
    model = _build_model()
    mel, text, lens = _sample_batch()
    prepared = cast(PreparedArgs, model._prepare_training_inputs(mel.clone(), text.clone(), lens.clone()))

    # Enable compile (eager backend keeps the test CPU-fast and deterministic).
    model.compile_training_core(backend="eager", fullgraph=False, dynamic=None)
    assert model.training_compile_state["enabled"] is True

    # Inject a compiled callable that raises a compiler failure, then run once to trigger
    # the runtime fallback (clear_training_compile + _compile_fallback_active=True).
    def raise_compile_error(*_args):
        raise _synthetic_compiler_error()

    object.__setattr__(model, "_compiled_loss_core", raise_compile_error)
    object.__setattr__(model, "_compile_runtime_fallback", True)
    loss_first, _, _ = model._run_loss_core(*prepared)
    assert torch.isfinite(loss_first)
    assert model.training_compile_state["fallback_active"] is True
    assert model.training_compile_state["enabled"] is False

    # Subsequent default-path calls must use the fp32 components reduction.
    loss_again, _, _ = model._run_loss_core(*prepared)
    components_loss, _sum, _denom, _cond, _pred = model._forward_loss_core_components(*prepared)
    assert loss_again.dtype == torch.float32
    assert torch.equal(loss_again, components_loss)


def test_post_fallback_fp32_reduction_matches_active_compiled_reduction():
    """The post-fallback fp32 reduction must equal the reduction the run used while
    compiled, so a mid-training fallback does not change the loss objective (only the
    backend, eager vs compiled). Compares the post-fallback loss against a freshly
    compiled-path loss from identical inputs.
    """
    model = _build_model()
    torch.manual_seed(0)
    mel = torch.randn(3, 37, 8)
    text = torch.randint(0, 32, (3, 11))
    lens = torch.tensor([37, 29, 33])
    prepared = cast(PreparedArgs, model._prepare_training_inputs(mel, text, lens))

    # Active-compiled reduction (eager backend: same reduction, no real compilation).
    model.compile_training_core(backend="eager", fullgraph=False, dynamic=None)
    compiled_loss, _, _ = model._run_loss_core(*prepared)

    # Force fallback.
    def raise_compile_error(*_args):
        raise _synthetic_compiler_error()

    object.__setattr__(model, "_compiled_loss_core", raise_compile_error)
    object.__setattr__(model, "_compile_runtime_fallback", True)
    model._run_loss_core(*prepared)  # triggers fallback
    assert model.training_compile_state["fallback_active"] is True

    # Same inputs -> post-fallback eager fp32 reduction must equal the compiled reduction.
    fallback_loss, _, _ = model._run_loss_core(*prepared)
    assert torch.equal(fallback_loss, compiled_loss)


def test_never_compiled_default_path_stays_bit_exact_upstream_after_fallback_fix():
    """The finding-6 fix must NOT change the never-compiled default path.

    A model whose compile was never enabled (_compile_fallback_active stays False) must
    still take the byte-exact upstream reduction, preserving the exact-default contract.
    """
    model = _build_model()
    assert model.training_compile_state["fallback_active"] is False
    assert model.training_compile_state["enabled"] is False

    mismatches = 0
    for seed in range(10):
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
    assert mismatches == 0, f"never-compiled default diverged from upstream on {mismatches}/10 seeds"
