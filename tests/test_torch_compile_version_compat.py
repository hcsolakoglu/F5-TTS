from __future__ import annotations

import functools
from typing import Any, cast

import pytest

import f5_tts.model.trainer as trainer_module
from f5_tts.model.trainer import Trainer


def _legacy_compile(
    model=None,
    *,
    fullgraph: bool = False,
    dynamic: bool = False,
    backend: str = "inductor",
    mode: str | None = None,
):
    return model


def _modern_compile(
    model=None,
    *,
    fullgraph: bool = False,
    dynamic: bool | None = None,
    backend: str = "inductor",
    mode: str | None = None,
):
    return model


@functools.wraps(_legacy_compile)
def _wrapped_legacy_compile(*args, **kwargs):
    return _legacy_compile(*args, **kwargs)


def _opaque_compile(*args, **kwargs):
    return args[0] if args else None


class _UninspectableCompile:
    @property
    def __signature__(self):
        raise RuntimeError("signature metadata is unavailable")

    def __call__(self, model=None, **kwargs):
        return model


class _RecordingModel:
    def __init__(self):
        self.compile_kwargs: dict[str, Any] | None = None

    def compile_training_core(self, **kwargs):
        self.compile_kwargs = kwargs


class _SingleProcessAccelerator:
    num_processes = 1
    is_main_process = False


def _configure_with_compile_callable(monkeypatch, compile_callable, requested_dynamic):
    monkeypatch.setattr(trainer_module.torch, "compile", compile_callable)

    model = _RecordingModel()
    trainer = Trainer.__new__(Trainer)
    trainer.compile_enabled = True
    trainer.compile_backend = "eager"
    trainer.compile_target = "cfm_loss_core"
    trainer.compile_mode = None
    trainer.compile_fullgraph = False
    trainer.compile_dynamic = requested_dynamic
    trainer.compile_fallback_to_eager = False
    trainer.compile_active = False
    trainer.compile_fallback_active = False
    trainer._unwrapped_model = cast(Any, model)
    trainer.accelerator = cast(Any, _SingleProcessAccelerator())

    trainer._configure_compile()
    assert model.compile_kwargs is not None
    return model.compile_kwargs


@pytest.mark.parametrize("compile_callable", [_legacy_compile, _wrapped_legacy_compile])
def test_auto_dynamic_enables_dynamic_shapes_for_legacy_false_default(monkeypatch, compile_callable):
    compile_kwargs = _configure_with_compile_callable(monkeypatch, compile_callable, None)

    assert compile_kwargs["dynamic"] is True


def test_auto_dynamic_keeps_modern_none_default(monkeypatch):
    compile_kwargs = _configure_with_compile_callable(monkeypatch, _modern_compile, None)

    assert "dynamic" not in compile_kwargs


@pytest.mark.parametrize("requested_dynamic", [False, True])
@pytest.mark.parametrize("compile_callable", [_legacy_compile, _modern_compile])
def test_explicit_dynamic_choice_is_preserved(monkeypatch, compile_callable, requested_dynamic):
    compile_kwargs = _configure_with_compile_callable(monkeypatch, compile_callable, requested_dynamic)

    assert compile_kwargs["dynamic"] is requested_dynamic


@pytest.mark.parametrize("compile_callable", [_opaque_compile, _UninspectableCompile()])
def test_auto_dynamic_does_not_guess_when_signature_is_unavailable(monkeypatch, compile_callable):
    compile_kwargs = _configure_with_compile_callable(monkeypatch, compile_callable, None)

    assert "dynamic" not in compile_kwargs
