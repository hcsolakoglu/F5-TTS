from copy import deepcopy
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from accelerate import Accelerator as RealAccelerator

import f5_tts.model.trainer as trainer_module
from f5_tts.model import CFM
from f5_tts.model.trainer import Trainer


class _Accelerator:
    is_main_process = False
    num_processes = 1

    def __init__(self, device_type="cpu"):
        self.device = torch.device(device_type)

    def prepare(self, *objects):
        return objects


class _LoadableModel:
    def load_state_dict(self, state_dict):
        self.loaded_state_dict = state_dict


def _patch_accelerator(monkeypatch, device_type="cpu"):
    monkeypatch.setattr(trainer_module, "Accelerator", lambda *args, **kwargs: _Accelerator(device_type))


def _model(device="cpu"):
    return cast(CFM, torch.nn.Linear(1, 1, device=device))


def test_trainer_prefers_fused_adamw_when_available(monkeypatch):
    _patch_accelerator(monkeypatch, "cuda")
    captured = {}

    def capture_adamw(params, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(trainer_module, "AdamW", capture_adamw)

    Trainer(_model(), epochs=1, learning_rate=1e-4, logger=None)

    assert captured["fused"] is True


def test_trainer_falls_back_when_fused_adamw_is_unavailable(monkeypatch):
    _patch_accelerator(monkeypatch, "cuda")
    calls = []

    def reject_fused_adamw(params, **kwargs):
        calls.append(kwargs)
        if kwargs.get("fused"):
            raise RuntimeError("`fused=True` requires all the params to be CUDA, floating point Tensor")
        return SimpleNamespace()

    monkeypatch.setattr(trainer_module, "AdamW", reject_fused_adamw)

    Trainer(_model(), epochs=1, learning_rate=1e-4, logger=None)

    assert calls[0]["fused"] is True
    assert "fused" not in calls[1]


@pytest.mark.parametrize(("target_device", "source_device"), [("cpu", "meta"), ("mps", "cpu")])
def test_trainer_does_not_try_fused_adamw_on_other_target_devices(monkeypatch, target_device, source_device):
    _patch_accelerator(monkeypatch, target_device)
    calls = []

    def capture_adamw(params, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(trainer_module, "AdamW", capture_adamw)

    Trainer(_model(source_device), epochs=1, learning_rate=1e-4, logger=None)

    assert len(calls) == 1
    assert "fused" not in calls[0]


def test_trainer_does_not_hide_unrelated_adamw_errors(monkeypatch):
    _patch_accelerator(monkeypatch, "cuda")

    def reject_adamw(params, **kwargs):
        raise RuntimeError("unexpected optimizer failure with fused=True enabled")

    monkeypatch.setattr(trainer_module, "AdamW", reject_adamw)

    with pytest.raises(RuntimeError, match="unexpected optimizer failure with fused=True enabled"):
        Trainer(_model(), epochs=1, learning_rate=1e-4, logger=None)


def test_resume_uses_current_adamw_fused_policy(monkeypatch, tmp_path):
    checkpoint_file = tmp_path / "model_last.pt"
    checkpoint_file.touch()

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter])
    optimizer_state_dict = optimizer.state_dict()
    optimizer_state_dict["param_groups"][0]["fused"] = True
    checkpoint = {
        "ema_model_state_dict": {},
        "model_state_dict": {},
        "optimizer_state_dict": optimizer_state_dict,
        "update": 7,
    }
    monkeypatch.setattr(trainer_module.torch, "load", lambda *args, **kwargs: checkpoint)

    model = _LoadableModel()
    accelerator = SimpleNamespace(
        is_main_process=False,
        wait_for_everyone=lambda: None,
        unwrap_model=lambda wrapped_model: wrapped_model,
    )
    trainer = Trainer.__new__(Trainer)
    trainer.__dict__.update(
        checkpoint_path=str(tmp_path),
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        _uses_torch_adamw=True,
    )

    assert trainer.load_checkpoint() == 7
    assert optimizer.param_groups[0]["fused"] is None

    parameter.grad = torch.ones_like(parameter)
    optimizer.step()


def test_resume_normalizes_all_accelerated_optimizer_groups():
    first = torch.nn.Parameter(torch.tensor([1.0]))
    second = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.AdamW([{"params": [first]}, {"params": [second]}])
    optimizer = RealAccelerator(cpu=True).prepare(optimizer)
    optimizer_state_dict = optimizer.state_dict()
    for param_group in optimizer_state_dict["param_groups"]:
        param_group["fused"] = True

    trainer = Trainer.__new__(Trainer)
    trainer.__dict__.update(optimizer=optimizer, _uses_torch_adamw=True)
    trainer._normalize_adamw_fused_policy(optimizer_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)

    assert [group["fused"] for group in optimizer.param_groups] == [None, None]
    first.grad = torch.ones_like(first)
    second.grad = torch.ones_like(second)
    optimizer.step()


def test_resume_does_not_change_bitsandbytes_optimizer_state():
    optimizer_state_dict = {"param_groups": [{"params": [0], "bnb_policy": "saved-8bit"}]}
    expected_state_dict = deepcopy(optimizer_state_dict)
    trainer = Trainer.__new__(Trainer)
    trainer.__dict__.update(
        optimizer=SimpleNamespace(param_groups=[{"params": [0], "bnb_policy": "current-8bit"}]),
        _uses_torch_adamw=False,
    )

    trainer._normalize_adamw_fused_policy(optimizer_state_dict)

    assert optimizer_state_dict == expected_state_dict
