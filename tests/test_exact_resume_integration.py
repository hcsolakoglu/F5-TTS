"""Trainer-path exact-resume coverage with lightweight tensor fixtures."""

from __future__ import annotations

import contextlib
from itertools import islice
from typing import Any, cast

import pytest
import torch
from accelerate.state import AcceleratorState
from accelerate.utils import DistributedType, GradientAccumulationPlugin
from torch import nn
from torch.utils.data import Dataset

from f5_tts.model.trainer import Trainer


class _ResumeDataset(Dataset):
    supports_exact_resume = True
    exact_resume_signature = "integration-resume-dataset-v1"

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {
            "mel_spec": torch.full((2, 3), float(index + 1)),
            "text": "resume",
        }


class _ResumeModel(nn.Module):
    supports_exact_resume = True
    exact_resume_signature = "test-resume-model-v1"

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))

    def forward(self, mel, *, text, lens, noise_scheduler):
        del text, lens, noise_scheduler
        target = mel.mean()
        loss = (self.weight - target).square()
        return loss, mel, mel


class _ExplicitAdjustPlugin(GradientAccumulationPlugin):
    def to_kwargs(self):
        return {
            "num_steps": self.num_steps,
            "adjust_scheduler": True,
            "sync_with_dataloader": self.sync_with_dataloader,
        }


class _ResumeEMA:
    def __init__(self):
        self.updates = 0

    def update(self):
        self.updates += 1

    def state_dict(self):
        return {"updates": self.updates}

    def load_state_dict(self, state):
        self.updates = state["updates"]


class _FakeScaler:
    def __init__(self, scale):
        self.scale = scale

    def state_dict(self):
        return {"scale": self.scale}

    def load_state_dict(self, state):
        self.scale = state["scale"]


class _ResumeAccelerator:
    device = torch.device("cpu")
    distributed_type = DistributedType.NO
    num_processes = 1
    num_machines = 1
    process_index = 0
    mixed_precision = "no"
    scaler = None
    dispatch_batches = False
    split_batches = False
    even_batches = False
    step_scheduler_with_optimizer = False
    sync_gradients = True
    optimizer_step_was_skipped = False
    is_main_process = True
    is_local_main_process = True

    def __init__(self, scaler_scale=128):
        self.scaler = _FakeScaler(scaler_scale)
        self.skip_calls = []
        self.end_calls = 0

    def prepare_data_loader(self, dataloader, device_placement):
        assert device_placement is True
        return dataloader

    def prepare(self, *objects):
        if len(objects) > 1:
            return objects
        return cast(Any, next(iter(objects)))

    def unwrap_model(self, model):
        return model

    def accumulate(self, model):
        del model
        return contextlib.nullcontext()

    def no_sync(self, model):
        del model
        return contextlib.nullcontext()

    def autocast(self):
        return contextlib.nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def skip_first_batches(self, dataloader, num_batches):
        self.skip_calls.append(num_batches)
        return islice(dataloader, num_batches, None)

    def wait_for_everyone(self):
        return None

    def save(self, state, path):
        torch.save(state, path)

    def reduce(self, value, reduction):
        assert reduction == "sum"
        return value

    def gather(self, value):
        return value

    def log(self, *args, **kwargs):
        return None

    def end_training(self):
        self.end_calls += 1


def _make_trainer(checkpoint_path, model, *, scaler_scale=128):
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.model = model
    trainer._unwrapped_model = model
    trainer.accelerator = _ResumeAccelerator(scaler_scale)
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer.ema_model = _ResumeEMA()
    trainer.global_masked_mean = False
    trainer.grad_accumulation_steps = 1
    trainer.max_grad_norm = 0
    trainer.log_samples = False
    trainer.batch_size_type = "sample"
    trainer.batch_size_per_gpu = 2
    trainer.max_samples = 2
    trainer.num_warmup_updates = 0
    trainer.epochs = 1
    trainer.duration_predictor = None
    trainer.noise_scheduler = None
    trainer.compile_active = False
    trainer.logger = None
    trainer.last_per_updates = 1
    trainer.save_per_updates = 100
    trainer.keep_last_n_checkpoints = -1
    trainer.checkpoint_path = str(checkpoint_path)
    trainer.vocoder_name = "vocos"
    return trainer


def _interrupt_after_first_checkpoint(trainer):
    save_checkpoint = trainer.save_checkpoint

    def save_then_interrupt(update, consumed_updates=None, last=False):
        save_checkpoint(update, consumed_updates, last)
        if update == 1 and last:
            raise RuntimeError("synthetic interruption after checkpoint")

    trainer.save_checkpoint = save_then_interrupt


def test_trainer_init_uses_effective_plugin_without_mutating_mapping(tmp_path):
    AcceleratorState._reset_state(reset_partial_state=True)
    plugin = GradientAccumulationPlugin(num_steps=2, sync_with_dataloader=False, adjust_scheduler=True)
    accelerate_kwargs = {"gradient_accumulation_plugin": plugin, "cpu": True}

    try:
        trainer = Trainer(
            cast(Any, _ResumeModel()),
            epochs=1,
            learning_rate=0.1,
            logger=None,
            checkpoint_path=str(tmp_path / "ckpts"),
            accelerate_kwargs=accelerate_kwargs,
        )

        assert accelerate_kwargs == {"gradient_accumulation_plugin": plugin, "cpu": True}
        assert trainer._get_effective_gradient_accumulation_contract() == (2, False, False)
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)


def _make_real_cpu_trainer(checkpoint_path, model, accelerate_kwargs=None):
    effective_accelerate_kwargs = {"cpu": True}
    if accelerate_kwargs is not None:
        effective_accelerate_kwargs.update(accelerate_kwargs)
    return Trainer(
        cast(Any, model),
        epochs=1,
        learning_rate=0.1,
        num_warmup_updates=0,
        save_per_updates=1,
        keep_last_n_checkpoints=-1,
        batch_size_per_gpu=2,
        logger=None,
        checkpoint_path=str(checkpoint_path),
        accelerate_kwargs=effective_accelerate_kwargs,
    )


def test_real_cpu_trainer_resume_matches_uninterrupted_trajectory(tmp_path):
    AcceleratorState._reset_state(reset_partial_state=True)
    dataset = _ResumeDataset()

    try:
        torch.manual_seed(123)
        uninterrupted = _make_real_cpu_trainer(tmp_path / "real-uninterrupted", _ResumeModel())
        uninterrupted.train(dataset, num_workers=0, resumable_with_seed=77, resume_mode="exact")

        torch.manual_seed(123)
        interrupted = _make_real_cpu_trainer(tmp_path / "real-resumed", _ResumeModel())
        _interrupt_after_first_checkpoint(interrupted)
        with pytest.raises(RuntimeError, match="synthetic interruption"):
            interrupted.train(dataset, num_workers=0, resumable_with_seed=77, resume_mode="exact")

        resumed = _make_real_cpu_trainer(tmp_path / "real-resumed", _ResumeModel())
        resumed.train(dataset, num_workers=0, resumable_with_seed=77, resume_mode="exact")

        torch.testing.assert_close(resumed.model.weight, uninterrupted.model.weight)
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)


def test_real_accelerator_adjust_scheduler_steps_each_microbatch(tmp_path):
    AcceleratorState._reset_state(reset_partial_state=True)
    plugin = _ExplicitAdjustPlugin(num_steps=2, sync_with_dataloader=True, adjust_scheduler=True)

    try:
        trainer = _make_real_cpu_trainer(
            tmp_path / "adjusted-scheduler",
            _ResumeModel(),
            {"gradient_accumulation_plugin": plugin},
        )
        trainer.train(_ResumeDataset(), num_workers=0, resumable_with_seed=77, resume_mode="best_effort")

        scheduler = cast(Any, trainer.scheduler).scheduler
        assert trainer._get_effective_gradient_accumulation_contract() == (2, True, True)
        assert trainer._scheduler_signature["total_raw_steps"] == 1
        assert scheduler._step_count == 3
        assert scheduler.last_epoch == 1
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)


def test_exact_resume_matches_uninterrupted_trainer_trajectory(tmp_path):
    dataset = _ResumeDataset()

    torch.manual_seed(123)
    uninterrupted = _make_trainer(tmp_path / "uninterrupted", _ResumeModel())
    uninterrupted.train(dataset, num_workers=0, resumable_with_seed=77, resume_mode="exact")

    torch.manual_seed(123)
    interrupted = _make_trainer(tmp_path / "resumed", _ResumeModel())
    _interrupt_after_first_checkpoint(interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        interrupted.train(dataset, num_workers=0, resumable_with_seed=77, resume_mode="exact")

    resumed = _make_trainer(tmp_path / "resumed", _ResumeModel(), scaler_scale=999)
    resumed.train(dataset, num_workers=0, resumable_with_seed=77, resume_mode="exact")

    torch.testing.assert_close(resumed.model.weight, uninterrupted.model.weight)
    assert cast(Any, resumed.accelerator).skip_calls == [1]
    assert cast(Any, resumed.accelerator).scaler.scale == 128
    assert resumed.ema_model.updates == uninterrupted.ema_model.updates
