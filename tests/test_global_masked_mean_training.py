"""End-to-end regressions for global masked-mean update accounting."""

from __future__ import annotations

import contextlib

import torch
from accelerate.utils import DistributedType
from torch import nn

import f5_tts.model.trainer as trainer_module
from f5_tts.model.trainer import Trainer


class _BatchLoader:
    def __init__(self, count=1):
        self.batch_sampler = object()
        self.count = count
        self.batch = {
            "mel": torch.ones((1, 2, 3)),
            "mel_lengths": torch.tensor([3]),
            "text": ["cursor"],
        }

    def __len__(self):
        return self.count

    def __iter__(self):
        for _ in range(self.count):
            yield self.batch


class _TrainingModel(nn.Module):
    num_channels = 2

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.forward_calls = 0

    def sample_training_mask(self, lens, seq_len):
        return torch.ones((lens.shape[0], seq_len), dtype=torch.bool)

    def forward(
        self,
        mel,
        *,
        text,
        lens,
        noise_scheduler,
        rand_span_mask,
        return_loss_components,
    ):
        del text, lens, noise_scheduler
        assert return_loss_components is True
        self.forward_calls += 1
        loss_count = rand_span_mask.sum(dtype=torch.int64) * mel.shape[-1]
        loss_sum = self.weight.square() * loss_count
        loss = loss_sum / loss_count
        return loss, loss_sum, loss_count, mel, mel


class _ControlledSGD(torch.optim.SGD):
    def __init__(self, params, *, accelerator, skip_update):
        super().__init__(params, lr=0.01)
        self.accelerator = accelerator
        self.skip_update = skip_update
        self.attempted_steps = 0
        self.last_gradient = None

    def step(self, closure=None):
        if not self.accelerator.sync_gradients:
            return None
        self.attempted_steps += 1
        [parameter] = self.param_groups[0]["params"]
        self.last_gradient = parameter.grad.detach().clone()
        if self.skip_update:
            return None
        return super().step(closure)


class _CountingEMA:
    def __init__(self):
        self.updates = 0

    def update(self):
        self.updates += 1


class _TrainingAccelerator:
    def __init__(self, *, optimizer_step_was_skipped):
        self.device = torch.device("cpu")
        self.num_processes = 1
        self.distributed_type = DistributedType.NO
        self.dispatch_batches = False
        self.sync_gradients = True
        self.optimizer_step_was_skipped = optimizer_step_was_skipped
        self.is_main_process = True
        self.is_local_main_process = True
        self.skip_calls = []
        self.end_calls = 0

    def prepare_data_loader(self, dataloader, device_placement):
        assert device_placement is False
        return dataloader

    def prepare(self, value):
        return value

    def gather(self, value):
        return value

    def reduce(self, value, reduction):
        assert reduction == "sum"
        return value

    def no_sync(self, model):
        del model
        return contextlib.nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def skip_first_batches(self, dataloader, num_batches):
        self.skip_calls.append(num_batches)
        if num_batches == 0:
            return dataloader
        return list(dataloader)[num_batches:]

    def log(self, *args, **kwargs):
        del args, kwargs

    def end_training(self):
        self.end_calls += 1


def _trainer(
    *,
    overflow=False,
    update=0,
    consumed_batches=0,
    grad_accumulation_steps=1,
    epochs=1,
    max_grad_norm: float = 0.0,
):
    model = _TrainingModel()
    accelerator = _TrainingAccelerator(optimizer_step_was_skipped=overflow)
    trainer = Trainer.__new__(Trainer)
    trainer.model = model
    trainer._unwrapped_model = model
    trainer.accelerator = accelerator
    trainer.optimizer = _ControlledSGD(model.parameters(), accelerator=accelerator, skip_update=overflow)
    trainer.ema_model = _CountingEMA()
    trainer.global_masked_mean = True
    trainer.grad_accumulation_steps = grad_accumulation_steps
    trainer.compile_active = False
    trainer._loaded_consumed_batches = 0
    trainer.max_grad_norm = max_grad_norm
    trainer.log_samples = False
    trainer.batch_size_type = "sample"
    trainer.batch_size_per_gpu = 1
    trainer.max_samples = 1
    trainer.num_warmup_updates = 0
    trainer.epochs = epochs
    trainer.duration_predictor = None
    trainer.noise_scheduler = None
    trainer.logger = None
    trainer.last_per_updates = 100
    trainer.save_per_updates = 100
    trainer.saved = []

    def load_checkpoint():
        trainer._loaded_consumed_batches = consumed_batches
        return update

    trainer.load_checkpoint = load_checkpoint
    trainer.save_checkpoint = lambda saved_update, last=False, consumed_batches=None: trainer.saved.append(
        (saved_update, consumed_batches, last)
    )
    return trainer


def test_successful_train_scales_loss_sum_before_backward(monkeypatch):
    monkeypatch.setattr(trainer_module, "DataLoader", lambda *args, **kwargs: _BatchLoader())
    trained = _trainer()

    trained.train(object(), num_workers=0)

    # loss_sum = weight**2 * 6 and production scale is 1 / 6.
    torch.testing.assert_close(trained.optimizer.last_gradient, torch.tensor(1.0))
    torch.testing.assert_close(trained.model.weight, torch.tensor(0.49))
    assert trained.ema_model.updates == 1
    assert trained.saved[-1] == (1, 1, True)


def test_gradient_clipping_runs_after_global_normalization(monkeypatch):
    monkeypatch.setattr(trainer_module, "DataLoader", lambda *args, **kwargs: _BatchLoader())
    trained = _trainer(max_grad_norm=0.5)

    trained.train(object(), num_workers=0)

    torch.testing.assert_close(trained.optimizer.last_gradient, torch.tensor(0.5))
    torch.testing.assert_close(trained.model.weight, torch.tensor(0.495))


def test_overflow_consumes_batch_without_advancing_update_or_ema(monkeypatch):
    monkeypatch.setattr(trainer_module, "DataLoader", lambda *args, **kwargs: _BatchLoader())
    overflowed = _trainer(overflow=True)

    overflowed.train(object(), num_workers=0)

    assert overflowed.optimizer.attempted_steps == 1
    assert overflowed.ema_model.updates == 0
    assert overflowed.model.forward_calls == 1
    assert overflowed.saved[-1] == (0, 1, True)

    resumed = _trainer(update=0, consumed_batches=1)
    resumed.train(object(), num_workers=0, resumable_with_seed=123)
    assert resumed.accelerator.skip_calls == [0]
    assert resumed.model.forward_calls == 0
    assert resumed.optimizer.attempted_steps == 0
    assert resumed.saved[-1] == (0, 1, True)


def test_partial_window_checkpoint_tracks_batches_not_windows_times_g(monkeypatch):
    monkeypatch.setattr(trainer_module, "DataLoader", lambda *args, **kwargs: _BatchLoader(count=5))
    first = _trainer(grad_accumulation_steps=3)

    first.train(object(), num_workers=0)

    assert first.model.forward_calls == 5
    assert first.saved[-1] == (2, 5, True)

    resumed = _trainer(update=2, consumed_batches=5, grad_accumulation_steps=3, epochs=2)
    resumed.train(object(), num_workers=0, resumable_with_seed=123)
    assert resumed.accelerator.skip_calls == [0]
    assert resumed.model.forward_calls == 5
    assert resumed.saved[-1] == (4, 10, True)
