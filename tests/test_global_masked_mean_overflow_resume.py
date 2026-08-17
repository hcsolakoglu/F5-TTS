"""Focused regression for overflow accounting and resume position."""

from __future__ import annotations

import contextlib

import torch
from accelerate.utils import DistributedType
from torch import nn

import f5_tts.model.trainer as trainer_module
from f5_tts.model.trainer import Trainer


class _OneBatchLoader:
    def __init__(self):
        self.batch_sampler = object()
        self.batch = {
            "mel": torch.ones((1, 2, 3)),
            "mel_lengths": torch.tensor([3]),
            "text": ["overflow"],
        }

    def __len__(self):
        return 1

    def __iter__(self):
        yield self.batch


class _OverflowModel(nn.Module):
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
        rand_span_mask=None,
        return_loss_components=False,
    ):
        del text, lens, noise_scheduler
        self.forward_calls += 1
        if rand_span_mask is None:
            rand_span_mask = torch.ones(mel.shape[0], mel.shape[-1], dtype=torch.bool)
        denominator = rand_span_mask.sum(dtype=torch.float32) * mel.shape[-1]
        loss_sum = self.weight.square() * denominator
        loss = loss_sum / denominator
        if return_loss_components:
            return loss, loss_sum, denominator, mel, mel
        return loss, mel, mel


class _ControlledSGD(torch.optim.SGD):
    """Capture the production gradient and optionally simulate an AMP skip."""

    def __init__(self, params, *, skip_update):
        super().__init__(params, lr=0.01)
        self.skip_update = skip_update
        self.attempted_steps = 0
        self.last_gradient = None

    def step(self, closure=None):
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


class _OverflowAccelerator:
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

    def prepare(self, *objects):
        return objects if len(objects) > 1 else objects[0]

    def gather(self, value):
        return value

    def reduce(self, value, reduction):
        assert reduction == "sum"
        return value

    def accumulate(self, model):
        del model
        return contextlib.nullcontext()

    def no_sync(self, model):
        return contextlib.nullcontext()

    def backward(self, loss):
        loss.backward()

    def skip_first_batches(self, dataloader, num_batches):
        self.skip_calls.append(num_batches)
        if num_batches == 0:
            return dataloader
        return list(dataloader)[num_batches:]

    def log(self, *args, **kwargs):
        return None

    def end_training(self):
        self.end_calls += 1


def _trainer(*, overflow, checkpoint_cursor, global_masked_mean=True):
    model = _OverflowModel()
    trainer = Trainer.__new__(Trainer)
    trainer.model = model
    trainer._unwrapped_model = model
    trainer.optimizer = _ControlledSGD(model.parameters(), skip_update=overflow)
    trainer.ema_model = _CountingEMA()
    trainer.accelerator = _OverflowAccelerator(optimizer_step_was_skipped=overflow)
    trainer.global_masked_mean = global_masked_mean
    trainer.grad_accumulation_steps = 1
    trainer._loaded_consumed_batches = 0
    trainer.max_grad_norm = 0
    trainer.log_samples = False
    trainer.batch_size_type = "sample"
    trainer.batch_size_per_gpu = 1
    trainer.max_samples = 1
    trainer.num_warmup_updates = 0
    trainer.epochs = 1
    trainer.duration_predictor = None
    trainer.noise_scheduler = None
    trainer.compile_active = False
    trainer.logger = None
    trainer.last_per_updates = 100
    trainer.save_per_updates = 100
    trainer.saved = []

    def _fake_load_checkpoint(*args, **kwargs):
        trainer._loaded_consumed_batches = checkpoint_cursor[1]
        return checkpoint_cursor[0]

    trainer.load_checkpoint = _fake_load_checkpoint
    trainer.save_checkpoint = lambda update, last=False, consumed_batches=None: trainer.saved.append(
        (update, consumed_batches, last)
    )
    return trainer


def test_successful_train_scales_the_production_loss_sum_before_backward(monkeypatch):
    monkeypatch.setattr(trainer_module, "DataLoader", lambda *args, **kwargs: _OneBatchLoader())
    trained = _trainer(overflow=False, checkpoint_cursor=(0, 0))

    trained.train(object(), num_workers=0)

    # loss_sum = weight**2 * 6 and the production scale is 1 / 6, so
    # d(loss_sum * scale) / d(weight) is exactly 2 * 0.5 = 1. A plausible
    # loss * scale mutation would instead produce 1 / 6.
    torch.testing.assert_close(trained.optimizer.last_gradient, torch.tensor(1.0))
    torch.testing.assert_close(trained.model.weight, torch.tensor(0.49))
    assert trained.ema_model.updates == 1
    assert trained.saved[-1] == (1, 1, True)


def test_overflow_consumes_window_without_advancing_update_or_ema_and_resume_skips_it(monkeypatch):
    monkeypatch.setattr(trainer_module, "DataLoader", lambda *args, **kwargs: _OneBatchLoader())
    overflowed = _trainer(overflow=True, checkpoint_cursor=(0, 0))

    overflowed.train(object(), num_workers=0)

    assert overflowed.optimizer.attempted_steps == 1
    assert overflowed.ema_model.updates == 0
    assert overflowed.model.forward_calls == 1
    assert overflowed.saved[-1] == (0, 1, True)

    # A restart has zero successful optimizer updates but one consumed
    # accumulation window. The consumed cursor must skip the completed epoch;
    # using global_update=0 would replay the overflowed batch.
    resumed = _trainer(overflow=False, checkpoint_cursor=(0, 1))
    resumed.train(object(), num_workers=0, resumable_with_seed=123)

    assert resumed.accelerator.skip_calls == [0]
    assert resumed.model.forward_calls == 0
    assert resumed.optimizer.attempted_steps == 0
    assert resumed.ema_model.updates == 0
    assert resumed.saved[-1] == (0, 1, True)


def test_non_global_masked_mean_path_keeps_upstream_ema_behavior_on_overflow(monkeypatch):
    # Pin stock upstream behavior: with global_masked_mean disabled, the legacy loop
    # is byte-identical to upstream and EMA still advances when AMP skips the step.
    # Overflow-aware legacy accounting is deliberately out of scope for this PR.
    monkeypatch.setattr(trainer_module, "DataLoader", lambda *args, **kwargs: _OneBatchLoader())
    overflowed = _trainer(overflow=True, checkpoint_cursor=(0, 0), global_masked_mean=False)

    overflowed.train(object(), num_workers=0)

    assert overflowed.optimizer.attempted_steps == 1
    assert overflowed.ema_model.updates == 1
    # Legacy checkpoints never carry the GMM consumption cursor.
    assert overflowed.saved[-1] == (1, None, True)
