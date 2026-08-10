from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from torch import nn
from torch.utils.data import Dataset

import f5_tts.model.trainer as trainer_module
from f5_tts.model.trainer import Trainer


class _ProbeDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {"mel_spec": torch.ones((2, 3)) * (index + 1), "text": "probe"}


class _ProbeModel(nn.Module):
    num_channels = 2

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.fail_forward = False

    def sample_training_mask(self, lens, seq_len):
        return torch.ones((lens.shape[0], seq_len), dtype=torch.bool)

    def forward(self, mel, *, text, lens, noise_scheduler, rand_span_mask=None, return_loss_components=False):
        del text, lens, noise_scheduler, rand_span_mask
        if self.fail_forward:
            raise RuntimeError("synthetic rank-zero forward failure")
        loss = self.weight.square()
        if return_loss_components:
            denominator = torch.tensor(mel.shape[0] * mel.shape[-1], dtype=torch.float32)
            return loss, loss * denominator, denominator, mel, mel
        return loss, mel, mel


class _ProbeEMA:
    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        del state

    def update(self):
        return None


def _make_trainer(accelerator, checkpoint_name, *, fail_forward=False):
    trainer = Trainer.__new__(Trainer)
    model = _ProbeModel()
    model.fail_forward = fail_forward
    trainer.model = model
    trainer._unwrapped_model = model
    trainer.accelerator = accelerator
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer.ema_model = cast(Any, _ProbeEMA())
    trainer.global_masked_mean = True
    trainer.grad_accumulation_steps = 1
    trainer.max_grad_norm = 0
    trainer.log_samples = False
    trainer.batch_size_type = "sample"
    trainer.batch_size_per_gpu = 1
    trainer.max_samples = 2
    trainer.num_warmup_updates = 0
    trainer.epochs = 1
    trainer.duration_predictor = None
    trainer.noise_scheduler = None
    trainer.compile_active = False
    trainer.logger = None
    trainer.last_per_updates = 100
    trainer.save_per_updates = 100
    trainer.keep_last_n_checkpoints = -1
    trainer.checkpoint_path = str(Path("/tmp") / checkpoint_name)
    trainer.vocoder_name = "vocos"
    return trainer


def _run_phase(name, expected_local_marker, fn, accelerator):
    if dist.get_world_size() != 2:
        raise RuntimeError(f"{name} probe requires exactly two ranks")
    local_error = None
    try:
        fn()
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, local_error)
    assert errors[0] is not None and expected_local_marker in errors[0], (name, errors)
    assert errors[1] is not None and "another rank raised" in errors[1], (name, errors)
    if accelerator.is_main_process:
        print(f"{name}_OK", errors)
    dist.barrier()


if int(os.environ.get("WORLD_SIZE", "1")) != 2:
    raise RuntimeError("coordinated train failure probe requires WORLD_SIZE=2")

accelerator = Accelerator(
    cpu=True,
    split_batches=False,
    dataloader_config=DataLoaderConfiguration(even_batches=False, dispatch_batches=False),
)
rank = dist.get_rank()


def construction_phase():
    original_loader = trainer_module.DataLoader
    try:
        if rank == 0:

            def fail_loader(*args, **kwargs):
                raise RuntimeError("synthetic rank-zero DataLoader construction failure")

            trainer_module.DataLoader = fail_loader
        else:
            trainer_module.DataLoader = lambda *args, **kwargs: object()
        trainer = _make_trainer(accelerator, "f5-construction-probe")
        trainer.train(_ProbeDataset(), num_workers=0)
    finally:
        trainer_module.DataLoader = original_loader


def preparation_phase():
    original_prepare = accelerator.prepare_data_loader
    try:
        if rank == 0:

            def fail_prepare(*args, **kwargs):
                raise RuntimeError("synthetic rank-zero DataLoader preparation failure")

            accelerator.prepare_data_loader = fail_prepare
        trainer = _make_trainer(accelerator, "f5-preparation-probe")
        trainer.train(_ProbeDataset(), num_workers=0)
    finally:
        accelerator.prepare_data_loader = original_prepare


def transfer_phase():
    original_send = trainer_module.send_to_device
    try:

        def send_or_fail(batch, *args, **kwargs):
            if rank == 0:
                raise RuntimeError("synthetic rank-zero device transfer failure")
            return original_send(batch, *args, **kwargs)

        trainer_module.send_to_device = send_or_fail
        trainer = _make_trainer(accelerator, "f5-transfer-probe")
        trainer.train(_ProbeDataset(), num_workers=0)
    finally:
        trainer_module.send_to_device = original_send


def forward_phase():
    trainer = _make_trainer(accelerator, "f5-forward-probe", fail_forward=rank == 0)
    trainer.train(_ProbeDataset(), num_workers=0)


_run_phase(
    "CONSTRUCTION",
    "synthetic rank-zero DataLoader construction failure",
    construction_phase,
    accelerator,
)
_run_phase(
    "PREPARATION",
    "synthetic rank-zero DataLoader preparation failure",
    preparation_phase,
    accelerator,
)
_run_phase("TRANSFER", "synthetic rank-zero device transfer failure", transfer_phase, accelerator)
_run_phase("FORWARD", "synthetic rank-zero forward failure", forward_phase, accelerator)
accelerator.end_training()
