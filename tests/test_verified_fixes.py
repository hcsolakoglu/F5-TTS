"""Regression tests for verified training, checkpoint, and cache fixes."""

from __future__ import annotations

import copy
import inspect
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import conftest  # noqa: F401
import pytest
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from f5_tts.model.backbones.dit import DiT
from f5_tts.model.backbones.mmdit import MMDiT
from f5_tts.model.backbones.unett import UNetT
from f5_tts.model.cfm import _compile_failure_types
from f5_tts.model.trainer import Trainer, _EpochRandomSampler


CUDA_OOM_TYPE = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)


class _PrepareAccelerator:
    def __init__(self):
        self.even_batches = True
        self.dispatch_batches = False
        self.num_processes = 1
        self.prepared = []

    def prepare_data_loader(self, dataloader, device_placement):
        self.prepared.append((dataloader, device_placement))
        return dataloader


class _SizedLoader:
    def __init__(self, batch_count):
        self.batch_count = batch_count

    def __len__(self):
        return self.batch_count


class _CheckpointAccelerator:
    is_main_process = True
    process_index = 0
    num_processes = 1

    def wait_for_everyone(self):
        pass

    def unwrap_model(self, model):
        return model

    def save(self, state, path):
        torch.save(state, path)


class _IteratorAccelerator:
    device = torch.device("cpu")
    num_processes = 1

    def reduce(self, value, reduction):
        assert reduction == "sum"
        return value


class _OomSampler:
    num_channels = 2
    vocab_char_map = None

    def sample_training_mask(self, lens, seq_len):
        del lens, seq_len
        oom_type = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
        raise oom_type("CUDA out of memory. synthetic")


def _batch():
    return {
        "mel": torch.zeros((1, 2, 3)),
        "mel_lengths": torch.tensor([3]),
        "text": ["test"],
    }


def test_global_masked_mean_preparation_disables_accelerate_duplicate_padding():
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.accelerator = _PrepareAccelerator()
    validated = []
    trainer._validate_global_masked_dataloader = lambda loader: validated.append(loader)
    dataloader = object()

    prepared = trainer._prepare_global_masked_mean_dataloader(dataloader)

    assert prepared is dataloader
    assert trainer.accelerator.even_batches is False
    assert validated == [dataloader]
    assert trainer.accelerator.prepared == [(dataloader, False)]


def test_global_masked_mean_rejects_incomplete_process_batch_group():
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.accelerator = _PrepareAccelerator()
    trainer.accelerator.num_processes = 2

    with pytest.raises(ValueError, match="divisible by the process count"):
        trainer._prepare_global_masked_mean_dataloader(_SizedLoader(3))

    assert trainer.accelerator.even_batches is False
    assert trainer.accelerator.prepared == []


def test_resumed_sample_loader_sets_epoch_on_dataloader_shard():
    def make_loader():
        dataset = list(range(8))
        sampler = _EpochRandomSampler(dataset, seed=1234)
        generator = torch.Generator().manual_seed(5678)
        loader = DataLoader(
            cast(Any, dataset),
            batch_size=2,
            sampler=sampler,
            shuffle=False,
            generator=generator,
        )
        accelerator = Accelerator(cpu=True)
        accelerator.even_batches = False
        return accelerator, accelerator.prepare_data_loader(loader, device_placement=False)

    _, expected_loader = make_loader()
    Trainer._set_dataloader_epoch(expected_loader, 0)
    epoch_zero = [tuple(batch.tolist()) for batch in expected_loader]
    Trainer._set_dataloader_epoch(expected_loader, 1)
    expected = [tuple(batch.tolist()) for batch in expected_loader]
    assert expected != epoch_zero

    accelerator, resumed_loader = make_loader()
    skipped_loader = accelerator.skip_first_batches(resumed_loader, num_batches=0)
    Trainer._set_dataloader_epoch(skipped_loader, 1)
    actual = [tuple(batch.tolist()) for batch in skipped_loader]

    assert actual == expected


def test_global_masked_mean_preserves_cuda_oom_type_during_mask_preparation():
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.grad_accumulation_steps = 1
    trainer.accelerator = _IteratorAccelerator()
    trainer._unwrapped_model = _OomSampler()

    with pytest.raises(CUDA_OOM_TYPE, match="CUDA out of memory"):
        list(trainer._iter_global_masked_mean_batches([_batch()]))


def test_checkpoint_rng_state_round_trip_restores_python_and_torch_rng():
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.accelerator = SimpleNamespace(process_index=0, num_processes=1)

    random.seed(8128)
    torch.manual_seed(2718)
    state = trainer._capture_rng_states()
    expected_python = random.random()
    expected_torch = torch.rand(())

    trainer._restore_rng_states(state)

    assert random.random() == expected_python
    torch.testing.assert_close(torch.rand(()), expected_torch)


def test_save_checkpoint_preserves_legacy_last_call_and_rng_payload(tmp_path):
    model = torch.nn.Linear(2, 2)
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.accelerator = _CheckpointAccelerator()
    trainer.model = model
    trainer.ema_model = torch.nn.Linear(2, 2)
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
    trainer.checkpoint_path = str(tmp_path)
    trainer.keep_last_n_checkpoints = 0

    trainer.save_checkpoint(10, True)

    checkpoint_path = Path(tmp_path) / "model_last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["update"] == 10
    assert checkpoint["consumed_updates"] == 10
    assert len(checkpoint["rng_state"]) == 1


def _checkpoint_trainer(checkpoint_path):
    model = torch.nn.Linear(2, 2)
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.accelerator = _CheckpointAccelerator()
    trainer.model = model
    trainer.ema_model = torch.nn.Linear(2, 2)
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
    trainer.checkpoint_path = str(checkpoint_path)
    trainer.keep_last_n_checkpoints = 0
    trainer.grad_accumulation_steps = 1
    trainer._train_dataloader_generator = torch.Generator().manual_seed(777)
    return trainer


def test_load_checkpoint_restores_cursor_and_rng_state(tmp_path):
    source = _checkpoint_trainer(tmp_path)
    target = _checkpoint_trainer(tmp_path)
    random.seed(101)
    torch.manual_seed(202)

    source.save_checkpoint(3, 5, True)
    expected_python = random.random()
    expected_torch = torch.rand(())
    expected_worker = torch.rand((), generator=source._train_dataloader_generator)

    random.seed(303)
    torch.manual_seed(404)
    assert target.load_checkpoint(return_cursor=True) == (3, 5)
    assert random.random() == expected_python
    torch.testing.assert_close(torch.rand(()), expected_torch)
    torch.testing.assert_close(torch.rand((), generator=target._train_dataloader_generator), expected_worker)


@pytest.mark.parametrize(
    "args,expected",
    [
        ((10,), (10, 10, False)),
        ((10, True), (10, 10, True)),
        ((10, 7), (10, 7, False)),
        ((10, 7, True), (10, 7, True)),
    ],
)
def test_checkpoint_arguments_preserve_legacy_and_cursor_forms(args, expected):
    assert Trainer._normalize_checkpoint_args(*args) == expected


@pytest.mark.parametrize("bad_value", [0, -1, True, 1.5])
def test_invalid_gradient_accumulation_steps_fail_before_accelerator_setup(bad_value):
    with pytest.raises(ValueError, match="gradient_accumulation_steps must be a positive integer"):
        Trainer._validate_gradient_accumulation_steps(bad_value)


def test_generic_runtime_error_is_not_a_compile_fallback_signal():
    assert not isinstance(RuntimeError("model failure"), _compile_failure_types())


def test_dynamo_wrapped_runtime_error_is_not_a_compile_fallback_signal():
    dynamo_exc = getattr(torch._dynamo, "exc", None)
    wrapped_error = getattr(dynamo_exc, "TorchRuntimeError", None) if dynamo_exc is not None else None
    if wrapped_error is None:
        pytest.skip("this PyTorch version does not expose TorchRuntimeError")
    try:
        error = wrapped_error("model runtime failure")
    except TypeError:
        pytest.skip("TorchRuntimeError constructor is not stable on this PyTorch version")
    assert not isinstance(error, _compile_failure_types())


def _small_dit():
    return DiT(
        dim=16,
        depth=2,
        heads=1,
        dim_head=4,
        dropout=0.0,
        mel_dim=2,
        text_num_embeds=8,
        text_dim=4,
    )


def _small_unett():
    return UNetT(
        dim=16,
        depth=2,
        heads=1,
        dim_head=4,
        dropout=0.0,
        mel_dim=2,
        text_num_embeds=8,
        text_dim=4,
    )


def _small_mmdit():
    return MMDiT(
        dim=16,
        depth=1,
        heads=1,
        dim_head=4,
        dropout=0.0,
        ff_mult=2,
        mel_dim=2,
        text_num_embeds=8,
    )


@pytest.mark.parametrize("factory", [_small_dit, _small_mmdit, _small_unett], ids=["dit", "mmdit", "unett"])
def test_inference_cache_does_not_block_model_deepcopy(factory):
    model = factory()
    model.text_cond = torch.ones((1, 2, 16))

    copied = copy.deepcopy(model)

    assert copied.text_cond is None


@pytest.mark.parametrize("factory", [_small_dit, _small_mmdit, _small_unett], ids=["dit", "mmdit", "unett"])
def test_inference_cache_does_not_block_model_pickle(factory, tmp_path):
    model = factory()
    model.text_cond = torch.ones((1, 2, 16))
    path = tmp_path / f"{factory.__name__}.pt"

    torch.save(model, path)
    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    loaded = torch.load(path, **load_kwargs)

    assert loaded.text_cond is None
