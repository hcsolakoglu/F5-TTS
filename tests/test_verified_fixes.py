"""Regression tests for verified training, checkpoint, and cache fixes."""

from __future__ import annotations

import builtins
import copy
import inspect
import io
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import conftest  # noqa: F401
import pytest
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
from torch.utils.data import DataLoader, Dataset

import f5_tts.model.dataset as dataset_module
from f5_tts.model.backbones.dit import DiT
from f5_tts.model.backbones.mmdit import MMDiT
from f5_tts.model.backbones.unett import UNetT
from f5_tts.model.cfm import CFM, _compile_failure_types
from f5_tts.model.dataset import _is_exact_resume_safe_dataset
from f5_tts.model.trainer import Trainer, _EpochRandomSampler


CUDA_OOM_TYPE = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)


class _PrepareAccelerator:
    def __init__(self):
        self.even_batches = True
        self.dispatch_batches = False
        self.num_processes = 1
        self.device = torch.device("cpu")
        self.prepared = []

    def prepare_data_loader(self, dataloader, device_placement):
        self.prepared.append((dataloader, device_placement))
        return dataloader

    def gather(self, value):
        return value


class _ExactResumeDataset(Dataset):
    supports_exact_resume = True
    exact_resume_signature = "test-single-item-v1"

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return index


class _SizedLoader:
    def __init__(self, batch_count):
        self.batch_count = batch_count

    def __len__(self):
        return self.batch_count


class _ExactSequenceDataset(Dataset):
    supports_exact_resume = True
    exact_resume_signature = "test-sequence-v1"

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return index


class _AudioPathDataset(Dataset):
    supports_exact_resume = True
    exact_resume_signature = "test-audio-path-v1"
    exact_resume_content_signature: str | None = None
    column_names = ("audio_path", "duration", "text")
    preprocessed_mel = False

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return index


class _NestedComponent(torch.nn.Module):
    supports_exact_resume = True

    def __init__(self, behavior):
        super().__init__()
        self.exact_resume_signature = {"behavior": behavior}


class _UnregisteredMutableComponent(torch.nn.Module):
    supports_exact_resume = True

    def __init__(self):
        super().__init__()
        self.runtime_counter = 0


class _IncompleteExtraStateComponent(torch.nn.Module):
    supports_exact_resume = True
    exact_resume_signature = "incomplete-extra-state-v1"

    def get_extra_state(self):
        return {"counter": 0}


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


class _PeerFailureAccelerator:
    device = torch.device("cpu")
    num_processes = 2

    def reduce(self, value, reduction):
        assert reduction == "sum"
        # Simulate the peer reporting a dataloader failure while this rank
        # successfully fetched a batch.
        if value.shape == (3,):
            return torch.tensor([1, 0, 0], device=value.device, dtype=value.dtype)
        if value.shape == (2,):
            return torch.tensor([1, 0], device=value.device, dtype=value.dtype)
        return value


class _IteratorCreationFailure:
    def __iter__(self):
        raise RuntimeError("synthetic worker startup failure")


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


def test_global_masked_mean_rejects_distributed_split_batches():
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.accelerator = _PrepareAccelerator()
    trainer.accelerator.num_processes = 2
    trainer.accelerator.split_batches = True

    with pytest.raises(ValueError, match="split_batches"):
        trainer._prepare_global_masked_mean_dataloader(_SizedLoader(4))

    assert trainer.accelerator.even_batches is False
    assert trainer.accelerator.prepared == []


def test_global_masked_mean_coordinates_peer_dataloader_failure_before_yield():
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.accelerator = _PeerFailureAccelerator()

    with pytest.raises(RuntimeError, match="another rank"):
        list(trainer._iter_coordinated_batches([_batch()]))


def test_global_masked_mean_coordinates_peer_prepared_length_failure():
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.accelerator = _PeerFailureAccelerator()
    trainer.global_masked_mean = True

    with pytest.raises(RuntimeError, match="another rank"):
        trainer._validate_global_masked_dataloader(_SizedLoader(4))


def test_global_masked_mean_coordinates_peer_length_failure_before_prepare():
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.accelerator = _PeerFailureAccelerator()
    trainer.global_masked_mean = True
    setattr(
        trainer.accelerator,
        "prepare_data_loader",
        lambda *args, **kwargs: pytest.fail("preparation must not start after a peer length failure"),
    )

    with pytest.raises(RuntimeError, match="another rank"):
        trainer._prepare_global_masked_mean_dataloader(_SizedLoader(4))


def test_global_masked_mean_coordinates_iterator_creation_failure():
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.accelerator = _IteratorAccelerator()

    with pytest.raises(RuntimeError, match="dataloader fetch failed on this rank"):
        list(trainer._iter_coordinated_batches(_IteratorCreationFailure()))


def test_global_masked_mean_coordinates_peer_forward_failure_before_backward():
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.accelerator = _PeerFailureAccelerator()
    trainer.global_masked_mean = True

    with pytest.raises(RuntimeError, match="another rank"):
        trainer._raise_if_global_masked_mean_failure("forward", None)


def test_inference_rng_sandbox_restores_python_and_torch_states():
    trainer = cast(Any, Trainer.__new__(Trainer))
    random.seed(901)
    torch.manual_seed(901)
    python_state = random.getstate()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    with trainer._preserve_inference_rng_state():
        random.random()
        torch.randn(5)
        if cuda_state is not None:
            torch.randn(5, device="cuda")

    assert random.getstate() == python_state
    torch.testing.assert_close(torch.get_rng_state(), torch_state)
    if cuda_state is not None:
        for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_state):
            torch.testing.assert_close(actual, expected)


def test_scheduler_horizon_uses_prepared_boundaries_and_wrapper_factor():
    total_raw_steps, warmup_raw_steps, factor = Trainer._compute_scheduler_horizon(
        prepared_batches=5,
        grad_accumulation_steps=4,
        epochs=1,
        num_warmup_updates=1,
        num_processes=2,
        split_batches=False,
    )
    assert (total_raw_steps, warmup_raw_steps, factor) == (4, 2, 2)

    total_raw_steps, warmup_raw_steps, factor = Trainer._compute_scheduler_horizon(
        prepared_batches=10,
        grad_accumulation_steps=4,
        epochs=1,
        num_warmup_updates=1,
        num_processes=2,
        split_batches=True,
    )
    assert (total_raw_steps, warmup_raw_steps, factor) == (3, 1, 1)

    total_raw_steps, warmup_raw_steps, factor = Trainer._compute_scheduler_horizon(
        prepared_batches=10,
        grad_accumulation_steps=4,
        epochs=1,
        num_warmup_updates=1,
        num_processes=2,
        split_batches=False,
        step_scheduler_with_optimizer=False,
    )
    assert (total_raw_steps, warmup_raw_steps, factor) == (3, 1, 1)


def test_scheduler_adjustment_bookkeeping_does_not_change_underlying_horizon():
    total_raw_steps, warmup_raw_steps, factor = Trainer._compute_scheduler_horizon(
        prepared_batches=5,
        grad_accumulation_steps=4,
        epochs=1,
        num_warmup_updates=1,
        num_processes=1,
        split_batches=False,
        adjust_scheduler=True,
    )

    assert (total_raw_steps, warmup_raw_steps, factor) == (2, 1, 1)


def test_scheduler_horizon_uses_global_raw_batches_without_dataloader_sync():
    total_raw_steps, warmup_raw_steps, factor = Trainer._compute_scheduler_horizon(
        prepared_batches=5,
        grad_accumulation_steps=4,
        epochs=2,
        num_warmup_updates=0,
        num_processes=1,
        split_batches=False,
        sync_with_dataloader=False,
    )
    assert (total_raw_steps, warmup_raw_steps, factor) == (2, 0, 1)


def test_effective_gradient_contract_comes_from_accelerate_plugin():
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer._requested_grad_accumulation_steps = 4
    trainer.accelerator = SimpleNamespace(
        gradient_state=SimpleNamespace(num_steps=2, sync_with_dataloader=False, adjust_scheduler=True),
        gradient_accumulation_steps=4,
    )

    assert trainer._get_effective_gradient_accumulation_contract() == (2, False, True)


def test_scheduler_horizon_rejects_warmup_that_consumes_decay():
    with pytest.raises(ValueError, match="warmup"):
        Trainer._compute_scheduler_horizon(
            prepared_batches=2,
            grad_accumulation_steps=1,
            epochs=1,
            num_warmup_updates=2,
            num_processes=1,
            split_batches=False,
        )


def test_exact_resume_rejects_unsupported_worker_configuration():
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.batch_size_type = "sample"

    with pytest.raises(ValueError, match="num_workers=0"):
        trainer._validate_resume_contract(_ExactResumeDataset(), 2, 123, "exact")


def test_exact_resume_rejects_compile_fallback():
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.resume_mode = "exact"
    trainer.compile_fallback_active = True

    with pytest.raises(RuntimeError, match="fell back to eager"):
        trainer._reject_exact_compile_fallback()


def test_exact_resume_rejects_unsupported_rng_backend_and_accumulation_sync():
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.batch_size_type = "sample"
    trainer.accelerator = SimpleNamespace(device=torch.device("mps"))

    with pytest.raises(ValueError, match="CPU and CUDA"):
        trainer._validate_resume_contract(_ExactResumeDataset(), 0, 123, "exact")

    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        gradient_state=SimpleNamespace(num_steps=1, sync_with_dataloader=False, adjust_scheduler=False),
    )
    with pytest.raises(ValueError, match="dataloader boundaries"):
        trainer._validate_resume_contract(_ExactResumeDataset(), 0, 123, "exact")


@pytest.mark.parametrize("distributed_type", [DistributedType.FSDP, DistributedType.DEEPSPEED])
def test_exact_resume_rejects_sharded_backends_before_checkpoint_loading(distributed_type):
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.batch_size_type = "sample"
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cuda"),
        distributed_type=distributed_type,
        gradient_state=SimpleNamespace(num_steps=1, sync_with_dataloader=True, adjust_scheduler=False),
    )

    with pytest.raises(ValueError, match="backend-aware checkpoint protocol"):
        trainer._validate_resume_contract(_ExactResumeDataset(), 0, 123, "exact")


def test_exact_resume_marker_rejects_arbitrary_wrappers():
    assert _is_exact_resume_safe_dataset(Dataset()) is False
    assert _is_exact_resume_safe_dataset(_ExactResumeDataset()) is True


def test_exact_resume_requires_explicit_dataset_capability_and_seed():
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.batch_size_type = "sample"

    with pytest.raises(ValueError, match="supports_exact_resume"):
        trainer._validate_resume_contract(Dataset(), 0, 123, "exact")
    with pytest.raises(ValueError, match="seed"):
        trainer._validate_resume_contract(_ExactResumeDataset(), 0, None, "exact")

    trainer._validate_resume_contract(_ExactResumeDataset(), 0, 123, "exact")


def test_exact_resume_rejects_external_audio_without_content_signature():
    trainer = cast(Any, Trainer.__new__(Trainer))
    dataset = _AudioPathDataset()

    with pytest.raises(ValueError, match="external audio data"):
        trainer._validate_exact_dataset_components(dataset)

    dataset.exact_resume_content_signature = "audio-manifest-v1"
    trainer._validate_exact_dataset_components(dataset)


def test_production_loader_passes_audio_content_signature_to_wrapper(monkeypatch):
    captured = {}

    class _CaptureDataset:
        def __init__(self, data, **kwargs):
            captured["data"] = data
            captured.update(kwargs)

    monkeypatch.setattr(dataset_module, "load_from_disk", lambda path: object())
    monkeypatch.setattr(dataset_module, "CustomDataset", _CaptureDataset)
    monkeypatch.setattr(
        builtins,
        "open",
        lambda *args, **kwargs: io.StringIO('{"duration": [1.0]}'),
    )

    dataset_module.load_dataset("demo", "pinyin", exact_resume_content_signature="audio-manifest-v1")

    assert captured["exact_resume_content_signature"] == "audio-manifest-v1"


def test_exact_resume_requires_explicit_component_capability_and_signature():
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.model_cfg_dict = {"model": {"name": "must-not-sign-custom-components"}}
    trainer._unwrapped_model = _UnregisteredMutableComponent()

    with pytest.raises(ValueError, match="exact_resume_signature"):
        trainer._validate_exact_component_signatures()

    trainer._unwrapped_model = _NestedComponent("first")
    trainer._unwrapped_model.exact_resume_signature = object()
    with pytest.raises(ValueError, match="primitive values"):
        trainer._validate_exact_component_signatures()

    trainer._unwrapped_model = _NestedComponent("first")
    trainer._validate_exact_component_signatures()
    first = trainer._component_resume_signature(trainer._unwrapped_model)
    second = trainer._component_resume_signature(_NestedComponent("second"))

    assert first != second

    trainer._unwrapped_model = _IncompleteExtraStateComponent()
    with pytest.raises(ValueError, match="both get_extra_state.*set_extra_state"):
        trainer._validate_exact_component_signatures()


def test_resolved_ema_signature_includes_defaults_and_explicit_set_values():
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.ema_kwargs = {"update_every": 1, "ignore_names": {"bias", "weight"}}

    signature = trainer._resolved_ema_config()

    assert signature["beta"] == 0.9999
    assert signature["update_every"] == 1
    assert signature["ignore_names"] == {"set": ["bias", "weight"]}
    assert signature["include_online_model"] is False


def test_cfm_exact_signature_covers_unregistered_behavior_and_requires_model_config():
    model = CFM.__new__(CFM)
    for name, value in {
        "sigma": 0.1,
        "audio_drop_prob": 0.3,
        "cond_drop_prob": 0.2,
        "frac_lengths_mask": (0.7, 1.0),
        "num_channels": 100,
        "odeint_kwargs": {"method": "euler"},
        "vocab_char_map": {"a": 1, "b": 2},
    }.items():
        object.__setattr__(model, name, value)

    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer._unwrapped_model = model
    trainer.model_cfg_dict = {}

    with pytest.raises(ValueError, match="production model configuration"):
        trainer._validate_exact_component_signatures()

    trainer.model_cfg_dict = {"model": {"backbone": "DiT"}}
    trainer._validate_exact_component_signatures()
    signature = trainer._component_resume_signature(
        model,
        configured_signature=trainer._model_configuration_signature(),
    )

    assert signature["explicit"]["vocab_char_map"] == {"a": 1, "b": 2}
    assert signature["config"] == {"backbone": "DiT"}


@pytest.mark.parametrize("duration_state", ["missing", "none"])
def test_exact_resume_rejects_missing_duration_predictor_state(duration_state):
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.resume_mode = "exact"
    trainer.duration_predictor = torch.nn.Linear(1, 1)
    setattr(cast(Any, trainer.duration_predictor), "supports_exact_resume", True)
    setattr(cast(Any, trainer.duration_predictor), "exact_resume_signature", "linear-duration-v1")
    trainer.accelerator = SimpleNamespace(scaler=None)
    checkpoint = {
        "update": 1,
        "resume_state_signature": {},
        "resume_signature": {},
        "rng_state": {},
        "scaler_state": None,
    }
    if duration_state == "none":
        checkpoint["duration_predictor_state_dict"] = None

    with pytest.raises(ValueError, match="duration_predictor_state_dict"):
        trainer._validate_resume_signatures(checkpoint)


def test_exact_resume_rejects_distributed_duration_predictor():
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.batch_size_type = "sample"
    trainer.duration_predictor = torch.nn.Linear(1, 1)
    setattr(cast(Any, trainer.duration_predictor), "supports_exact_resume", True)
    setattr(cast(Any, trainer.duration_predictor), "exact_resume_signature", "linear-duration-v1")
    trainer._unwrapped_model = _NestedComponent("stable")
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        num_processes=2,
        gradient_state=SimpleNamespace(num_steps=1, sync_with_dataloader=True, adjust_scheduler=False),
    )

    with pytest.raises(ValueError, match="duration_predictor with multiple processes"):
        trainer._validate_resume_contract(_ExactSequenceDataset(), 0, 123, "exact")


def test_exact_resume_rejects_scheduler_signature_mismatch():
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.resume_mode = "exact"
    trainer._scheduler_signature = {"prepared_batches": 5, "total_raw_steps": 4}

    with pytest.raises(ValueError, match="scheduler signature"):
        trainer._validate_scheduler_signature(
            {"update": 3, "scheduler_signature": {"prepared_batches": 6, "total_raw_steps": 4}}
        )

    trainer._validate_scheduler_signature(
        {"update": 3, "scheduler_signature": {"prepared_batches": 5, "total_raw_steps": 4}}
    )


def test_legacy_best_effort_restores_rng_but_not_unvalidated_scaler():
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.resume_mode = "best_effort"
    trainer.accelerator = SimpleNamespace(scaler=object())

    with pytest.warns(RuntimeWarning, match="legacy checkpoint"):
        compatibility = trainer._validate_resume_signatures(
            {"update": 1, "rng_state": {}, "scaler_state": {"scale": 1}}
        )

    assert compatibility == (True, False)


def test_exact_resume_rejects_missing_or_mismatched_persisted_signatures():
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.resume_mode = "exact"
    trainer._resume_state_signature = {"mixed_precision": "fp16", "num_processes": 1}
    trainer._resume_signature = {"dataset": "dataset-v1", "prepared_batches": 4}

    with pytest.raises(ValueError, match="resume signature"):
        trainer._validate_resume_signatures(
            {
                "update": 1,
                "resume_state_signature": {"mixed_precision": "no", "num_processes": 1},
                "resume_signature": {"dataset": "dataset-v2", "prepared_batches": 4},
                "scaler_state": None,
            }
        )

    with pytest.raises(ValueError, match="resume signature"):
        trainer._validate_resume_signatures({"update": 1})


def test_global_masked_mean_reuses_collectively_validated_prepared_length():
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.accelerator = _PrepareAccelerator()
    trainer._validate_global_masked_dataloader = lambda loader: len(loader)

    trainer._prepare_global_masked_mean_dataloader(_SizedLoader(4))

    assert trainer._prepared_global_masked_mean_batches == 4


def test_exact_resume_replays_supported_dataset_trajectory():
    dataset = _ExactSequenceDataset()

    def make_batches():
        sampler = _EpochRandomSampler(dataset, seed=1234)
        loader = DataLoader(dataset, batch_size=2, sampler=sampler, shuffle=False)
        Trainer._set_dataloader_epoch(loader, 0)
        return [batch.clone() for batch in loader]

    def apply_batch(model, optimizer, batch):
        values = batch.float().unsqueeze(1)
        loss = (model(values) - values.square()).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.manual_seed(8128)
    initial_model = torch.nn.Linear(1, 1, bias=False)
    initial_state = copy.deepcopy(initial_model.state_dict())

    uninterrupted = torch.nn.Linear(1, 1, bias=False)
    uninterrupted.load_state_dict(initial_state)
    uninterrupted_optimizer = torch.optim.SGD(uninterrupted.parameters(), lr=0.01)
    uninterrupted_batches = make_batches()
    for batch in uninterrupted_batches:
        apply_batch(uninterrupted, uninterrupted_optimizer, batch)

    resumed = torch.nn.Linear(1, 1, bias=False)
    resumed.load_state_dict(initial_state)
    resumed_optimizer = torch.optim.SGD(resumed.parameters(), lr=0.01)
    resumed_batches = make_batches()
    for batch in resumed_batches[:2]:
        apply_batch(resumed, resumed_optimizer, batch)

    checkpoint = {
        "model": copy.deepcopy(resumed.state_dict()),
        "optimizer": copy.deepcopy(resumed_optimizer.state_dict()),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "consumed_batches": 2,
    }
    restored = torch.nn.Linear(1, 1, bias=False)
    restored.load_state_dict(checkpoint["model"])
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.01)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    random.setstate(checkpoint["python_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])

    for batch in resumed_batches[checkpoint["consumed_batches"] :]:
        apply_batch(restored, restored_optimizer, batch)

    assert [batch.tolist() for batch in resumed_batches] == [batch.tolist() for batch in uninterrupted_batches]
    torch.testing.assert_close(restored.weight, uninterrupted.weight)


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


def test_best_effort_resume_skips_rng_restore_when_topology_changes():
    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.accelerator = SimpleNamespace(process_index=0, num_processes=2)

    with pytest.warns(RuntimeWarning, match="Skipping incompatible RNG restoration"):
        trainer._restore_rng_states([{}])

    trainer.resume_mode = "exact"
    with pytest.raises(ValueError, match="same process count"):
        trainer._restore_rng_states([{}])


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


def test_save_and_load_checkpoint_restores_duration_predictor_state(tmp_path):
    source = _checkpoint_trainer(tmp_path)
    target = _checkpoint_trainer(tmp_path)
    source.duration_predictor = torch.nn.BatchNorm1d(2)
    target.duration_predictor = torch.nn.BatchNorm1d(2)
    cast(Any, source.duration_predictor).running_mean.copy_(torch.tensor([3.0, 4.0]))

    source.save_checkpoint(3, 3, True)
    target.load_checkpoint(return_cursor=True)

    torch.testing.assert_close(target.duration_predictor.running_mean, torch.tensor([3.0, 4.0]))


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
