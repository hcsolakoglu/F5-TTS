"""Direct contract tests for the F5-TTS global masked-mean implementation."""

from __future__ import annotations

import datetime
import tempfile
from types import SimpleNamespace

# Multiprocessing "spawn" imports this module outside pytest's normal conftest
# loading path. Import the lightweight optional-dependency stubs before F5-TTS.
import conftest  # noqa: F401
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate.data_loader import BatchSamplerShard
from accelerate.utils import DistributedType
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from f5_tts.model.cfm import CFM
from f5_tts.model.dataset import DynamicBatchSampler
from f5_tts.model.trainer import Trainer


class _DummyMelSpec(nn.Module):
    n_mel_channels = 2
    target_sample_rate = 24_000
    hop_length = 256

    def forward(self, wav):  # pragma: no cover - all contract inputs are mel tensors
        raise AssertionError("audio frontend must not run")


class _TinyTransformer(nn.Module):
    dim = 2

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.75))

    def forward(self, *, x, **kwargs):
        return x * self.gain


def _tiny_cfm() -> CFM:
    return CFM(
        _TinyTransformer(),
        num_channels=2,
        mel_spec_module=_DummyMelSpec(),
        frac_lengths_mask=(0.5, 0.5),
        audio_drop_prob=0.0,
        cond_drop_prob=0.0,
    )


def test_sample_training_mask_is_boolean_and_never_selects_padding():
    model = _tiny_cfm()
    lens = torch.tensor([5, 2], dtype=torch.int32)

    mask = model.sample_training_mask(lens, seq_len=6)

    assert mask.shape == (2, 6)
    assert mask.dtype == torch.bool
    assert mask.device == model.device
    assert not mask[0, 5]
    assert not mask[1, 2:].any()


@pytest.mark.parametrize(
    "lens,seq_len,message",
    [
        (torch.tensor([-1, 2]), 4, "between 0 and seq_len"),
        (torch.tensor([5, 2]), 4, "between 0 and seq_len"),
        (torch.tensor([1, 2]), -1, "seq_len must be non-negative"),
    ],
    ids=["negative-length", "length-exceeds-sequence", "negative-sequence-length"],
)
def test_sample_training_mask_rejects_invalid_length_bounds(lens, seq_len, message):
    model = _tiny_cfm()

    with pytest.raises(ValueError, match=message):
        model.sample_training_mask(lens, seq_len)


def test_forward_reuses_presampled_mask_without_resampling(monkeypatch):
    model = _tiny_cfm()
    inp = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2) / 8
    text = torch.tensor([[1, 2], [3, 4]])
    lens = torch.tensor([3, 1])
    # Padded selections must be removed by the valid-length mask.
    presampled = torch.ones((2, 4), dtype=torch.bool)

    def fail_if_resampled(*args, **kwargs):
        raise AssertionError("forward resampled a precomputed training mask")

    monkeypatch.setattr(model, "_sample_training_mask", fail_if_resampled)
    loss, loss_sum, denominator, _, _ = model(
        inp,
        text,
        lens=lens,
        rand_span_mask=presampled,
        return_loss_components=True,
    )

    # Four valid time positions, two mel channels each.
    assert denominator == 8
    torch.testing.assert_close(loss, loss_sum / denominator)


def test_forward_rejects_presampled_mask_with_wrong_shape():
    model = _tiny_cfm()
    inp = torch.zeros((2, 4, 2))
    text = torch.tensor([[1], [2]])

    with pytest.raises(ValueError, match=r"rand_span_mask must have shape \(2, 4\)"):
        model(
            inp,
            text,
            lens=torch.tensor([4, 4]),
            rand_span_mask=torch.ones((2, 3), dtype=torch.bool),
            return_loss_components=True,
        )


@pytest.mark.parametrize(
    "bad_mask",
    [
        torch.tensor([[0.0, 1.0], [float("nan"), 0.0]]),
        torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
    ],
    ids=["float-with-nan", "integer"],
)
def test_forward_rejects_non_boolean_presampled_mask(bad_mask):
    model = _tiny_cfm()
    inp = torch.zeros((2, 2, 2))
    text = torch.tensor([[1], [2]])

    with pytest.raises(TypeError, match="rand_span_mask must have dtype torch.bool"):
        model(
            inp,
            text,
            lens=torch.tensor([2, 2]),
            rand_span_mask=bad_mask,
            return_loss_components=True,
        )


class _MaskSampler:
    def __init__(self, masks, *, vocab_char_map=None):
        self.masks = iter(masks)
        self.calls = []
        self.vocab_char_map = vocab_char_map

    def sample_training_mask(self, lens, seq_len):
        self.calls.append((lens.clone(), seq_len))
        return next(self.masks)


class _IteratorAccelerator:
    def __init__(self, *, num_processes=1, remote_denominator=0):
        self.device = torch.device("cpu")
        self.num_processes = num_processes
        self.remote_denominator = remote_denominator
        self.reduced = []

    def reduce(self, value, reduction):
        assert reduction == "sum"
        self.reduced.append(value.clone())
        remote = torch.zeros_like(value)
        if remote.ndim == 0:
            remote = remote + self.remote_denominator
        else:
            remote[0] = self.remote_denominator
        return value + remote


def _batch(*, mel_channels=2, seq_len=5):
    return {
        "mel": torch.zeros((1, mel_channels, seq_len)),
        "mel_lengths": torch.tensor([seq_len]),
        "text": ["test"],
    }


def _iterator_trainer(
    masks,
    *,
    gradient_accumulation_steps=3,
    num_processes=1,
    remote_denominator=0,
    vocab_char_map=None,
):
    trainer = Trainer.__new__(Trainer)
    trainer.grad_accumulation_steps = gradient_accumulation_steps
    trainer.accelerator = _IteratorAccelerator(
        num_processes=num_processes,
        remote_denominator=remote_denominator,
    )
    trainer._unwrapped_model = _MaskSampler(masks, vocab_char_map=vocab_char_map)
    return trainer


def test_window_iterator_uses_int64_denominator_tensor_and_exact_boundaries():
    masks = [
        torch.tensor([[True, False, False, False, False]]),
        torch.tensor([[True, True, True, False, False]]),
        torch.tensor([[True, True, False, False, False]]),
        torch.tensor([[True, True, True, True, False]]),
        torch.tensor([[True, False, False, False, False]]),
    ]
    trainer = _iterator_trainer(masks, gradient_accumulation_steps=3)

    yielded = list(trainer._iter_global_masked_mean_batches([_batch() for _ in masks]))

    assert [entry[4] for entry in yielded] == [False, False, True, False, True]
    assert [entry[1] for entry in yielded] == masks
    # Counts are multiplied by two mel channels: first window 12, second 10.
    assert [int(entry[3]) for entry in yielded] == [12, 12, 12, 10, 10]
    assert [entry[3].dtype for entry in yielded] == [torch.int64] * 5
    assert [float(entry[2]) for entry in yielded] == pytest.approx([3 / 12] * 3 + [3 / 10] * 2)
    assert [value.dtype for value in trainer.accelerator.reduced] == [torch.int64, torch.int64]


def test_window_iterator_accepts_preprocessed_pinyin_token_lists():
    mask = torch.tensor([[True, True, False]])
    trainer = _iterator_trainer(
        [mask],
        gradient_accumulation_steps=1,
        vocab_char_map={" ": 0, "ni3": 1, "hao3": 2},
    )
    batch = _batch(mel_channels=2, seq_len=3)
    batch["text"] = [[" ", "ni3", "hao3"]]

    [yielded] = list(trainer._iter_global_masked_mean_batches([batch]))

    assert yielded[0] is batch
    assert yielded[1] is mask
    assert yielded[4] is True


def test_window_iterator_rejects_nested_tokens_without_vocabulary_map():
    trainer = _iterator_trainer([torch.ones((1, 3), dtype=torch.bool)], gradient_accumulation_steps=1)
    batch = _batch(mel_channels=2, seq_len=3)
    batch["text"] = [[" ", "ni3", "hao3"]]

    with pytest.raises(ValueError, match="nested text token lists require a vocabulary map"):
        list(trainer._iter_global_masked_mean_batches([batch]))


def test_window_iterator_rejects_non_string_preprocessed_text_tokens():
    trainer = _iterator_trainer([torch.ones((1, 3), dtype=torch.bool)], gradient_accumulation_steps=1)
    batch = _batch(mel_channels=2, seq_len=3)
    batch["text"] = [["ni3", 3]]

    with pytest.raises(ValueError, match="strings or lists of strings"):
        list(trainer._iter_global_masked_mean_batches([batch]))


def test_window_scale_includes_configured_g_and_data_parallel_world_size():
    masks = [torch.tensor([[True, True, False]])]
    trainer = _iterator_trainer(
        masks,
        gradient_accumulation_steps=4,
        num_processes=2,
        remote_denominator=6,
    )

    [(_, _, loss_scale, global_denominator, is_boundary)] = list(
        trainer._iter_global_masked_mean_batches([_batch(mel_channels=2, seq_len=3)])
    )

    # Local denominator 2 tokens * 2 channels = 4; emulated peer contributes 6.
    assert global_denominator == 10
    assert loss_scale.dtype == torch.float32
    assert loss_scale.device == trainer.accelerator.device
    torch.testing.assert_close(loss_scale, torch.tensor(4 * 2 / 10, dtype=torch.float32))
    assert is_boundary is True


class _NoItemTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, value):
        return value.as_subclass(cls)

    def item(self):
        raise AssertionError("denominator path performed a host-synchronizing Tensor.item()")


class _NoItemAccelerator(_IteratorAccelerator):
    def reduce(self, value, reduction):
        result = super().reduce(value, reduction)
        return _NoItemTensor(result)


def test_window_scale_stays_on_device_without_tensor_item():
    trainer = _iterator_trainer([torch.tensor([[True, False]])], gradient_accumulation_steps=2)
    trainer.accelerator = _NoItemAccelerator()

    [entry] = list(trainer._iter_global_masked_mean_batches([_batch(mel_channels=2, seq_len=2)]))

    assert isinstance(entry[2], torch.Tensor)
    assert entry[2].dtype == torch.float32


def test_all_zero_global_denominator_fails_before_yielding_or_backward():
    trainer = _iterator_trainer(
        [torch.zeros((1, 3), dtype=torch.bool), torch.zeros((1, 3), dtype=torch.bool)],
        gradient_accumulation_steps=2,
    )

    with pytest.raises(RuntimeError, match="empty accumulation window"):
        list(
            trainer._iter_global_masked_mean_batches(
                [_batch(mel_channels=2, seq_len=3), _batch(mel_channels=2, seq_len=3)]
            )
        )


class _RecordingContext:
    def __init__(self, events, name):
        self.events = events
        self.name = name

    def __enter__(self):
        self.events.append(f"enter:{self.name}")

    def __exit__(self, exc_type, exc, traceback):
        self.events.append(f"exit:{self.name}")


class _ContextAccelerator:
    def __init__(self, distributed_type):
        self.distributed_type = distributed_type
        self.sync_gradients = True
        self.events = []

    def accumulate(self, model):
        return _RecordingContext(self.events, "accumulate")

    def no_sync(self, model):
        return _RecordingContext(self.events, "no_sync")


class _BoundaryModel:
    def __init__(self, *, gradient_accumulation_steps=4):
        self.boundaries = []
        self.dp_world_size = 2
        self._gradient_accumulation_steps = gradient_accumulation_steps

    def set_gradient_accumulation_boundary(self, is_boundary):
        self.boundaries.append(is_boundary)

    def gradient_accumulation_steps(self):
        return self._gradient_accumulation_steps


def _context_trainer(distributed_type):
    trainer = Trainer.__new__(Trainer)
    trainer.accelerator = _ContextAccelerator(distributed_type)
    trainer.model = _BoundaryModel()
    return trainer


def test_default_path_keeps_accelerate_automatic_accumulation():
    trainer = _context_trainer(DistributedType.NO)

    with trainer._accumulation_context(None):
        trainer.accelerator.events.append("body")

    assert trainer.accelerator.events == ["enter:accumulate", "body", "exit:accumulate"]
    assert trainer.accelerator.sync_gradients is True
    assert trainer.model.boundaries == []


@pytest.mark.parametrize(
    "is_boundary,expected_events",
    [
        (False, ["enter:no_sync", "body", "exit:no_sync"]),
        (True, ["body"]),
    ],
)
def test_ddp_explicit_boundary_controls_no_sync(is_boundary, expected_events):
    trainer = _context_trainer(DistributedType.MULTI_CPU)

    with trainer._accumulation_context(is_boundary):
        trainer.accelerator.events.append("body")

    assert trainer.accelerator.events == expected_events
    assert trainer.accelerator.sync_gradients is is_boundary
    assert trainer.model.boundaries == []


def test_context_propagates_body_exception_without_corrupting_boundary_state():
    trainer = _context_trainer(DistributedType.MULTI_CPU)

    with pytest.raises(RuntimeError, match="forward failed"):
        with trainer._accumulation_context(False):
            raise RuntimeError("forward failed")

    assert trainer.accelerator.events == ["enter:no_sync", "exit:no_sync"]
    assert trainer.accelerator.sync_gradients is False


class _DeepSpeedPluginStub:
    def __init__(self, *, zero_stage=0, gradient_clipping=1.0):
        self.zero_stage = zero_stage
        self.gradient_clipping = gradient_clipping
        self.deepspeed_config = {
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": gradient_clipping,
        }

    def get_value(self, key):
        assert key in {"gradient_accumulation_steps", "gradient_clipping"}
        return self.deepspeed_config[key]


class _BackendAccelerator:
    def __init__(
        self,
        *,
        distributed_type=DistributedType.DEEPSPEED,
        num_processes=2,
        gas=4,
        zero_stage=0,
    ):
        self.distributed_type = distributed_type
        self.num_processes = num_processes
        self.gradient_accumulation_steps = gas
        self.state = SimpleNamespace(deepspeed_plugin=_DeepSpeedPluginStub(zero_stage=zero_stage))


def _backend_trainer(model=None, *, trainer_gas=4, accelerator_gas=4, zero_stage=0):
    trainer = Trainer.__new__(Trainer)
    trainer.global_masked_mean = True
    trainer.grad_accumulation_steps = trainer_gas
    trainer.max_grad_norm = 1.0
    trainer.accelerator = _BackendAccelerator(gas=accelerator_gas, zero_stage=zero_stage)
    trainer.model = _BoundaryModel(gradient_accumulation_steps=trainer_gas) if model is None else model
    return trainer


class _GatherAccelerator:
    def __init__(self, gathered_lengths):
        self.device = torch.device("cpu")
        self.gathered_lengths = torch.tensor(gathered_lengths, dtype=torch.int64)

    def gather(self, value):
        assert value.dtype == torch.int64
        return self.gathered_lengths.to(value.device)


def test_global_dataloader_validation_accepts_equal_rank_lengths():
    trainer = Trainer.__new__(Trainer)
    trainer.accelerator = _GatherAccelerator([7, 7])

    trainer._validate_global_masked_dataloader(range(7))


def test_global_dataloader_validation_rejects_unequal_rank_lengths_collectively():
    trainer = Trainer.__new__(Trainer)
    trainer.accelerator = _GatherAccelerator([7, 6])

    with pytest.raises(RuntimeError, match=r"received \[7, 6\]"):
        trainer._validate_global_masked_dataloader(range(7))


@pytest.mark.parametrize("num_processes", [2, 3])
@pytest.mark.parametrize("num_dynamic_batches", range(1, 8))
def test_dynamic_batch_sampler_drop_last_gives_every_rank_equal_batch_count(
    num_processes,
    num_dynamic_batches,
):
    # DynamicBatchSampler intentionally advertises drop_last=True to
    # BatchSamplerShard even when it retained its own residual sample batch.
    # Accelerate then drops only the final incomplete *group of rank batches*.
    sampler = DynamicBatchSampler.__new__(DynamicBatchSampler)
    sampler.batches = [[index] for index in range(num_dynamic_batches)]
    sampler.drop_last = True
    sampler.random_seed = None
    sampler.epoch = 0

    shards = [
        BatchSamplerShard(
            sampler,
            num_processes=num_processes,
            process_index=rank,
            split_batches=False,
            even_batches=False,
        )
        for rank in range(num_processes)
    ]
    per_rank_batches = [list(shard) for shard in shards]

    assert {len(batches) for batches in per_rank_batches} == {num_dynamic_batches // num_processes}
    assert {len(shard) for shard in shards} == {num_dynamic_batches // num_processes}


class _DirectDDPAccelerator:
    def __init__(self, *, world_size, gradient_accumulation_steps):
        self.device = torch.device("cpu")
        self.num_processes = world_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.distributed_type = DistributedType.MULTI_CPU
        self.sync_gradients = True

    def reduce(self, value, reduction):
        assert reduction == "sum"
        reduced = value.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        return reduced

    def no_sync(self, model):
        return model.no_sync()

    def backward(self, loss):
        (loss / self.gradient_accumulation_steps).backward()


def _direct_model():
    model = nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.25, -0.5], [0.4, 0.1]], dtype=torch.float32))
        model.bias.copy_(torch.tensor([0.125, -0.2], dtype=torch.float32))
    return model


def _direct_batch(rank, index, selected):
    seq_len = max(selected + 1, 2)
    raw = torch.arange(seq_len * 2, dtype=torch.float32).reshape(seq_len, 2)
    x = (raw + rank * 3 + index) / 7
    target = torch.stack((0.3 * x[:, 0] - 0.2, -0.4 * x[:, 1] + 0.1), dim=-1)
    mask = torch.zeros((1, seq_len), dtype=torch.bool)
    mask[0, :selected] = True
    return {
        "mel": x.transpose(0, 1).unsqueeze(0),
        "mel_lengths": torch.tensor([seq_len]),
        "target": target,
        "text": ["test"],
    }, mask


def _run_direct_ddp_case(rank, world_size, *, gradient_accumulation_steps, rank_counts):
    pairs = [_direct_batch(rank, index, selected) for index, selected in enumerate(rank_counts[rank])]
    batches = [pair[0] for pair in pairs]
    masks = [pair[1] for pair in pairs]
    model = _direct_model()
    ddp_model = DistributedDataParallel(model)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.03, momentum=0.8)
    trainer = Trainer.__new__(Trainer)
    trainer.grad_accumulation_steps = gradient_accumulation_steps
    trainer.accelerator = _DirectDDPAccelerator(
        world_size=world_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    trainer.model = ddp_model
    trainer._unwrapped_model = _MaskSampler(masks)

    boundaries = []
    for batch, mask, loss_scale, _, is_boundary in trainer._iter_global_masked_mean_batches(batches):
        boundaries.append(is_boundary)
        with trainer._accumulation_context(is_boundary):
            x = batch["mel"].permute(0, 2, 1).squeeze(0)
            residual = ddp_model(x) - batch["target"]
            loss_sum = (residual.square() * mask.squeeze(0)[:, None]).sum()
            trainer.accelerator.backward(loss_sum * loss_scale)
            if trainer.accelerator.sync_gradients:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    assert boundaries[-1] is True

    reference = _direct_model()
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03, momentum=0.8)
    global_loss_sum = torch.zeros((), dtype=torch.float32)
    global_denominator = 0
    for peer_rank, counts in enumerate(rank_counts):
        for index, selected in enumerate(counts):
            batch, mask = _direct_batch(peer_rank, index, selected)
            x = batch["mel"].permute(0, 2, 1).squeeze(0)
            residual = reference(x) - batch["target"]
            global_loss_sum = global_loss_sum + (residual.square() * mask.squeeze(0)[:, None]).sum()
            global_denominator += selected * 2
    (global_loss_sum / global_denominator).backward()
    reference_optimizer.step()

    torch.testing.assert_close(model.weight, reference.weight, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(model.bias, reference.bias, atol=1e-6, rtol=1e-6)


def _direct_ddp_worker(rank, world_size, init_method):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        _run_direct_ddp_case(
            rank,
            world_size,
            gradient_accumulation_steps=2,
            rank_counts=((1, 0), (4, 2)),
        )
        dist.barrier()
        _run_direct_ddp_case(
            rank,
            world_size,
            gradient_accumulation_steps=4,
            rank_counts=((0, 0), (5, 1)),
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _malformed_ddp_worker(rank, world_size, init_method):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        batch, mask = _direct_batch(rank, 0, selected=2)
        if rank == 0:
            batch["mel_lengths"] = batch["mel_lengths"].to(dtype=torch.float32)
        trainer = Trainer.__new__(Trainer)
        trainer.grad_accumulation_steps = 1
        trainer.accelerator = _DirectDDPAccelerator(world_size=world_size, gradient_accumulation_steps=1)
        trainer._unwrapped_model = _MaskSampler([mask])

        with pytest.raises(ValueError, match="global_masked_mean rejected an accumulation window"):
            list(trainer._iter_global_masked_mean_batches([batch]))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _rank_oom_worker(rank, world_size, init_method):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        timeout=datetime.timedelta(seconds=30),
    )
    try:

        class _RankMaskSampler:
            num_channels = 2
            vocab_char_map = None

            def sample_training_mask(self, lens, seq_len):
                if rank == 0:
                    oom_type = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
                    raise oom_type("CUDA out of memory. synthetic rank failure")
                return torch.ones((lens.shape[0], seq_len), dtype=torch.bool)

        batch, _ = _direct_batch(rank, 0, selected=2)
        trainer = Trainer.__new__(Trainer)
        trainer.grad_accumulation_steps = 1
        trainer.accelerator = _DirectDDPAccelerator(world_size=world_size, gradient_accumulation_steps=1)
        trainer._unwrapped_model = _RankMaskSampler()
        try:
            list(trainer._iter_global_masked_mean_batches([batch]))
        except BaseException as exc:
            status = (type(exc).__name__, str(exc))
        else:  # pragma: no cover - a missing collective error would land here
            status = ("no-error", "")

        statuses = [None] * world_size
        dist.all_gather_object(statuses, status)
        assert all(status[1] for status in statuses)
        assert "out of memory" in statuses[0][1].lower()
        assert "another rank" in statuses[1][1].lower()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="gloo backend unavailable")
def test_rank_local_cuda_oom_is_preserved_without_peer_hang():
    with tempfile.TemporaryDirectory(prefix="f5-global-oom-gloo-") as tmpdir:
        mp.start_processes(
            _rank_oom_worker,
            args=(2, f"file://{tmpdir}/store"),
            nprocs=2,
            join=True,
            start_method="spawn",
        )


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="gloo backend unavailable")
def test_production_helpers_match_global_batch_on_two_rank_gloo():
    with tempfile.TemporaryDirectory(prefix="f5-global-contract-gloo-") as tmpdir:
        mp.start_processes(
            _direct_ddp_worker,
            args=(2, f"file://{tmpdir}/store"),
            nprocs=2,
            join=True,
            start_method="spawn",
        )


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="gloo backend unavailable")
def test_rank_local_malformed_batch_raises_collectively_before_backward():
    with tempfile.TemporaryDirectory(prefix="f5-global-malformed-gloo-") as tmpdir:
        mp.start_processes(
            _malformed_ddp_worker,
            args=(2, f"file://{tmpdir}/store"),
            nprocs=2,
            join=True,
            start_method="spawn",
        )
