"""Adversarial reference tests for accumulation-wide masked-mean training.

This is intentionally a validation-branch harness, not an upstream unit test.
It provides an independent, tiny-model oracle for the training-loop contract:

* denominators are counted exactly before any backward call;
* every numerator is scaled before backward;
* a partial final accumulation window still performs exactly one update;
* an all-empty window performs no optimizer/scheduler/EMA update;
* DeepSpeed may consume/partition/clear gradients inside ``backward``;
* the result matches one concatenated global masked-mean batch.

Production integration should replace ``_pre_backward_update`` with the F5-TTS
window helper while retaining ``_concatenated_reference_update`` as the oracle.
No F5-TTS model import is needed, keeping the harness CPU/gloo safe.
"""

from __future__ import annotations

import contextlib
import datetime
import tempfile
from dataclasses import dataclass
from typing import Iterable, Sequence

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class _Batch:
    x: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor

    @property
    def denominator(self) -> torch.Tensor:
        return self.mask.count_nonzero().to(dtype=torch.int64)


@dataclass(frozen=True)
class _UpdateResult:
    optimizer_steps: int
    scheduler_steps: int
    ema_steps: int
    skipped: bool


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


class _CountingScheduler:
    def __init__(self):
        self.step_count = 0

    def step(self):
        self.step_count += 1


class _CountingEMA:
    def __init__(self):
        self.step_count = 0

    def update(self):
        self.step_count += 1


def _make_model() -> nn.Linear:
    model = nn.Linear(2, 1, bias=True)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.25, -0.5]]))
        model.bias.copy_(torch.tensor([0.125]))
    return model


def _make_batches(mask_counts: Sequence[int]) -> list[_Batch]:
    batches = []
    cursor = 0
    for batch_index, count in enumerate(mask_counts):
        # Leave at least one unselected item so a mean-of-means implementation
        # cannot accidentally pass all parameterizations.
        size = max(count + 1, 2)
        values = torch.arange(cursor, cursor + size * 2, dtype=torch.float64).reshape(size, 2)
        x = ((values % 11) - 5.0) / 4.0
        target = (0.3 * x[:, :1]) - (0.2 * x[:, 1:]) + (batch_index + 1) * 0.17
        mask = torch.zeros(size, dtype=torch.bool)
        mask[:count] = True
        batches.append(_Batch(x=x, target=target, mask=mask))
        cursor += size * 2
    return batches


def _loss_sum(model: nn.Module, batch: _Batch) -> torch.Tensor:
    residual = model(batch.x) - batch.target
    return residual.square().squeeze(-1).masked_fill(~batch.mask, 0.0).sum()


def _exact_denominator(batches: Iterable[_Batch]) -> torch.Tensor:
    counts = [batch.denominator for batch in batches]
    if not counts:
        return torch.zeros((), dtype=torch.int64)
    total = torch.stack(counts).sum(dtype=torch.int64)
    if total < 0:  # defensive contract for a future externally supplied count
        raise ValueError("masked-element denominator cannot be negative")
    return total


def _pre_backward_update(
    model: nn.Module,
    batches: Sequence[_Batch],
    *,
    gradient_accumulation_steps: int,
    optimizer: _CountingSGD,
    scheduler: _CountingScheduler,
    ema: _CountingEMA,
    max_grad_norm: float | None = None,
) -> _UpdateResult:
    """Candidate window algorithm with Accelerate's configured-G division.

    The buffered data contain no autograd graph. Only after the exact denominator
    is known do we run one forward/backward per microbatch.
    """
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if len(batches) > gradient_accumulation_steps:
        raise ValueError("one update cannot contain more than one accumulation window")

    denominator = _exact_denominator(batches)
    if int(denominator) == 0:
        optimizer.zero_grad(set_to_none=True)
        return _UpdateResult(optimizer.step_count, scheduler.step_count, ema.step_count, skipped=True)

    # Accelerate divides non-DeepSpeed losses by the configured accumulation
    # count even for a short final window. Multiplying before backward cancels
    # that division. Crucially, it occurs before clipping or optimizer.step.
    scale = torch.tensor(
        gradient_accumulation_steps / int(denominator),
        dtype=next(model.parameters()).dtype,
    )
    for batch in batches:
        scaled_loss = _loss_sum(model, batch) * scale
        (scaled_loss / gradient_accumulation_steps).backward()

    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    ema.update()
    optimizer.zero_grad(set_to_none=True)
    return _UpdateResult(optimizer.step_count, scheduler.step_count, ema.step_count, skipped=False)


def _concatenated_reference_update(
    model: nn.Module,
    batches: Sequence[_Batch],
    *,
    optimizer: _CountingSGD,
    scheduler: _CountingScheduler,
    ema: _CountingEMA,
    max_grad_norm: float | None = None,
) -> _UpdateResult:
    denominator = _exact_denominator(batches)
    if int(denominator) == 0:
        optimizer.zero_grad(set_to_none=True)
        return _UpdateResult(optimizer.step_count, scheduler.step_count, ema.step_count, skipped=True)

    loss = torch.stack([_loss_sum(model, batch) for batch in batches]).sum() / int(denominator)
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    ema.update()
    optimizer.zero_grad(set_to_none=True)
    return _UpdateResult(optimizer.step_count, scheduler.step_count, ema.step_count, skipped=False)


def _run_pair(
    mask_counts: Sequence[int],
    *,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None = None,
    weight_decay: float = 0.0,
):
    batches = _make_batches(mask_counts)
    candidate = _make_model().double()
    reference = _make_model().double()
    reference.load_state_dict(candidate.state_dict())

    candidate_optimizer = _CountingSGD(candidate.parameters(), lr=0.03, momentum=0.8, weight_decay=weight_decay)
    reference_optimizer = _CountingSGD(reference.parameters(), lr=0.03, momentum=0.8, weight_decay=weight_decay)
    candidate_scheduler, reference_scheduler = _CountingScheduler(), _CountingScheduler()
    candidate_ema, reference_ema = _CountingEMA(), _CountingEMA()

    candidate_result = _pre_backward_update(
        candidate,
        batches,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optimizer=candidate_optimizer,
        scheduler=candidate_scheduler,
        ema=candidate_ema,
        max_grad_norm=max_grad_norm,
    )
    reference_result = _concatenated_reference_update(
        reference,
        batches,
        optimizer=reference_optimizer,
        scheduler=reference_scheduler,
        ema=reference_ema,
        max_grad_norm=max_grad_norm,
    )
    return candidate, reference, candidate_result, reference_result


@pytest.mark.parametrize(
    "mask_counts,gradient_accumulation_steps",
    [
        ([3], 1),  # G=1
        ([1, 5, 2, 7], 4),  # full window, unequal microbatch counts
        ([0, 4, 1], 4),  # partial final window with one empty microbatch
        ([2, 6], 8),  # entire dataloader shorter than configured G
    ],
)
def test_pre_backward_window_matches_concatenated_parameter_update(mask_counts, gradient_accumulation_steps):
    candidate, reference, candidate_result, reference_result = _run_pair(
        mask_counts, gradient_accumulation_steps=gradient_accumulation_steps
    )

    torch.testing.assert_close(candidate.weight, reference.weight, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(candidate.bias, reference.bias, atol=1e-12, rtol=1e-12)
    assert candidate_result == reference_result == _UpdateResult(1, 1, 1, skipped=False)


def test_normalization_precedes_gradient_clipping():
    candidate, reference, candidate_result, reference_result = _run_pair(
        [1, 17, 2],
        gradient_accumulation_steps=4,
        max_grad_norm=0.08,
    )

    torch.testing.assert_close(candidate.weight, reference.weight, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(candidate.bias, reference.bias, atol=1e-12, rtol=1e-12)
    assert candidate_result == reference_result == _UpdateResult(1, 1, 1, skipped=False)


def test_all_zero_window_skips_weight_decay_optimizer_scheduler_and_ema():
    candidate, reference, candidate_result, reference_result = _run_pair(
        [0, 0],
        gradient_accumulation_steps=4,
        weight_decay=0.2,
    )

    torch.testing.assert_close(candidate.weight, reference.weight, atol=0, rtol=0)
    torch.testing.assert_close(candidate.bias, reference.bias, atol=0, rtol=0)
    assert candidate_result == reference_result == _UpdateResult(0, 0, 0, skipped=True)
    assert all(parameter.grad is None for parameter in candidate.parameters())


def test_denominator_remains_exact_beyond_float32_integer_range():
    counts = torch.tensor([2**24, 1, 9], dtype=torch.int64)
    exact = counts.sum(dtype=torch.int64)
    lossy = counts.to(dtype=torch.float32).sum()

    assert exact.dtype == torch.int64
    assert int(exact) == 2**24 + 10
    assert int(lossy) != int(exact)


@pytest.mark.parametrize("gradient_accumulation_steps", [0, -1])
def test_invalid_accumulation_count_fails_before_backward(gradient_accumulation_steps):
    model = _make_model().double()
    optimizer = _CountingSGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="must be positive"):
        _pre_backward_update(
            model,
            _make_batches([1]),
            gradient_accumulation_steps=gradient_accumulation_steps,
            optimizer=optimizer,
            scheduler=_CountingScheduler(),
            ema=_CountingEMA(),
        )
    assert optimizer.step_count == 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_oversized_window_fails_instead_of_silently_changing_update_boundaries():
    model = _make_model().double()
    optimizer = _CountingSGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="more than one accumulation window"):
        _pre_backward_update(
            model,
            _make_batches([1, 2, 3]),
            gradient_accumulation_steps=2,
            optimizer=optimizer,
            scheduler=_CountingScheduler(),
            ema=_CountingEMA(),
        )
    assert optimizer.step_count == 0


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_compiled_loss_sum_keeps_window_normalization_outside_graph(backend):
    if backend == "inductor":
        pytest.importorskip("torch._inductor")
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")

    class _LossCore(nn.Module):
        def forward(self, prediction, target, mask):
            return (prediction.sub(target).square() * mask[:, None]).sum()

    batches = _make_batches([1, 5, 2])
    eager_model = _make_model().double()
    compiled_model = _make_model().double()
    compiled_model.load_state_dict(eager_model.state_dict())
    eager_optimizer = _CountingSGD(eager_model.parameters(), lr=0.03)
    compiled_optimizer = _CountingSGD(compiled_model.parameters(), lr=0.03)
    compiled_core = torch.compile(_LossCore(), backend=backend, fullgraph=True, dynamic=True)
    denominator = _exact_denominator(batches)

    for batch in batches:
        (_loss_sum(eager_model, batch) / int(denominator)).backward()
        compiled_sum = compiled_core(compiled_model(batch.x), batch.target, batch.mask)
        (compiled_sum / int(denominator)).backward()
    eager_optimizer.step()
    compiled_optimizer.step()

    torch.testing.assert_close(compiled_model.weight, eager_model.weight, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(compiled_model.bias, eager_model.bias, atol=1e-10, rtol=1e-10)


class _DeepSpeedEngineMock:
    """Small semantic mock of the engine boundary contract, not ZeRO internals."""

    def __init__(self, model: nn.Module, *, gradient_accumulation_steps: int):
        self.model = model
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optimizer = _CountingSGD(model.parameters(), lr=0.03)
        self.boundary_override: bool | None = None
        self.micro_steps = 0
        self.step_calls = 0
        self.actual_updates = 0
        self.boundary_events: list[bool] = []

    def set_gradient_accumulation_boundary(self, is_boundary: bool):
        self.boundary_override = is_boundary
        self.boundary_events.append(is_boundary)

    def backward(self, loss: torch.Tensor):
        # DeepSpeed owns configured-G loss scaling.
        (loss / self.gradient_accumulation_steps).backward()

    def step(self):
        self.step_calls += 1
        default_boundary = (self.micro_steps + 1) % self.gradient_accumulation_steps == 0
        boundary = default_boundary if self.boundary_override is None else self.boundary_override
        self.micro_steps += 1
        if not boundary:
            return
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.actual_updates += 1


class _Accelerate033DeepSpeedWrapperMock:
    """Accelerate 0.33: backward always invokes engine.step()."""

    def __init__(self, engine: _DeepSpeedEngineMock):
        self.engine = engine

    def backward(self, loss: torch.Tensor, *, sync_gradients: bool):
        del sync_gradients
        self.engine.backward(loss)
        self.engine.step()


class _AccelerateCurrentDeepSpeedWrapperMock:
    """Accelerate 1.14: boundary follows sync_gradients and step is conditional."""

    def __init__(self, engine: _DeepSpeedEngineMock):
        self.engine = engine

    def backward(self, loss: torch.Tensor, *, sync_gradients: bool):
        self.engine.set_gradient_accumulation_boundary(is_boundary=sync_gradients)
        self.engine.backward(loss)
        if sync_gradients:
            self.engine.step()


def _deepspeed_pre_backward_update(wrapper, batches: Sequence[_Batch], *, gradient_accumulation_steps: int):
    engine = wrapper.engine
    denominator = _exact_denominator(batches)
    if int(denominator) == 0:
        return
    scale = gradient_accumulation_steps / int(denominator)
    for index, batch in enumerate(batches):
        is_boundary = index == len(batches) - 1
        # The explicit engine boundary is required for a partial window under
        # Accelerate 0.33; current Accelerate sets the same value again.
        engine.set_gradient_accumulation_boundary(is_boundary=is_boundary)
        wrapper.backward(_loss_sum(engine.model, batch) * scale, sync_gradients=is_boundary)


@pytest.mark.parametrize(
    "wrapper_type,expected_step_calls",
    [
        (_Accelerate033DeepSpeedWrapperMock, 2),
        (_AccelerateCurrentDeepSpeedWrapperMock, 1),
    ],
    ids=["accelerate-0.33", "accelerate-current"],
)
def test_deepspeed_boundary_adapter_flushes_partial_window_once(wrapper_type, expected_step_calls):
    batches = _make_batches([1, 6])
    candidate = _make_model().double()
    reference = _make_model().double()
    reference.load_state_dict(candidate.state_dict())
    engine = _DeepSpeedEngineMock(candidate, gradient_accumulation_steps=4)

    _deepspeed_pre_backward_update(wrapper_type(engine), batches, gradient_accumulation_steps=4)
    _concatenated_reference_update(
        reference,
        batches,
        optimizer=_CountingSGD(reference.parameters(), lr=0.03),
        scheduler=_CountingScheduler(),
        ema=_CountingEMA(),
    )

    assert engine.actual_updates == 1
    assert engine.step_calls == expected_step_calls
    assert engine.boundary_events[-1] is True
    torch.testing.assert_close(candidate.weight, reference.weight, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(candidate.bias, reference.bias, atol=1e-12, rtol=1e-12)


def test_post_backward_gradient_scaling_is_too_late_for_deepspeed_step():
    batches = _make_batches([1, 7])
    broken = _make_model().double()
    reference = _make_model().double()
    reference.load_state_dict(broken.state_dict())
    engine = _DeepSpeedEngineMock(broken, gradient_accumulation_steps=2)
    wrapper = _AccelerateCurrentDeepSpeedWrapperMock(engine)
    denominator = _exact_denominator(batches)

    # Reproduce the rejected design: backward raw sums and mutate .grad only
    # after the boundary backward. Current Accelerate has already called
    # engine.step() and DeepSpeed has cleared the gradients at that point.
    for index, batch in enumerate(batches):
        wrapper.backward(_loss_sum(broken, batch), sync_gradients=index == len(batches) - 1)
    for parameter in broken.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(2 / int(denominator))

    _concatenated_reference_update(
        reference,
        batches,
        optimizer=_CountingSGD(reference.parameters(), lr=0.03),
        scheduler=_CountingScheduler(),
        ema=_CountingEMA(),
    )

    assert engine.actual_updates == 1
    assert all(parameter.grad is None for parameter in broken.parameters())
    with pytest.raises(AssertionError):
        torch.testing.assert_close(broken.weight, reference.weight, atol=1e-12, rtol=1e-12)


def test_no_autograd_graph_is_created_while_precomputing_denominator():
    batches = _make_batches([1, 3, 2])
    for batch in batches:
        batch.x.requires_grad_(True)

    denominator = _exact_denominator(batches)

    assert denominator.dtype == torch.int64
    assert denominator.grad_fn is None
    assert all(batch.x.grad_fn is None and batch.x.grad is None for batch in batches)


def _ddp_case(rank: int, world_size: int, *, gradient_accumulation_steps: int, rank_counts):
    local_batches = _make_batches(rank_counts[rank])
    model = _make_model().double()
    ddp_model = DistributedDataParallel(model)
    optimizer = _CountingSGD(ddp_model.parameters(), lr=0.03, momentum=0.8)

    local_denominator = _exact_denominator(local_batches)
    global_denominator = local_denominator.clone()
    dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM)
    assert global_denominator.dtype == torch.int64
    assert int(global_denominator) > 0

    for index, batch in enumerate(local_batches):
        is_boundary = index == len(local_batches) - 1
        sync_context = contextlib.nullcontext() if is_boundary else ddp_model.no_sync()
        with sync_context:
            # Accelerate divides by configured G; DDP averages by world size.
            # Both are cancelled before backward.
            loss = _loss_sum(ddp_model, batch)
            scaled_loss = loss * gradient_accumulation_steps * world_size / int(global_denominator)
            (scaled_loss / gradient_accumulation_steps).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Independent global-batch oracle uses every rank's data in one process.
    reference = _make_model().double()
    reference_optimizer = _CountingSGD(reference.parameters(), lr=0.03, momentum=0.8)
    global_batches = [batch for counts in rank_counts for batch in _make_batches(counts)]
    _concatenated_reference_update(
        reference,
        global_batches,
        optimizer=reference_optimizer,
        scheduler=_CountingScheduler(),
        ema=_CountingEMA(),
    )

    torch.testing.assert_close(model.weight, reference.weight, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(model.bias, reference.bias, atol=1e-12, rtol=1e-12)

    # Detect rank divergence even if both ranks happened to miss the oracle in
    # the same way. This also catches an unmatched final reduction boundary.
    flattened = torch.cat([model.weight.detach().reshape(-1), model.bias.detach().reshape(-1)])
    gathered = [torch.empty_like(flattened) for _ in range(world_size)]
    dist.all_gather(gathered, flattened)
    for peer in gathered:
        torch.testing.assert_close(flattened, peer, atol=0, rtol=0)


def _ddp_worker(rank: int, world_size: int, init_method: str):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        # Full G=2 window with unequal counts and one empty microbatch.
        _ddp_case(
            rank,
            world_size,
            gradient_accumulation_steps=2,
            rank_counts=((1, 0), (4, 2)),
        )
        dist.barrier()
        # Partial R=2 < G=4 window where rank 0 selects nothing at all.
        _ddp_case(
            rank,
            world_size,
            gradient_accumulation_steps=4,
            rank_counts=((0, 0), (5, 1)),
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="gloo backend unavailable")
def test_two_rank_gloo_matches_global_batch_for_full_partial_and_zero_local_denominator():
    with tempfile.TemporaryDirectory(prefix="f5-global-mean-gloo-") as tmpdir:
        init_method = f"file://{tmpdir}/store"
        mp.spawn(_ddp_worker, args=(2, init_method), nprocs=2, join=True)
