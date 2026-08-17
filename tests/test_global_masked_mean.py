import inspect
import random
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from accelerate.utils import DataLoaderConfiguration
from omegaconf import OmegaConf
from torch import nn

from f5_tts.model.cfm import CFM
from f5_tts.model.trainer import Trainer
from f5_tts.model.utils import lens_to_mask, mask_from_frac_lengths, mask_from_start_end_indices


class _MelSpec(nn.Module):
    n_mel_channels = 2


class _ConstantTransformer(nn.Module):
    dim = 2

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.25))

    def forward(self, *, x, cond, text, time, drop_audio_cond, drop_text, mask):
        del cond, text, time, drop_audio_cond, drop_text, mask
        return self.bias.expand_as(x)


def _cfm():
    return CFM(
        transformer=_ConstantTransformer(),
        mel_spec_module=_MelSpec(),
        frac_lengths_mask=(0.5, 0.5),
        audio_drop_prob=0.0,
        cond_drop_prob=0.0,
    )


def test_mask_from_start_end_indices_preserves_implicit_and_explicit_lengths():
    seq_len = torch.tensor([3, 5])
    start = torch.tensor([1, 2])
    end = torch.tensor([3, 5])

    implicit = mask_from_start_end_indices(seq_len, start, end)
    explicit = mask_from_start_end_indices(seq_len, start, end, length=7)

    assert implicit.shape == (2, 5)
    assert explicit.shape == (2, 7)
    torch.testing.assert_close(explicit[:, :5], implicit)
    assert not explicit[:, 5:].any()


def test_global_masked_mean_is_opt_in_in_api_and_shipped_configs():
    assert inspect.signature(Trainer).parameters["global_masked_mean"].default is False
    checkpoint_parameters = list(inspect.signature(Trainer.save_checkpoint).parameters.values())
    assert checkpoint_parameters[2].name == "last"
    assert checkpoint_parameters[3].name == "consumed_batches"
    assert checkpoint_parameters[3].kind is inspect.Parameter.KEYWORD_ONLY
    config_dir = Path(__file__).parents[1] / "src" / "f5_tts" / "configs"
    configs = sorted(config_dir.glob("*.yaml"))
    assert len(configs) == 6
    for config_path in configs:
        config = OmegaConf.load(config_path)
        assert config.optim.global_masked_mean is False, config_path


def test_global_masked_mean_disables_accelerate_duplicate_padding(tmp_path):
    trainer = Trainer(
        _cfm(),
        epochs=1,
        learning_rate=1e-4,
        num_warmup_updates=0,
        checkpoint_path=tmp_path,
        logger=None,
        global_masked_mean=True,
        accelerate_kwargs={"cpu": True},
    )
    assert trainer.accelerator.even_batches is False


def test_global_masked_mean_rejects_duplicate_padding_configuration(tmp_path):
    with pytest.raises(ValueError, match="even_batches=False"):
        Trainer(
            _cfm(),
            epochs=1,
            learning_rate=1e-4,
            num_warmup_updates=0,
            checkpoint_path=tmp_path,
            logger=None,
            global_masked_mean=True,
            accelerate_kwargs={"dataloader_config": DataLoaderConfiguration(even_batches=True)},
        )


def _batch(frames=4, channels=2):
    return {
        "mel": torch.zeros((1, channels, frames)),
        "mel_lengths": torch.tensor([frames]),
        "text": ["text"],
    }


def test_sample_training_mask_is_boolean_and_never_selects_padding():
    model = _cfm()
    lens = torch.tensor([5, 2], dtype=torch.int32)

    mask = model.sample_training_mask(lens, seq_len=7)

    assert mask.shape == (2, 7)
    assert mask.dtype == torch.bool
    assert not mask[0, 5:].any()
    assert not mask[1, 2:].any()


@pytest.mark.parametrize(
    ("lens", "seq_len", "message"),
    [
        (torch.tensor([-1]), 3, "between 0 and seq_len"),
        (torch.tensor([4]), 3, "between 0 and seq_len"),
        (torch.tensor([1.0]), 3, "integer dtype"),
    ],
)
def test_sample_training_mask_rejects_invalid_lengths(lens, seq_len, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _cfm().sample_training_mask(lens, seq_len)


def test_presampled_mask_returns_fp32_sum_and_denominator(monkeypatch):
    model = _cfm()
    inp = torch.tensor(
        [
            [[1.0, -2.0], [3.0, 4.0], [5.0, -6.0], [7.0, 8.0]],
            [[-1.0, 2.0], [3.0, -4.0], [5.0, 6.0], [7.0, -8.0]],
        ]
    )
    text = torch.zeros((2, 2), dtype=torch.long)
    lens = torch.tensor([4, 3])
    span = torch.tensor(
        [
            [True, False, True, False],
            [False, True, True, False],
        ]
    )
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.zeros_like(value))
    monkeypatch.setattr(
        torch,
        "rand",
        lambda shape, *, dtype, device: torch.full(shape, 0.5, dtype=dtype, device=device),
    )

    _, loss_sum, loss_count, cond, pred = model(
        inp,
        text=text,
        lens=lens,
        rand_span_mask=span,
        return_loss_components=True,
    )

    expected_elementwise = F.mse_loss(pred.float(), inp.float(), reduction="none")
    expected_sum = torch.where(span[..., None], expected_elementwise, torch.zeros_like(expected_elementwise)).sum()
    expected_count = span.sum(dtype=torch.int64) * inp.shape[-1]
    torch.testing.assert_close(loss_sum, expected_sum)
    assert loss_sum.dtype == torch.float32
    # The merged core returns a float denominator for the per-batch mean; exact
    # int64 global counting lives in the trainer's own accumulation window math.
    assert torch.equal(loss_count, expected_count.to(loss_count.dtype))
    assert loss_count.dtype.is_floating_point
    assert torch.equal(cond == 0, span[..., None].expand_as(cond))

    loss_sum.backward()
    assert torch.isfinite(model.transformer.bias.grad)


def test_presampled_default_path_matches_masked_mean(monkeypatch):
    model = _cfm()
    inp = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
    text = torch.zeros((2, 2), dtype=torch.long)
    lens = torch.tensor([4, 4])
    span = torch.tensor([[True, False, True, False], [False, True, False, True]])
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.zeros_like(value))
    monkeypatch.setattr(
        torch,
        "rand",
        lambda shape, *, dtype, device: torch.full(shape, 0.5, dtype=dtype, device=device),
    )

    loss, _, pred = model(inp, text=text, lens=lens, rand_span_mask=span)
    expected = F.mse_loss(pred, inp, reduction="none")[span].mean()

    torch.testing.assert_close(loss, expected, atol=0, rtol=0)


def test_unselected_nonfinite_values_do_not_poison_loss_sum(monkeypatch):
    model = _cfm()
    inp = torch.ones((1, 4, 2))
    span = torch.tensor([[True, False, True, False]])
    inp[:, ~span[0]] = torch.nan
    text = torch.zeros((1, 2), dtype=torch.long)
    lens = torch.tensor([4])
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.zeros_like(value))
    monkeypatch.setattr(
        torch,
        "rand",
        lambda shape, *, dtype, device: torch.full(shape, 0.5, dtype=dtype, device=device),
    )

    _, loss_sum, loss_count, _, _ = model(
        inp,
        text=text,
        lens=lens,
        rand_span_mask=span,
        return_loss_components=True,
    )

    assert loss_count == 4
    assert torch.isfinite(loss_sum)


def test_default_path_preserves_historical_rng_order_and_reduction():
    model = _cfm()
    inp = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
    text = torch.zeros((2, 2), dtype=torch.long)
    lens = torch.tensor([4, 3])
    seed = 912

    random.seed(seed)
    torch.manual_seed(seed)
    loss, cond, pred = model(inp, text=text, lens=lens)

    random.seed(seed)
    torch.manual_seed(seed)
    valid_mask = lens_to_mask(lens, length=inp.shape[1])
    frac_lengths = torch.zeros((inp.shape[0],)).uniform_(*model.frac_lengths_mask)
    span = mask_from_frac_lengths(lens, frac_lengths, length=inp.shape[1]) & valid_mask
    x0 = torch.randn_like(inp)
    torch.rand((inp.shape[0],), dtype=inp.dtype)
    random.random()
    random.random()
    flow = inp - x0

    torch.testing.assert_close(cond, torch.where(span[..., None], torch.zeros_like(inp), inp), atol=0, rtol=0)
    torch.testing.assert_close(loss, F.mse_loss(pred, flow, reduction="none")[span].mean(), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_low_precision_components_and_gradients_are_finite(dtype, monkeypatch):
    model = _cfm().to(device="cuda", dtype=dtype)
    inp = torch.full((2, 4, 2), 100.0, device="cuda", dtype=dtype)
    text = torch.zeros((2, 2), device="cuda", dtype=torch.long)
    lens = torch.tensor([4, 3], device="cuda")
    span = torch.tensor([[True, False, True, False], [False, True, True, False]], device="cuda", dtype=torch.bool)
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.zeros_like(value))
    monkeypatch.setattr(
        torch,
        "rand",
        lambda shape, *, dtype, device: torch.full(shape, 0.5, dtype=dtype, device=device),
    )

    _, loss_sum, loss_count, _, _ = model(
        inp,
        text=text,
        lens=lens,
        rand_span_mask=span,
        return_loss_components=True,
    )
    (loss_sum / loss_count).backward()

    assert loss_sum.dtype == torch.float32
    assert torch.isfinite(loss_sum)
    assert torch.isfinite(model.transformer.bias.grad)


@pytest.mark.parametrize(
    ("bad_mask", "message"),
    [
        (torch.ones((2, 3), dtype=torch.bool), "must have shape"),
        (torch.ones((2, 4)), "dtype torch.bool"),
        (torch.ones((2, 4), dtype=torch.int64), "dtype torch.bool"),
    ],
    ids=["shape", "float", "integer"],
)
def test_forward_rejects_malformed_presampled_mask(bad_mask, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _cfm()(
            torch.zeros((2, 4, 2)),
            text=torch.zeros((2, 2), dtype=torch.long),
            lens=torch.tensor([4, 3]),
            rand_span_mask=bad_mask,
        )


class _MaskSampler:
    def __init__(self, masks):
        self.masks = iter(masks)

    def sample_training_mask(self, lens, seq_len):
        del lens, seq_len
        return next(self.masks)


class _ReducingAccelerator:
    def __init__(self, world_size=2):
        self.device = torch.device("cpu")
        self.num_processes = world_size

    def reduce(self, value, reduction):
        assert reduction == "sum"
        return value * self.num_processes


def _iterator_trainer(masks, gradient_accumulation_steps=3, world_size=2):
    trainer = Trainer.__new__(Trainer)
    trainer.__dict__.update(
        grad_accumulation_steps=gradient_accumulation_steps,
        accelerator=_ReducingAccelerator(world_size),
        _unwrapped_model=_MaskSampler(masks),
    )
    return trainer


def test_window_iterator_uses_global_int64_count_and_exact_boundaries():
    masks = [
        torch.tensor([[True, False, False, False]]),
        torch.tensor([[True, True, False, False]]),
        torch.tensor([[True, True, True, False]]),
        torch.tensor([[True, False, False, False]]),
        torch.tensor([[False, True, False, False]]),
    ]
    trainer = _iterator_trainer(masks)

    rows = list(trainer._iter_global_masked_mean_batches([_batch() for _ in masks]))

    assert [row[-1] for row in rows] == [False, False, True, False, True]
    first_global_count = rows[0][3]
    partial_global_count = rows[3][3]
    assert first_global_count.dtype == torch.int64
    assert int(first_global_count) == (1 + 2 + 3) * 2 * 2
    assert int(partial_global_count) == (1 + 1) * 2 * 2
    assert float(rows[0][2]) == pytest.approx(3 * 2 / int(first_global_count))
    assert float(rows[3][2]) == pytest.approx(3 * 2 / int(partial_global_count))
    assert all(row[3] is first_global_count for row in rows[:3])
    assert all(row[3] is partial_global_count for row in rows[3:])


def test_all_zero_global_window_fails_before_forward():
    masks = [torch.zeros((1, 4), dtype=torch.bool) for _ in range(2)]
    trainer = _iterator_trainer(masks, gradient_accumulation_steps=2)

    with pytest.raises(RuntimeError, match="empty accumulation window"):
        list(trainer._iter_global_masked_mean_batches([_batch(), _batch()]))
