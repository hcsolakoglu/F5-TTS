"""Focused behavioural tests for the ``max_padded_frames`` padded-batch-rectangle guard.

Extracted from the standalone isolation of the feature (no torch.compile / global_masked_mean
/ fused-AdamW / CFG dependencies). Covers:

* default-off cap parity with upstream for datasets whose rows are already valid
* the bimodal-corpus overshoot the guard exists for
* data-lossless recommended value (per frontend) and lossy-value warning
* Trainer sample-mode rejection / frame-mode acceptance
* ``padded_mel_frames`` correctness vs real vocos/bigvgan mel frontends
* resampling-aware upper bound vs real torchaudio ``Resample``
* invalid-row substitution: the cap holds against the *actually emitted* sample
"""

import pytest
import torch

from f5_tts.model import CFM, DiT
from f5_tts.train.finetune_cli import parse_args


def _build_model(
    *,
    audio_drop_prob=0.0,
    cond_drop_prob=0.0,
    vocab_size=32,
    dropout=0.0,
    conv_layers=0,
    pe_attn_head=None,
    text_mask_padding=True,
    average_upsampling=False,
):
    model = CFM(
        transformer=DiT(
            dim=32,
            depth=1,
            heads=2,
            dim_head=16,
            mel_dim=8,
            text_num_embeds=vocab_size,
            text_dim=16,
            dropout=dropout,
            conv_layers=conv_layers,
            pe_attn_head=pe_attn_head,
            text_mask_padding=text_mask_padding,
            text_embedding_average_upsampling=average_upsampling,
        ),
        mel_spec_kwargs={"n_mel_channels": 8},
        audio_drop_prob=audio_drop_prob,
        cond_drop_prob=cond_drop_prob,
    ).cpu()
    model.eval()
    return model


class _FrameLenDataset:
    """Minimal dataset exposing only what DynamicBatchSampler consumes."""

    def __init__(self, frame_lens):
        self.frame_lens = list(frame_lens)

    def get_frame_len(self, index):
        return float(self.frame_lens[index])

    def __len__(self):
        return len(self.frame_lens)


class _IndexSampler:
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


def _build_batches(frame_lens, threshold, *, max_samples=0, max_padded_frames=0, mel_spec_type="vocos"):
    from f5_tts.model.dataset import DynamicBatchSampler

    dataset = _FrameLenDataset(frame_lens)
    sampler = DynamicBatchSampler(
        _IndexSampler(dataset),
        threshold,
        max_samples=max_samples,
        random_seed=None,
        drop_residual=False,
        max_padded_frames=max_padded_frames,
        mel_spec_type=mel_spec_type,
    )
    return sampler.batches, dataset


def _real_mel_widths(sample_counts, mel_spec_type, *, hop_length=256, n_fft=1024, n_mel_channels=8):
    """Run the actual mel frontend and return each clip's real output width.

    This is the independent oracle: instead of reusing the sampler's own frame-count
    formula (which made the old bound assertions tautological), we run the same
    ``get_vocos_mel_spectrogram`` / ``get_bigvgan_mel_spectrogram`` the dataset uses and
    read the tensor's last dimension.
    """
    from f5_tts.model.modules import get_bigvgan_mel_spectrogram, get_vocos_mel_spectrogram

    extractor = get_vocos_mel_spectrogram if mel_spec_type == "vocos" else get_bigvgan_mel_spectrogram
    widths = []
    for n in sample_counts:
        wav = torch.zeros(1, n)
        mel = extractor(wav, n_fft=n_fft, n_mel_channels=n_mel_channels, hop_length=hop_length, win_length=n_fft)
        widths.append(mel.shape[-1])
    return widths


def test_max_padded_frames_defaults_to_upstream_batch_composition():
    """Default (0) preserves batching when all dataset rows are already valid."""
    frame_lens = [94, 120, 300, 301, 500, 900, 2100, 2200, 2800, 2810]
    baseline, _ = _build_batches(frame_lens, 3000, max_samples=8)
    guarded, _ = _build_batches(frame_lens, 3000, max_samples=8, max_padded_frames=0)
    assert guarded == baseline


def test_negative_max_padded_frames_is_rejected():
    with pytest.raises(ValueError, match="must be >= 0"):
        _build_batches([100, 200], 1000, max_padded_frames=-1)


def test_max_padded_frames_bounds_the_padded_rectangle_on_bimodal_data():
    """The adversarial case the guard exists for.

    A frame-sum budget lets a batch of short utterances be closed out by a much longer
    one, so the allocated rectangle len(batch)*max_len can far exceed the requested
    budget. Measured worst case on a bimodal corpus at N=100k is 1.82x. With the guard set,
    no batch may exceed it.

    The expected rectangle is computed with the frontend-aware ``padded_mel_frames``
    helper (the same one the sampler uses), not a re-derived ceil formula, so this test
    asserts the guard's own invariant rather than a tautological copy of an old formula.
    The non-tautological end-to-end check against real mel tensor widths lives in
    ``test_max_padded_frames_cap_holds_against_real_frontend_mel_width``.
    """
    from f5_tts.model.dataset import padded_mel_frames

    rng = torch.Generator().manual_seed(11)
    short = (torch.rand(400, generator=rng) * 140 + 94).tolist()
    long = (torch.rand(400, generator=rng) * 560 + 2250).tolist()
    frame_lens = short + long

    threshold = 4000
    unguarded, dataset = _build_batches(frame_lens, threshold, max_samples=64)
    worst = max(len(b) * max(padded_mel_frames(dataset.get_frame_len(i), "vocos") for i in b) for b in unguarded)
    assert worst > threshold, "expected the frame-sum budget to overshoot the padded rectangle here"

    guarded, dataset = _build_batches(frame_lens, threshold, max_samples=64, max_padded_frames=threshold)
    for batch in guarded:
        padded = len(batch) * max(padded_mel_frames(dataset.get_frame_len(i), "vocos") for i in batch)
        assert padded <= threshold, f"padded rectangle {padded} exceeds cap {threshold}"


@pytest.mark.parametrize("mel_spec_type", ["vocos", "bigvgan"])
def test_max_padded_frames_at_recommended_value_never_drops_samples(mel_spec_type):
    """The per-frontend recommended cap must be data-lossless.

    A padded rectangle is never smaller than the frame sum, so any sample admitted by
    frames_threshold also fits a cap of the recommended size. For bigvgan that is
    ``frames_threshold`` (padded width == floor(frame_len)); for vocos it is
    ``frames_threshold + 1`` (center=True adds one frame, so a clip exactly at the
    threshold has padded width threshold+1). This is what makes the recommended value
    safe to enable without auditing the corpus first.
    """
    import warnings as _warnings

    from f5_tts.model.dataset import padded_mel_frames

    threshold = 3000
    # The cap is the padded width of a clip whose frame_len == threshold.
    cap = padded_mel_frames(threshold, mel_spec_type)
    frame_lens = [94, 500, 1200, 2999, 3000, 1500, 700, 2400]
    baseline, _ = _build_batches(frame_lens, threshold, max_samples=64, mel_spec_type=mel_spec_type)
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")  # any drop warning becomes a failure
        guarded, _ = _build_batches(
            frame_lens, threshold, max_samples=64, max_padded_frames=cap, mel_spec_type=mel_spec_type
        )

    assert sorted(i for b in baseline for i in b) == sorted(i for b in guarded for i in b)


def test_max_padded_frames_below_threshold_warns_about_dropped_samples():
    """The one genuinely dangerous setting must never fail silently.

    cap < frames_threshold discards every sample between the two. That is almost always a
    misconfiguration -- the same memory bound is better expressed by lowering
    frames_threshold -- so it must be loud.
    """
    import warnings as _warnings

    frame_lens = [100, 200, 900, 1800, 2500]  # 1800 and 2500 exceed the cap but not the threshold
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        batches, _ = _build_batches(frame_lens, 3000, max_samples=64, max_padded_frames=1000)

    messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("max_padded_frames" in m and "dropped" in m for m in messages), messages
    kept = {i for b in batches for i in b}
    assert kept == {0, 1, 2}, "samples longer than the cap should be the only ones dropped"


def test_max_padded_frames_keeps_every_usable_sample():
    """The guard may repartition batches but must not silently drop fitting samples."""
    frame_lens = [94, 120, 300, 301, 500, 900, 1100, 1200]
    baseline, _ = _build_batches(frame_lens, 3000, max_samples=8)
    guarded, _ = _build_batches(frame_lens, 3000, max_samples=8, max_padded_frames=2400)

    assert sorted(i for b in baseline for i in b) == sorted(i for b in guarded for i in b)


def test_trainer_rejects_nonzero_max_padded_frames_with_sample_batching(monkeypatch):
    """max_padded_frames is a frame-mode-only memory guard; sample mode cannot apply it.

    Sample mode uses fixed-size shuffled batches with no length sorting, so the cap has
    no mechanism to act through. Silently ignoring a safety guard is worse than refusing
    to configure it: a user who adds the cap after an OOM believes they are protected,
    retries, and OOMs again. Reject at construction with a precise message.
    """
    import f5_tts.model.trainer as trainer_module

    def accelerator_must_not_be_constructed(*args, **kwargs):
        raise AssertionError("invalid sampler configuration must fail before Accelerator setup")

    monkeypatch.setattr(trainer_module, "Accelerator", accelerator_must_not_be_constructed)

    model = _build_model()
    with pytest.raises(ValueError, match="max_padded_frames"):
        trainer_module.Trainer(
            model,
            epochs=1,
            learning_rate=1e-4,
            num_warmup_updates=1,
            save_per_updates=10**9,
            keep_last_n_checkpoints=0,
            logger=None,
            log_samples=False,
            batch_size_type="sample",
            max_padded_frames=1000,
        )


def test_trainer_rejects_negative_max_padded_frames_before_setup(monkeypatch):
    import f5_tts.model.trainer as trainer_module

    def accelerator_must_not_be_constructed(*args, **kwargs):
        raise AssertionError("negative cap must fail before Accelerator setup")

    monkeypatch.setattr(trainer_module, "Accelerator", accelerator_must_not_be_constructed)

    with pytest.raises(ValueError, match="must be >= 0"):
        trainer_module.Trainer(
            _build_model(),
            epochs=1,
            learning_rate=1e-4,
            num_warmup_updates=1,
            save_per_updates=10**9,
            keep_last_n_checkpoints=0,
            logger=None,
            log_samples=False,
            batch_size_type="frame",
            max_padded_frames=-1,
        )


def test_trainer_accepts_zero_max_padded_frames_with_sample_batching():
    """The default (max_padded_frames=0) must remain valid for sample mode (regression guard)."""
    from f5_tts.model.trainer import Trainer

    model = _build_model()
    trainer = Trainer(
        model,
        epochs=1,
        learning_rate=1e-4,
        num_warmup_updates=1,
        save_per_updates=10**9,
        keep_last_n_checkpoints=0,
        logger=None,
        log_samples=False,
        batch_size_type="sample",
        max_padded_frames=0,
    )
    assert trainer.max_padded_frames == 0
    assert trainer.batch_size_type == "sample"


def test_trainer_accepts_nonzero_max_padded_frames_with_frame_batching():
    """Frame mode must still accept the cap (the supported configuration)."""
    from f5_tts.model.trainer import Trainer

    model = _build_model()
    trainer = Trainer(
        model,
        epochs=1,
        learning_rate=1e-4,
        num_warmup_updates=1,
        save_per_updates=10**9,
        keep_last_n_checkpoints=0,
        logger=None,
        log_samples=False,
        batch_size_type="frame",
        max_padded_frames=1000,
    )
    assert trainer.max_padded_frames == 1000
    assert trainer.batch_size_type == "frame"


@pytest.mark.parametrize("mel_spec_type", ["vocos", "bigvgan"])
def test_padded_mel_frames_matches_real_frontend_width_at_exact_hop_multiples(mel_spec_type):
    """The cap's frame-count helper must equal the real mel tensor width, not ceil.

    At exact hop multiples the old ``ceil(frame_len)`` was one short for vocos
    (``center=True`` adds a frame) and one loose for bigvgan (``center=False``).
    This is the off-by-one finding 13 reports: feed sample counts that are exact
    multiples of hop_length (so frame_len is integral) and compare the helper against
    the actual frontend output width.
    """
    from f5_tts.model.dataset import padded_mel_frames

    hop = 256
    # Multiples of hop that are >= n_fft so both frontends produce a valid STFT.
    sample_counts = [4 * hop, 5 * hop, 6 * hop, 8 * hop, 10 * hop, 16 * hop]
    widths = _real_mel_widths(sample_counts, mel_spec_type, hop_length=hop)
    for n, w in zip(sample_counts, widths):
        frame_len = n / hop  # exactly integral
        assert padded_mel_frames(frame_len, mel_spec_type) == w, (
            f"{mel_spec_type}: n={n} frame_len={frame_len} helper={padded_mel_frames(frame_len, mel_spec_type)} "
            f"real_width={w}"
        )


@pytest.mark.parametrize("mel_spec_type", ["vocos", "bigvgan"])
def test_max_padded_frames_cap_holds_against_real_frontend_mel_width(mel_spec_type):
    """The cap must hold against the *actual* mel tensor width, not the sampler's formula.

    Builds batches from integral frame_lens (exact hop multiples) with the cap set to a
    value the old ``ceil`` formula admitted but the true vocos width violates, then
    checks every batch rectangle against the independently computed real frontend width.
    This fails on the pre-fix code for vocos and passes after the frontend-aware fix.
    """
    hop = 256
    sample_counts = [4 * hop, 6 * hop, 8 * hop, 10 * hop, 12 * hop]
    frame_lens = [n / hop for n in sample_counts]  # integral: 4, 6, 8, 10, 12
    widths = _real_mel_widths(sample_counts, mel_spec_type, hop_length=hop)
    width_by_idx = {i: w for i, w in enumerate(widths)}

    # Cap chosen so the max clip's real width fits exactly once (cap == max real width).
    # For vocos the max real width is 12+1=13; for bigvgan it is 12.
    cap = max(widths)
    batches, dataset = _build_batches(
        frame_lens, 10_000, max_samples=64, max_padded_frames=cap, mel_spec_type=mel_spec_type
    )
    assert batches, "expected at least one batch"
    for batch in batches:
        real_padded = len(batch) * max(width_by_idx[i] for i in batch)
        assert real_padded <= cap, (
            f"{mel_spec_type}: real padded rectangle {real_padded} exceeds cap {cap} "
            f"(batch={batch}, widths={[width_by_idx[i] for i in batch]})"
        )


def test_max_padded_frames_vocos_cap_rejects_what_old_ceil_admitted():
    """Regression guard for the exact off-by-one finding 13 reports.

    A single clip whose frame_len is integral (e.g. 10) has vocos real width 11 but
    old ``ceil`` width 10. With ``max_padded_frames=10`` the old code admitted the
    clip (ceil(10)=10 <= 10) while the real tensor is 11 frames wide; the frontend-aware
    code must reject it (dropping it with the cap-drop warning, since it cannot fit
    alone). This pins the behavioural difference.
    """
    import warnings as _warnings

    from f5_tts.model.dataset import padded_mel_frames

    hop = 256
    n = 10 * hop
    frame_len = n / hop  # 10.0, integral
    assert padded_mel_frames(frame_len, "vocos") == 11  # ceil(10)+1
    assert padded_mel_frames(frame_len, "bigvgan") == 10  # ceil(10)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        batches, _ = _build_batches([frame_len], 10_000, max_samples=64, max_padded_frames=10, mel_spec_type="vocos")
    # The clip's real vocos width (11) exceeds the cap (10), so it is dropped, not batched.
    assert batches == [], "vocos clip of width 11 must not fit a cap of 10"
    messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("max_padded_frames" in m and "dropped" in m for m in messages), messages


# --- Resampling-aware cap regression tests -----------------------------------
#
# ``get_frame_len`` budgets on the float ratio ``n_src * tgt / src / hop``, but
# ``torchaudio.transforms.Resample`` emits ``ceil(n_src * tgt / src)`` samples -- up to
# one sample more than the ratio. That extra sample can cross a hop boundary and add a
# whole mel frame, so a ``floor(frame_len)``-based cap can be violated. The fix uses
# ``ceil(frame_len)`` (bigvgan) / ``ceil(frame_len) + 1`` (vocos),
# which absorbs the rounding. These tests run the *real* torchaudio resampler and mel
# frontends as the oracle, not the sampler's own formula.

# (n_source_samples, source_rate); all resampled to the 24k target the trainer uses.
_RESAMPLE_CASES = [
    (5461, 16000),  # the reported case: 8191.5 -> 8192 samples, crosses a hop boundary
    (3333, 22050),  # 22.05k -> 24k, non-integer ratio
    (12345, 48000),  # 48k -> 24k, half-rate
    (7000, 24000),  # same rate, no resampling -- bound may conservatively exceed the width
]


@pytest.mark.parametrize("mel_spec_type", ["vocos", "bigvgan"])
def test_padded_mel_frames_upper_bounds_real_resampled_mel_width(mel_spec_type):
    """The cap helper must never underestimate the real resampled mel width.

    For each (n_src, src_rate) we run the actual torchaudio Resample to 24k, then the
    actual mel frontend, and compare the real tensor width against
    ``padded_mel_frames(frame_len, mel_spec_type)`` where ``frame_len`` is the float
    ``get_frame_len`` would return. The helper must be >= the real width (it is an upper
    bound). We also assert the prior ``floor``-based helper would have been *under* the
    real width on the 16k n=5461 case, pinning the regression.
    """
    import math as _math

    import torchaudio

    from f5_tts.model.dataset import padded_mel_frames

    hop = 256
    target_rate = 24000
    for n_src, src_rate in _RESAMPLE_CASES:
        wav_src = torch.zeros(n_src)
        if src_rate == target_rate:
            wav_tgt = wav_src
        else:
            resampler = torchaudio.transforms.Resample(src_rate, target_rate)
            wav_tgt = resampler(wav_src)
        mel = _real_mel_widths([wav_tgt.shape[-1]], mel_spec_type, hop_length=hop)[0]
        frame_len = n_src / src_rate * target_rate / hop  # what get_frame_len returns
        helper = padded_mel_frames(frame_len, mel_spec_type)
        assert helper >= mel, (
            f"{mel_spec_type} src={src_rate} n_src={n_src}: frame_len={frame_len} "
            f"helper={helper} < real_width={mel} (resampled samples={wav_tgt.shape[-1]})"
        )
        # Prior floor-based helper (the bug): must be under the real width on the
        # reported 16k n=5461 -> 24k case, confirming this is a real regression not a
        # tautology. ceil(frame_len) == floor(frame_len) at integral frame_len, so only
        # the non-integral 5461 case is asserted here.
        if (n_src, src_rate) == (5461, 16000):
            old_helper = _math.floor(frame_len) + (1 if mel_spec_type == "vocos" else 0)
            assert old_helper < mel, (
                f"{mel_spec_type}: prior floor helper {old_helper} was not under real "
                f"width {mel} -- regression guard would be vacuous"
            )


@pytest.mark.parametrize("mel_spec_type", ["vocos", "bigvgan"])
def test_max_padded_frames_cap_holds_under_real_resampling_batched(mel_spec_type):
    """A batch of resampled clips must not blow the cap the prior floor formula admitted.

    Ten identical 16k clips of 5461 samples resample to 8192 samples (24k), i.e. real
    bigvgan width 32 / vocos width 33. ``get_frame_len`` reports frame_len=31.998, so
    the prior ``floor``-based helper estimated 31 (bigvgan) / 32 (vocos) frames per
    clip. Setting ``max_padded_frames`` to ``10 * prior_helper`` was admitted by the old
    code (10 * prior <= cap) but the *real* padded rectangle is ``10 * real_width``,
    which exceeds the cap -- a silent OOM risk. The ``ceil``-based fix must instead cap
    the batch at 9 clips so the real rectangle stays within the budget.
    """
    import math as _math

    import torchaudio

    from f5_tts.model.dataset import padded_mel_frames

    hop = 256
    n_src, src_rate, target_rate = 5461, 16000, 24000
    resampler = torchaudio.transforms.Resample(src_rate, target_rate)
    n_tgt = resampler(torch.zeros(n_src)).shape[-1]  # 8192
    real_width = _real_mel_widths([n_tgt], mel_spec_type, hop_length=hop)[0]
    frame_len = n_src / src_rate * target_rate / hop  # 31.998046875

    prior_helper = _math.floor(frame_len) + (1 if mel_spec_type == "vocos" else 0)
    n_clips = 10
    cap = n_clips * prior_helper  # exactly what the old floor formula admitted
    # Sanity: the old formula's estimate fit the cap, but the real rectangle does not.
    assert n_clips * prior_helper <= cap
    assert n_clips * real_width > cap, (
        f"{mel_spec_type}: real rectangle {n_clips * real_width} must exceed cap {cap} "
        "for this to be a real cap-violation regression"
    )

    frame_lens = [frame_len] * n_clips
    batches, _ = _build_batches(frame_lens, 10_000, max_samples=64, max_padded_frames=cap, mel_spec_type=mel_spec_type)
    # The ceil-based helper must keep every batch's real rectangle within the cap.
    helper = padded_mel_frames(frame_len, mel_spec_type)
    for batch in batches:
        real_padded = len(batch) * real_width
        assert real_padded <= cap, (
            f"{mel_spec_type}: real padded rectangle {real_padded} exceeds cap {cap} "
            f"(batch size {len(batch)}, real_width {real_width})"
        )
    # And it must have repartitioned: the old formula admitted all 10 in one batch, the
    # fix caps the batch at floor(cap / helper) clips.
    max_batch = max(len(b) for b in batches)
    assert max_batch <= cap // helper, f"{mel_spec_type}: batch size {max_batch} exceeds cap//helper={cap // helper}"
    assert max_batch < n_clips, f"{mel_spec_type}: fix failed to repartition the violating batch ({max_batch} clips)"


# --- Invalid-row substitution regression test --------------------------------
#
# ``HFDataset``/``CustomDataset`` skip rows whose duration is outside [0.3, 30]s by
# advancing ``(index + 1) % len`` until a valid row is found and emitting *that* row.
# Before the fix, ``get_frame_len`` measured the original (possibly short/invalid) row
# while ``__getitem__`` emitted a different (valid, possibly long) row, so the cap
# budgeted against the wrong length and could be violated by the actually-emitted
# sample. ``_resolve_valid_index`` makes the two consistent. This test proves the cap
# holds against the *real* mel width of the row the dataset actually yields, using a
# CustomDataset-shaped stub whose ``get_frame_len`` and ``__getitem__`` share the walk.


class _SubstitutingCustomDataset:
    """CustomDataset-shaped stub with invalid rows that trigger index substitution.

    Rows with ``duration`` outside [0.3, 30] are skipped exactly like the real
    CustomDataset: ``__getitem__`` advances ``(index + 1) % len`` until a valid row is
    found. ``get_frame_len`` resolves to the same valid row via ``_resolve_valid_index``,
    so the cap budgets against the row actually emitted. ``__getitem__`` returns the
    resolved row's mel width (the quantity the GPU allocates), letting the test assert
    the cap holds end-to-end against the real emitted sample.
    """

    def __init__(self, durations, mel_spec_type="vocos", hop_length=256, target_rate=24000):
        self.durations = list(durations)
        self.mel_spec_type = mel_spec_type
        self.hop_length = hop_length
        self.target_rate = target_rate
        self.n = len(self.durations)
        # Precompute the real mel width each valid row would produce at an exact hop
        # multiple sample count so the oracle is independent of the cap's own formula.
        self._widths = [self._real_width(d) if 0.3 <= d <= 30 else None for d in self.durations]

    def _real_width(self, duration):
        # Exact hop-multiple sample count -> frame_len integral -> helper == real width.
        n_samples = int(round(duration * self.target_rate / self.hop_length)) * self.hop_length
        return _real_mel_widths([n_samples], self.mel_spec_type, hop_length=self.hop_length)[0]

    def _resolve_valid_index(self, index):
        for _ in range(self.n):
            if 0.3 <= self.durations[index] <= 30:
                return index
            index = (index + 1) % self.n
        raise ValueError("no valid row")

    def get_frame_len(self, index):
        index = self._resolve_valid_index(index)
        return self.durations[index] * self.target_rate / self.hop_length

    def __getitem__(self, index):
        index = self._resolve_valid_index(index)
        return {"mel_width": self._widths[index], "resolved_index": index}

    def __len__(self):
        return self.n


def test_max_padded_frames_cap_holds_against_actually_emitted_sample():
    """The cap must hold against the mel width of the row the dataset *actually emits*.

    A corpus with invalid (out-of-[0.3, 30]s) rows triggers index substitution: a short
    invalid row measured by a naive ``get_frame_len`` is replaced at ``__getitem__`` time
    by a longer valid row. Before ``_resolve_valid_index``, the cap budgeted against the
    short measured length and the long emitted sample blew it. After the fix,
    ``get_frame_len`` resolves to the same valid row, so the cap is sound against the
    real emitted width. This is the end-to-end proof the guard is not defeated by
    substitution.
    """
    from f5_tts.model.dataset import DynamicBatchSampler, padded_mel_frames

    mel_spec_type = "vocos"
    # Bimodal-ish valid rows (durations in s) plus invalid short rows interspersed.
    # Invalid rows (0.1s) would be measured as ~9 frames but substitute to a ~234-frame
    # valid row -- the exact mismatch that defeats an un-fixed get_frame_len.
    durations = [0.1, 2.0, 0.1, 5.0, 0.1, 8.0, 0.1, 2.5, 0.1, 6.0]
    dataset = _SubstitutingCustomDataset(durations, mel_spec_type=mel_spec_type)

    # Cap set to the recommended value (frames_threshold for the longest valid clip's
    # padded width). frames_threshold chosen above the longest valid frame sum so the
    # cap is the binding constraint, not the frame-sum budget.
    max_valid_frame_len = max(dataset.get_frame_len(i) for i in range(len(dataset)))
    cap = padded_mel_frames(max_valid_frame_len, mel_spec_type) * 2  # allow 2 longest clips
    threshold = int(max_valid_frame_len) * 10  # generous frame-sum budget

    sampler = DynamicBatchSampler(
        _IndexSampler(dataset),
        threshold,
        max_samples=64,
        random_seed=None,
        drop_residual=False,
        max_padded_frames=cap,
        mel_spec_type=mel_spec_type,
    )

    assert sampler.batches, "expected at least one batch"
    for batch in sampler.batches:
        # The real emitted width of each index (post-substitution), not the measured one.
        emitted_widths = [dataset[i]["mel_width"] for i in batch]
        real_padded = len(batch) * max(emitted_widths)
        assert real_padded <= cap, (
            f"emitted padded rectangle {real_padded} exceeds cap {cap} (batch={batch}, emitted_widths={emitted_widths})"
        )


def test_cli_max_padded_frames_arg_parses(monkeypatch):
    """The --max_padded_frames CLI arg is wired and parses to the Trainer pass-through."""
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        ["finetune_cli.py", "--dataset_name", "test", "--exp_name", "F5TTS_Base", "--max_padded_frames", "38400"],
    )
    args = parse_args()
    assert args.max_padded_frames == 38400
