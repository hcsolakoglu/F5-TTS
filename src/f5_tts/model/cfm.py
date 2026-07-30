"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)


TRAINING_COMPILE_TARGETS = ("cfm_loss_core", "dit_blocks")


def _is_cuda_oom(exc: BaseException) -> bool:
    """Return True only for a genuine CUDA allocator out-of-memory error.

    Both checks require the message to mention CUDA. ``torch.cuda.OutOfMemoryError``
    is an alias of the generic ``torch.OutOfMemoryError`` in current PyTorch, so the
    type alone also matches host/CPU allocator failures. And a bare ``"out of memory"``
    substring match would misclassify compiler-side failures (for example a Triton
    autotune worker reporting *host* OOM) as a GPU capacity problem, hard-failing a run
    that should have fallen back to eager.
    """
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return "cuda" in str(exc).lower()
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        return "out of memory" in message and "cuda" in message
    return False


def _note_oom_compile_context(exc: BaseException) -> None:
    """Attach a compile-context note to a CUDA OOM without changing its type/message.

    ``add_note`` (Python 3.11+) appends to ``__notes__`` and is not part of
    ``str(exc)``, so ``except`` type checks and message-based batch-size reducers
    keep matching the original exception. Idempotent: never duplicates the note.
    """
    note = "torch.compile: GPU OOM; not falling back to eager (reduce batch size / sequence length)."
    add_note = getattr(exc, "add_note", None)
    if add_note is None:
        return
    existing = getattr(exc, "__notes__", None) or ()
    if note in existing:
        return
    add_note(note)


def _compile_failure_types() -> tuple[type[BaseException], ...]:
    """Exception types that mean 'the compiler failed', not 'the model is wrong'.

    Only these justify an eager retry. A compiler failure is raised while Dynamo is
    tracing or while the backend is lowering, i.e. *before* the graph executes, so a
    retry cannot double-apply side effects. Ordinary model errors (shape mismatch, bad
    vocabulary index, dtype error, assertion) must propagate unchanged: swallowing them
    would hide a real bug behind a misleading "compile fallback" message and then raise
    a second, more confusing error from the eager retry.
    """
    types: list[type[BaseException]] = []
    try:
        from torch._dynamo import exc as dynamo_exc

        for name in ("BackendCompilerFailed", "Unsupported", "InternalTorchDynamoError", "TorchRuntimeError"):
            candidate = getattr(dynamo_exc, name, None)
            if isinstance(candidate, type) and issubclass(candidate, BaseException):
                types.append(candidate)
    except Exception:  # pragma: no cover - torch without _dynamo
        pass
    try:
        from torch._inductor import exc as inductor_exc

        for name in ("InductorError", "CppCompileError", "CUDACompileError"):
            candidate = getattr(inductor_exc, name, None)
            if isinstance(candidate, type) and issubclass(candidate, BaseException):
                types.append(candidate)
    except Exception:  # pragma: no cover - torch without _inductor
        pass
    return tuple(types)


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

        # torch.compile state for optional training targets. object.__setattr__ bypasses
        # nn.Module.__setattr__ so these are never registered as parameters/buffers;
        # state_dict is unchanged and EMA deepcopy (which runs before any compile call)
        # never sees a compiled callable.
        object.__setattr__(self, "_compiled_loss_core", None)
        object.__setattr__(self, "_compile_target", None)
        object.__setattr__(self, "_compile_runtime_fallback", True)
        object.__setattr__(self, "_compile_fallback_active", False)
        object.__setattr__(self, "_compile_error", None)

    # torch.compile returns Python callables that close over this exact CFM instance.
    # deepcopy treats functions as atomic, so copying a compiled CFM without stripping
    # this state would leave the copy executing the source model's parameters. The same
    # wrapper is not pickleable as part of a whole-model torch.save. Copies and loaded
    # models therefore intentionally resume eager, matching the regional DiT policy.
    _CFM_COMPILE_ONLY_ATTRS = (
        "_compiled_loss_core",
        "_compile_target",
        "_compile_runtime_fallback",
        "_compile_fallback_active",
        "_compile_error",
    )

    def __getstate__(self):
        # Match nn.Module.__getstate__ on newer PyTorch releases without relying on
        # that method: torch 2.0 lacks it, while Python 3.10's object also provides no
        # parent implementation. Explicitly stripping PyTorch's compiled call hook
        # preserves the behavior of torch 2.1+.
        state = self.__dict__.copy()
        state.pop("_compiled_call_impl", None)
        for attr in self._CFM_COMPILE_ONLY_ATTRS:
            state.pop(attr, None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        object.__setattr__(self, "_compiled_loss_core", None)
        object.__setattr__(self, "_compile_target", None)
        object.__setattr__(self, "_compile_runtime_fallback", True)
        object.__setattr__(self, "_compile_fallback_active", False)
        object.__setattr__(self, "_compile_error", None)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def training_compile_state(self):
        enabled = self._training_compile_enabled()
        return {
            "enabled": enabled,
            "target": self._compile_target if enabled else None,
            "fallback_active": self._compile_fallback_active,
            "error": self._compile_error,
        }

    def compile_training_core(self, *, target: str = "cfm_loss_core", runtime_fallback: bool = True, **compile_kwargs):
        """Compile an explicit training target for an optional speedup.

        ``target=cfm_loss_core`` preserves the original behavior: only the deterministic
        core (xt interpolation, transformer forward, masked MSE) is compiled; stochastic
        preparation (mel extraction, span masking, noise/time sampling, CFG dropout) stays
        eager in ``forward`` so RNG state is unaffected.

        ``target=dit_blocks`` compiles each repeated DiT block independently and leaves
        embeddings, rotary setup, final projection, and loss reduction eager. This regional
        target is useful for variable-length training where the larger CFM loss-core graph
        has excessive cold compile/recompile overhead.

        ``runtime_fallback`` (default True) lets a *forward* compile failure permanently
        switch this module to eager. Set it False under DDP so a failure raises on every
        rank instead of desynchronising the gradient all-reduce. Note: this fallback
        covers forward failures only; a compile failure during backward is not caught and
        will raise. To surface such errors, set ``runtime_fallback=False`` (Trainer:
        ``compile_fallback_to_eager=False``) so a compile failure raises instead of
        silently switching to eager; compilation itself stays active.
        """
        if target not in TRAINING_COMPILE_TARGETS:
            valid = ", ".join(TRAINING_COMPILE_TARGETS)
            raise ValueError(f"Unknown torch.compile training target {target!r}; expected one of: {valid}")

        self.clear_training_compile()

        if target == "cfm_loss_core":
            self._check_compile_compatible_transformer(target=target)
            compiled = torch.compile(self._forward_loss_core_components, **compile_kwargs)
            object.__setattr__(self, "_compiled_loss_core", compiled)
        else:
            compile_fn = getattr(self.transformer, "compile_training_target", None)
            if compile_fn is None:
                raise ValueError(
                    "torch.compile target='dit_blocks' requires a DiT transformer exposing "
                    "compile_training_target(); use target='cfm_loss_core' for other backbones."
                )
            compiled = compile_fn(target, **compile_kwargs)

        object.__setattr__(self, "_compile_target", target)
        object.__setattr__(self, "_compile_runtime_fallback", bool(runtime_fallback))
        object.__setattr__(self, "_compile_fallback_active", False)
        object.__setattr__(self, "_compile_error", None)
        return compiled

    def clear_training_compile(self):
        """Remove any compiled training target, reverting to eager execution."""
        clear_transformer = getattr(self.transformer, "clear_training_compile", None)
        if clear_transformer is not None:
            clear_transformer()
        object.__setattr__(self, "_compiled_loss_core", None)
        object.__setattr__(self, "_compile_target", None)
        object.__setattr__(self, "_compile_runtime_fallback", True)
        object.__setattr__(self, "_compile_fallback_active", False)
        object.__setattr__(self, "_compile_error", None)

    def _transformer_training_compile_state(self):
        state = getattr(self.transformer, "training_compile_state", None)
        if state is None:
            return {"enabled": False}
        return state

    def _training_compile_enabled(self):
        return self._compiled_loss_core is not None or bool(
            self._transformer_training_compile_state().get("enabled", False)
        )

    def _check_compile_compatible_transformer(self, *, target: str):
        """Reject transformer paths known to graph-break inside compiled training.

        DiT's ``text_embedding_average_upsampling=True`` calls
        ``average_upsample_text_by_mask`` from the compiled loss core. That path
        contains a per-sample Python loop plus ``Tensor.item()`` calls, which graph-break
        under torch.compile and hard-fail with ``fullgraph=True``. Failing at setup gives
        Trainer a clean eager fallback path instead of a late Dynamo error.
        """
        if target != "cfm_loss_core":
            return

        text_embed = getattr(self.transformer, "text_embed", None)
        if text_embed is not None and getattr(text_embed, "average_upsampling", False):
            raise ValueError(
                "torch.compile training is incompatible with "
                "DiT text_embedding_average_upsampling=True: the per-sample "
                "average_upsample_text_by_mask loop uses Tensor.item() and breaks "
                "the compiled graph. Disable compile (fallback_to_eager=True) or set "
                "text_embedding_average_upsampling=False to enable compiled training."
            )

    def _run_loss_core_components(self, *args):
        """Run the deterministic loss core via the compiled callable, with eager fallback.

        No RNG save/restore: the stochastic inputs (x0, time, rand_span_mask, drop_*) are
        already materialised in ``args`` before this call, so an eager retry reuses them
        exactly. The only RNG inside the compiled region is nn.Dropout in the transformer;
        torch.get_rng_state does not round-trip the Philox offset used by compiled code, so
        restoring it would give a false guarantee. A fresh dropout draw on fallback is valid.

        Only *compiler* failures trigger the eager retry (see ``_compile_failure_types``).
        Ordinary model errors propagate unchanged so a genuine bug is not disguised as a
        compile problem and then re-raised from a second, partially-executed forward.
        """
        compiled = self.__dict__.get("_compiled_loss_core")
        compiled_active = self._training_compile_enabled()
        if not compiled_active:
            return self._forward_loss_core_components(*args)
        try:
            if compiled is not None:
                return compiled(*args)
            # Regional DiT targets execute through the eager loss-core path; the DiT
            # blocks invoked from the transformer are the compiled callables.
            return self._forward_loss_core_components(*args)
        except Exception as exc:
            # A compiled CUDA OOM is a capacity failure, not a compiler failure.
            # Retrying eagerly usually repeats the same allocation pressure and can hide
            # the real problem behind a fallback state, so never fall back on a real GPU
            # OOM. Re-raise the ORIGINAL exception (bare ``raise``) so its concrete type
            # (``torch.cuda.OutOfMemoryError``) and the standard ``CUDA out of memory``
            # message survive: ``except torch.cuda.OutOfMemoryError`` handlers and
            # message-based batch-size reducers depend on them. A note adds compile
            # context without altering type or message matching.
            if _is_cuda_oom(exc):
                _note_oom_compile_context(exc)
                raise
            if not self._compile_runtime_fallback:
                raise
            if not isinstance(exc, _compile_failure_types()):
                raise
            # Permanently disable compile for the rest of this run.
            self.clear_training_compile()
            object.__setattr__(self, "_compile_fallback_active", True)
            object.__setattr__(self, "_compile_error", repr(exc))
            return self._forward_loss_core_components(*args)

    def _run_loss_core(self, *args):
        """Run the loss core and preserve the public training return shape.

        With compile disabled this dispatches to the bit-exact upstream reduction so the
        default training path is byte-identical to F5-TTS without this feature. The
        compile-friendly reduction is algebraically identical but reassociates the sum,
        which changes the result by ~1 ULP -- acceptable for an opt-in speedup, not
        acceptable for users who never asked for it.

        After a *runtime compile fallback* (``_compile_fallback_active``), the module is
        back to eager, but we deliberately keep the fp32 masked-multiply reduction
        (``_forward_loss_core_components``) instead of the upstream-exact path. The
        upstream path computes ``F.mse_loss(pred, flow)`` in the input dtype, so an fp16
        AMP run that fell back mid-training would switch from the compiled fp32
        accumulation to fp16 squared error and could turn a finite loss into ``inf``/
        ``nan``. The fp32 components path is numerically safe and matches the reduction
        the run was using while compiled. The never-compiled default (``_compile_fallback_
        active`` stays False) still takes the byte-exact upstream path.
        """
        if not self._training_compile_enabled():
            if self._compile_fallback_active:
                loss, _, _, cond, pred = self._forward_loss_core_components(*args)
                return loss, cond, pred
            return self._forward_loss_core_upstream_exact(*args)
        loss, _, _, cond, pred = self._run_loss_core_components(*args)
        return loss, cond, pred

    def _forward_loss_core_upstream_exact(
        self,
        x1: torch.Tensor,
        text: torch.Tensor,
        mask: torch.Tensor,
        rand_span_mask: torch.Tensor,
        x0: torch.Tensor,
        time: torch.Tensor,
        drop_audio_cond: bool | torch.Tensor,
        drop_text: bool | torch.Tensor,
    ):
        """Reproduce upstream's loss reduction exactly (never compiled).

        ``loss[rand_span_mask]`` produces a data-dependent shape, which graph-breaks under
        torch.compile -- that is precisely why the compiled path uses the masked-multiply
        form instead. Keeping this expression for the default path is what makes
        compile-disabled training bit-for-bit identical to upstream, including dtype: the
        compiled path accumulates in fp32 (needed so a raw ``loss_sum`` cannot overflow
        fp16), whereas upstream returns the input dtype.
        """
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        pred = self.transformer(
            x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text, mask=mask
        )

        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred

    def _prepare_training_inputs(self, inp, text, lens):
        """Stochastic preparation shared by ``forward`` (stays eager; not compiled)."""
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask: long dtype for mask index arithmetic
        if not exists(lens):  # if lens not acquired by trainer from collate_fn
            lens = torch.full((batch,), seq_len, device=device, dtype=torch.long)
        else:
            lens = lens.to(device=device, dtype=torch.long)
        mask = lens_to_mask(lens, length=seq_len)

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths, length=seq_len)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1; x0 is gaussian noise; time step
        x1 = inp
        x0 = torch.randn_like(x1)
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # Tensor-aware backbones consume CFG flags with branchless embedding masks.
        # Keeping these flags as 0-D tensors avoids separate torch.compile graphs for
        # each CFG bool combination while preserving the same sampled drop decisions.
        # Tensorize only when a compile target is actually active so the never-compiled
        # default path stays byte-identical to upstream (Python bools, zero allocation
        # on no-drop steps). Runtime fallback clears compile state, so flags revert to
        # bools automatically. Backbones without this capability keep historical bools.
        if getattr(self.transformer, "supports_tensor_cfg_training_flags", False) and self._training_compile_enabled():
            drop_audio_cond = torch.as_tensor(drop_audio_cond, device=device, dtype=torch.bool)
            drop_text = torch.as_tensor(drop_text, device=device, dtype=torch.bool)

        return x1, text, mask, rand_span_mask, x0, time, drop_audio_cond, drop_text

    def _forward_loss_core_components(
        self,
        x1: torch.Tensor,
        text: torch.Tensor,
        mask: torch.Tensor,
        rand_span_mask: torch.Tensor,
        x0: torch.Tensor,
        time: torch.Tensor,
        drop_audio_cond: bool | torch.Tensor,
        drop_text: bool | torch.Tensor,
    ):
        # Deterministic loss core: sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # apply mask will use more memory; might adjust batchsize or batchsampler long sequence threshold
        pred = self.transformer(
            x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text, mask=mask
        )

        # flow matching loss: masked mean over valid span tokens and feature dim.
        # Boolean indexing (loss[rand_span_mask]) causes a graph break under torch.compile;
        # the elementwise multiply form is numerically equivalent and compile-friendly.
        # denom.clamp(min=1.0) keeps the loss finite when no token is selected.
        # Accumulate the squared error in fp32 so fp16 AMP/global loss-sum
        # training cannot overflow before GradScaler sees the loss.
        loss = F.mse_loss(pred.float(), flow.float(), reduction="none")
        loss_mask = rand_span_mask[..., None].to(loss.dtype)
        loss_sum = (loss * loss_mask).sum()
        denom = (loss_mask.sum() * loss.shape[-1]).clamp(min=1.0)
        loss = loss_sum / denom

        return loss, loss_sum, denom, cond, pred

    def _forward_loss_core(self, *args):
        """Return the historical mean-loss tuple for callers outside the trainer."""
        loss, _, _, cond, pred = self._forward_loss_core_components(*args)
        return loss, cond, pred

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str],
        duration: int | int["b"],
        *,
        lens: int["b"] | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=65536,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()
        # raw wave

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # text

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # duration

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow (cond)
            if cfg_strength < 1e-5:
                pred = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=True,
                )
                return pred

            # predict flow (cond and uncond), for classifier-free guidance
            pred_cfg = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                cfg_infer=True,
                cache=True,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        if t_start == 0 and use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave
        text: int["b nt"] | list[str],
        *,
        lens: int["b"] | None = None,
        noise_scheduler: str | None = None,
    ):
        # Stochastic preparation stays eager; deterministic loss core may be compiled.
        x1, text, mask, rand_span_mask, x0, time, drop_audio_cond, drop_text = self._prepare_training_inputs(
            inp, text, lens
        )
        loss, cond, pred = self._run_loss_core(x1, text, mask, rand_span_mask, x0, time, drop_audio_cond, drop_text)
        return loss, cond, pred
