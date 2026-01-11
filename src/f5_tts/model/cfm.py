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
from typing import Callable, Literal

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
        objective: Literal["mse", "gaussian_nll", "rl_grpo"] = "mse",
        ln_sig_clamp: tuple[float, float] = (-10.0, 10.0),  # clamp range for log sigma
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

        # training objective
        self.objective = objective
        self.ln_sig_clamp = ln_sig_clamp

    @property
    def device(self):
        return next(self.parameters()).device

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
        max_duration=4096,
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
        objective: str | None = None,  # override self.objective if provided
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):  # if lens not acquired by trainer from collate_fn
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # select objective
        obj = objective if objective is not None else self.objective

        if obj == "mse":
            # Standard MSE flow matching loss
            pred = self.transformer(
                x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text, mask=mask
            )
            loss = F.mse_loss(pred, flow, reduction="none")
            loss = loss[rand_span_mask]
            return loss.mean(), cond, pred

        elif obj == "gaussian_nll":
            # Gaussian NLL loss for probabilistic training
            return self._forward_gaussian_nll(
                φ=φ,
                cond=cond,
                text=text,
                time=time,
                flow=flow,
                rand_span_mask=rand_span_mask,
                mask=mask,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
            )

        elif obj == "rl_grpo":
            # RL GRPO objective - this requires special handling via GRPOTrainer
            raise ValueError(
                "rl_grpo objective should not be used directly in forward(). "
                "Use GRPOTrainer for RL training."
            )

        else:
            raise ValueError(f"Unknown objective: {obj}. Must be 'mse', 'gaussian_nll', or 'rl_grpo'.")

    def _forward_gaussian_nll(
        self,
        φ: torch.Tensor,
        cond: torch.Tensor,
        text: torch.Tensor,
        time: torch.Tensor,
        flow: torch.Tensor,
        rand_span_mask: torch.Tensor,
        mask: torch.Tensor,
        drop_audio_cond: bool,
        drop_text: bool,
    ):
        """Compute Gaussian NLL loss for probabilistic training.

        The loss is computed as:
            -log p(flow | φ, cond, text, time)
            = 0.5 * (ln_sig + (flow - mu)^2 / exp(2 * ln_sig))

        Plus a constant term that we drop.
        """
        # Check if transformer supports gaussian output
        if not hasattr(self.transformer, "forward_prob"):
            raise RuntimeError(
                "Gaussian NLL objective requires a transformer with forward_prob method. "
                "Initialize the transformer with output_dist='gaussian'."
            )

        # Get mu and log sigma from transformer
        mu, ln_sig = self.transformer.forward_prob(
            x=φ, cond=cond, text=text, time=time, mask=mask,
            drop_audio_cond=drop_audio_cond, drop_text=drop_text,
        )

        # Clamp log sigma for numerical stability
        ln_sig = torch.clamp(ln_sig, self.ln_sig_clamp[0], self.ln_sig_clamp[1])

        # Compute Gaussian NLL
        # -log N(flow; mu, sigma) = 0.5 * log(2π) + ln_sig + 0.5 * (flow - mu)^2 / sigma^2
        # We drop the constant 0.5 * log(2π) term
        var = torch.exp(2 * ln_sig)
        nll = ln_sig + 0.5 * (flow - mu) ** 2 / (var + 1e-8)

        # Apply mask
        nll = nll[rand_span_mask]

        # Check for NaN/Inf
        if not torch.isfinite(nll).all():
            raise ValueError(
                f"NaN or Inf in Gaussian NLL loss. "
                f"mu range: [{mu.min():.4f}, {mu.max():.4f}], "
                f"ln_sig range: [{ln_sig.min():.4f}, {ln_sig.max():.4f}]"
            )

        return nll.mean(), cond, mu

    def forward_rl(
        self,
        inp: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str],
        *,
        lens: int["b"] | None = None,
        return_logprob: bool = True,
    ):
        """Forward pass for RL training, returning samples and log probabilities.

        This method is used by GRPOTrainer to generate samples and compute
        log probabilities for the policy gradient update.

        Args:
            inp: Input mel spectrogram or raw waveform
            text: Text tokens or strings
            lens: Optional sequence lengths
            return_logprob: Whether to return log probabilities

        Returns:
            Tuple of (samples, log_probs, cond, mask) where:
                samples: Generated samples
                log_probs: Log probabilities of the samples (if return_logprob)
                cond: Conditioning signal
                mask: The mask used for generation
        """
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, device = *inp.shape[:2], self.device

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)

        # get a random span to mask out
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1 (target)
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # Sample trajectory and compute log probabilities
        samples, log_probs = self._sample_rl_trajectory(
            x0=x0,
            cond=cond,
            text=text,
            mask=mask,
            rand_span_mask=rand_span_mask,
            return_logprob=return_logprob,
        )

        return samples, log_probs, cond, rand_span_mask

    def _sample_rl_trajectory(
        self,
        x0: torch.Tensor,
        cond: torch.Tensor,
        text: torch.Tensor,
        mask: torch.Tensor,
        rand_span_mask: torch.Tensor,
        return_logprob: bool = True,
        num_steps: int = 10,
    ):
        """Sample a trajectory for RL training with log probability computation.

        Uses simple Euler integration with gaussian sampling at each step.
        """
        if not hasattr(self.transformer, "forward_prob"):
            raise RuntimeError(
                "RL training requires a transformer with forward_prob method. "
                "Initialize the transformer with output_dist='gaussian'."
            )

        batch, seq_len = x0.shape[:2]
        device = x0.device
        dtype = x0.dtype

        # Time steps for integration
        dt = 1.0 / num_steps
        t_steps = torch.linspace(0, 1 - dt, num_steps, device=device, dtype=dtype)

        x = x0.clone()
        total_log_prob = torch.zeros(batch, device=device, dtype=dtype) if return_logprob else None

        for t_val in t_steps:
            t = torch.full((batch,), t_val, device=device, dtype=dtype)

            # Get mu and log sigma from transformer
            mu, ln_sig = self.transformer.forward_prob(
                x=x, cond=cond, text=text, time=t, mask=mask,
                drop_audio_cond=False, drop_text=False,
            )

            # Clamp log sigma
            ln_sig = torch.clamp(ln_sig, self.ln_sig_clamp[0], self.ln_sig_clamp[1])
            sigma = torch.exp(ln_sig)

            # Sample next step: x_next = x + dt * (mu + sigma * noise)
            noise = torch.randn_like(x)
            dx = mu + sigma * noise
            x_next = x + dt * dx

            # Compute log probability of this step
            if return_logprob:
                # log p(noise) = -0.5 * noise^2 - 0.5 * log(2π)
                # We only compute the -0.5 * noise^2 part and sum over dims
                log_prob_step = -0.5 * (noise ** 2).sum(dim=(1, 2))
                total_log_prob = total_log_prob + log_prob_step

            x = x_next

        return x, total_log_prob

    def compute_logprob(
        self,
        samples: torch.Tensor,
        x0: torch.Tensor,
        cond: torch.Tensor,
        text: torch.Tensor,
        mask: torch.Tensor,
        num_steps: int = 10,
    ):
        """Compute log probability of given samples under the model.

        Used for computing KL divergence in GRPO.
        """
        if not hasattr(self.transformer, "forward_prob"):
            raise RuntimeError(
                "compute_logprob requires a transformer with forward_prob method."
            )

        batch = samples.shape[0]
        device = samples.device
        dtype = samples.dtype

        dt = 1.0 / num_steps
        t_steps = torch.linspace(0, 1 - dt, num_steps, device=device, dtype=dtype)

        # Reconstruct trajectory from x0 to samples
        # This is approximate - we interpolate linearly
        total_log_prob = torch.zeros(batch, device=device, dtype=dtype)

        for i, t_val in enumerate(t_steps):
            t = torch.full((batch,), t_val, device=device, dtype=dtype)

            # Interpolate x at time t
            alpha = t_val
            x_t = (1 - alpha) * x0 + alpha * samples

            # Get mu and log sigma from transformer
            mu, ln_sig = self.transformer.forward_prob(
                x=x_t, cond=cond, text=text, time=t, mask=mask,
                drop_audio_cond=False, drop_text=False,
            )

            ln_sig = torch.clamp(ln_sig, self.ln_sig_clamp[0], self.ln_sig_clamp[1])
            sigma = torch.exp(ln_sig)

            # Compute target velocity
            target_velocity = samples - x0

            # Log probability under the gaussian
            # log N(target_velocity; mu, sigma)
            log_prob = -0.5 * ((target_velocity - mu) / (sigma + 1e-8)) ** 2 - ln_sig
            log_prob = log_prob.sum(dim=(1, 2))
            total_log_prob = total_log_prob + log_prob * dt

        return total_log_prob
