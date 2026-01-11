"""
GRPO (Group Relative Policy Optimization) Trainer for RL fine-tuning.

This trainer implements GRPO-style reinforcement learning for TTS models,
using reward signals from configurable reward providers.
"""

from __future__ import annotations

import copy
import gc
import math
import os
from typing import Any

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from f5_tts.model import CFM
from f5_tts.model.dataset import collate_fn
from f5_tts.model.utils import default, exists
from f5_tts.rewards import RewardCombiner, RewardInput


class GRPOTrainer:
    """GRPO Trainer for RL fine-tuning of TTS models.

    Implements Group Relative Policy Optimization using reward signals
    from configurable reward providers.

    The training loop:
    1. Sample multiple trajectories from the policy model
    2. Compute rewards for generated audio
    3. Compute advantages using group normalization
    4. Update policy using GRPO objective with KL penalty to reference model

    Attributes:
        model: The CFM model being trained (policy)
        ref_model: Frozen reference model for KL computation
        reward_combiner: Combined reward providers
    """

    def __init__(
        self,
        model: CFM,
        epochs: int,
        learning_rate: float,
        reward_configs: list[dict[str, Any]] | None = None,
        reward_combine_config: dict[str, Any] | None = None,
        num_warmup_updates: int = 1000,
        save_per_updates: int = 1000,
        keep_last_n_checkpoints: int = -1,
        checkpoint_path: str | None = None,
        batch_size_per_gpu: int = 8,
        grad_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        # GRPO specific parameters
        group_size: int = 4,  # number of samples per input for group normalization
        kl_coef: float = 0.1,  # KL penalty coefficient
        reward_scale: float = 1.0,  # scale factor for rewards
        clip_range: float = 0.2,  # PPO-style clipping
        # Logging
        logger: str | None = "wandb",
        wandb_project: str = "f5-tts-rl",
        wandb_run_name: str = "grpo_run",
        wandb_resume_id: str | None = None,
        # Vocoder for reward computation
        vocoder_name: str = "vocos",
        is_local_vocoder: bool = False,
        local_vocoder_path: str = "",
        # Other
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
    ):
        """Initialize the GRPO trainer.

        Args:
            model: The CFM model to train (must have gaussian output).
            epochs: Number of training epochs.
            learning_rate: Learning rate for the optimizer.
            reward_configs: List of reward provider configurations.
            reward_combine_config: Configuration for reward combination.
            num_warmup_updates: Number of warmup updates.
            save_per_updates: Save checkpoint every N updates.
            keep_last_n_checkpoints: Number of checkpoints to keep.
            checkpoint_path: Path to save checkpoints.
            batch_size_per_gpu: Batch size per GPU.
            grad_accumulation_steps: Gradient accumulation steps.
            max_grad_norm: Maximum gradient norm for clipping.
            group_size: Number of samples per input for GRPO.
            kl_coef: KL divergence penalty coefficient.
            reward_scale: Scale factor for rewards.
            clip_range: PPO-style clip range for policy ratio.
            logger: Logger type ('wandb' or None).
            wandb_project: W&B project name.
            wandb_run_name: W&B run name.
            wandb_resume_id: W&B resume ID.
            vocoder_name: Vocoder type for audio generation.
            is_local_vocoder: Whether vocoder is local.
            local_vocoder_path: Path to local vocoder.
            accelerate_kwargs: Additional kwargs for Accelerator.
            ema_kwargs: Additional kwargs for EMA.
        """
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        if logger == "wandb" and not wandb.api.api_key:
            logger = None

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.logger = logger
        if self.logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config={
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "group_size": group_size,
                    "kl_coef": kl_coef,
                    "reward_scale": reward_scale,
                    "clip_range": clip_range,
                },
            )

        self.model = model
        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.checkpoint_path = default(checkpoint_path, "ckpts/grpo_f5-tts")
        self.batch_size_per_gpu = batch_size_per_gpu
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # GRPO parameters
        self.group_size = group_size
        self.kl_coef = kl_coef
        self.reward_scale = reward_scale
        self.clip_range = clip_range

        # Vocoder config
        self.vocoder_name = vocoder_name
        self.is_local_vocoder = is_local_vocoder
        self.local_vocoder_path = local_vocoder_path
        self._vocoder = None

        # Create reference model (frozen copy)
        if self.is_main:
            self.ref_model = self._create_reference_model(model)
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)
            print(f"Using logger: {logger}")

        # Setup reward system
        self.reward_combiner = self._setup_rewards(reward_configs, reward_combine_config)

        # Optimizer
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def vocoder(self):
        """Lazy load vocoder."""
        if self._vocoder is None:
            from f5_tts.infer.utils_infer import load_vocoder

            self._vocoder = load_vocoder(
                vocoder_name=self.vocoder_name,
                is_local=self.is_local_vocoder,
                local_path=self.local_vocoder_path,
            )
        return self._vocoder

    def _create_reference_model(self, model: CFM) -> CFM:
        """Create a frozen copy of the model for KL computation."""
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        return ref_model

    def _setup_rewards(
        self,
        reward_configs: list[dict[str, Any]] | None,
        combine_config: dict[str, Any] | None,
    ) -> RewardCombiner | None:
        """Setup reward combiner from configs."""
        if not reward_configs:
            return None

        return RewardCombiner.from_config(reward_configs, combine_config)

    def save_checkpoint(self, update: int, last: bool = False):
        """Save a training checkpoint."""
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at update {update}")
            else:
                if self.keep_last_n_checkpoints == 0:
                    return
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{update}.pt")
                if self.keep_last_n_checkpoints > 0:
                    checkpoints = [
                        f
                        for f in os.listdir(self.checkpoint_path)
                        if f.startswith("model_") and f.endswith(".pt") and f != "model_last.pt"
                    ]
                    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                    while len(checkpoints) > self.keep_last_n_checkpoints:
                        oldest_checkpoint = checkpoints.pop(0)
                        os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                        print(f"Removed old checkpoint: {oldest_checkpoint}")

    def load_checkpoint(self) -> int:
        """Load a checkpoint and return the update number."""
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(filename.endswith((".pt", ".safetensors")) for filename in os.listdir(self.checkpoint_path))
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            checkpoints = [f for f in os.listdir(self.checkpoint_path) if f.startswith("model_") and f.endswith(".pt")]
            if not checkpoints:
                return 0
            latest_checkpoint = sorted(checkpoints, key=lambda x: int("".join(filter(str.isdigit, x))))[-1]

        checkpoint = torch.load(
            f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu"
        )

        if self.is_main:
            if "ema_model_state_dict" in checkpoint:
                self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        if "model_state_dict" in checkpoint:
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        update = checkpoint.get("update", 0)
        del checkpoint
        gc.collect()
        return update

    def _compute_rewards(
        self,
        generated_audio: torch.Tensor,
        texts: list[str],
        ref_audio: torch.Tensor | None = None,
        sample_rate: int = 24000,
    ) -> torch.Tensor:
        """Compute rewards for generated audio."""
        if self.reward_combiner is None:
            # Return dummy rewards if no reward system configured
            return torch.ones(generated_audio.shape[0], device=generated_audio.device)

        # Prepare reward inputs
        batch_size = generated_audio.shape[0]
        inputs = []
        for i in range(batch_size):
            audio = generated_audio[i]
            text = texts[i] if i < len(texts) else ""
            speaker_ref = ref_audio[i] if ref_audio is not None and i < ref_audio.shape[0] else None

            inputs.append(
                RewardInput(
                    audio=audio,
                    text=text,
                    speaker_ref=speaker_ref,
                    sample_rate=sample_rate,
                )
            )

        # Compute rewards
        outputs = self.reward_combiner.compute(inputs)

        # Extract total rewards
        rewards = torch.stack([out.total_reward for out in outputs])
        return rewards * self.reward_scale

    def _compute_advantages(self, rewards: torch.Tensor, group_size: int) -> torch.Tensor:
        """Compute advantages using group normalization.

        Groups rewards by input and normalizes within each group.
        """
        batch_size = rewards.shape[0]
        num_groups = batch_size // group_size

        if num_groups == 0:
            # Not enough samples for group normalization
            return rewards - rewards.mean()

        # Reshape to groups
        rewards_grouped = rewards[:num_groups * group_size].view(num_groups, group_size)

        # Normalize within each group
        mean = rewards_grouped.mean(dim=1, keepdim=True)
        std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards_grouped - mean) / std

        # Flatten back
        advantages = advantages.view(-1)

        # Handle remaining samples
        if batch_size > num_groups * group_size:
            remaining = rewards[num_groups * group_size:]
            remaining_adv = remaining - remaining.mean()
            advantages = torch.cat([advantages, remaining_adv])

        return advantages

    def _grpo_step(
        self,
        batch: dict,
    ) -> dict[str, torch.Tensor]:
        """Perform a single GRPO step.

        Args:
            batch: Training batch with 'text', 'mel', 'mel_lengths'.

        Returns:
            Dictionary with loss components.
        """
        text_inputs = batch["text"]
        mel_spec = batch["mel"].permute(0, 2, 1)
        mel_lengths = batch["mel_lengths"]

        # Sample from policy
        with torch.no_grad():
            samples, log_probs, cond, mask = self.model.forward_rl(
                inp=mel_spec,
                text=text_inputs,
                lens=mel_lengths,
                return_logprob=True,
            )

        # Generate audio for reward computation
        with torch.no_grad():
            # Use the samples directly as mel spectrogram
            # Convert to audio using vocoder
            gen_mel = samples.permute(0, 2, 1)
            if self.vocoder_name == "vocos":
                gen_audio = self.vocoder.decode(gen_mel).cpu()
            else:
                gen_audio = self.vocoder(gen_mel).squeeze(1).cpu()

        # Compute rewards
        target_sample_rate = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
        texts = text_inputs if isinstance(text_inputs, list) else [str(t) for t in text_inputs]
        rewards = self._compute_rewards(
            gen_audio,
            texts,
            sample_rate=target_sample_rate,
        )

        # Compute advantages
        advantages = self._compute_advantages(rewards, self.group_size)

        # Compute policy loss
        # Re-compute log probs under current policy (with gradients)
        _, current_log_probs, _, _ = self.model.forward_rl(
            inp=mel_spec,
            text=text_inputs,
            lens=mel_lengths,
            return_logprob=True,
        )

        # Compute log probs under reference policy
        with torch.no_grad():
            _, ref_log_probs, _, _ = self.ref_model.forward_rl(
                inp=mel_spec,
                text=text_inputs,
                lens=mel_lengths,
                return_logprob=True,
            )

        # Policy gradient loss with clipping
        ratio = torch.exp(current_log_probs - log_probs.detach())
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)

        policy_loss_unclipped = -advantages * ratio
        policy_loss_clipped = -advantages * clipped_ratio
        policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean()

        # KL penalty
        kl = current_log_probs - ref_log_probs
        kl_loss = self.kl_coef * kl.mean()

        # Total loss
        total_loss = policy_loss + kl_loss

        return {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "kl_loss": kl_loss,
            "reward_mean": rewards.mean(),
            "advantage_mean": advantages.mean(),
        }

    def train(self, train_dataset: Dataset, num_workers: int = 4):
        """Run GRPO training loop.

        Args:
            train_dataset: Training dataset.
            num_workers: Number of dataloader workers.
        """
        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            batch_size=self.batch_size_per_gpu,
            shuffle=True,
        )

        # Setup scheduler
        warmup_updates = self.num_warmup_updates * self.accelerator.num_processes
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        decay_updates = total_updates - warmup_updates

        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )

        train_dataloader, self.scheduler = self.accelerator.prepare(train_dataloader, self.scheduler)
        start_update = self.load_checkpoint()
        global_update = start_update

        for epoch in range(self.epochs):
            self.model.train()

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
            )

            for batch in train_dataloader:
                with self.accelerator.accumulate(self.model):
                    losses = self._grpo_step(batch)
                    loss = losses["loss"]

                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        update=str(global_update),
                        loss=loss.item(),
                        reward=losses["reward_mean"].item(),
                    )

                if self.accelerator.is_local_main_process:
                    log_dict = {
                        "loss": loss.item(),
                        "policy_loss": losses["policy_loss"].item(),
                        "kl_loss": losses["kl_loss"].item(),
                        "reward_mean": losses["reward_mean"].item(),
                        "advantage_mean": losses["advantage_mean"].item(),
                        "lr": self.scheduler.get_last_lr()[0],
                    }
                    self.accelerator.log(log_dict, step=global_update)

                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)

        self.save_checkpoint(global_update, last=True)
        self.accelerator.end_training()
