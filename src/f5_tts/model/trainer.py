from __future__ import annotations

import gc
import math
import os
from contextlib import contextmanager, nullcontext
from typing import cast

import torch
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs, DistributedType, send_to_device
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from f5_tts.model import CFM
from f5_tts.model.dataset import DynamicBatchSampler, collate_fn
from f5_tts.model.utils import default, exists


# trainer


class Trainer:
    def __init__(
        self,
        model: CFM,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        keep_last_n_checkpoints: int = -1,  # -1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints
        checkpoint_path=None,
        batch_size_per_gpu=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        logger: str | None = "wandb",  # "wandb" | "tensorboard" | None
        wandb_project="test_f5-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        log_samples: bool = False,
        last_per_updates=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        mel_spec_type: str = "vocos",  # "vocos" | "bigvgan"
        is_local_vocoder: bool = False,  # use local path vocoder
        local_vocoder_path: str = "",  # local vocoder path
        model_cfg_dict: dict = dict(),  # training config
        global_masked_mean: bool = False,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        if logger == "wandb" and not wandb.api.api_key:
            logger = None
        self.log_samples = log_samples

        accelerate_kwargs = dict(accelerate_kwargs)
        if global_masked_mean:
            dataloader_config = accelerate_kwargs.get("dataloader_config")
            if dataloader_config is not None:
                if dataloader_config.even_batches:
                    raise ValueError("global_masked_mean requires dataloader_config.even_batches=False")
            elif "even_batches" in accelerate_kwargs:
                if accelerate_kwargs["even_batches"] is not False:
                    raise ValueError("global_masked_mean requires even_batches=False")
            else:
                accelerate_kwargs["dataloader_config"] = DataLoaderConfiguration(even_batches=False)

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )
        self.global_masked_mean = global_masked_mean
        if self.global_masked_mean and grad_accumulation_steps < 1:
            raise ValueError("global_masked_mean requires grad_accumulation_steps >= 1")
        self._validate_global_masked_mean_backend()

        self.logger = logger
        if self.logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            if not model_cfg_dict:
                model_cfg_dict = {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size_per_gpu": batch_size_per_gpu,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "noise_scheduler": noise_scheduler,
                    "bnb_optimizer": bnb_optimizer,
                    "global_masked_mean": global_masked_mean,
                }
            model_cfg_dict["gpus"] = self.accelerator.num_processes
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config=model_cfg_dict,
            )

        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = None
            if self.accelerator.is_main_process:
                self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")

        self.model = model

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)

            print(f"Using logger: {logger}")
            if grad_accumulation_steps > 1:
                print(
                    "Gradient accumulation checkpointing with per_updates now, old logic per_steps used with before f992c4e"
                )

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_f5-tts")

        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # mel vocoder config
        self.vocoder_name = mel_spec_type
        self.is_local_vocoder = is_local_vocoder
        self.local_vocoder_path = local_vocoder_path

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate, fused=True)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self._unwrapped_model = self.accelerator.unwrap_model(self.model)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def _validate_global_masked_mean_backend(self):
        if not self.global_masked_mean:
            return

        supported = {DistributedType.NO, DistributedType.MULTI_CPU, DistributedType.MULTI_GPU}
        if self.accelerator.distributed_type not in supported:
            raise NotImplementedError(
                "global_masked_mean currently supports single-process and vanilla DDP training only; "
                f"received {self.accelerator.distributed_type.value}."
            )
        if self.accelerator.even_batches:
            raise ValueError("global_masked_mean requires Accelerate even_batches=False to prevent duplicates.")
        if self.accelerator.split_batches:
            raise NotImplementedError("global_masked_mean does not support split_batches=True.")
        if self.accelerator.dispatch_batches:
            raise NotImplementedError("global_masked_mean does not support dispatch_batches=True.")

    def _reduce_global_masked_value(self, local_value: torch.Tensor) -> torch.Tensor:
        local_value = local_value.detach().to(device=self.accelerator.device)
        return cast(torch.Tensor, self.accelerator.reduce(local_value, reduction="sum"))

    def _iter_global_masked_mean_batches(self, dataloader):
        """Yield accumulation windows normalized before any backward pass."""
        dataloader_iter = iter(dataloader)
        sample_mask = self._unwrapped_model.sample_training_mask

        while True:
            batches = []
            for _ in range(self.grad_accumulation_steps):
                try:
                    batches.append(next(dataloader_iter))
                except StopIteration:
                    break
            if not batches:
                return

            window = []
            local_loss_count = torch.zeros((), device=self.accelerator.device, dtype=torch.int64)
            local_error = None
            for batch in batches:
                batch_error = self._validate_global_masked_batch(batch)
                rand_span_mask = None
                if batch_error is None:
                    try:
                        mel = batch["mel"]
                        rand_span_mask = sample_mask(batch["mel_lengths"], mel.shape[-1])
                        local_loss_count += rand_span_mask.sum(dtype=torch.int64) * mel.shape[1]
                    except Exception as exc:
                        batch_error = f"training-mask preparation failed: {exc}"
                if batch_error is not None and local_error is None:
                    local_error = batch_error
                window.append((batch, rand_span_mask))

            local_stats = torch.stack(
                (
                    local_loss_count,
                    torch.tensor(int(local_error is not None), device=self.accelerator.device, dtype=torch.int64),
                )
            )
            global_stats = self._reduce_global_masked_value(local_stats)
            global_loss_count_value, global_error_count = global_stats.cpu().tolist()
            if global_error_count:
                detail = local_error or "another rank reported malformed training input"
                raise ValueError(f"global_masked_mean rejected an accumulation window: {detail}")
            if global_loss_count_value == 0:
                raise RuntimeError(
                    "global_masked_mean produced an empty accumulation window; "
                    "no optimizer update can be defined without masked elements."
                )

            global_loss_count = global_stats[0]
            loss_scale = global_loss_count.to(dtype=torch.float32).reciprocal()
            loss_scale *= self.grad_accumulation_steps * self.accelerator.num_processes
            for index, (batch, rand_span_mask) in enumerate(window):
                assert rand_span_mask is not None
                yield batch, rand_span_mask, loss_scale, global_loss_count, index == len(window) - 1

    def _validate_global_masked_batch(self, batch) -> str | None:
        """Return rank-local input errors without raising before the count collective."""
        if not isinstance(batch, dict):
            return f"expected a batch dictionary, received {type(batch).__name__}"
        if "mel" not in batch or "mel_lengths" not in batch or "text" not in batch:
            return "batch must contain mel, mel_lengths, and text"

        mel = batch["mel"]
        mel_lengths = batch["mel_lengths"]
        if not isinstance(mel, torch.Tensor) or mel.ndim != 3 or not mel.is_floating_point():
            return "mel must be a floating-point rank-3 tensor shaped [batch, channels, frames]"
        if mel.shape[0] == 0:
            return "mel batch must not be empty"
        if not isinstance(mel_lengths, torch.Tensor) or mel_lengths.ndim != 1:
            return "mel_lengths must be a rank-1 tensor"
        if mel_lengths.shape[0] != mel.shape[0]:
            return "mel_lengths count must match the mel batch size"
        if mel_lengths.dtype == torch.bool or mel_lengths.is_floating_point() or mel_lengths.is_complex():
            return f"mel_lengths must use an integer dtype, received {mel_lengths.dtype}"
        if bool(((mel_lengths <= 0) | (mel_lengths > mel.shape[-1])).any().item()):
            return f"mel_lengths values must be between 1 and the padded frame length ({mel.shape[-1]})"

        text = batch["text"]
        if isinstance(text, torch.Tensor):
            if text.ndim == 0 or text.shape[0] != mel.shape[0]:
                return "tensor text batch size must match the mel batch size"
            if text.dtype == torch.bool or text.is_floating_point() or text.is_complex():
                return f"tensor text must use an integer dtype, received {text.dtype}"
        elif isinstance(text, list):
            if len(text) != mel.shape[0]:
                return "text list size must match the mel batch size"
            if any(
                not isinstance(item, str)
                and (not isinstance(item, list) or any(not isinstance(token, str) for token in item))
                for item in text
            ):
                return "text list entries must be strings or lists of strings"
            # list_str_to_idx supports nested token lists (pinyin style), but only
            # when the model has a vocabulary map. Without one CFM falls back to
            # ByT5 byte encoding (list_str_to_tensor), which requires plain strings
            # and would otherwise raise a cryptic TypeError deep inside forward.
            if any(isinstance(item, list) for item in text) and not (
                getattr(self._unwrapped_model, "vocab_char_map", None) is not None
            ):
                return "nested text token lists require a vocabulary map"
        else:
            return "text must be a tensor or list"
        return None

    @contextmanager
    def _accumulation_context(self, is_boundary: bool | None):
        if is_boundary is None:
            with self.accelerator.accumulate(self.model):
                yield
            return

        self.accelerator.sync_gradients = is_boundary
        sync_context = nullcontext() if is_boundary else self.accelerator.no_sync(self.model)
        with sync_context:
            yield

    def _validate_global_masked_dataloader(self, dataloader):
        local_length = torch.tensor([len(dataloader)], device=self.accelerator.device, dtype=torch.int64)
        gathered_lengths = cast(torch.Tensor, self.accelerator.gather(local_length))
        if int(gathered_lengths[0]) == 0:
            raise ValueError("global_masked_mean requires a non-empty training dataloader.")
        if bool((gathered_lengths != gathered_lengths[0]).any().item()):
            raise RuntimeError(
                "global_masked_mean requires equal dataloader lengths on every rank; "
                f"received {gathered_lengths.cpu().tolist()}."
            )

    def save_checkpoint(self, update, last=False, *, consumed_batches: int | None = None):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
            )
            if consumed_batches is not None:
                checkpoint["consumed_batches"] = consumed_batches
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
                    # Updated logic to exclude pretrained model from rotation
                    checkpoints = [
                        f
                        for f in os.listdir(self.checkpoint_path)
                        if f.startswith("model_")
                        and not f.startswith("pretrained_")  # Exclude pretrained models
                        and f.endswith(".pt")
                        and f != "model_last.pt"
                    ]
                    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                    while len(checkpoints) > self.keep_last_n_checkpoints:
                        oldest_checkpoint = checkpoints.pop(0)
                        os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                        print(f"Removed old checkpoint: {oldest_checkpoint}")

    def load_checkpoint(self):
        self._loaded_consumed_batches = 0
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
            # Updated to consider pretrained models for loading but prioritize training checkpoints
            all_checkpoints = [
                f
                for f in os.listdir(self.checkpoint_path)
                if (f.startswith("model_") or f.startswith("pretrained_")) and f.endswith((".pt", ".safetensors"))
            ]

            # First try to find regular training checkpoints
            training_checkpoints = [f for f in all_checkpoints if f.startswith("model_") and f != "model_last.pt"]
            if training_checkpoints:
                latest_checkpoint = sorted(
                    training_checkpoints,
                    key=lambda x: int("".join(filter(str.isdigit, x))),
                )[-1]
            else:
                # If no training checkpoints, use pretrained model
                latest_checkpoint = next(f for f in all_checkpoints if f.startswith("pretrained_"))

        if latest_checkpoint.endswith(".safetensors"):  # always a pretrained checkpoint
            from safetensors.torch import load_file

            checkpoint = load_file(f"{self.checkpoint_path}/{latest_checkpoint}", device="cpu")
            checkpoint = {"ema_model_state_dict": checkpoint}
        elif latest_checkpoint.endswith(".pt"):
            # checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location=self.accelerator.device)  # rather use accelerator.load_state ಥ_ಥ
            checkpoint = torch.load(
                f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu"
            )

        # patch for backward compatibility, 305e3ea
        for key in ["ema_model.mel_spec.mel_stft.mel_scale.fb", "ema_model.mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["ema_model_state_dict"]:
                del checkpoint["ema_model_state_dict"][key]

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        if "update" in checkpoint or "step" in checkpoint:
            # patch for backward compatibility, with before f992c4e
            if "step" in checkpoint:
                checkpoint["update"] = checkpoint["step"] // self.grad_accumulation_steps
                if self.grad_accumulation_steps > 1 and self.is_main:
                    print(
                        "F5-TTS WARNING: Loading checkpoint saved with per_steps logic (before f992c4e), will convert to per_updates according to grad_accumulation_steps setting, may have unexpected behaviour."
                    )
            # patch for backward compatibility, 305e3ea
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]

            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            update = checkpoint["update"]
            self._loaded_consumed_batches = checkpoint.get("consumed_batches", update * self.grad_accumulation_steps)
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            update = 0

        del checkpoint
        gc.collect()
        return update

    def train(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):
        if self.log_samples:
            from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef

            vocoder = load_vocoder(
                vocoder_name=self.vocoder_name, is_local=self.is_local_vocoder, local_path=self.local_vocoder_path
            )
            target_sample_rate = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,  # This enables reproducible shuffling
                drop_residual=False,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")

        if self.global_masked_mean:
            train_dataloader = self.accelerator.prepare_data_loader(train_dataloader, device_placement=False)
            self._validate_global_masked_dataloader(train_dataloader)

        # Scheduler length must use the prepared rank-local loader. AcceleratedScheduler
        # advances its wrapped scheduler once per process at each synchronized update.
        warmup_updates = self.num_warmup_updates * self.accelerator.num_processes
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        scheduler_updates = total_updates * self.accelerator.num_processes if self.global_masked_mean else total_updates
        decay_updates = scheduler_updates - warmup_updates
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        if self.global_masked_mean:
            self.scheduler = self.accelerator.prepare(self.scheduler)
        else:
            train_dataloader, self.scheduler = self.accelerator.prepare(
                train_dataloader, self.scheduler
            )  # actual multi_gpu updates = single_gpu updates / gpu nums
        start_update = self.load_checkpoint()
        global_update = start_update
        consumed_batches = self._loaded_consumed_batches if self.global_masked_mean else 0

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = consumed_batches if self.global_masked_mean else start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            # Set epoch for the batch sampler if it exists
            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            if self.global_masked_mean:
                training_batches = self._iter_global_masked_mean_batches(current_dataloader)
            else:
                training_batches = ((batch, None, None, None, None) for batch in current_dataloader)
            window_loss_sum = torch.zeros((), device=self.accelerator.device, dtype=torch.float32)

            for batch, rand_span_mask, loss_scale, global_loss_count, is_boundary in training_batches:
                if self.global_masked_mean:
                    batch = send_to_device(batch, self.accelerator.device)
                with self._accumulation_context(is_boundary):
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]

                    # TODO. add duration predictor training
                    if self.duration_predictor is not None and self.accelerator.is_local_main_process:
                        dur_loss = self.duration_predictor(mel_spec, lens=batch.get("durations"))
                        self.accelerator.log({"duration loss": dur_loss.item()}, step=global_update)

                    if self.global_masked_mean:
                        loss_sum, _, cond, pred = self.model(
                            mel_spec,
                            text=text_inputs,
                            lens=mel_lengths,
                            noise_scheduler=self.noise_scheduler,
                            rand_span_mask=rand_span_mask,
                            return_loss_components=True,
                        )
                        window_loss_sum += loss_sum.detach()
                        loss = loss_sum * loss_scale
                    else:
                        loss, cond, pred = self.model(
                            mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=self.noise_scheduler
                        )
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.global_masked_mean:
                    consumed_batches += 1
                update_completed = self.accelerator.sync_gradients and (
                    not self.global_masked_mean or not self.accelerator.optimizer_step_was_skipped
                )
                logged_loss = loss.detach()
                if self.global_masked_mean:
                    if self.accelerator.sync_gradients:
                        global_loss_sum = self._reduce_global_masked_value(window_loss_sum)
                        logged_loss = global_loss_sum / global_loss_count
                        window_loss_sum.zero_()
                    else:
                        logged_loss = None

                if update_completed:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(update=str(global_update), loss=logged_loss.item())

                if logged_loss is not None and self.accelerator.is_local_main_process:
                    self.accelerator.log(
                        {"loss": logged_loss.item(), "lr": self.scheduler.get_last_lr()[0]}, step=global_update
                    )
                if logged_loss is not None and self.logger == "tensorboard" and self.accelerator.is_main_process:
                    self.writer.add_scalar("loss", logged_loss.item(), global_update)
                    self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)

                if global_update % self.last_per_updates == 0 and update_completed:
                    self.save_checkpoint(
                        global_update,
                        last=True,
                        consumed_batches=consumed_batches if self.global_masked_mean else None,
                    )

                if global_update % self.save_per_updates == 0 and update_completed:
                    self.save_checkpoint(
                        global_update,
                        consumed_batches=consumed_batches if self.global_masked_mean else None,
                    )

                    if self.log_samples and self.accelerator.is_local_main_process:
                        ref_audio_len = mel_lengths[0]
                        infer_text = [
                            text_inputs[0] + ([" "] if isinstance(text_inputs[0], list) else " ") + text_inputs[0]
                        ]
                        with torch.inference_mode(), self.accelerator.autocast():
                            generated, _ = self.accelerator.unwrap_model(self.model).sample(
                                cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
                                text=infer_text,
                                duration=ref_audio_len * 2,
                                steps=nfe_step,
                                cfg_strength=cfg_strength,
                                sway_sampling_coef=sway_sampling_coef,
                            )
                            generated = generated.to(torch.float32)
                            gen_mel_spec = generated[:, ref_audio_len:, :].permute(0, 2, 1).to(self.accelerator.device)
                            ref_mel_spec = batch["mel"][0, :, :ref_audio_len].unsqueeze(0)
                            if self.vocoder_name == "vocos":
                                gen_audio = vocoder.decode(gen_mel_spec).cpu()
                                ref_audio = vocoder.decode(ref_mel_spec).cpu()
                            elif self.vocoder_name == "bigvgan":
                                gen_audio = vocoder(gen_mel_spec).squeeze(0).cpu()
                                ref_audio = vocoder(ref_mel_spec).squeeze(0).cpu()

                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate
                        )
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_ref.wav", ref_audio, target_sample_rate
                        )
                        self.model.train()

        self.save_checkpoint(
            global_update,
            last=True,
            consumed_batches=consumed_batches if self.global_masked_mean else None,
        )

        self.accelerator.end_training()
