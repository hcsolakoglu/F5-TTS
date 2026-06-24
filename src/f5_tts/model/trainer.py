from __future__ import annotations

import gc
import math
import os
import time

import torch
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
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
        metrics_enabled: bool = False,
        metrics_log_every: int = 100,
        metrics_warmup_updates: int = 1,
        metrics_sync_cuda: bool = True,
        metrics_include_memory: bool = True,
        compile_enabled: bool = False,
        compile_target: str = "cfm_loss_core",
        compile_backend: str | None = "inductor",
        compile_mode: str | None = None,
        compile_fullgraph: bool = False,
        compile_dynamic: bool | None = None,
        compile_fallback_to_eager: bool = True,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        if logger == "wandb" and not wandb.api.api_key:
            logger = None
        self.log_samples = log_samples

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
        self.metrics_enabled = metrics_enabled
        self.metrics_log_every = max(1, int(metrics_log_every))
        self.metrics_warmup_updates = max(0, int(metrics_warmup_updates))
        self.metrics_sync_cuda = metrics_sync_cuda
        self.metrics_include_memory = metrics_include_memory
        self.compile_enabled = compile_enabled
        self.compile_target = compile_target
        self.compile_backend = compile_backend
        self.compile_mode = compile_mode
        self.compile_fullgraph = compile_fullgraph
        self.compile_dynamic = compile_dynamic
        self.compile_fallback_to_eager = compile_fallback_to_eager
        self.compile_active = False
        self.compile_setup_time = 0.0
        self.compile_fallback_active = False
        self.compile_first_forward_time = 0.0
        self._compile_first_forward_pending = False

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate, fused=True)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self._configure_compile()

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def _sync_metrics_device(self):
        if (
            self.metrics_enabled
            and self.metrics_sync_cuda
            and self.accelerator.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(self.accelerator.device)

    def _batch_training_metrics(self, mel_lengths, update_time, padded_frames=None):
        mel_lengths = mel_lengths.detach().float().cpu()
        batch_size = int(mel_lengths.numel())
        total_frames = float(mel_lengths.sum())
        max_frames = float(mel_lengths.max())
        mean_frames = float(mel_lengths.mean())
        p95_frames = float(torch.quantile(mel_lengths, 0.95))
        padded_frames = batch_size * max_frames if padded_frames is None else float(padded_frames)
        padding_ratio = 1.0 - (total_frames / padded_frames) if padded_frames > 0 else 0.0
        metrics = {
            "train/batch_size": batch_size,
            "train/local_samples_per_update": batch_size,
            "train/effective_batch_size": batch_size * self.accelerator.num_processes,
            "train/batch_total_frames": total_frames,
            "train/batch_mean_frames": mean_frames,
            "train/batch_max_frames": max_frames,
            "train/batch_p95_frames": p95_frames,
            "train/padding_ratio": padding_ratio,
        }
        if update_time > 0:
            metrics["train/samples_per_s"] = batch_size * self.accelerator.num_processes / update_time
            metrics["train/frames_per_s"] = total_frames * self.accelerator.num_processes / update_time
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            mel_spec = getattr(unwrapped_model, "mel_spec", None)
            hop_length = getattr(mel_spec, "hop_length", None)
            target_sample_rate = getattr(mel_spec, "target_sample_rate", None)
            if hop_length and target_sample_rate:
                audio_seconds = total_frames * hop_length / target_sample_rate
                metrics["train/audio_seconds_per_s"] = (
                    audio_seconds * self.accelerator.num_processes / update_time
                )
        return metrics

    def _memory_training_metrics(self):
        if not (
            self.metrics_include_memory and self.accelerator.device.type == "cuda" and torch.cuda.is_available()
        ):
            return {}
        device = self.accelerator.device
        return {
            "train/gpu_allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
            "train/gpu_reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
            "train/gpu_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        }

    def _log_training_metrics(self, metrics, step):
        if not (self.metrics_enabled and self.accelerator.is_local_main_process):
            return
        self.accelerator.log(metrics, step=step)
        if self.logger == "tensorboard" and self.accelerator.is_main_process and self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, step)

    def _configure_compile(self):
        if not self.compile_enabled:
            return
        if self.compile_target not in {"cfm_loss_core", "model"}:
            raise ValueError(
                f"Unsupported compile target: {self.compile_target}. Supported target: cfm_loss_core"
            )
        if not hasattr(torch, "compile"):
            if self.compile_fallback_to_eager:
                if self.is_main:
                    print("torch.compile is unavailable; falling back to eager training.")
                self.compile_fallback_active = True
                return
            raise RuntimeError("torch.compile is unavailable in this PyTorch build")

        compile_kwargs = {
            "backend": self.compile_backend,
            "mode": self.compile_mode,
            "fullgraph": self.compile_fullgraph,
            "dynamic": self.compile_dynamic,
        }
        compile_kwargs = {key: value for key, value in compile_kwargs.items() if value is not None}

        compile_start = time.perf_counter()
        try:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            compile_training_core = getattr(unwrapped_model, "compile_training_core", None)
            if compile_training_core is None:
                raise TypeError("The training model does not expose compile_training_core()")
            compile_training_core(
                fallback_to_eager=self.compile_fallback_to_eager,
                **compile_kwargs,
            )
            self.compile_active = True
            self._compile_first_forward_pending = True
        except Exception as exc:
            if not self.compile_fallback_to_eager:
                raise
            if self.is_main:
                print(f"torch.compile setup failed; falling back to eager training. Error: {exc}")
            clear_training_compile = getattr(self.accelerator.unwrap_model(self.model), "clear_training_compile", None)
            if clear_training_compile is not None:
                clear_training_compile()
            self.compile_active = False
            self.compile_fallback_active = True
        finally:
            self.compile_setup_time = time.perf_counter() - compile_start

        if self.compile_active and self.is_main:
            print(
                "torch.compile enabled "
                f"(target={self.compile_target}, backend={self.compile_backend}, mode={self.compile_mode}, "
                f"fullgraph={self.compile_fullgraph}, dynamic={self.compile_dynamic})"
            )

    def _forward_model(self, *args, **kwargs):
        measure_first_forward = self.metrics_enabled and self._compile_first_forward_pending
        if measure_first_forward:
            self._sync_metrics_device()
            first_forward_start = time.perf_counter()
        result = self.model(*args, **kwargs)
        if measure_first_forward:
            self._sync_metrics_device()
            self.compile_first_forward_time = time.perf_counter() - first_forward_start
            self._compile_first_forward_pending = False
        if self.compile_active:
            compile_state = getattr(self.accelerator.unwrap_model(self.model), "training_compile_state", None)
            if compile_state is not None and compile_state["fallback_active"]:
                if self.is_main:
                    print(
                        "torch.compile runtime failed inside the CFM loss core; "
                        f"continuing eagerly with the same prepared tensors. Error: {compile_state['error']}"
                    )
                self.compile_active = False
                self.compile_fallback_active = True
        return result

    def save_checkpoint(self, update, last=False):
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
                persistent_workers=num_workers > 0,
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
                persistent_workers=num_workers > 0,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")

        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_updates = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        # otherwise by default with split_batches=False, warmup steps change with num_processes
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        decay_updates = total_updates - warmup_updates
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual multi_gpu updates = single_gpu updates / gpu nums
        start_update = self.load_checkpoint()
        global_update = start_update
        if (
            self.metrics_enabled
            and self.metrics_include_memory
            and self.accelerator.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.reset_peak_memory_stats(self.accelerator.device)

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        metrics_window = {
            "data_wait_time": 0.0,
            "forward_time": 0.0,
            "backward_time": 0.0,
            "optimizer_time": 0.0,
            "scheduler_time": 0.0,
            "zero_grad_time": 0.0,
            "checkpoint_time": 0.0,
            "sampling_time": 0.0,
            "microbatches": 0,
            "padded_frames": 0,
            "loss_sum": 0.0,
            "lengths": [],
        }
        metrics_update_start = None

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

            data_wait_start = time.perf_counter()
            for batch in current_dataloader:
                metrics_active = self.metrics_enabled
                if metrics_active:
                    self._sync_metrics_device()
                    data_wait_time = time.perf_counter() - data_wait_start
                    if metrics_update_start is None:
                        metrics_update_start = data_wait_start
                    step_start = time.perf_counter()
                else:
                    data_wait_time = 0.0
                    step_start = 0.0
                forward_time = 0.0
                backward_time = 0.0
                optimizer_time = 0.0
                scheduler_time = 0.0
                zero_grad_time = 0.0
                checkpoint_time = 0.0
                sampling_time = 0.0
                grad_norm = None

                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]
                    if metrics_active:
                        metrics_window["data_wait_time"] += data_wait_time
                        metrics_window["microbatches"] += 1
                        metrics_window["padded_frames"] += int(mel_lengths.numel()) * int(mel_spec.shape[1])
                        metrics_window["lengths"].append(mel_lengths.detach().float().cpu())

                    # TODO. add duration predictor training
                    if metrics_active:
                        self._sync_metrics_device()
                        forward_start = time.perf_counter()
                    if self.duration_predictor is not None and self.accelerator.is_local_main_process:
                        dur_loss = self.duration_predictor(mel_spec, lens=batch.get("durations"))
                        self.accelerator.log({"duration loss": dur_loss.item()}, step=global_update)

                    loss, cond, pred = self._forward_model(
                        mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=self.noise_scheduler
                    )
                    if metrics_active:
                        self._sync_metrics_device()
                        forward_time = time.perf_counter() - forward_start
                        metrics_window["forward_time"] += forward_time
                        metrics_window["loss_sum"] += float(loss.detach().float().cpu())

                    if metrics_active:
                        self._sync_metrics_device()
                        backward_start = time.perf_counter()
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    if metrics_active:
                        self._sync_metrics_device()
                        backward_time = time.perf_counter() - backward_start
                        metrics_window["backward_time"] += backward_time

                    if metrics_active:
                        self._sync_metrics_device()
                        optimizer_start = time.perf_counter()
                    self.optimizer.step()
                    if metrics_active:
                        self._sync_metrics_device()
                        optimizer_time = time.perf_counter() - optimizer_start
                        metrics_window["optimizer_time"] += optimizer_time

                    if metrics_active:
                        self._sync_metrics_device()
                        scheduler_start = time.perf_counter()
                    self.scheduler.step()
                    if metrics_active:
                        self._sync_metrics_device()
                        scheduler_time = time.perf_counter() - scheduler_start
                        metrics_window["scheduler_time"] += scheduler_time

                    if metrics_active:
                        self._sync_metrics_device()
                        zero_grad_start = time.perf_counter()
                    self.optimizer.zero_grad()
                    if metrics_active:
                        self._sync_metrics_device()
                        zero_grad_time = time.perf_counter() - zero_grad_start
                        metrics_window["zero_grad_time"] += zero_grad_time
                        step_time = time.perf_counter() - step_start
                    else:
                        step_time = 0.0

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(update=str(global_update), loss=loss.item())

                if self.accelerator.is_local_main_process:
                    self.accelerator.log(
                        {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}, step=global_update
                    )
                if self.logger == "tensorboard" and self.accelerator.is_main_process:
                    self.writer.add_scalar("loss", loss.item(), global_update)
                    self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)

                if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    if metrics_active:
                        self._sync_metrics_device()
                        checkpoint_start = time.perf_counter()
                    self.save_checkpoint(global_update, last=True)
                    if metrics_active:
                        self._sync_metrics_device()
                        checkpoint_elapsed = time.perf_counter() - checkpoint_start
                        checkpoint_time += checkpoint_elapsed
                        metrics_window["checkpoint_time"] += checkpoint_elapsed

                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    if metrics_active:
                        self._sync_metrics_device()
                        checkpoint_start = time.perf_counter()
                    self.save_checkpoint(global_update)
                    if metrics_active:
                        self._sync_metrics_device()
                        checkpoint_elapsed = time.perf_counter() - checkpoint_start
                        checkpoint_time += checkpoint_elapsed
                        metrics_window["checkpoint_time"] += checkpoint_elapsed

                    if self.log_samples and self.accelerator.is_local_main_process:
                        if metrics_active:
                            self._sync_metrics_device()
                            sampling_start = time.perf_counter()
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
                        if metrics_active:
                            self._sync_metrics_device()
                            sampling_time += time.perf_counter() - sampling_start
                            metrics_window["sampling_time"] += sampling_time

                if metrics_active and self.accelerator.sync_gradients:
                    self._sync_metrics_device()
                    update_time = time.perf_counter() - metrics_update_start
                    should_log_metrics = (
                        global_update > start_update + self.metrics_warmup_updates
                        and global_update % self.metrics_log_every == 0
                    )
                    if should_log_metrics:
                        compute_time = sum(
                            metrics_window[name]
                            for name in [
                                "forward_time",
                                "backward_time",
                                "optimizer_time",
                                "scheduler_time",
                                "zero_grad_time",
                            ]
                        )
                        metrics = {
                            "train/step_time_s": update_time,
                            "train/update_time_s": update_time,
                            "train/compute_time_s": compute_time,
                            "train/data_wait_time_s": metrics_window["data_wait_time"],
                            "train/forward_time_s": metrics_window["forward_time"],
                            "train/backward_time_s": metrics_window["backward_time"],
                            "train/optimizer_time_s": metrics_window["optimizer_time"],
                            "train/scheduler_time_s": metrics_window["scheduler_time"],
                            "train/zero_grad_time_s": metrics_window["zero_grad_time"],
                            "train/checkpoint_time_s": metrics_window["checkpoint_time"],
                            "train/sampling_time_s": metrics_window["sampling_time"],
                            "train/microbatches_per_update": metrics_window["microbatches"],
                            "train/loss": metrics_window["loss_sum"] / metrics_window["microbatches"],
                            "train/lr": self.scheduler.get_last_lr()[0],
                            "train/compile_enabled": float(self.compile_active),
                            "train/compile_fallback_active": float(self.compile_fallback_active),
                            "train/compile_setup_time_s": self.compile_setup_time,
                            "train/compile_first_forward_time_s": self.compile_first_forward_time,
                        }
                        if grad_norm is not None:
                            metrics["train/grad_norm"] = float(grad_norm.detach().float().cpu())
                        metrics.update(
                            self._batch_training_metrics(
                                torch.cat(metrics_window["lengths"]),
                                update_time,
                                padded_frames=metrics_window["padded_frames"],
                            )
                        )
                        metrics.update(self._memory_training_metrics())
                        self._log_training_metrics(metrics, global_update)

                if metrics_active:
                    data_wait_start = time.perf_counter()
                    if self.accelerator.sync_gradients:
                        metrics_window = {
                            "data_wait_time": 0.0,
                            "forward_time": 0.0,
                            "backward_time": 0.0,
                            "optimizer_time": 0.0,
                            "scheduler_time": 0.0,
                            "zero_grad_time": 0.0,
                            "checkpoint_time": 0.0,
                            "sampling_time": 0.0,
                            "microbatches": 0,
                            "padded_frames": 0,
                            "loss_sum": 0.0,
                            "lengths": [],
                        }
                        metrics_update_start = None

        self.save_checkpoint(global_update, last=True)

        self.accelerator.end_training()
