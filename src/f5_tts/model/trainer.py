from __future__ import annotations

import gc
import inspect
import math
import operator
import os
import random
from contextlib import contextmanager, nullcontext
from typing import Any, cast

import torch
import torch.distributed as dist
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, send_to_device
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler
from tqdm import tqdm

from f5_tts.model import CFM
from f5_tts.model.cfm import _is_cuda_oom
from f5_tts.model.dataset import DynamicBatchSampler, collate_fn
from f5_tts.model.utils import default, exists


def _resolve_compile_dynamic(
    requested: bool | None,
    compile_fn: Any,
) -> bool | None:
    """Resolve ``dynamic=None`` across torch.compile API generations.

    PyTorch 2.0 exposed ``dynamic=False`` as the callable default, while newer
    releases use ``None`` for automatic dynamic-shape detection. Preserve an
    explicit user choice on every release. For the auto setting, opt into
    ``dynamic=True`` only when the callable's published signature proves that
    omission would select the legacy static default.

    Signature inspection follows ``functools.wraps`` metadata. If a custom or
    monkeypatched callable has no trustworthy signature, leave the argument
    unset rather than guessing at an API it may not support.
    """
    if requested is not None:
        return requested

    try:
        dynamic_parameter = inspect.signature(compile_fn, follow_wrapped=True).parameters.get("dynamic")
    except Exception:
        return None

    if dynamic_parameter is not None and dynamic_parameter.default is False:
        return True
    return None


try:
    from torch.optim.optimizer import _get_fused_kernels_supported_devices as _fused_devices

    FUSED_ADAMW_DEVICE_TYPES = frozenset(_fused_devices())
except ImportError:
    FUSED_ADAMW_DEVICE_TYPES = frozenset(("cuda",))


class _EpochRandomSampler(Sampler[int]):
    """Random sampler whose order is stable and addressable by epoch."""

    def __init__(self, data_source, seed: int):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_source), generator=generator).tolist()
        return iter(indices)

    def __len__(self):
        return len(self.data_source)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)


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
        compile_enabled: bool = False,
        compile_backend: str | None = "inductor",
        compile_target: str = "cfm_loss_core",
        compile_mode: str | None = None,
        compile_fullgraph: bool = False,
        compile_dynamic: bool | None = None,
        compile_fallback_to_eager: bool = True,
        global_masked_mean: bool = False,
    ):
        grad_accumulation_steps = self._validate_gradient_accumulation_steps(grad_accumulation_steps)
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
        self.global_masked_mean = global_masked_mean
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self._validate_global_masked_mean_config()

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

        # mel vocoder config
        self.vocoder_name = mel_spec_type
        self.is_local_vocoder = is_local_vocoder
        self.local_vocoder_path = local_vocoder_path

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor

        # torch.compile configuration (optional, default-off)
        self.compile_enabled = compile_enabled
        self.compile_backend = compile_backend
        self.compile_target = compile_target
        self.compile_mode = compile_mode
        self.compile_fullgraph = compile_fullgraph
        self.compile_dynamic = compile_dynamic
        self.compile_fallback_to_eager = compile_fallback_to_eager
        self.compile_active = False
        self.compile_fallback_active = False
        self._unwrapped_model = None  # cached after accelerator.prepare

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            use_fused = self.accelerator.device.type in FUSED_ADAMW_DEVICE_TYPES
            self.optimizer = AdamW(model.parameters(), lr=learning_rate, fused=use_fused)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self._unwrapped_model = self.accelerator.unwrap_model(self.model)
        self._validate_global_masked_mean_backend()
        self._configure_compile()

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @staticmethod
    def _validate_gradient_accumulation_steps(value):
        """Normalize and validate the count before constructing Accelerate."""
        if isinstance(value, bool):
            raise ValueError("gradient_accumulation_steps must be a positive integer")
        try:
            normalized = operator.index(value)
        except TypeError as exc:
            raise ValueError("gradient_accumulation_steps must be a positive integer") from exc
        if normalized < 1:
            raise ValueError("gradient_accumulation_steps must be a positive integer")
        return normalized

    @staticmethod
    def _normalize_checkpoint_args(update, consumed_updates=None, last=False):
        """Keep the pre-cursor ``save_checkpoint(update, last=False)`` call valid."""
        if isinstance(consumed_updates, bool):
            if last:
                raise TypeError("last was provided twice to save_checkpoint")
            last = consumed_updates
            consumed_updates = update
        if consumed_updates is None:
            consumed_updates = update
        return update, consumed_updates, last

    @staticmethod
    def _set_dataloader_epoch(dataloader, epoch: int):
        """Set epoch on Accelerate's wrapper and every nested sampler."""
        pending = [dataloader]
        seen = set()
        while pending:
            current = pending.pop(0)
            if current is None or id(current) in seen:
                continue
            seen.add(id(current))
            set_epoch = getattr(current, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch)
            for attribute in ("batch_sampler", "sampler"):
                nested = getattr(current, attribute, None)
                if nested is not None:
                    pending.append(nested)

    def _prepare_global_masked_mean_dataloader(self, dataloader):
        """Prepare GMM data without Accelerate's duplicate-batch padding.

        ``even_batches=True`` can replay samples when the number of batches is
        not divisible by the process count. GMM has no validity mask for such
        synthetic samples, so reject an incomplete process group rather than
        silently biasing the denominator or dropping a tail batch.
        """
        self.accelerator.even_batches = False
        if getattr(self.accelerator, "dispatch_batches", False):
            raise ValueError(
                "global_masked_mean=True requires dispatch_batches=False so accumulation windows "
                "can remain on the host until each microbatch is processed."
            )
        num_processes = int(getattr(self.accelerator, "num_processes", 1))
        split_batches = bool(getattr(self.accelerator, "split_batches", False))
        if num_processes > 1 and not split_batches and len(dataloader) % num_processes:
            raise ValueError(
                "global_masked_mean requires the prepared dataloader batch count to be divisible by "
                f"the process count ({num_processes}); received {len(dataloader)} batches. "
                "Increase/drop the final batch explicitly instead of allowing duplicate padding."
            )
        prepared = self.accelerator.prepare_data_loader(dataloader, device_placement=False)
        self._validate_global_masked_dataloader(prepared)
        return prepared

    def _capture_rng_states(self):
        """Capture per-process RNG state so resumed training does not replay masks."""
        local_state = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            local_state["cuda"] = torch.cuda.get_rng_state_all()
        dataloader_generator = getattr(self, "_train_dataloader_generator", None)
        if dataloader_generator is not None:
            local_state["dataloader_generator"] = dataloader_generator.get_state()

        if dist.is_available() and dist.is_initialized():
            states = [None] * dist.get_world_size()
            dist.all_gather_object(states, local_state)
            return states
        return [local_state]

    def _restore_rng_states(self, states):
        """Restore the current process RNG state when checkpoint topology matches."""
        if states is None:
            return
        expected_processes = int(getattr(self.accelerator, "num_processes", 1))
        if len(states) != expected_processes:
            raise ValueError(
                "checkpoint RNG state was saved for "
                f"{len(states)} processes, but the current run uses {expected_processes}; "
                "resume with the same process count or start from model weights only."
            )
        process_index = int(getattr(self.accelerator, "process_index", 0))
        state = states[process_index]
        random.setstate(state["python"])
        torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            cuda_state = state["cuda"]
            if isinstance(cuda_state, (list, tuple)):
                torch.cuda.set_rng_state_all(list(cuda_state))
            else:  # backward compatibility with checkpoints containing one CUDA state
                torch.cuda.set_rng_state(cuda_state)
        dataloader_generator = getattr(self, "_train_dataloader_generator", None)
        if dataloader_generator is not None and "dataloader_generator" in state:
            dataloader_generator.set_state(state["dataloader_generator"])

    # ---- torch.compile support (optional, default-off) ----

    def _validate_global_masked_mean_config(self):
        """Reject incompatible distributed configurations before model setup."""
        if not self.global_masked_mean:
            return
        if self.accelerator.distributed_type in (DistributedType.FSDP, DistributedType.MEGATRON_LM):
            raise NotImplementedError(
                f"global_masked_mean=True does not support {self.accelerator.distributed_type.value}; "
                "the current F5-TTS EMA and checkpoint path is not compatible with sharded model parameters."
            )
        if self.accelerator.distributed_type != DistributedType.DEEPSPEED:
            return

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage = int(getattr(deepspeed_plugin, "zero_stage", 0))
        if zero_stage != 0:
            raise NotImplementedError(
                "global_masked_mean currently supports DeepSpeed ZeRO stage 0 only. "
                "F5-TTS checkpointing saves optimizer state on rank 0 and its EMA requires full parameters, "
                "so enabling sharded ZeRO stages would produce incomplete checkpoints or invalid EMA updates."
            )

        configured_gas = deepspeed_plugin.get_value("gradient_accumulation_steps")
        if configured_gas not in (None, "auto") and int(configured_gas) != self.grad_accumulation_steps:
            raise ValueError(
                "DeepSpeed and Trainer gradient accumulation steps must match for global_masked_mean=True."
            )

        deepspeed_clip = deepspeed_plugin.get_value("gradient_clipping")
        expected_clip = self.max_grad_norm if self.max_grad_norm > 0 else 0.0
        if deepspeed_clip in (None, "auto"):
            deepspeed_plugin.deepspeed_config["gradient_clipping"] = expected_clip
        elif float(deepspeed_clip) != float(expected_clip):
            raise ValueError(
                "DeepSpeed gradient_clipping must match Trainer.max_grad_norm for global_masked_mean=True "
                "because DeepSpeed steps inside Accelerator.backward()."
            )

    def _validate_global_masked_mean_backend(self):
        """Validate the prepared DeepSpeed engine required by manual boundaries."""
        if not self.global_masked_mean or self.accelerator.distributed_type != DistributedType.DEEPSPEED:
            return

        set_boundary = getattr(self.model, "set_gradient_accumulation_boundary", None)
        if not callable(set_boundary):
            raise NotImplementedError(
                "global_masked_mean=True requires DeepSpeedEngine.set_gradient_accumulation_boundary()."
            )

        # One denominator is correct only when every process is a replica in
        # the same data-parallel group. F5-TTS does not configure these
        # model-parallel modes, but reject externally supplied combinations
        # rather than silently over-counting their replicated samples.
        unsupported_modes = []
        if getattr(self.model, "sequence_parallel_size", 1) != 1:
            unsupported_modes.append("sequence parallelism")
        if getattr(self.model, "mp_world_size", 1) != 1:
            unsupported_modes.append("model parallelism")
        if getattr(self.model, "pipeline_parallelism", False):
            unsupported_modes.append("pipeline parallelism")
        if getattr(self.model, "has_moe_layers", False):
            unsupported_modes.append("MoE expert parallelism")
        if getattr(self.model, "dp_world_size", self.accelerator.num_processes) != self.accelerator.num_processes:
            unsupported_modes.append("a non-world data-parallel group")
        if unsupported_modes:
            modes = ", ".join(unsupported_modes)
            raise NotImplementedError(f"global_masked_mean=True does not yet support DeepSpeed {modes}.")

        accelerator_gas = int(self.accelerator.gradient_accumulation_steps)
        if accelerator_gas != self.grad_accumulation_steps:
            raise ValueError(
                "Accelerator and Trainer gradient accumulation steps must match for global_masked_mean=True."
            )
        deepspeed_gas = getattr(self.model, "gradient_accumulation_steps", None)
        if callable(deepspeed_gas) and int(deepspeed_gas()) != self.grad_accumulation_steps:
            raise ValueError(
                "DeepSpeed and Trainer gradient accumulation steps must match for global_masked_mean=True."
            )

    def _configure_compile(self):
        """Set up torch.compile on the CFM loss core if enabled.

        Compilation is lazy (triggered by the first real training batch); there is no
        synthetic preflight, which avoids shape/vocab mismatches with arbitrary models.
        In DDP (num_processes > 1) runtime fallback is disabled: a per-rank eager fallback
        would desynchronise the gradient all-reduce. Setup-time fallback is still allowed
        and synchronised across ranks via ``_sync_compile_setup_ddp``.

        Strict mode (``compile_fallback_to_eager=False``): a rank-local setup exception is
        deferred until *every* rank has participated in the status collective
        (``_sync_compile_setup_ddp``), then all ranks raise coherently -- the failing rank
        re-raises its original exception, peers raise a collective-failure error. This
        prevents a rank-divergent setup failure from stranding healthy ranks in an
        unmatched collective. Single-process preserves the original immediate re-raise.
        """
        if not self.compile_enabled:
            return

        if not hasattr(torch, "compile"):
            # torch.compile unavailable: participate in the collective before raising.
            self.compile_active = False
            if self.compile_fallback_to_eager:
                self.compile_fallback_active = True
                if self.is_main:
                    print("torch.compile is unavailable; falling back to eager training.")
            self._sync_compile_setup_ddp(local_failed=True)
            if not self.compile_fallback_to_eager:
                raise RuntimeError("torch.compile is unavailable in this PyTorch build")
            return

        effective_compile_dynamic = _resolve_compile_dynamic(self.compile_dynamic, torch.compile)
        compile_kwargs = {
            "backend": self.compile_backend,
            "mode": self.compile_mode,
            "fullgraph": self.compile_fullgraph,
            "dynamic": effective_compile_dynamic,
        }
        compile_kwargs = {k: v for k, v in compile_kwargs.items() if v is not None}

        # Under DDP, disable runtime fallback so a compile failure raises on all ranks
        # (collective-safe) instead of one rank silently going eager.
        runtime_fallback = self.compile_fallback_to_eager and self.accelerator.num_processes <= 1

        setup_exc: Exception | None = None
        try:
            compile_fn = getattr(self._unwrapped_model, "compile_training_core", None)
            if compile_fn is None:
                raise TypeError("The training model does not expose compile_training_core()")
            compile_target = getattr(self, "compile_target", "cfm_loss_core")
            compile_fn(target=compile_target, runtime_fallback=runtime_fallback, **compile_kwargs)
            self.compile_active = True
        except Exception as exc:
            # Defer the strict-mode re-raise until after the collective so every rank
            # participates before any rank raises. Store the exception; in fallback mode
            # also mark/clear for eager continuation. In strict mode compile_fallback_active
            # stays False (we are not falling back — we are about to raise).
            setup_exc = exc
            self.compile_active = False
            if self.compile_fallback_to_eager:
                self.compile_fallback_active = True
                if self.is_main:
                    print(f"torch.compile setup failed; falling back to eager training. Error: {exc}")
                clear_fn = getattr(self._unwrapped_model, "clear_training_compile", None)
                if clear_fn is not None:
                    clear_fn()

        # Every rank participates in the status exchange before any rank raises.
        any_failed = self._sync_compile_setup_ddp(local_failed=setup_exc is not None)

        # Strict mode: raise coherently on ALL ranks after the collective.
        if any_failed and not self.compile_fallback_to_eager:
            if setup_exc is not None:
                raise setup_exc
            raise RuntimeError("torch.compile setup failed on at least one rank; aborting (strict mode).")

        if self.compile_active and self.is_main:
            compile_target = getattr(self, "compile_target", "cfm_loss_core")
            print(
                f"torch.compile enabled (target={compile_target}, backend={self.compile_backend}, "
                f"mode={self.compile_mode}, fullgraph={self.compile_fullgraph}, "
                f"dynamic={effective_compile_dynamic})"
            )
            if self.accelerator.num_processes > 1 and self.compile_fallback_to_eager:
                print("DDP detected: runtime compile fallback disabled (errors will raise on all ranks).")

    def _sync_compile_setup_ddp(self, local_failed: bool = False) -> bool:
        """Exchange setup-time compile status across DDP ranks via a single reduce.

        Every rank participates before any rank raises, so a rank-divergent setup failure
        cannot strand peers in an unmatched collective. ``local_failed`` carries strict-
        mode failures (where ``compile_fallback_active`` is not set because we are about to
        raise, not fall back). Returns ``True`` if any rank reported a setup failure. In
        fallback mode, all ranks switch to eager so the gradient all-reduce stays
        consistent. In strict mode the caller raises after this returns (see
        ``_configure_compile``).

        Uses ``accelerator.reduce(..., reduction='max')`` on a 0/1 flag rather than a raw
        ``torch.distributed.all_reduce`` so the collective goes through Accelerate's
        dispatch layer (handles DeepSpeed/FSDP process groups correctly). Single-process
        returns the local flag directly with no collective.

        This covers setup only. A compile failure that first surfaces at *runtime* on a
        single rank is deliberately fatal for that rank (runtime fallback is disabled under
        DDP, see ``_configure_compile``): the alternative, one rank silently going eager,
        would desynchronise gradients and corrupt training silently. A dying rank aborts
        the job through the launcher, which is the intended fail-fast behaviour -- it is
        not a graceful, collective-safe recovery, and no per-step collective is added to
        make it one because that cost would be paid by every healthy step.
        """
        failed = local_failed or self.compile_fallback_active
        if self.accelerator.num_processes <= 1:
            return failed
        flag = torch.tensor(1.0 if failed else 0.0, device=self.accelerator.device)
        flag = cast(torch.Tensor, self.accelerator.reduce(flag, reduction="max"))
        any_failed = float(flag) > 0.0
        if any_failed and self.compile_active:
            if self.is_main:
                print("torch.compile setup failed on at least one rank; switching all ranks to eager.")
            clear_fn = getattr(self._unwrapped_model, "clear_training_compile", None)
            if clear_fn is not None:
                clear_fn()
            self.compile_active = False
        if any_failed:
            self.compile_fallback_active = True
        return any_failed

    def _check_compile_runtime_fallback(self):
        """Detect a runtime compile failure surfaced by the CFM module and update trainer state."""
        if not self.compile_active:
            return
        state = getattr(self._unwrapped_model, "training_compile_state", None)
        if state is not None and state["fallback_active"]:
            if self.is_main:
                print(f"torch.compile runtime failed; continuing eagerly. Error: {state['error']}")
            self.compile_active = False
            self.compile_fallback_active = True

    def _reduce_global_masked_value(self, local_value: torch.Tensor) -> torch.Tensor:
        """Sum a detached scalar over the supported data-parallel group."""
        local_value = local_value.detach().to(device=self.accelerator.device)
        return cast(torch.Tensor, self.accelerator.reduce(local_value, reduction="sum"))

    def _iter_global_masked_mean_batches(self, dataloader):
        """Yield pre-normalized accumulation windows without retaining graphs."""
        dataloader_iter = iter(dataloader)
        sample_mask = getattr(self._unwrapped_model, "sample_training_mask", None)
        if not callable(sample_mask):
            raise TypeError("The training model does not expose sample_training_mask().")

        while True:
            batches = []

            for _ in range(self.grad_accumulation_steps):
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    break
                batches.append(batch)

            if not batches:
                return

            window = []
            local_loss_denom = torch.zeros((), device=self.accelerator.device, dtype=torch.int64)
            local_error = None
            local_oom = None
            local_exception = None
            for batch in batches:
                batch_error = self._validate_global_masked_batch(batch)
                rand_span_mask = None
                if batch_error is None:
                    try:
                        mel = batch["mel"]
                        rand_span_mask = sample_mask(batch["mel_lengths"], mel.shape[-1])
                        batch_loss_denom = rand_span_mask.sum(dtype=torch.int64) * mel.shape[1]
                        local_loss_denom = local_loss_denom + batch_loss_denom
                    except Exception as exc:
                        if local_exception is None:
                            local_exception = exc
                        if _is_cuda_oom(exc):
                            local_oom = exc
                        batch_error = f"training-mask preparation failed: {exc}"
                if batch_error is not None and local_error is None:
                    local_error = batch_error
                window.append((batch, rand_span_mask))

            local_stats = torch.stack(
                (
                    local_loss_denom,
                    torch.tensor(int(local_error is not None), device=self.accelerator.device, dtype=torch.int64),
                    torch.tensor(int(local_oom is not None), device=self.accelerator.device, dtype=torch.int64),
                )
            )
            global_stats = self._reduce_global_masked_value(local_stats)
            global_denom_value, global_error_count, global_oom_count = global_stats.cpu().tolist()
            if global_error_count:
                if global_oom_count:
                    if local_oom is not None:
                        raise local_oom
                    raise RuntimeError(
                        "global_masked_mean rejected an accumulation window because another rank reported "
                        "CUDA out of memory during training-mask preparation."
                    )
                detail = local_error or "another rank reported malformed training input"
                message = f"global_masked_mean rejected an accumulation window: {detail}"
                if local_exception is not None:
                    raise ValueError(message) from local_exception
                raise ValueError(message)
            if global_denom_value == 0:
                raise RuntimeError(
                    "global_masked_mean produced an empty accumulation window; "
                    "no optimizer update can be defined without masked elements."
                )
            global_loss_denom = global_stats[0]
            loss_scale = global_loss_denom.to(dtype=torch.float32).reciprocal()
            loss_scale = loss_scale * (self.grad_accumulation_steps * self.accelerator.num_processes)

            for index, (batch, rand_span_mask) in enumerate(window):
                assert rand_span_mask is not None
                is_boundary = index == len(window) - 1
                yield batch, rand_span_mask, loss_scale, global_loss_denom, is_boundary

    def _validate_global_masked_batch(self, batch) -> str | None:
        """Return a rank-local validation error without raising before collectives."""
        if not isinstance(batch, dict):
            return f"expected a batch dictionary, received {type(batch).__name__}"
        if "mel" not in batch or "mel_lengths" not in batch or "text" not in batch:
            return "batch must contain mel, mel_lengths, and text"

        mel = batch["mel"]
        mel_lengths = batch["mel_lengths"]
        if not isinstance(mel, torch.Tensor) or mel.ndim != 3:
            return "mel must be a rank-3 tensor shaped [batch, channels, frames]"
        if not mel.is_floating_point():
            return f"mel must use a floating-point dtype, received {mel.dtype}"
        if mel.shape[0] == 0:
            return "mel batch must not be empty"
        expected_channels = getattr(self._unwrapped_model, "num_channels", mel.shape[1])
        if mel.shape[1] != expected_channels:
            return f"mel channel count must be {expected_channels}, received {mel.shape[1]}"
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
            if any(isinstance(item, list) for item in text) and getattr(
                self._unwrapped_model, "vocab_char_map", None
            ) is None:
                return "nested text token lists require a vocabulary map"
        else:
            return "text must be a tensor or list"
        return None

    @contextmanager
    def _accumulation_context(self, is_boundary: bool | None):
        """Select automatic or explicit accumulation without backend ambiguity."""
        if is_boundary is None:
            with self.accelerator.accumulate(self.model):
                yield
            return

        self.accelerator.sync_gradients = is_boundary
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            # Accelerate 0.33 always calls DeepSpeedEngine.step() from backward,
            # while newer releases propagate sync_gradients themselves. Setting
            # the public engine boundary works in both implementations and is
            # required to flush a final partial accumulation window.
            self.model.set_gradient_accumulation_boundary(is_boundary)
            sync_context = nullcontext()
        else:
            sync_context = nullcontext() if is_boundary else self.accelerator.no_sync(self.model)

        with sync_context:
            yield

    def _validate_global_masked_dataloader(self, dataloader):
        """Fail collectively if ranks would execute different forward counts."""
        local_length = torch.tensor([len(dataloader)], device=self.accelerator.device, dtype=torch.int64)
        gathered_lengths = cast(torch.Tensor, self.accelerator.gather(local_length))
        if gathered_lengths[0].item() == 0:
            raise ValueError("global_masked_mean requires a non-empty training dataloader.")
        if bool((gathered_lengths != gathered_lengths[0]).any().item()):
            lengths = gathered_lengths.cpu().tolist()
            raise RuntimeError(
                "global_masked_mean requires the same number of dataloader batches on every rank; "
                f"received per-rank lengths {lengths}."
            )

    def save_checkpoint(self, update, consumed_updates=None, last=False):
        update, consumed_updates, last = self._normalize_checkpoint_args(update, consumed_updates, last)
        rng_states = self._capture_rng_states()
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
                consumed_updates=consumed_updates,
                rng_state=rng_states,
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

    def load_checkpoint(self, *, return_cursor=False):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(filename.endswith((".pt", ".safetensors")) for filename in os.listdir(self.checkpoint_path))
        ):
            return (0, 0) if return_cursor else 0

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
            load_kwargs: dict[str, Any] = {"map_location": "cpu"}
            if "weights_only" in inspect.signature(torch.load).parameters:
                load_kwargs["weights_only"] = True
            checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", **load_kwargs)

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
            # Optimizer overflows consume an accumulation window without
            # completing an update. Older checkpoints predate that distinction
            # and advanced "update" for every consumed window.
            consumed_updates = checkpoint.get("consumed_updates", update)
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            update = 0
            consumed_updates = 0

        self._restore_rng_states(checkpoint.get("rng_state"))
        del checkpoint
        gc.collect()
        return (update, consumed_updates) if return_cursor else update

    def train(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):
        if self.log_samples:
            from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef

            vocoder = load_vocoder(
                vocoder_name=self.vocoder_name, is_local=self.is_local_vocoder, local_path=self.local_vocoder_path
            )
            target_sample_rate = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

        self._train_dataloader_generator = None
        persistent_workers = num_workers > 0

        if self.batch_size_type == "sample":
            sample_sampler = None
            if exists(resumable_with_seed):
                self._train_dataloader_generator = torch.Generator()
                self._train_dataloader_generator.manual_seed(resumable_with_seed)
                sample_sampler = _EpochRandomSampler(train_dataset, resumable_with_seed)
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=persistent_workers,
                batch_size=self.batch_size_per_gpu,
                shuffle=sample_sampler is None,
                sampler=sample_sampler,
                generator=self._train_dataloader_generator,
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
                persistent_workers=persistent_workers,
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
        if self.global_masked_mean:
            # Buffer accumulation windows on the host. A normally prepared
            # dataloader places each yielded batch on the accelerator, which
            # would retain G large mel batches and undermine accumulation's
            # memory bound. The window iterator transfers one batch at a time.
            train_dataloader = self._prepare_global_masked_mean_dataloader(train_dataloader)
            self.scheduler = self.accelerator.prepare(self.scheduler)
        else:
            train_dataloader, self.scheduler = self.accelerator.prepare(
                train_dataloader, self.scheduler
            )  # actual multi_gpu updates = single_gpu updates / gpu nums
        start_update, start_consumed_updates = self.load_checkpoint(return_cursor=True)
        global_update = start_update
        consumed_updates = start_consumed_updates

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            updates_per_epoch = math.ceil(orig_epoch_step / self.grad_accumulation_steps)
            skipped_epoch = int(start_consumed_updates // updates_per_epoch)
            updates_into_epoch = start_consumed_updates % updates_per_epoch
            skipped_batch = min(updates_into_epoch * self.grad_accumulation_steps, orig_epoch_step)
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

            self._set_dataloader_epoch(current_dataloader, epoch)

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

            loss_sum_accum = None
            for batch, rand_span_mask, loss_scale, global_loss_denom, is_boundary in training_batches:
                if self.global_masked_mean:
                    batch = send_to_device(batch, self.accelerator.device, non_blocking=True)
                with self._accumulation_context(is_boundary):
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]

                    # TODO. add duration predictor training
                    if self.duration_predictor is not None and self.accelerator.is_local_main_process:
                        dur_loss = self.duration_predictor(mel_spec, lens=batch.get("durations"))
                        self.accelerator.log({"duration loss": dur_loss.item()}, step=global_update)

                    if self.global_masked_mean:
                        assert rand_span_mask is not None
                        assert loss_scale is not None
                        loss, loss_sum, loss_denom, cond, pred = self.model(
                            mel_spec,
                            text=text_inputs,
                            lens=mel_lengths,
                            noise_scheduler=self.noise_scheduler,
                            rand_span_mask=rand_span_mask,
                            return_loss_components=True,
                        )
                        loss_sum_accum = (
                            loss_sum.detach() if loss_sum_accum is None else loss_sum_accum + loss_sum.detach()
                        )
                        self._check_compile_runtime_fallback()
                        self.accelerator.backward(loss_sum * loss_scale)
                    else:
                        loss, cond, pred = self.model(
                            mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=self.noise_scheduler
                        )
                        self._check_compile_runtime_fallback()
                        self.accelerator.backward(loss)

                    deepspeed_clips_in_backward = (
                        self.global_masked_mean and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                    )
                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients and not deepspeed_clips_in_backward:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    optimizer_step_was_skipped = bool(getattr(self.accelerator, "optimizer_step_was_skipped", False))
                    if not optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.global_masked_mean and self.accelerator.sync_gradients:
                    assert loss_sum_accum is not None
                    assert global_loss_denom is not None
                    global_loss_sum = self._reduce_global_masked_value(loss_sum_accum.to(dtype=torch.float32))
                    loss_to_log = (global_loss_sum / global_loss_denom.to(dtype=torch.float32)).item()
                    loss_sum_accum = None
                elif self.global_masked_mean:
                    loss_to_log = None
                else:
                    loss_to_log = loss.item()

                update_completed = self.accelerator.sync_gradients and not bool(
                    getattr(self.accelerator, "optimizer_step_was_skipped", False)
                )
                if self.accelerator.sync_gradients:
                    consumed_updates += 1
                    progress_bar.update(1)

                if update_completed:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                if self.accelerator.sync_gradients:
                    assert loss_to_log is not None
                    if self.global_masked_mean:
                        progress_bar.set_postfix(
                            update=str(global_update),
                            loss=loss_to_log,
                            skipped=not update_completed,
                        )
                    else:
                        progress_bar.set_postfix(update=str(global_update), loss=loss_to_log)

                should_log = not self.global_masked_mean or self.accelerator.sync_gradients
                if self.accelerator.is_local_main_process and should_log:
                    assert loss_to_log is not None
                    self.accelerator.log(
                        {"loss": loss_to_log, "lr": self.scheduler.get_last_lr()[0]}, step=global_update
                    )
                if self.logger == "tensorboard" and self.accelerator.is_main_process and should_log:
                    assert loss_to_log is not None
                    self.writer.add_scalar("loss", loss_to_log, global_update)
                    self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)

                if global_update % self.last_per_updates == 0 and update_completed:
                    self.save_checkpoint(global_update, consumed_updates, last=True)

                if global_update % self.save_per_updates == 0 and update_completed:
                    self.save_checkpoint(global_update, consumed_updates)

                    if self.log_samples and self.accelerator.is_local_main_process:
                        ref_audio_len = mel_lengths[0]
                        infer_text = [
                            text_inputs[0] + ([" "] if isinstance(text_inputs[0], list) else " ") + text_inputs[0]
                        ]
                        # Switch the unwrapped model to eval so regional compile dispatch
                        # (which keys on Module.training) routes inference through the eager
                        # path. Logged samples use cfg_infer-doubled batches and swept
                        # durations; routing them through the training compile cache would
                        # exhaust its per-code-object recompile budget and silently push new
                        # training shapes back to eager while compile is still reported active.
                        sample_model = self.accelerator.unwrap_model(self.model)
                        was_training = sample_model.training
                        sample_model.eval()
                        try:
                            with torch.inference_mode(), self.accelerator.autocast():
                                generated, _ = sample_model.sample(
                                    cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
                                    text=infer_text,
                                    duration=ref_audio_len * 2,
                                    steps=nfe_step,
                                    cfg_strength=cfg_strength,
                                    sway_sampling_coef=sway_sampling_coef,
                                )
                                generated = generated.to(torch.float32)
                                gen_mel_spec = (
                                    generated[:, ref_audio_len:, :].permute(0, 2, 1).to(self.accelerator.device)
                                )
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
                        finally:
                            sample_model.train(was_training)

        self.save_checkpoint(global_update, consumed_updates, last=True)

        self.accelerator.end_training()
