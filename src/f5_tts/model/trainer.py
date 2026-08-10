from __future__ import annotations

import gc
import inspect
import math
import operator
import os
import random
import warnings
from contextlib import contextmanager, nullcontext
from enum import Enum
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
from f5_tts.model.dataset import (
    DynamicBatchSampler,
    _get_exact_resume_dataset_signature,
    collate_fn,
)
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
        accelerate_kwargs: dict | None = None,
        ema_kwargs: dict | None = None,
        bnb_optimizer: bool = False,
        mel_spec_type: str = "vocos",  # "vocos" | "bigvgan"
        is_local_vocoder: bool = False,  # use local path vocoder
        local_vocoder_path: str = "",  # local vocoder path
        model_cfg_dict: dict | None = None,  # training config
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
        accelerate_kwargs = {} if accelerate_kwargs is None else dict(accelerate_kwargs)
        ema_kwargs = {} if ema_kwargs is None else dict(ema_kwargs)
        model_cfg_dict = {} if model_cfg_dict is None else dict(model_cfg_dict)
        self.model_cfg_dict = self._resume_primitive(model_cfg_dict)
        self.ema_kwargs = self._resume_primitive(ema_kwargs)
        self._ema_component_signature = {
            "type": f"{EMA.__module__}.{EMA.__qualname__}",
            "config": self._resolved_ema_config(),
        }
        gradient_accumulation_plugin = accelerate_kwargs.pop("gradient_accumulation_plugin", None)
        configured_accumulation_steps = accelerate_kwargs.pop("gradient_accumulation_steps", None)
        if gradient_accumulation_plugin is not None:
            if configured_accumulation_steps is not None:
                raise ValueError("pass either gradient_accumulation_steps or gradient_accumulation_plugin, not both")
            accumulation_kwargs = {"gradient_accumulation_plugin": gradient_accumulation_plugin}
        else:
            if configured_accumulation_steps is not None:
                grad_accumulation_steps = self._validate_gradient_accumulation_steps(configured_accumulation_steps)
            accumulation_kwargs = {"gradient_accumulation_steps": grad_accumulation_steps}

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            **accumulation_kwargs,
            **accelerate_kwargs,
        )
        self.global_masked_mean = global_masked_mean
        self._requested_grad_accumulation_steps = grad_accumulation_steps
        (
            self.grad_accumulation_steps,
            self._sync_with_dataloader,
            self._adjust_scheduler,
        ) = self._get_effective_gradient_accumulation_contract()
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
        self.learning_rate = learning_rate
        self.bnb_optimizer = bnb_optimizer
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

    def _get_effective_gradient_accumulation_contract(self):
        gradient_state = getattr(self.accelerator, "gradient_state", None)
        requested = getattr(
            self,
            "_requested_grad_accumulation_steps",
            getattr(self, "grad_accumulation_steps", 1),
        )
        effective_steps = getattr(
            gradient_state,
            "num_steps",
            getattr(self.accelerator, "gradient_accumulation_steps", requested),
        )
        effective_steps = self._validate_gradient_accumulation_steps(effective_steps)
        sync_with_dataloader = bool(getattr(gradient_state, "sync_with_dataloader", True))
        adjust_scheduler = bool(getattr(gradient_state, "adjust_scheduler", False))
        return effective_steps, sync_with_dataloader, adjust_scheduler

    @staticmethod
    def _resume_primitive(value):
        if isinstance(value, Enum):
            return Trainer._resume_primitive(value.value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (tuple, list)):
            return [Trainer._resume_primitive(item) for item in value]
        if isinstance(value, (set, frozenset)):
            items = [Trainer._resume_primitive(item) for item in value]
            return {"set": sorted(items, key=repr)}
        if isinstance(value, dict):
            return {str(key): Trainer._resume_primitive(item) for key, item in value.items() if key != "params"}
        if isinstance(value, torch.dtype):
            return str(value)
        return f"{type(value).__module__}.{type(value).__qualname__}"

    @staticmethod
    def _exact_signature_primitive(value):
        if isinstance(value, Enum):
            return Trainer._exact_signature_primitive(value.value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (tuple, list)):
            return [Trainer._exact_signature_primitive(item) for item in value]
        if isinstance(value, (set, frozenset)):
            items = [Trainer._exact_signature_primitive(item) for item in value]
            return {"set": sorted(items, key=repr)}
        if isinstance(value, dict):
            return {str(key): Trainer._exact_signature_primitive(item) for key, item in value.items()}
        if isinstance(value, torch.dtype):
            return str(value)
        raise ValueError(
            "exact resume signatures must contain only primitive values, mappings, sequences, sets, enums, or dtypes"
        )

    @staticmethod
    def _qualified_type(value):
        if value is None:
            return None
        return f"{type(value).__module__}.{type(value).__qualname__}"

    def _resolved_ema_config(self):
        config = {}
        for name, parameter in inspect.signature(EMA).parameters.items():
            if name in {"model", "ema_model"} or parameter.default is inspect.Parameter.empty:
                continue
            config[name] = parameter.default
        config.update(getattr(self, "ema_kwargs", {}) or {})
        config["include_online_model"] = False
        return self._resume_primitive(config)

    def _model_configuration_signature(self):
        config = getattr(self, "model_cfg_dict", None)
        if isinstance(config, dict) and "model" in config:
            return config["model"]
        return config or None

    def _component_resume_signature(self, component, configured_signature=None):
        """Describe an opted-in component without inspecting arbitrary Python state."""
        if component is None:
            return None
        explicit = getattr(component, "exact_resume_signature", None)
        if callable(explicit):
            explicit = explicit()
        signature: dict[str, Any] = {
            "type": self._qualified_type(component),
            "state_protocol": "torch.nn.Module.state_dict",
        }
        if explicit is not None:
            signature["explicit"] = self._exact_signature_primitive(explicit)
        if configured_signature:
            signature["config"] = self._exact_signature_primitive(configured_signature)
        return signature

    @staticmethod
    def _validate_component_state_protocol(component, name: str):
        get_extra_state = getattr(type(component), "get_extra_state", None)
        set_extra_state = getattr(type(component), "set_extra_state", None)
        has_get_extra_state = get_extra_state is not None and get_extra_state is not torch.nn.Module.get_extra_state
        has_set_extra_state = set_extra_state is not None and set_extra_state is not torch.nn.Module.set_extra_state
        if has_get_extra_state != has_set_extra_state:
            raise ValueError(
                f"resume_mode='exact' requires {name} to implement both get_extra_state() and set_extra_state()"
            )

    def _build_optimizer_signature(self):
        optimizer = getattr(self, "optimizer", None)
        underlying = getattr(optimizer, "optimizer", optimizer)
        param_groups = []
        for group in getattr(underlying, "param_groups", []):
            param_groups.append(self._resume_primitive(group))
        return {
            "type": self._qualified_type(underlying),
            "bnb_optimizer": bool(getattr(self, "bnb_optimizer", False)),
            "learning_rate": self._resume_primitive(getattr(self, "learning_rate", None)),
            "param_groups": param_groups,
        }

    def _build_dataset_processing_signature(self, train_dataset):
        mel_spectrogram = getattr(train_dataset, "mel_spectrogram", None)
        return {
            "preprocessed_mel": getattr(train_dataset, "preprocessed_mel", None),
            "content_signature": self._resume_primitive(getattr(train_dataset, "exact_resume_content_signature", None)),
            "preprocessing_signature": self._resume_primitive(
                getattr(
                    train_dataset,
                    "exact_resume_preprocessing_signature",
                    getattr(mel_spectrogram, "exact_resume_signature", None),
                )
            ),
            "target_sample_rate": self._resume_primitive(getattr(train_dataset, "target_sample_rate", None)),
            "n_mel_channels": self._resume_primitive(getattr(train_dataset, "n_mel_channels", None)),
            "hop_length": self._resume_primitive(getattr(train_dataset, "hop_length", None)),
            "n_fft": self._resume_primitive(getattr(train_dataset, "n_fft", None)),
            "win_length": self._resume_primitive(getattr(train_dataset, "win_length", None)),
            "mel_spec_type": self._resume_primitive(getattr(train_dataset, "mel_spec_type", None)),
            "mel_spectrogram_type": self._qualified_type(mel_spectrogram),
        }

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
        if num_processes > 1 and split_batches:
            raise ValueError(
                "global_masked_mean=True does not support split_batches=True in distributed training; "
                "set split_batches=False to keep per-rank collective counts symmetric."
            )
        if num_processes > 1 and not split_batches:
            dataloader_length = None
            length_exception = None
            try:
                dataloader_length = len(dataloader)
            except Exception as exc:
                length_exception = exc
            else:
                if dataloader_length % num_processes:
                    length_exception = ValueError(
                        "global_masked_mean requires the prepared dataloader batch count to be divisible by "
                        f"the process count ({num_processes}); received {dataloader_length} batches. "
                        "Increase/drop the final batch explicitly instead of allowing duplicate padding."
                    )
            self._raise_if_global_masked_mean_failure("dataloader length", length_exception)
            if dataloader_length is None:  # pragma: no cover - helper raises on the failed rank
                raise RuntimeError("global_masked_mean dataloader length was unavailable")
        prepared = None
        prepare_exception = None
        try:
            prepared = self.accelerator.prepare_data_loader(dataloader, device_placement=False)
        except Exception as exc:
            prepare_exception = exc
        self._raise_if_global_masked_mean_failure("dataloader preparation", prepare_exception)
        if prepared is None:  # pragma: no cover - helper raises on every failed rank
            raise RuntimeError("global_masked_mean dataloader preparation returned no loader")
        self._prepared_global_masked_mean_batches = self._validate_global_masked_dataloader(prepared)
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
            message = (
                "checkpoint RNG state was saved for "
                f"{len(states)} processes, but the current run uses {expected_processes}; "
                "resume with the same process count or start from model weights only."
            )
            if getattr(self, "resume_mode", "best_effort") == "exact":
                raise ValueError(message)
            warnings.warn(
                f"{message} Skipping incompatible RNG restoration in best-effort mode.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
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

    def _capture_scaler_state(self):
        scaler = getattr(self.accelerator, "scaler", None)
        return None if scaler is None else scaler.state_dict()

    def _restore_scaler_state(self, state):
        if state is None:
            return
        scaler = getattr(self.accelerator, "scaler", None)
        if scaler is not None:
            scaler.load_state_dict(state)

    def _build_resume_state_signature(self):
        accelerator = self.accelerator
        device = getattr(accelerator, "device", None)
        device_type = getattr(device, "type", str(device))
        mixed_precision = getattr(accelerator, "mixed_precision", "no")
        mixed_precision = getattr(mixed_precision, "value", mixed_precision)
        distributed_type = getattr(accelerator, "distributed_type", DistributedType.NO)
        distributed_type = getattr(distributed_type, "value", distributed_type)
        scaler = getattr(accelerator, "scaler", None)
        return {
            "version": 1,
            "device_type": str(device_type),
            "cuda_device_count": torch.cuda.device_count() if device_type == "cuda" else 0,
            "mixed_precision": str(mixed_precision),
            "scaler_type": None if scaler is None else type(scaler).__qualname__,
            "distributed_type": str(distributed_type),
            "num_processes": int(getattr(accelerator, "num_processes", 1)),
            "num_machines": int(getattr(accelerator, "num_machines", 1)),
            "rng_types": self._resume_primitive(getattr(accelerator, "rng_types", None)),
            "use_seedable_sampler": self._resume_primitive(getattr(accelerator, "use_seedable_sampler", None)),
        }

    def _build_resume_signature(
        self,
        train_dataset,
        num_workers: int,
        resumable_with_seed,
        prepared_batches: int,
        split_batches: bool,
        step_scheduler_with_optimizer: bool,
        sync_with_dataloader: bool,
        adjust_scheduler: bool,
    ):
        accelerator = self.accelerator
        model = getattr(self, "_unwrapped_model", None) or getattr(self, "model", None)
        ema_signature = getattr(self, "_ema_component_signature", None)
        if ema_signature is None:
            ema_signature = {
                "type": self._qualified_type(getattr(self, "ema_model", None)),
                "config": self._resume_primitive(getattr(self, "ema_kwargs", {}) or {}),
            }
        compile_config = {
            "active": bool(getattr(self, "compile_active", False)),
            "backend": self._resume_primitive(getattr(self, "compile_backend", None)),
            "target": self._resume_primitive(getattr(self, "compile_target", None)),
            "mode": self._resume_primitive(getattr(self, "compile_mode", None)),
            "fullgraph": self._resume_primitive(getattr(self, "compile_fullgraph", None)),
            "dynamic": self._resume_primitive(
                getattr(self, "compile_effective_dynamic", getattr(self, "compile_dynamic", None))
            ),
            "fallback_to_eager": bool(getattr(self, "compile_fallback_to_eager", True)),
            "fallback_active": bool(getattr(self, "compile_fallback_active", False)),
        }
        return {
            "version": 2,
            "dataset_type": f"{type(train_dataset).__module__}.{type(train_dataset).__qualname__}",
            "dataset": _get_exact_resume_dataset_signature(train_dataset),
            "dataset_processing": self._build_dataset_processing_signature(train_dataset),
            "batch_size_type": self.batch_size_type,
            "batch_size_per_gpu": self.batch_size_per_gpu,
            "max_samples": self.max_samples,
            "grad_accumulation_steps": self.grad_accumulation_steps,
            "sync_with_dataloader": bool(sync_with_dataloader),
            "adjust_scheduler": bool(adjust_scheduler),
            "epochs": self.epochs,
            "num_warmup_updates": self.num_warmup_updates,
            "max_grad_norm": self._resume_primitive(getattr(self, "max_grad_norm", None)),
            "global_masked_mean": bool(self.global_masked_mean),
            "num_workers": int(num_workers),
            "persistent_workers": bool(num_workers > 0),
            "resumable_with_seed": None if resumable_with_seed is None else int(resumable_with_seed),
            "sampler": "epoch_random_sample" if self.batch_size_type == "sample" else "dynamic_frame",
            "prepared_batches": int(prepared_batches),
            "split_batches": bool(split_batches),
            "dispatch_batches": bool(getattr(accelerator, "dispatch_batches", False)),
            "even_batches": getattr(accelerator, "even_batches", None),
            "device_placement": not bool(self.global_masked_mean),
            "step_scheduler_with_optimizer": bool(step_scheduler_with_optimizer),
            "optimizer": self._build_optimizer_signature(),
            "model_type": self._qualified_type(model),
            "model_component": self._component_resume_signature(
                model,
                configured_signature=self._model_configuration_signature() if isinstance(model, CFM) else None,
            ),
            "ema_component": ema_signature,
            "ema_kwargs": self._resume_primitive(getattr(self, "ema_kwargs", None)),
            "duration_predictor_type": self._qualified_type(getattr(self, "duration_predictor", None)),
            "duration_predictor_component": self._component_resume_signature(getattr(self, "duration_predictor", None)),
            "noise_scheduler": self._resume_primitive(getattr(self, "noise_scheduler", None)),
            "compile": compile_config,
            "resume_state": self._build_resume_state_signature(),
        }

    def _validate_exact_component_signatures(self):
        model = getattr(self, "_unwrapped_model", None) or getattr(self, "model", None)
        duration_predictor = getattr(self, "duration_predictor", None)
        components = {
            "model": (
                model,
                self._component_resume_signature(
                    model,
                    configured_signature=self._model_configuration_signature() if isinstance(model, CFM) else None,
                ),
            ),
            "duration predictor": (
                duration_predictor,
                self._component_resume_signature(duration_predictor),
            ),
        }
        for name, (component, signature) in components.items():
            if signature is None:
                continue
            if getattr(component, "supports_exact_resume", False) is not True:
                raise ValueError(
                    f"resume_mode='exact' requires {name}.supports_exact_resume=True; "
                    "the opt-in certifies that state_dict() contains every training-affecting mutable value"
                )
            if not any(key in signature for key in ("explicit", "config")):
                raise ValueError(
                    f"resume_mode='exact' requires {name}.exact_resume_signature or a production model configuration"
                )
            if isinstance(component, CFM) and "config" not in signature:
                raise ValueError("resume_mode='exact' requires the production model configuration for CFM")
            self._validate_component_state_protocol(component, name)

    def _validate_exact_dataset_components(self, train_dataset):
        source = getattr(train_dataset, "data", train_dataset)
        features = getattr(source, "features", None)
        uses_audio_feature = features is not None and "audio" in features
        source_columns = getattr(source, "column_names", None)
        source_features = getattr(source, "features", None)
        uses_audio_path = (source_columns is not None and "audio_path" in source_columns) or (
            source_features is not None and "audio_path" in source_features
        )
        if (uses_audio_feature or uses_audio_path) and not getattr(train_dataset, "preprocessed_mel", False):
            if getattr(train_dataset, "exact_resume_content_signature", None) is None:
                raise ValueError("resume_mode='exact' requires exact_resume_content_signature for external audio data")

        mel_spectrogram = getattr(train_dataset, "mel_spectrogram", None)
        if mel_spectrogram is None or getattr(train_dataset, "preprocessed_mel", False):
            return
        if self._qualified_type(mel_spectrogram) == "f5_tts.model.modules.MelSpec":
            return
        preprocessing_signature = getattr(train_dataset, "exact_resume_preprocessing_signature", None)
        if preprocessing_signature is None:
            preprocessing_signature = getattr(mel_spectrogram, "exact_resume_signature", None)
        if preprocessing_signature is None:
            raise ValueError("resume_mode='exact' requires exact_resume_preprocessing_signature for custom mel modules")

    def _validate_resume_signatures(self, checkpoint):
        """Validate persisted state that is required for exact resume."""
        if "update" not in checkpoint and "step" not in checkpoint:
            return True, True
        expected_state = getattr(self, "_resume_state_signature", None)
        expected_resume = getattr(self, "_resume_signature", None)
        saved_state = checkpoint.get("resume_state_signature")
        saved_resume = checkpoint.get("resume_signature")
        missing = (
            expected_state is None
            or expected_resume is None
            or saved_state is None
            or saved_resume is None
            or "rng_state" not in checkpoint
            or "scaler_state" not in checkpoint
            or (
                getattr(self, "duration_predictor", None) is not None
                and checkpoint.get("duration_predictor_state_dict") is None
            )
        )
        legacy_signature = (
            expected_state is None or expected_resume is None or saved_state is None or saved_resume is None
        )
        mismatches = []
        if not missing:
            if saved_state != expected_state:
                mismatches.append("execution topology or mixed-precision state")
            if saved_resume != expected_resume:
                mismatches.append("dataset or dataloader training contract")
            if (getattr(self.accelerator, "scaler", None) is not None) != (checkpoint.get("scaler_state") is not None):
                mismatches.append("AMP scaler state")
        if missing or mismatches:
            details_parts = list(mismatches)
            if missing:
                details_parts.append("missing persisted resume signature/state")
            if (
                getattr(self, "duration_predictor", None) is not None
                and checkpoint.get("duration_predictor_state_dict") is None
            ):
                details_parts.append("duration_predictor_state_dict")
            details = ", ".join(details_parts)
            message = f"resume signature is incompatible: {details}"
            if getattr(self, "resume_mode", "best_effort") == "exact":
                raise ValueError(f"exact resume requires a matching {message}")
            if legacy_signature and "rng_state" in checkpoint:
                warnings.warn(
                    "legacy checkpoint lacks full resume signatures; restoring its legacy RNG state in "
                    "best-effort mode; exact resume is unavailable.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return True, False
            warnings.warn(
                f"{message}; skipping scaler and RNG restoration in best-effort mode.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False, False
        return True, True

    @staticmethod
    def _capture_local_rng_state():
        """Capture RNG streams without entering a distributed collective."""
        state = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_local_rng_state(state):
        random.setstate(state["python"])
        torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(list(state["cuda"]))

    @contextmanager
    def _preserve_inference_rng_state(self):
        """Prevent rank-local sample logging from advancing training RNG streams."""
        state = self._capture_local_rng_state()
        try:
            yield
        finally:
            self._restore_local_rng_state(state)

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
        self.compile_effective_dynamic = effective_compile_dynamic
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

    def _raise_if_global_masked_mean_failure(self, phase: str, local_exception: Exception | None):
        """Raise a phase error only after global masked-mean ranks synchronize status."""
        if not getattr(self, "global_masked_mean", False):
            if local_exception is not None:
                raise local_exception
            return

        num_processes = int(getattr(self.accelerator, "num_processes", 1))
        if num_processes <= 1:
            if local_exception is not None:
                raise local_exception
            return

        local_oom = local_exception is not None and _is_cuda_oom(local_exception)
        local_stats = torch.tensor(
            (int(local_exception is not None), int(local_oom)),
            device=self.accelerator.device,
            dtype=torch.int32,
        )
        global_stats = self._reduce_global_masked_value(local_stats)
        global_error_count, global_oom_count = global_stats.cpu().tolist()
        if not global_error_count:
            return

        if local_exception is not None:
            if local_oom:
                raise local_exception
            raise RuntimeError(
                f"global_masked_mean {phase} failed on this rank: {local_exception}"
            ) from local_exception
        if global_oom_count:
            raise RuntimeError(f"global_masked_mean {phase} failed because another rank reported CUDA out of memory.")
        raise RuntimeError(f"global_masked_mean {phase} failed because another rank raised an exception.")

    def _iter_coordinated_batches(self, dataloader):
        """Yield batches only after all ranks agree that fetching succeeded."""
        dataloader_iter = None
        iterator_exception = None
        try:
            dataloader_iter = cast(Any, iter(dataloader))
        except Exception as exc:
            iterator_exception = exc

        num_processes = int(getattr(self.accelerator, "num_processes", 1))
        while True:
            batch = None
            local_stop = False
            local_exception = iterator_exception
            local_oom = None
            if local_exception is not None and _is_cuda_oom(local_exception):
                local_oom = local_exception
            if local_exception is None:
                try:
                    batch = next(cast(Any, dataloader_iter))
                except StopIteration:
                    local_stop = True
                except Exception as exc:
                    local_exception = exc
                    if _is_cuda_oom(exc):
                        local_oom = exc

            local_stats = torch.tensor(
                (
                    int(local_exception is not None),
                    int(local_stop),
                    int(local_oom is not None),
                ),
                device=self.accelerator.device,
                dtype=torch.int32,
            )
            global_stats = self._reduce_global_masked_value(local_stats)
            global_error_count, global_stop_count, global_oom_count = global_stats.cpu().tolist()

            if global_error_count:
                if local_exception is not None:
                    if local_oom is not None:
                        raise local_oom
                    raise RuntimeError(
                        f"global_masked_mean dataloader fetch failed on this rank: {local_exception}"
                    ) from local_exception
                if global_oom_count:
                    raise RuntimeError(
                        "global_masked_mean dataloader fetch failed because another rank reported CUDA out of memory."
                    )
                raise RuntimeError(
                    "global_masked_mean dataloader fetch failed because another rank raised an exception."
                )

            if global_stop_count:
                if global_stop_count != num_processes:
                    raise RuntimeError(
                        "global_masked_mean dataloader yielded different numbers of batches across ranks."
                    )
                return

            yield batch

    def _iter_global_masked_mean_batches(self, dataloader):
        """Yield pre-normalized accumulation windows without retaining graphs."""
        dataloader_iter = iter(self._iter_coordinated_batches(dataloader))
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
            if (
                any(isinstance(item, list) for item in text)
                and getattr(self._unwrapped_model, "vocab_char_map", None) is None
            ):
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
        dataloader_length = None
        length_exception = None
        try:
            dataloader_length = len(dataloader)
        except Exception as exc:
            length_exception = exc
        self._raise_if_global_masked_mean_failure("prepared dataloader length", length_exception)
        if dataloader_length is None:  # pragma: no cover - helper raises on the failed rank
            raise RuntimeError("global_masked_mean prepared dataloader length was unavailable")
        local_length = torch.tensor([dataloader_length], device=self.accelerator.device, dtype=torch.int64)
        gathered_lengths = cast(torch.Tensor, self.accelerator.gather(local_length))
        if gathered_lengths[0].item() == 0:
            raise ValueError("global_masked_mean requires a non-empty training dataloader.")
        if bool((gathered_lengths != gathered_lengths[0]).any().item()):
            lengths = gathered_lengths.cpu().tolist()
            raise RuntimeError(
                "global_masked_mean requires the same number of dataloader batches on every rank; "
                f"received per-rank lengths {lengths}."
            )
        return dataloader_length

    def save_checkpoint(self, update, consumed_updates=None, last=False):
        update, consumed_updates, last = self._normalize_checkpoint_args(update, consumed_updates, last)
        rng_states = self._capture_rng_states()
        scaler_state = self._capture_scaler_state()
        resume_state_signature = self._build_resume_state_signature()
        self.accelerator.wait_for_everyone()
        if self.is_main:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            duration_predictor = getattr(self, "duration_predictor", None)
            duration_predictor_state = None if duration_predictor is None else duration_predictor.state_dict()
            checkpoint = dict(
                model_state_dict=unwrapped_model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                duration_predictor_state_dict=duration_predictor_state,
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
                consumed_updates=consumed_updates,
                rng_state=rng_states,
                scaler_state=scaler_state,
                resume_state_signature=resume_state_signature,
                resume_signature=getattr(self, "_resume_signature", None),
            )
            scheduler_signature = getattr(self, "_scheduler_signature", None)
            if scheduler_signature is not None:
                checkpoint["scheduler_signature"] = scheduler_signature
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

        checkpoint: dict[str, Any] = {}
        if latest_checkpoint.endswith(".safetensors"):  # always a pretrained checkpoint
            from safetensors.torch import load_file

            checkpoint = load_file(f"{self.checkpoint_path}/{latest_checkpoint}", device="cpu")
            checkpoint = {"ema_model_state_dict": checkpoint}
        elif latest_checkpoint.endswith(".pt"):
            load_kwargs: dict[str, Any] = {"map_location": "cpu"}
            if "weights_only" in inspect.signature(torch.load).parameters:
                load_kwargs["weights_only"] = True
            checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", **load_kwargs)

        self._validate_scheduler_signature(checkpoint)
        resume_rng_compatible, resume_scaler_compatible = self._validate_resume_signatures(checkpoint)

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

            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if resume_scaler_compatible:
                self._restore_scaler_state(checkpoint.get("scaler_state"))
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
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(checkpoint["model_state_dict"])
            update = 0
            consumed_updates = 0

        duration_predictor = getattr(self, "duration_predictor", None)
        if duration_predictor is not None and checkpoint.get("duration_predictor_state_dict") is not None:
            duration_predictor.load_state_dict(checkpoint["duration_predictor_state_dict"])

        if resume_rng_compatible:
            self._restore_rng_states(checkpoint.get("rng_state"))
        del checkpoint
        gc.collect()
        return (update, consumed_updates) if return_cursor else update

    def _validate_scheduler_signature(self, checkpoint):
        """Reject exact resume when the saved LR horizon is absent or changed."""
        if getattr(self, "resume_mode", "best_effort") != "exact":
            return
        if "update" not in checkpoint and "step" not in checkpoint:
            return
        expected = getattr(self, "_scheduler_signature", None)
        saved = checkpoint.get("scheduler_signature")
        if expected is None or saved != expected:
            raise ValueError(
                "exact resume requires a matching scheduler signature; "
                "the prepared dataloader or schedule configuration changed"
            )

    def _reject_exact_compile_fallback(self):
        if getattr(self, "resume_mode", "best_effort") == "exact" and getattr(self, "compile_fallback_active", False):
            raise RuntimeError(
                "resume_mode='exact' cannot continue after torch.compile fell back to eager; "
                "the uninterrupted and resumed execution contracts would differ"
            )

    def _validate_resume_contract(self, train_dataset, num_workers: int, resumable_with_seed, resume_mode: str):
        """Validate the promised fidelity of checkpoint resume before creating workers."""
        self.resume_mode = resume_mode
        self._reject_exact_compile_fallback()
        if resume_mode not in {"best_effort", "exact"}:
            raise ValueError("resume_mode must be either 'best_effort' or 'exact'")
        if resume_mode == "best_effort":
            return
        if num_workers != 0:
            raise ValueError("resume_mode='exact' currently requires num_workers=0")
        if resumable_with_seed is None:
            raise ValueError("resume_mode='exact' requires resumable_with_seed")
        if getattr(self, "batch_size_type", None) != "sample":
            raise ValueError("resume_mode='exact' currently supports batch_size_type='sample' only")
        if not isinstance(train_dataset, Dataset):
            raise ValueError("resume_mode='exact' requires a map-style torch Dataset")
        if getattr(train_dataset, "supports_exact_resume", False) is not True:
            raise ValueError(
                "resume_mode='exact' requires train_dataset.supports_exact_resume=True; "
                "arbitrary dataset and augmentation state cannot be inferred safely"
            )
        if _get_exact_resume_dataset_signature(train_dataset) is None:
            raise ValueError(
                "resume_mode='exact' requires train_dataset.exact_resume_signature or an immutable HF fingerprint"
            )
        self._validate_exact_component_signatures()
        self._validate_exact_dataset_components(train_dataset)
        accelerator = getattr(self, "accelerator", None)
        device = getattr(accelerator, "device", torch.device("cpu"))
        device_type = getattr(device, "type", str(device))
        if device_type not in {"cpu", "cuda"}:
            raise ValueError(
                "resume_mode='exact' supports only CPU and CUDA RNG streams; "
                f"backend {device_type!r} requires an explicit backend RNG contract"
            )
        distributed_type = getattr(accelerator, "distributed_type", DistributedType.NO)
        distributed_type = getattr(distributed_type, "value", str(distributed_type))
        if distributed_type not in {
            DistributedType.NO.value,
            DistributedType.MULTI_CPU.value,
            DistributedType.MULTI_GPU.value,
        }:
            raise ValueError(
                "resume_mode='exact' supports only unsharded single-process or CPU/CUDA DDP; "
                f"distributed backend {distributed_type!r} requires a backend-aware checkpoint protocol"
            )
        if accelerator is None:
            sync_with_dataloader = True
        else:
            _, sync_with_dataloader, _ = self._get_effective_gradient_accumulation_contract()
        if not sync_with_dataloader:
            raise ValueError(
                "resume_mode='exact' requires gradient accumulation to synchronize at dataloader boundaries"
            )
        if getattr(self, "duration_predictor", None) is not None and getattr(accelerator, "num_processes", 1) > 1:
            raise ValueError(
                "resume_mode='exact' does not support a duration_predictor with multiple processes; "
                "its per-process state is not globally checkpointed"
            )

    @staticmethod
    def _compute_scheduler_horizon(
        *,
        prepared_batches: int,
        grad_accumulation_steps: int,
        epochs: int,
        num_warmup_updates: int,
        num_processes: int,
        split_batches: bool,
        step_scheduler_with_optimizer: bool = True,
        sync_with_dataloader: bool = True,
        adjust_scheduler: bool = False,
    ) -> tuple[int, int, int]:
        """Compute raw scheduler steps after Accelerate data-loader preparation."""
        if prepared_batches <= 0:
            raise ValueError("the prepared training dataloader must contain at least one batch")
        if grad_accumulation_steps <= 0 or epochs <= 0 or num_processes <= 0:
            raise ValueError("scheduler horizon arguments must be positive")
        if num_warmup_updates < 0:
            raise ValueError("num_warmup_updates must be non-negative")

        del adjust_scheduler  # Wrapper call cadence does not change underlying scheduler horizon.
        scheduler_step_factor = 1 if split_batches or not step_scheduler_with_optimizer else num_processes
        if sync_with_dataloader:
            updates_per_epoch = math.ceil(prepared_batches / grad_accumulation_steps)
            total_updates = updates_per_epoch * epochs
        else:
            total_updates = (prepared_batches * epochs) // grad_accumulation_steps
        if total_updates <= 0:
            raise ValueError("the training horizon must contain at least one optimizer update")
        total_raw_steps = total_updates * scheduler_step_factor
        warmup_raw_steps = num_warmup_updates * scheduler_step_factor
        if warmup_raw_steps >= total_raw_steps:
            raise ValueError(
                "num_warmup_updates must be smaller than the total prepared scheduler horizon "
                f"({warmup_raw_steps} >= {total_raw_steps} raw steps)."
            )
        return total_raw_steps, warmup_raw_steps, scheduler_step_factor

    def train(
        self,
        train_dataset: Dataset,
        num_workers=16,
        resumable_with_seed: int | None = None,
        resume_mode: str = "best_effort",
    ):
        self.resume_mode = resume_mode
        self._validate_resume_contract(train_dataset, num_workers, resumable_with_seed, resume_mode)
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

        train_dataloader = None
        dataloader_exception = None
        try:
            if self.batch_size_type == "sample":
                sample_sampler = None
                if exists(resumable_with_seed):
                    resume_seed = cast(int, resumable_with_seed)
                    self._train_dataloader_generator = torch.Generator()
                    self._train_dataloader_generator.manual_seed(resume_seed)
                    sample_sampler = _EpochRandomSampler(train_dataset, resume_seed)
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
                raise ValueError(
                    f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}"
                )
        except Exception as exc:
            dataloader_exception = exc
        self._raise_if_global_masked_mean_failure("dataloader construction", dataloader_exception)
        if train_dataloader is None:  # pragma: no cover - helper raises on the failed rank
            raise RuntimeError("global_masked_mean dataloader construction returned no loader")

        # Prepare the dataloader before sizing the scheduler. Accelerate changes
        # both its length and the number of raw scheduler steps per optimizer update.
        if self.global_masked_mean:
            # Buffer accumulation windows on the host. A normally prepared
            # dataloader places each yielded batch on the accelerator, which
            # would retain G large mel batches and undermine accumulation's
            # memory bound. The window iterator transfers one batch at a time.
            train_dataloader = self._prepare_global_masked_mean_dataloader(train_dataloader)
            prepared_batches = self._prepared_global_masked_mean_batches
        else:
            train_dataloader = self.accelerator.prepare_data_loader(train_dataloader, device_placement=True)
            prepared_batches = len(train_dataloader)

        effective_grad_accumulation_steps, sync_with_dataloader, adjust_scheduler = (
            self._get_effective_gradient_accumulation_contract()
        )
        self.grad_accumulation_steps = effective_grad_accumulation_steps
        if self.global_masked_mean and not sync_with_dataloader:
            raise ValueError(
                "global_masked_mean requires gradient accumulation to synchronize at dataloader boundaries"
            )

        split_batches = bool(getattr(self.accelerator, "split_batches", False))
        step_scheduler_with_optimizer = bool(getattr(self.accelerator, "step_scheduler_with_optimizer", True))
        total_updates, warmup_updates, scheduler_step_factor = self._compute_scheduler_horizon(
            prepared_batches=prepared_batches,
            grad_accumulation_steps=self.grad_accumulation_steps,
            epochs=self.epochs,
            num_warmup_updates=self.num_warmup_updates,
            num_processes=self.accelerator.num_processes,
            split_batches=split_batches,
            step_scheduler_with_optimizer=step_scheduler_with_optimizer,
            sync_with_dataloader=sync_with_dataloader,
            adjust_scheduler=adjust_scheduler,
        )
        self._scheduler_signature = {
            "version": 2,
            "prepared_batches": prepared_batches,
            "grad_accumulation_steps": self.grad_accumulation_steps,
            "epochs": self.epochs,
            "num_warmup_updates": self.num_warmup_updates,
            "num_processes": self.accelerator.num_processes,
            "split_batches": split_batches,
            "step_scheduler_with_optimizer": step_scheduler_with_optimizer,
            "sync_with_dataloader": sync_with_dataloader,
            "adjust_scheduler": adjust_scheduler,
            "scheduler_step_factor": scheduler_step_factor,
            "total_raw_steps": total_updates,
            "warmup_raw_steps": warmup_updates,
        }
        self._resume_state_signature = self._build_resume_state_signature()
        self._resume_signature = self._build_resume_signature(
            train_dataset=train_dataset,
            num_workers=num_workers,
            resumable_with_seed=resumable_with_seed,
            prepared_batches=prepared_batches,
            split_batches=split_batches,
            step_scheduler_with_optimizer=step_scheduler_with_optimizer,
            sync_with_dataloader=sync_with_dataloader,
            adjust_scheduler=adjust_scheduler,
        )
        decay_updates = total_updates - warmup_updates
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        if warmup_updates > 0:
            warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
            self.scheduler = SequentialLR(
                self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
            )
        else:
            # Torch 2.5 leaves the optimizer at the warmup start factor when a
            # zero-length scheduler is included in SequentialLR.
            self.scheduler = decay_scheduler
        self.scheduler = self.accelerator.prepare(self.scheduler)
        start_update, start_consumed_updates = self.load_checkpoint(return_cursor=True)
        global_update = start_update
        consumed_updates = start_consumed_updates

        skipped_batch = 0
        skipped_dataloader = train_dataloader
        orig_epoch_step = len(train_dataloader)
        if exists(resumable_with_seed):
            if sync_with_dataloader:
                updates_per_epoch = math.ceil(orig_epoch_step / self.grad_accumulation_steps)
                skipped_epoch = int(start_consumed_updates // updates_per_epoch)
                updates_into_epoch = start_consumed_updates % updates_per_epoch
                skipped_batch = min(updates_into_epoch * self.grad_accumulation_steps, orig_epoch_step)
            else:
                consumed_batches = start_consumed_updates * self.grad_accumulation_steps
                skipped_epoch, skipped_batch = divmod(consumed_batches, orig_epoch_step)
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                if sync_with_dataloader:
                    progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                else:
                    progress_bar_initial = (
                        epoch * len(train_dataloader) + skipped_batch
                    ) // self.grad_accumulation_steps - (epoch * len(train_dataloader)) // self.grad_accumulation_steps
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            self._set_dataloader_epoch(current_dataloader, epoch)

            if sync_with_dataloader:
                epoch_update_count = math.ceil(len(train_dataloader) / self.grad_accumulation_steps)
            else:
                epoch_start = epoch * len(train_dataloader)
                epoch_update_count = (
                    epoch_start + len(train_dataloader)
                ) // self.grad_accumulation_steps - epoch_start // self.grad_accumulation_steps
            progress_bar = tqdm(
                range(epoch_update_count),
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
                    device_exception = None
                    device_batch = None
                    try:
                        device_batch = send_to_device(batch, self.accelerator.device, non_blocking=True)
                    except Exception as exc:
                        device_exception = exc
                    self._raise_if_global_masked_mean_failure("batch device transfer", device_exception)
                    if device_batch is None:  # pragma: no cover - helper raises on every failed rank
                        raise RuntimeError("global_masked_mean device transfer returned no batch")
                    batch = device_batch
                with self._accumulation_context(is_boundary):
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]

                    duration_exception = None
                    if self.duration_predictor is not None and self.accelerator.is_local_main_process:
                        try:
                            dur_loss = self.duration_predictor(mel_spec, lens=batch.get("durations"))
                            self.accelerator.log({"duration loss": dur_loss.item()}, step=global_update)
                        except Exception as exc:
                            duration_exception = exc
                    if self.global_masked_mean and self.duration_predictor is not None:
                        self._raise_if_global_masked_mean_failure("duration predictor", duration_exception)
                    elif duration_exception is not None:
                        raise duration_exception

                    if self.global_masked_mean:
                        assert rand_span_mask is not None
                        assert loss_scale is not None
                        forward_exception = None
                        forward_result = None
                        try:
                            forward_result = self.model(
                                mel_spec,
                                text=text_inputs,
                                lens=mel_lengths,
                                noise_scheduler=self.noise_scheduler,
                                rand_span_mask=rand_span_mask,
                                return_loss_components=True,
                            )
                        except Exception as exc:
                            forward_exception = exc
                        self._raise_if_global_masked_mean_failure("forward", forward_exception)
                        if forward_result is None:  # pragma: no cover - helper raises on every failed rank
                            raise RuntimeError("global_masked_mean forward returned no result")
                        loss, loss_sum, loss_denom, cond, pred = forward_result
                        loss_sum_accum = (
                            loss_sum.detach() if loss_sum_accum is None else loss_sum_accum + loss_sum.detach()
                        )
                        self._check_compile_runtime_fallback()
                        self._reject_exact_compile_fallback()
                        self.accelerator.backward(loss_sum * loss_scale)
                    else:
                        loss, cond, pred = self.model(
                            mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=self.noise_scheduler
                        )
                        self._check_compile_runtime_fallback()
                        self._reject_exact_compile_fallback()
                        self.accelerator.backward(loss)

                    deepspeed_clips_in_backward = (
                        self.global_masked_mean and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                    )
                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients and not deepspeed_clips_in_backward:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    optimizer_step_was_skipped = bool(getattr(self.accelerator, "optimizer_step_was_skipped", False))
                    if self.accelerator.sync_gradients or adjust_scheduler:
                        accelerated_scheduler = hasattr(self.scheduler, "gradient_state")
                        if accelerated_scheduler or not optimizer_step_was_skipped:
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
                            with (
                                self._preserve_inference_rng_state(),
                                torch.inference_mode(),
                                self.accelerator.autocast(),
                            ):
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
