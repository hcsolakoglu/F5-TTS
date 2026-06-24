#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from f5_tts.model import CFM, DiT  # noqa: E402
from f5_tts.model.dataset import CustomDataset, collate_fn  # noqa: E402


MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "tiny": {
        "dim": 64,
        "depth": 2,
        "heads": 4,
        "dim_head": 16,
        "ff_mult": 2,
        "text_dim": 64,
        "text_mask_padding": True,
        "conv_layers": 0,
        "pe_attn_head": None,
        "qk_norm": None,
    },
    "f5tts-small": {
        "dim": 768,
        "depth": 18,
        "heads": 12,
        "dim_head": 64,
        "ff_mult": 2,
        "text_dim": 512,
        "text_mask_padding": False,
        "conv_layers": 4,
        "pe_attn_head": 1,
        "qk_norm": None,
    },
    "f5tts-base": {
        "dim": 1024,
        "depth": 22,
        "heads": 16,
        "dim_head": 64,
        "ff_mult": 2,
        "text_dim": 512,
        "text_mask_padding": False,
        "conv_layers": 4,
        "pe_attn_head": 1,
        "qk_norm": None,
    },
    "f5tts-v1-small": {
        "dim": 768,
        "depth": 18,
        "heads": 12,
        "dim_head": 64,
        "ff_mult": 2,
        "text_dim": 512,
        "text_mask_padding": True,
        "conv_layers": 4,
        "pe_attn_head": None,
        "qk_norm": None,
    },
    "f5tts-v1-base": {
        "dim": 1024,
        "depth": 22,
        "heads": 16,
        "dim_head": 64,
        "ff_mult": 2,
        "text_dim": 512,
        "text_mask_padding": True,
        "conv_layers": 4,
        "pe_attn_head": None,
        "qk_norm": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F5-TTS training-step benchmark over a CustomDataset.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-cwd",
        type=Path,
        default=None,
        help="Working directory used to resolve relative audio paths stored in the dataset.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--model-profile", default="tiny", choices=sorted(MODEL_PROFILES))
    parser.add_argument("--mel-dim", type=int, default=100)
    parser.add_argument("--target-sample-rate", type=int, default=24000)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--mel-spec-type", default="vocos", choices=["vocos", "bigvgan"])
    parser.add_argument("--precision", default="float32", choices=["float32", "bf16"])
    parser.add_argument("--fused-adamw", default="auto", choices=["auto", "true", "false"])
    parser.add_argument(
        "--pad-to-frames",
        type=int,
        default=None,
        help=(
            "Benchmark-only shape-stability probe: pad every collated mel tensor to this frame count. "
            "Real mel_lengths stay unchanged; values below the batch max frame length raise."
        ),
    )
    parser.add_argument("--compile-enabled", action="store_true")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-mode", default=None)
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--compile-dynamic", default="default", choices=["default", "true", "false"])
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def run_git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_durations(dataset_root: Path) -> list[float]:
    with (dataset_root / "duration.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [float(item) for item in payload["duration"]]


def duration_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def build_dataset(args: argparse.Namespace) -> CustomDataset:
    hf_dataset = load_from_disk(str(args.dataset_root / "raw"))
    durations = read_durations(args.dataset_root)
    return CustomDataset(
        hf_dataset,
        durations=durations,
        target_sample_rate=args.target_sample_rate,
        hop_length=args.hop_length,
        n_mel_channels=args.mel_dim,
        n_fft=args.n_fft,
        win_length=args.win_length,
        mel_spec_type=args.mel_spec_type,
        preprocessed_mel=False,
    )


def build_model(args: argparse.Namespace, device: torch.device) -> CFM:
    profile = MODEL_PROFILES[args.model_profile]
    transformer = DiT(
        dim=profile["dim"],
        depth=profile["depth"],
        heads=profile["heads"],
        dim_head=profile["dim_head"],
        ff_mult=profile["ff_mult"],
        mel_dim=args.mel_dim,
        text_num_embeds=256,
        text_dim=profile["text_dim"],
        text_mask_padding=profile["text_mask_padding"],
        qk_norm=profile["qk_norm"],
        conv_layers=profile["conv_layers"],
        pe_attn_head=profile["pe_attn_head"],
        attn_backend="torch",
        attn_mask_enabled=False,
        checkpoint_activations=False,
    )
    model = CFM(
        transformer=transformer,
        mel_spec_kwargs={
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "win_length": args.win_length,
            "n_mel_channels": args.mel_dim,
            "target_sample_rate": args.target_sample_rate,
            "mel_spec_type": args.mel_spec_type,
        },
        vocab_char_map=None,
    )
    return model.to(device)


def adamw_fused_value(requested: str, device: torch.device) -> bool:
    if requested == "true":
        return True
    if requested == "false":
        return False
    return device.type == "cuda"


def compile_dynamic_value(value: str) -> bool | None:
    if value == "default":
        return None
    return value == "true"


def autocast_context(device: torch.device, precision: str):
    if precision == "float32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError(f"--precision {precision} requires CUDA")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def configure_compile(model: CFM, args: argparse.Namespace) -> tuple[torch.nn.Module, str, float]:
    if not args.compile_enabled:
        return model, "disabled", 0.0
    compile_kwargs = {
        "backend": args.compile_backend,
        "mode": args.compile_mode,
        "fullgraph": args.compile_fullgraph,
        "dynamic": compile_dynamic_value(args.compile_dynamic),
    }
    compile_kwargs = {key: value for key, value in compile_kwargs.items() if value is not None}

    compile_start = time.perf_counter()
    compile_training_core = getattr(model, "compile_training_core", None)
    if compile_training_core is not None:
        compile_training_core(**compile_kwargs)
        compile_target = "cfm_training_core"
        compiled_model = model
    else:
        compiled_model = torch.compile(model, **compile_kwargs)
        compile_target = "module_forward"
    return compiled_model, compile_target, time.perf_counter() - compile_start


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "mel": batch["mel"].to(device, non_blocking=True),
        "mel_lengths": batch["mel_lengths"].to(device, non_blocking=True),
        "text": batch["text"],
        "text_lengths": batch["text_lengths"].to(device, non_blocking=True),
    }


def maybe_pad_batch_frames(batch: dict[str, Any], pad_to_frames: int | None) -> dict[str, Any]:
    if pad_to_frames is None:
        return batch
    if pad_to_frames <= 0:
        raise ValueError("--pad-to-frames must be positive when set")
    mel = batch["mel"]
    current_frames = int(mel.shape[-1])
    if current_frames > pad_to_frames:
        raise ValueError(
            f"--pad-to-frames={pad_to_frames} is smaller than the current batch frame length {current_frames}"
        )
    if current_frames == pad_to_frames:
        return batch
    padded = dict(batch)
    padded["mel"] = F.pad(mel, (0, pad_to_frames - current_frames), value=0.0)
    return padded


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def environment(device: torch.device) -> dict[str, Any]:
    status = run_git(["status", "--porcelain=v1"]) or ""
    env: dict[str, Any] = {
        "repo_root": str(REPO_ROOT),
        "git_commit": run_git(["rev-parse", "HEAD"]),
        "git_branch": run_git(["branch", "--show-current"]),
        "git_dirty": bool(status),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        env.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            }
        )
    return env


def batch_stats(batch: dict[str, Any]) -> dict[str, float | int]:
    lengths = batch["mel_lengths"].detach().cpu().float()
    max_len = float(lengths.max().item())
    batch_size = int(lengths.numel())
    total_frames = float(lengths.sum().item())
    input_frames = int(batch["mel"].shape[-1])
    padded_frames = float(batch_size * max_len)
    compute_padded_frames = float(batch_size * input_frames)
    padding_ratio = 1.0 - (total_frames / padded_frames) if padded_frames > 0 else 0.0
    compute_padding_ratio = (
        1.0 - (total_frames / compute_padded_frames) if compute_padded_frames > 0 else 0.0
    )
    return {
        "batch_size": batch_size,
        "total_frames": total_frames,
        "max_frames": max_len,
        "mean_frames": float(lengths.mean().item()),
        "input_frames": input_frames,
        "padding_ratio": padding_ratio,
        "compute_padding_ratio": compute_padding_ratio,
    }


def run_train_step(
    model: CFM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    batch: dict[str, Any],
    device: torch.device,
    max_grad_norm: float,
    pad_to_frames: int | None,
    precision: str,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)

    sync(device)
    step_start = time.perf_counter()

    pad_start = time.perf_counter()
    batch = maybe_pad_batch_frames(batch, pad_to_frames)
    pad_end = time.perf_counter()
    stats = batch_stats(batch)

    h2d_start = time.perf_counter()
    batch = move_batch(batch, device)
    sync(device)
    h2d_end = time.perf_counter()

    mel_spec = batch["mel"].permute(0, 2, 1)
    mel_lengths = batch["mel_lengths"]
    text_inputs = batch["text"]

    forward_start = time.perf_counter()
    with autocast_context(device, precision):
        loss, _, _ = model(mel_spec, text=text_inputs, lens=mel_lengths)
    sync(device)
    forward_end = time.perf_counter()

    backward_start = time.perf_counter()
    loss.backward()
    if max_grad_norm > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    else:
        grad_norm = torch.tensor(0.0, device=device)
    sync(device)
    backward_end = time.perf_counter()

    optimizer_start = time.perf_counter()
    optimizer.step()
    sync(device)
    optimizer_end = time.perf_counter()

    scheduler_start = time.perf_counter()
    scheduler.step()
    sync(device)
    scheduler_end = time.perf_counter()

    zero_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    sync(device)
    zero_end = time.perf_counter()

    step_end = time.perf_counter()
    return {
        **stats,
        "loss": float(loss.detach().cpu()),
        "grad_norm": float(grad_norm.detach().cpu()),
        "step_time_s": step_end - step_start,
        "host_to_device_time_s": h2d_end - h2d_start,
        "forward_time_s": forward_end - forward_start,
        "backward_time_s": backward_end - backward_start,
        "optimizer_time_s": optimizer_end - optimizer_start,
        "scheduler_time_s": scheduler_end - scheduler_start,
        "zero_grad_time_s": zero_end - zero_start,
        "shape_pad_time_s": pad_end - pad_start,
        "samples_per_s": stats["batch_size"] / (step_end - step_start),
        "frames_per_s": stats["total_frames"] / (step_end - step_start),
    }


def next_batch(iterator: Any, loader: DataLoader) -> tuple[Any, Any, bool]:
    try:
        return next(iterator), iterator, False
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator, True


def dynamo_counters() -> dict[str, Any] | None:
    try:
        from torch._dynamo.utils import counters
    except Exception:
        return None

    def normalize(value):
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    return normalize(counters)


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.dataset_cwd is not None:
        os.chdir(args.dataset_cwd)

    device = select_device(args.device)
    seed_everything(args.seed)
    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = build_model(args, device)
    model, compile_target, compile_setup_time_s = configure_compile(model, args)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        fused=adamw_fused_value(args.fused_adamw, device),
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=1.0,
        total_iters=max(1, args.steps + args.warmup_steps),
    )
    model.train()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    iterator = iter(loader)
    warmup_results = []
    measured_results = []
    data_waits = []
    epoch_wraps = 0

    for step_index in range(args.warmup_steps + args.steps):
        data_start = time.perf_counter()
        batch, iterator, wrapped = next_batch(iterator, loader)
        data_end = time.perf_counter()
        if wrapped:
            epoch_wraps += 1
        row = run_train_step(
            model,
            optimizer,
            scheduler,
            batch,
            device,
            args.max_grad_norm,
            args.pad_to_frames,
            args.precision,
        )
        row["data_wait_time_s"] = data_end - data_start
        if step_index < args.warmup_steps:
            warmup_results.append(row)
        else:
            measured_results.append(row)
            data_waits.append(data_end - data_start)

    metric_keys = [
        "step_time_s",
        "data_wait_time_s",
        "host_to_device_time_s",
        "forward_time_s",
        "backward_time_s",
        "optimizer_time_s",
        "scheduler_time_s",
        "zero_grad_time_s",
        "shape_pad_time_s",
        "samples_per_s",
        "frames_per_s",
        "padding_ratio",
        "compute_padding_ratio",
        "mean_frames",
        "max_frames",
        "input_frames",
    ]
    result: dict[str, Any] = {
        "benchmark": "training_step_baseline",
        "claim_scope": "small_real_audio_baseline_not_final_representative_throughput",
        "compile": {
            "enabled": args.compile_enabled,
            "target": compile_target,
            "backend": args.compile_backend if args.compile_enabled else None,
            "mode": args.compile_mode if args.compile_enabled else None,
            "fullgraph": args.compile_fullgraph if args.compile_enabled else None,
            "dynamic": args.compile_dynamic if args.compile_enabled else None,
            "setup_time_s": compile_setup_time_s,
            "cold_start_note": "torch.compile is lazy; first warmup forward/step includes graph capture and codegen.",
            "first_warmup_step_time_s": warmup_results[0]["step_time_s"] if warmup_results else None,
            "first_warmup_forward_time_s": warmup_results[0]["forward_time_s"] if warmup_results else None,
        },
        "environment": environment(device),
        "dataset": {
            "root": str(args.dataset_root),
            "cwd": str(args.dataset_cwd) if args.dataset_cwd is not None else os.getcwd(),
            "rows": len(dataset),
            "duration_s": duration_summary(read_durations(args.dataset_root)),
            "num_workers": args.num_workers,
            "batch_size": args.batch_size,
            "shuffle": True,
            "drop_last": False,
            "pad_to_frames": args.pad_to_frames,
            "epoch_wraps": epoch_wraps,
        },
        "config": {
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "seed": args.seed,
            "lr": args.lr,
            "max_grad_norm": args.max_grad_norm,
            "model_profile": args.model_profile,
            "model_arch": MODEL_PROFILES[args.model_profile],
            "precision": args.precision,
            "optimizer": "AdamW",
            "fused_adamw": adamw_fused_value(args.fused_adamw, device),
            "scheduler": "LinearLR_constant",
            "mel_dim": args.mel_dim,
            "target_sample_rate": args.target_sample_rate,
            "hop_length": args.hop_length,
            "mel_spec_type": args.mel_spec_type,
            "tokenizer": "byte",
        },
        "dynamo_counters": dynamo_counters(),
        "metrics": {key: summarize([float(row[key]) for row in measured_results]) for key in metric_keys},
        "loss": summarize([row["loss"] for row in measured_results]),
        "grad_norm": summarize([row["grad_norm"] for row in measured_results]),
        "raw_warmup": warmup_results,
        "raw_measured": measured_results,
    }
    if device.type == "cuda":
        result["memory"] = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
