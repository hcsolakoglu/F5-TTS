#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from f5_tts.model import CFM, DiT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic F5-TTS eager training-step smoke benchmark.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--mel-dim", type=int, default=16)
    parser.add_argument("--text-len", type=int, default=24)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
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


def build_model(args: argparse.Namespace, device: torch.device) -> CFM:
    transformer = DiT(
        dim=64,
        depth=2,
        heads=4,
        dim_head=16,
        ff_mult=2,
        mel_dim=args.mel_dim,
        text_num_embeds=args.vocab_size,
        text_dim=32,
        text_mask_padding=True,
        conv_layers=0,
        attn_backend="torch",
        attn_mask_enabled=False,
        checkpoint_activations=False,
    )
    model = CFM(transformer=transformer, mel_spec_kwargs={"n_mel_channels": args.mel_dim})
    return model.to(device)


def make_batch(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mel = torch.randn(args.batch_size, args.frames, args.mel_dim, device=device)
    text = torch.randint(0, args.vocab_size, (args.batch_size, args.text_len), device=device)
    if args.text_len > 4:
        text[1::2, -(args.text_len // 4) :] = -1
    lens = torch.full((args.batch_size,), args.frames, dtype=torch.long, device=device)
    if args.batch_size > 1 and args.frames > 8:
        lens[1::2] = args.frames - 8
    return mel, text, lens


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


def run_step(
    model: CFM,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    max_grad_norm: float,
) -> dict[str, float]:
    mel, text, lens = batch
    optimizer.zero_grad(set_to_none=True)

    sync(device)
    step_start = time.perf_counter()

    forward_start = time.perf_counter()
    loss, _, _ = model(mel, text=text, lens=lens)
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

    zero_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    sync(device)
    zero_end = time.perf_counter()

    step_end = time.perf_counter()
    return {
        "loss": float(loss.detach().cpu()),
        "grad_norm": float(grad_norm.detach().cpu()),
        "step_time_s": step_end - step_start,
        "forward_time_s": forward_end - forward_start,
        "backward_time_s": backward_end - backward_start,
        "optimizer_time_s": optimizer_end - optimizer_start,
        "zero_grad_time_s": zero_end - zero_start,
    }


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")

    device = select_device(args.device)
    seed_everything(args.seed)
    model = build_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batch = make_batch(args, device)
    model.train()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    warmup_results = [
        run_step(model, optimizer, batch, device, args.max_grad_norm) for _ in range(args.warmup_steps)
    ]
    measured_results = [run_step(model, optimizer, batch, device, args.max_grad_norm) for _ in range(args.steps)]

    metrics = {
        key: summarize([row[key] for row in measured_results])
        for key in [
            "step_time_s",
            "forward_time_s",
            "backward_time_s",
            "optimizer_time_s",
            "zero_grad_time_s",
        ]
    }
    result: dict[str, Any] = {
        "benchmark": "synthetic_training_step_smoke",
        "claim_scope": "mechanics_only_not_real_training_throughput",
        "compile": "disabled",
        "environment": environment(device),
        "config": {
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "batch_size": args.batch_size,
            "frames": args.frames,
            "mel_dim": args.mel_dim,
            "text_len": args.text_len,
            "vocab_size": args.vocab_size,
            "seed": args.seed,
            "lr": args.lr,
            "max_grad_norm": args.max_grad_norm,
            "precision": "float32",
            "optimizer": "AdamW",
        },
        "metrics": metrics,
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
