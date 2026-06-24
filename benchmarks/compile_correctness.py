#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
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
    parser = argparse.ArgumentParser(description="Compare eager and torch.compile F5-TTS training math.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--backend", default="inductor")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--dynamic", default="default", choices=["default", "true", "false"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--mel-dim", type=int, default=16)
    parser.add_argument("--text-len", type=int, default=24)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--precision", default="float32", choices=["float32", "bf16"])
    parser.add_argument("--checkpoint-activations", action="store_true")
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def run_git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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
        checkpoint_activations=args.checkpoint_activations,
    )
    # Set CFG drop probabilities to zero for this correctness harness. The
    # production path still supports dropout; this isolates tensor math from
    # intentionally stochastic branch selection.
    model = CFM(
        transformer=transformer,
        mel_spec_kwargs={"n_mel_channels": args.mel_dim},
        audio_drop_prob=0.0,
        cond_drop_prob=0.0,
    )
    return model.to(device)


def make_batch(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seed_everything(args.seed + 1)
    mel = torch.randn(args.batch_size, args.frames, args.mel_dim, device=device)
    text = torch.randint(0, args.vocab_size, (args.batch_size, args.text_len), device=device)
    if args.text_len > 4:
        text[1::2, -(args.text_len // 4) :] = -1
    lens = torch.full((args.batch_size,), args.frames, dtype=torch.long, device=device)
    if args.batch_size > 1 and args.frames > 8:
        lens[1::2] = args.frames - 8
    return mel, text, lens


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


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


def maybe_compile(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    compile_kwargs = {
        "backend": args.backend,
        "mode": args.mode,
        "fullgraph": args.fullgraph,
        "dynamic": compile_dynamic_value(args.dynamic),
    }
    compile_kwargs = {key: value for key, value in compile_kwargs.items() if value is not None}
    compile_training_core = getattr(model, "compile_training_core", None)
    if compile_training_core is not None:
        compile_training_core(**compile_kwargs)
        return model
    return torch.compile(model, **compile_kwargs)


def comparable_name(name: str) -> str:
    return name.removeprefix("_orig_mod.")


def run_one_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    seed: int,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    mel, text, lens = batch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    seed_everything(seed)
    sync(device)
    start = time.perf_counter()
    with autocast_context(device, precision):
        loss, cond, pred = model(mel, text=text, lens=lens)
    sync(device)
    forward_time = time.perf_counter() - start
    loss.backward()
    sync(device)
    backward_time = time.perf_counter() - start - forward_time
    grads = {
        comparable_name(name): param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    optimizer.step()
    sync(device)
    total_time = time.perf_counter() - start
    params = {comparable_name(name): param.detach().clone() for name, param in model.named_parameters()}
    return {
        "loss": loss.detach().clone(),
        "cond": cond.detach().clone(),
        "pred": pred.detach().clone(),
        "grads": grads,
        "params": params,
        "forward_time_s": forward_time,
        "backward_time_s": backward_time,
        "total_time_s": total_time,
    }


def tensor_compare(
    eager: torch.Tensor,
    compiled: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    eager_cpu = eager.detach().float().cpu()
    compiled_cpu = compiled.detach().float().cpu()
    abs_diff = (eager_cpu - compiled_cpu).abs()
    rel_diff = abs_diff / eager_cpu.abs().clamp_min(1e-12)
    return {
        "allclose": bool(torch.allclose(eager_cpu, compiled_cpu, atol=atol, rtol=rtol)),
        "max_abs": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "max_rel": float(rel_diff.max().item()) if rel_diff.numel() else 0.0,
        "mean_abs": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
        "shape": list(eager_cpu.shape),
    }


def mapping_compare(
    eager: dict[str, torch.Tensor],
    compiled: dict[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    common = sorted(set(eager) & set(compiled))
    missing_from_compiled = sorted(set(eager) - set(compiled))
    extra_in_compiled = sorted(set(compiled) - set(eager))
    entries = {name: tensor_compare(eager[name], compiled[name], atol=atol, rtol=rtol) for name in common}
    failed = {name: value for name, value in entries.items() if not value["allclose"]}
    max_abs_name = max(common, key=lambda name: entries[name]["max_abs"], default=None)
    max_rel_name = max(common, key=lambda name: entries[name]["max_rel"], default=None)
    diff_l2_sq = sum(
        float((eager[name].detach().float().cpu() - compiled[name].detach().float().cpu()).square().sum())
        for name in common
    )
    eager_l2_sq = sum(float(eager[name].detach().float().cpu().square().sum()) for name in common)
    return {
        "allclose": not failed and not missing_from_compiled and not extra_in_compiled,
        "count": len(common),
        "failed_count": len(failed),
        "failed_names": list(failed)[:20],
        "missing_from_compiled": missing_from_compiled[:20],
        "extra_in_compiled": extra_in_compiled[:20],
        "max_abs": entries[max_abs_name]["max_abs"] if max_abs_name is not None else 0.0,
        "max_abs_name": max_abs_name,
        "max_rel": entries[max_rel_name]["max_rel"] if max_rel_name is not None else 0.0,
        "max_rel_name": max_rel_name,
        "mean_abs": statistics.fmean(value["mean_abs"] for value in entries.values()) if entries else 0.0,
        "l2_abs": diff_l2_sq**0.5,
        "l2_relative": (diff_l2_sq / max(eager_l2_sq, 1e-24)) ** 0.5,
        "sample": {name: entries[name] for name in common[:5]},
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
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
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


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    device = select_device(args.device)
    default_tolerance = 5e-3 if args.precision == "bf16" else (1e-4 if device.type == "cuda" else 1e-5)
    atol = args.atol if args.atol is not None else default_tolerance
    rtol = args.rtol if args.rtol is not None else default_tolerance

    seed_everything(args.seed)
    eager_model = build_model(args, device)
    model_state = clone_state_dict(eager_model)
    compiled_model = build_model(args, device)
    compiled_model.load_state_dict(model_state)
    compiled_model = maybe_compile(compiled_model, args)

    batch = make_batch(args, device)
    eager_optimizer = torch.optim.AdamW(eager_model.parameters(), lr=args.lr)
    compiled_optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=args.lr)

    eager_steps = [
        run_one_step(eager_model, eager_optimizer, batch, args.seed + 2 + step, device, args.precision)
        for step in range(args.steps)
    ]
    compiled_steps = [
        run_one_step(compiled_model, compiled_optimizer, batch, args.seed + 2 + step, device, args.precision)
        for step in range(args.steps)
    ]
    eager = eager_steps[-1]
    compiled = compiled_steps[-1]

    loss_trajectory = tensor_compare(
        torch.stack([step["loss"] for step in eager_steps]),
        torch.stack([step["loss"] for step in compiled_steps]),
        atol=atol,
        rtol=rtol,
    )
    if args.steps == 1:
        comparisons = {
            "loss": tensor_compare(eager["loss"], compiled["loss"], atol=atol, rtol=rtol),
            "cond": tensor_compare(eager["cond"], compiled["cond"], atol=atol, rtol=rtol),
            "pred": tensor_compare(eager["pred"], compiled["pred"], atol=atol, rtol=rtol),
            "grads": mapping_compare(eager["grads"], compiled["grads"], atol=atol, rtol=rtol),
            "params_after_update": mapping_compare(eager["params"], compiled["params"], atol=atol, rtol=rtol),
        }
        gating_names = list(comparisons)
        diagnostic_only = []
    else:
        first_eager = eager_steps[0]
        first_compiled = compiled_steps[0]
        comparisons = {
            "first_step_loss": tensor_compare(
                first_eager["loss"], first_compiled["loss"], atol=atol, rtol=rtol
            ),
            "first_step_cond": tensor_compare(
                first_eager["cond"], first_compiled["cond"], atol=atol, rtol=rtol
            ),
            "first_step_pred": tensor_compare(
                first_eager["pred"], first_compiled["pred"], atol=atol, rtol=rtol
            ),
            "first_step_grads": mapping_compare(
                first_eager["grads"], first_compiled["grads"], atol=atol, rtol=rtol
            ),
            "first_step_params_after_update": mapping_compare(
                first_eager["params"], first_compiled["params"], atol=atol, rtol=rtol
            ),
            "loss_trajectory": loss_trajectory,
            "final_loss": tensor_compare(eager["loss"], compiled["loss"], atol=atol, rtol=rtol),
            "final_cond": tensor_compare(eager["cond"], compiled["cond"], atol=atol, rtol=rtol),
            "final_pred": tensor_compare(eager["pred"], compiled["pred"], atol=atol, rtol=rtol),
            "final_grads": mapping_compare(eager["grads"], compiled["grads"], atol=atol, rtol=rtol),
            "final_params_after_update": mapping_compare(
                eager["params"], compiled["params"], atol=atol, rtol=rtol
            ),
        }
        gating_names = [
            "first_step_loss",
            "first_step_cond",
            "first_step_pred",
            "first_step_grads",
            "first_step_params_after_update",
            "loss_trajectory",
            "final_loss",
            "final_cond",
            "final_pred",
        ]
        diagnostic_only = ["final_grads", "final_params_after_update"]
    passed = all(comparisons[name]["allclose"] for name in gating_names)
    result: dict[str, Any] = {
        "benchmark": "compile_correctness",
        "passed": passed,
        "gating_comparisons": gating_names,
        "diagnostic_only_comparisons": diagnostic_only,
        "all_comparisons_allclose": all(item["allclose"] for item in comparisons.values()),
        "environment": environment(device),
        "config": {
            "backend": args.backend,
            "mode": args.mode,
            "fullgraph": args.fullgraph,
            "dynamic": args.dynamic,
            "batch_size": args.batch_size,
            "frames": args.frames,
            "mel_dim": args.mel_dim,
            "text_len": args.text_len,
            "vocab_size": args.vocab_size,
            "seed": args.seed,
            "steps": args.steps,
            "lr": args.lr,
            "precision": args.precision,
            "checkpoint_activations": args.checkpoint_activations,
            "audio_drop_prob": 0.0,
            "cond_drop_prob": 0.0,
            "atol": atol,
            "rtol": rtol,
        },
        "timings": {
            "eager_forward_s": sum(step["forward_time_s"] for step in eager_steps),
            "eager_backward_s": sum(step["backward_time_s"] for step in eager_steps),
            "eager_total_s": sum(step["total_time_s"] for step in eager_steps),
            "compiled_forward_s": sum(step["forward_time_s"] for step in compiled_steps),
            "compiled_backward_s": sum(step["backward_time_s"] for step in compiled_steps),
            "compiled_total_s": sum(step["total_time_s"] for step in compiled_steps),
        },
        "comparisons": comparisons,
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
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
