#!/usr/bin/env python3
"""Colab real-data Trainer benchmark for F5-TTS torch.compile candidates.

This script is intentionally self-contained for `colab run --gpu G4`:
- clones the requested F5-TTS branch on the Colab VM,
- installs the package,
- builds a real LibriSpeech subset with precomputed mel spectrograms,
- runs each candidate in a fresh Python process with the real Trainer/Accelerate stack,
- records timing, memory, fallback, loss, and shape-cardinality evidence.

It does not modify the source-of-truth checkout. Candidate-only behavior that is not
exposed by the current PR CLI, such as torch.compile options and cudagraph mark-step,
is injected by the benchmark driver around Trainer construction so it can be validated
before deciding whether to add upstream API fields.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

REPO_URL = "https://github.com/hcsolakoglu/F5-TTS.git"
BRANCH = "torch-compile-upstream-integration"
COMMIT = "e04c641e6cc38f280d1987e58e853313c3cc44db"
DATASET_NAME = "openslr/librispeech_asr"
DATASET_CONFIG = "clean"
DATASET_SPLIT = "train.100"

CANDIDATES: dict[str, dict[str, Any]] = {
    "eager": {"compile": False, "mark_step": False},
    "compiled_default": {"compile": True, "mode": None, "options": None, "mark_step": False},
    "compiled_dynamic_true": {"compile": True, "mode": None, "options": None, "mark_step": False, "dynamic": True},
    "compiled_mark_dynamic": {
        "compile": True,
        "mode": None,
        "options": None,
        "mark_step": False,
        "dynamic": None,
        "mark_dynamic": True,
        "automatic_dynamic_shapes": False,
    },
    "reduce_overhead_markstep": {"compile": True, "mode": "reduce-overhead", "options": None, "mark_step": True},
    "combo_cudagraphs_markstep": {
        "compile": True,
        "mode": None,
        "mark_step": True,
        "options": {
            "triton.cudagraphs": True,
            "combo_kernels": True,
            "benchmark_combo_kernel": True,
            "combo_kernels_autotune": 2,
            "combo_kernel_allow_mixed_sizes": 2,
        },
    },
}


def run(cmd: list[str] | str, cwd: str | Path | None = None, env: dict[str, str] | None = None) -> None:
    printable = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"\n$ {printable}", flush=True)
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        shell=isinstance(cmd, str),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, cmd)


def capture(cmd: list[str] | str, cwd: str | Path | None = None, env: dict[str, str] | None = None) -> str:
    printable = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"\n$ {printable}", flush=True)
    return subprocess.check_output(cmd, cwd=cwd, env=env, shell=isinstance(cmd, str), text=True, stderr=subprocess.STDOUT)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def setup_repo(work_dir: Path, repo_url: str, branch: str, commit: str | None) -> Path:
    repo = work_dir / "F5-TTS"
    if repo.exists():
        run(["git", "-C", str(repo), "fetch", "origin", branch, "--depth", "1"])
    else:
        run(["git", "clone", "--branch", branch, "--single-branch", "--depth", "1", repo_url, str(repo)])
    if commit:
        run(["git", "-C", str(repo), "fetch", "origin", commit, "--depth", "1"])
        run(["git", "-C", str(repo), "checkout", commit])
    head = capture(["git", "rev-parse", "HEAD"], cwd=repo).strip()
    print(f"Repo HEAD: {head}", flush=True)
    return repo


def install_repo(repo: Path) -> None:
    # Colab normally ships CUDA torch/torchaudio. Keep them, install project deps.
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip", "setuptools", "wheel"])
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo)])


def prepare_librispeech_subset(
    repo: Path,
    out_file: Path,
    samples: int,
    fetch_rows: int,
    max_duration: float,
    min_duration: float,
) -> dict[str, Any]:
    sys.path.insert(0, str(repo / "src"))
    import torch
    import torchaudio
    from datasets import Audio, load_dataset
    from f5_tts.model.modules import MelSpec

    if out_file.exists():
        print(f"Using existing preprocessed dataset: {out_file}", flush=True)
        obj = torch.load(out_file, map_location="cpu")
        return obj["metadata"]

    print(
        f"Streaming real dataset {DATASET_NAME}/{DATASET_CONFIG} split={DATASET_SPLIT} scan_rows<={fetch_rows}",
        flush=True,
    )
    # Streaming avoids materialising the full LibriSpeech config. Non-streaming split slices
    # still resolve/download many parquet files on Colab, which is slow and wasteful.
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT, streaming=True)
    ds = ds.shuffle(buffer_size=min(max(fetch_rows, samples), 5000), seed=666)
    # Decode at source sample rate first. Repo dataset wrapper resamples to 24 kHz.
    ds = ds.cast_column("audio", Audio(decode=True))

    mel_spec = MelSpec(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24_000,
        mel_spec_type="vocos",
    )
    resamplers: dict[int, Any] = {}
    mels: list[torch.Tensor] = []
    texts: list[str] = []
    durations: list[float] = []
    speakers: set[str] = set()
    frame_lengths: list[int] = []

    start = time.perf_counter()
    with torch.no_grad():
        scanned = 0
        for row in ds:
            scanned += 1
            if scanned > fetch_rows:
                break
            audio = row["audio"]
            array = audio["array"]
            sr = int(audio["sampling_rate"])
            duration = float(len(array) / sr)
            if duration < min_duration or duration > max_duration:
                continue
            wav = torch.as_tensor(array, dtype=torch.float32).unsqueeze(0)
            if sr != 24_000:
                if sr not in resamplers:
                    resamplers[sr] = torchaudio.transforms.Resample(sr, 24_000)
                wav = resamplers[sr](wav)
            mel = mel_spec(wav).squeeze(0).contiguous().cpu()
            text = str(row.get("text") or row.get("sentence") or "").strip()
            if not text:
                continue
            mels.append(mel)
            texts.append(text)
            durations.append(duration)
            frame_lengths.append(int(mel.shape[-1]))
            if "speaker_id" in row:
                speakers.add(str(row["speaker_id"]))
            elif "speaker" in row:
                speakers.add(str(row["speaker"]))
            if len(mels) >= samples:
                break

    if len(mels) < samples:
        raise RuntimeError(f"Only prepared {len(mels)} valid samples, requested {samples}")

    metadata = {
        "dataset": DATASET_NAME,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "samples": len(mels),
        "fetch_rows": fetch_rows,
        "streaming": True,
        "min_duration_s": min(durations),
        "median_duration_s": statistics.median(durations),
        "max_duration_s": max(durations),
        "total_hours": sum(durations) / 3600,
        "unique_speakers": len(speakers) if speakers else None,
        "min_frames": min(frame_lengths),
        "median_frames": statistics.median(frame_lengths),
        "max_frames": max(frame_lengths),
        "unique_frame_lengths": len(set(frame_lengths)),
        "preprocess_wall_s": time.perf_counter() - start,
        "dtype": "float32",
        "sample_rate": 24_000,
        "n_mel_channels": 100,
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mels": mels, "texts": texts, "durations": durations, "metadata": metadata}, out_file)
    write_json(out_file.with_suffix(".metadata.json"), metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    return metadata


def candidate_worker(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    data_file = Path(args.data_file).resolve()
    out_dir = Path(args.out_dir).resolve()
    candidate = args.candidate
    cfg = CANDIDATES[candidate]

    env_info: dict[str, Any] = {}
    sys.path.insert(0, str(repo / "src"))

    import torch
    from torch.utils.data import Dataset, SequentialSampler
    from f5_tts.model import CFM, DiT, Trainer
    import f5_tts.model.trainer as trainer_mod
    from f5_tts.model.dataset import DynamicBatchSampler
    from f5_tts.model.utils import get_tokenizer

    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass

    try:
        import torch._dynamo as dynamo

        dynamo.reset()
        dynamo.utils.counters.clear()
        if cfg.get("automatic_dynamic_shapes") is not None:
            dynamo.config.automatic_dynamic_shapes = bool(cfg["automatic_dynamic_shapes"])
    except Exception:
        dynamo = None

    class PrecomputedMelDataset(Dataset):
        def __init__(self, path: Path):
            obj = torch.load(path, map_location="cpu")
            self.mels = obj["mels"]
            self.texts = obj["texts"]
            self.durations = obj["durations"]
            self.metadata = obj["metadata"]

        def __len__(self) -> int:
            return len(self.mels)

        def get_frame_len(self, index: int) -> int:
            return int(self.mels[index].shape[-1])

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {"mel_spec": self.mels[index], "text": self.texts[index]}

    dataset = PrecomputedMelDataset(data_file)
    sampler = SequentialSampler(dataset)
    batch_sampler = DynamicBatchSampler(
        sampler,
        args.batch_size_per_gpu,
        max_samples=args.max_samples,
        random_seed=args.seed,
        drop_residual=False,
    )
    batches_per_epoch = len(batch_sampler)
    updates_expected = batches_per_epoch * args.epochs
    batch_shapes = []
    for batch in batch_sampler.batches:
        lengths = [dataset.get_frame_len(i) for i in batch]
        texts = [len(dataset.texts[i]) for i in batch]
        batch_shapes.append((len(batch), max(lengths), max(texts)))

    vocab_char_map, vocab_size = get_tokenizer("", "byte")
    model_cfg = dict(
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        ff_mult=2,
        text_dim=args.text_dim,
        text_mask_padding=True,
        qk_norm=None,
        conv_layers=4,
        pe_attn_head=None,
        attn_backend="torch",
        attn_mask_enabled=False,
        checkpoint_activations=False,
    )
    mel_spec_kwargs = dict(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24_000,
        mel_spec_type="vocos",
    )
    model = CFM(
        transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    compile_enabled_for_trainer = bool(cfg["compile"] and cfg.get("options") is None)
    trainer = Trainer(
        model,
        epochs=args.epochs,
        learning_rate=7.5e-5,
        num_warmup_updates=1,
        save_per_updates=10**9,
        keep_last_n_checkpoints=0,
        checkpoint_path=str(out_dir / "ckpts" / candidate),
        batch_size_per_gpu=args.batch_size_per_gpu,
        batch_size_type="frame",
        max_samples=args.max_samples,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        logger=None,
        wandb_project="f5tts-real-trainer-bench",
        wandb_run_name=candidate,
        log_samples=False,
        last_per_updates=10**9,
        bnb_optimizer=False,
        mel_spec_type="vocos",
        compile_enabled=compile_enabled_for_trainer,
        compile_backend="inductor",
        compile_mode=cfg.get("mode"),
        compile_fullgraph=False,
        compile_dynamic=cfg.get("dynamic"),
        compile_fallback_to_eager=False,
        global_masked_mean=False,
    )
    # Avoid checkpoint I/O dominating short benchmark timing while preserving Trainer/Accelerate training behavior.
    trainer.save_checkpoint = lambda *a, **k: None  # type: ignore[method-assign]

    if cfg["compile"] and cfg.get("options") is not None:
        trainer._unwrapped_model.compile_training_core(
            backend="inductor",
            fullgraph=False,
            dynamic=cfg.get("dynamic"),
            options=cfg["options"],
            runtime_fallback=False,
        )
        trainer.compile_active = True

    if cfg.get("mark_dynamic"):
        if dynamo is None or not hasattr(dynamo, "mark_dynamic"):
            raise RuntimeError("torch._dynamo.mark_dynamic is unavailable")
        module = trainer._unwrapped_model
        original_run_loss_core = module._run_loss_core

        def run_loss_core_with_mark_dynamic(*loss_args: Any):
            # _run_loss_core args: x1, text, mask, rand_span_mask, x0, time, drop_audio_cond, drop_text.
            # The real LibriSpeech dynamic sampler produced 65 unique (mel_seq, text_seq) shapes;
            # marking sequence dims lets Dynamo generalize those dimensions instead of specialising every batch.
            for index, dim in ((0, 1), (1, 1), (2, 1), (3, 1), (4, 1)):
                value = loss_args[index]
                if torch.is_tensor(value) and value.dim() > dim:
                    dynamo.mark_dynamic(value, dim)
            return original_run_loss_core(*loss_args)

        module._run_loss_core = run_loss_core_with_mark_dynamic  # type: ignore[method-assign]

    if cfg.get("mark_step"):
        mark_step = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
        if mark_step is None:
            raise RuntimeError("torch.compiler.cudagraph_mark_step_begin is unavailable")
        module = trainer._unwrapped_model
        original_forward = module.forward

        def forward_with_mark_step(*forward_args: Any, **forward_kwargs: Any):
            if trainer.compile_active:
                mark_step()
            return original_forward(*forward_args, **forward_kwargs)

        module.forward = forward_with_mark_step  # type: ignore[method-assign]

    update_records: list[dict[str, Any]] = []
    original_tqdm = trainer_mod.tqdm

    class BenchProgress:
        def __init__(self, iterable, desc=None, unit=None, disable=False, initial=0, **kwargs):
            self.iterable = iterable
            self.desc = desc or ""
            self.disable = disable
            self.count = int(initial or 0)
            self.last = time.perf_counter()

        def update(self, n=1):
            now = time.perf_counter()
            self.count += int(n)
            rec = {"desc": self.desc, "update_in_epoch": self.count, "dt_s": now - self.last}
            if torch.cuda.is_available():
                rec["max_alloc_mb"] = torch.cuda.max_memory_allocated() / 1024**2
                rec["max_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024**2
            update_records.append(rec)
            self.last = now

        def set_postfix(self, **kwargs):
            if update_records:
                update_records[-1]["postfix"] = kwargs

    trainer_mod.tqdm = BenchProgress
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    error = None
    try:
        trainer.train(dataset, num_workers=args.num_workers, resumable_with_seed=args.seed)
    except Exception as exc:  # record failures as evidence, then re-raise after JSON write
        error = repr(exc)
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - start
        trainer_mod.tqdm = original_tqdm

    losses = []
    for rec in update_records:
        postfix = rec.get("postfix") or {}
        try:
            losses.append(float(postfix.get("loss")))
        except Exception:
            pass

    counters = {}
    if dynamo is not None:
        try:
            counters = {k: dict(v) for k, v in dynamo.utils.counters.items()}
        except Exception:
            counters = {"error": "could not serialize dynamo counters"}

    state = getattr(trainer._unwrapped_model, "training_compile_state", None)
    result = {
        "candidate": candidate,
        "error": error,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "precision": {
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else None,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "amp_or_quantization": "none",
        },
        "model_cfg": model_cfg,
        "params": sum(p.numel() for p in trainer._unwrapped_model.parameters()),
        "dataset_metadata": dataset.metadata,
        "batch_size_per_gpu_frames": args.batch_size_per_gpu,
        "max_samples_per_batch": args.max_samples,
        "epochs": args.epochs,
        "batches_per_epoch": batches_per_epoch,
        "updates_expected": updates_expected,
        "updates_recorded": len(update_records),
        "unique_batch_shapes": len(set(batch_shapes)),
        "first_12_batch_shapes": batch_shapes[:12],
        "wall_s": wall,
        "mean_update_s": wall / max(1, len(update_records)),
        "median_recorded_update_s": statistics.median([r["dt_s"] for r in update_records]) if update_records else None,
        "p90_recorded_update_s": statistics.quantiles([r["dt_s"] for r in update_records], n=10)[8]
        if len(update_records) >= 10
        else None,
        "first_recorded_update_s": update_records[0]["dt_s"] if update_records else None,
        "last_recorded_update_s": update_records[-1]["dt_s"] if update_records else None,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_finite": all(map(lambda x: x == x and abs(x) < float("inf"), losses)) if losses else None,
        "compile_active_end": bool(getattr(trainer, "compile_active", False)),
        "compile_fallback_active_end": bool(getattr(trainer, "compile_fallback_active", False)),
        "model_training_compile_state": state,
        "dynamo_counters": counters,
        "candidate_config": cfg,
    }
    if torch.cuda.is_available():
        result.update(
            {
                "max_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
                "max_memory_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
            }
        )
    write_json(out_dir / f"{candidate}.json", result)
    with (out_dir / "results.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result, sort_keys=True) + "\n")
    print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    if error is not None:
        raise RuntimeError(error)


def run_correctness_probe(repo: Path, data_file: Path, out_dir: Path, candidate_names: list[str], args: argparse.Namespace) -> None:
    # Run correctness probes as separate workers with tiny one-batch script code to avoid contaminating training processes.
    probe = out_dir / "correctness_probe.py"
    code = textwrap.dedent(
        f"""
        import copy
        import json
        import sys
        from pathlib import Path

        import torch

        repo = Path({str(repo)!r})
        data_file = Path({str(data_file)!r})
        out_dir = Path({str(out_dir)!r})
        sys.path.insert(0, str(repo / "src"))

        from f5_tts.model import CFM, DiT
        from f5_tts.model.dataset import collate_fn
        from f5_tts.model.utils import get_tokenizer

        torch.manual_seed({args.seed})
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all({args.seed})
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")

        obj = torch.load(data_file, map_location="cpu")
        batch = [{{"mel_spec": obj["mels"][i], "text": obj["texts"][i]}} for i in range(min(4, len(obj["mels"])))]
        b = collate_fn(batch)
        mel = b["mel"].permute(0, 2, 1).cuda()
        lens = b["mel_lengths"].cuda()
        text = b["text"]
        vocab_char_map, vocab_size = get_tokenizer("", "byte")
        model_cfg = dict(
            dim={args.dim},
            depth={args.depth},
            heads={args.heads},
            ff_mult=2,
            text_dim={args.text_dim},
            text_mask_padding=True,
            qk_norm=None,
            conv_layers=4,
            pe_attn_head=None,
            attn_backend="torch",
            attn_mask_enabled=False,
            checkpoint_activations=False,
        )
        mel_spec_kwargs = dict(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=100,
            target_sample_rate=24000,
            mel_spec_type="vocos",
        )

        def build():
            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            return CFM(
                transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
                mel_spec_kwargs=mel_spec_kwargs,
                vocab_char_map=vocab_char_map,
            ).cuda().train()

        base = build()
        cand_template = copy.deepcopy(base)
        torch.manual_seed(999)
        torch.cuda.manual_seed_all(999)
        prepared = base._prepare_training_inputs(mel.clone(), text, lens.clone())
        loss, cond, pred = base._run_loss_core(*prepared)
        loss.backward()
        base_grads = [p.grad.detach().clone() if p.grad is not None else None for p in base.parameters()]
        candidates = json.loads({json.dumps(json.dumps(CANDIDATES))})
        selected = set({candidate_names!r})
        results = []
        for name, cfg in candidates.items():
            if name not in selected or not cfg.get("compile"):
                continue
            m = copy.deepcopy(cand_template).cuda().train()
            kwargs = {{
                "backend": "inductor",
                "fullgraph": False,
                "dynamic": cfg.get("dynamic"),
                "runtime_fallback": False,
            }}
            if cfg.get("options") is not None:
                kwargs["options"] = cfg["options"]
            elif cfg.get("mode") is not None:
                kwargs["mode"] = cfg["mode"]
            m.compile_training_core(**kwargs)
            if cfg.get("mark_step"):
                torch.compiler.cudagraph_mark_step_begin()
            args2 = tuple(x.detach().clone() if torch.is_tensor(x) else x for x in prepared)
            closs, ccond, cpred = m._run_loss_core(*args2)
            closs.backward()
            max_grad = 0.0
            for bg, p in zip(base_grads, m.parameters()):
                if bg is not None and p.grad is not None:
                    max_grad = max(max_grad, float((bg - p.grad).abs().max().detach().cpu()))
            results.append(
                {{
                    "candidate": name,
                    "loss_abs_diff": float(abs(loss.detach().cpu() - closs.detach().cpu())),
                    "pred_max_abs_diff": float((pred.detach().cpu() - cpred.detach().cpu()).abs().max()),
                    "grad_max_abs_diff": max_grad,
                    "state": m.training_compile_state,
                }}
            )

        (out_dir / "correctness.json").write_text(json.dumps(results, indent=2, sort_keys=True) + chr(10))
        print("CORRECTNESS_JSON=" + json.dumps(results, sort_keys=True), flush=True)
        """
    ).lstrip()
    probe.write_text(code, encoding="utf-8")
    run([sys.executable, str(probe)])


def orchestrate(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir).resolve()
    out_dir = work_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=== host ===", flush=True)
    print(capture("uname -a || true"), flush=True)
    print(capture("nvidia-smi || true"), flush=True)

    repo = setup_repo(work_dir, args.repo_url, args.branch, args.commit)
    install_repo(repo)
    metadata = prepare_librispeech_subset(
        repo=repo,
        out_file=work_dir / "data" / f"librispeech_{args.samples}_mel.pt",
        samples=args.samples,
        fetch_rows=args.fetch_rows,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
    )

    selected = [name.strip() for name in args.candidates.split(",") if name.strip()]
    for name in selected:
        if name not in CANDIDATES:
            raise ValueError(f"Unknown candidate: {name}")

    if args.skip_correctness:
        print("Skipping correctness probe; relying on prior same-script probe run.", flush=True)
    else:
        run_correctness_probe(repo, work_dir / "data" / f"librispeech_{args.samples}_mel.pt", out_dir, selected, args)

    common = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--repo",
        str(repo),
        "--data-file",
        str(work_dir / "data" / f"librispeech_{args.samples}_mel.pt"),
        "--out-dir",
        str(out_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size-per-gpu",
        str(args.batch_size_per_gpu),
        "--max-samples",
        str(args.max_samples),
        "--num-workers",
        str(args.num_workers),
        "--dim",
        str(args.dim),
        "--depth",
        str(args.depth),
        "--heads",
        str(args.heads),
        "--text-dim",
        str(args.text_dim),
        "--seed",
        str(args.seed),
    ]
    for name in selected:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo / "src")
        env["TORCHINDUCTOR_CACHE_DIR"] = str(out_dir / "inductor_cache" / name)
        if not env.get("TORCH_LOGS"):
            env.pop("TORCH_LOGS", None)
        env["WANDB_DISABLED"] = "true"
        print(f"\n=== candidate {name} ===", flush=True)
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                common + ["--candidate", name],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if completed.stdout:
                print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
            if completed.returncode:
                raise subprocess.CalledProcessError(completed.returncode, common + ["--candidate", name])
        finally:
            print(f"candidate {name} elapsed_s={time.perf_counter()-start:.3f}", flush=True)

    rows = []
    results_path = out_dir / "results.jsonl"
    if results_path.exists():
        rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    by_name = {r["candidate"]: r for r in rows}
    base = by_name.get("eager")
    compiled = by_name.get("compiled_default")
    summary = {
        "repo": {"url": args.repo_url, "branch": args.branch, "commit": capture(["git", "rev-parse", "HEAD"], cwd=repo).strip()},
        "dataset_metadata": metadata,
        "args": vars(args),
        "correctness": json.loads((out_dir / "correctness.json").read_text()) if (out_dir / "correctness.json").exists() else None,
        "results": rows,
    }
    for r in rows:
        if base and r.get("wall_s") and base.get("wall_s"):
            r["speedup_vs_eager_wall"] = base["wall_s"] / r["wall_s"]
        if compiled and r.get("wall_s") and compiled.get("wall_s"):
            r["speedup_vs_compiled_default_wall"] = compiled["wall_s"] / r["wall_s"]
    write_json(out_dir / "summary.json", summary)
    print("\n=== FINAL SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True)[:60000], flush=True)
    print(f"ARTIFACT_DIR={out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument("--commit", default=COMMIT)
    parser.add_argument("--work-dir", default="/content/f5tts_real_trainer_bench")
    parser.add_argument("--repo")
    parser.add_argument("--data-file")
    parser.add_argument("--out-dir")
    parser.add_argument("--candidate", choices=sorted(CANDIDATES))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--fetch-rows", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size-per-gpu", type=int, default=12000)
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=12.0)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--text-dim", type=int, default=512)
    parser.add_argument(
        "--candidates",
        default="eager,compiled_default,compiled_dynamic_true,reduce_overhead_markstep,combo_cudagraphs_markstep",
    )
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--skip-correctness", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        candidate_worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
