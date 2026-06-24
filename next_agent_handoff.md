# Next Agent Handoff

## Start Here

Read these files first:

1. `goal.md`
2. `goal_progress.md`
3. `memory.md`
4. `next_agent_handoff.md`
5. `compile_optimization_analysis.md`
6. `src/f5_tts/model/trainer.py`
7. `src/f5_tts/model/cfm.py`
8. `src/f5_tts/model/dataset.py`
9. `src/f5_tts/model/backbones/dit.py`

## Repo State

- Project root: `/media/mithex/NVME 2/Codex Linux/f5-tts-prs/F5-TTS`
- Branch: `main`
- Starting HEAD: `2ae2c9bd9b64dab2cb069c4b97e5e7673c521e01`
- Starting `F5-TTS/` git status: clean before tracking docs were added.
- Parent workspace git status is noisy because sibling project folders are untracked; ignore it unless the user asks about the parent repo.

## Current Task

The high-quality, backward-compatible `torch.compile` training support is
complete. Resume only for review feedback or the optional future work listed
below.

## Future-Agent Rules

Use this file as the single resume document for this branch. The former
`future_agent_instructions.md` content was consolidated here to avoid two
overlapping handoff files drifting apart.

- Keep compile and metrics default-off unless the user explicitly asks to run
  or benchmark them.
- Treat `compile.dynamic` as an experiment knob. Compare `null`, `true`, and
  `false` against the same baseline instead of assuming `dynamic=True` is best.
- Baseline first. Do not change hot-path behavior, dataset, sampler, padding,
  precision, batch size, gradient accumulation, optimizer, scheduler, logging,
  checkpointing, or measured region before collecting a comparable eager run.
- Pre-register any new benchmark hypothesis in `goal_progress.md`, then record
  raw numbers, compile startup, steady-state throughput, memory, limitations,
  graph breaks, recompiles, and failed experiments.
- Correctness comparisons must control seed, model config, initial weights or
  checkpoint, batch, precision, optimizer, scheduler, accumulation, and
  distributed setup. Include loss, selected gradients, parameter update, and
  longer smoke behavior where feasible.
- If dataset, sampler, bucketing, padding, collation, or data movement is
  touched, preserve ordering, seeded randomness, distributed sharding, epoch
  boundaries, resume semantics, filtering, masks, padding, and large-dataset
  memory behavior.
- CPU checks are acceptable for import/config/smoke correctness. Representative
  performance claims require realistic GPU training runs; big GPU runs must use
  Colab CLI with a named session, explicit artifact plan, cleanup, and final
  `colab sessions` verification.
- Keep dependency environments, caches, datasets, checkpoints, benchmark logs,
  notebooks, and temporary artifacts out of git unless intentionally tracked.
- Before stopping future work, inspect `git status` and `git diff`, update
  `goal_progress.md`, `memory.md`, and this handoff with durable decisions,
  benchmark evidence, caveats, and the exact stopping point.

## What Was Tried

- Performed initial audit of training entry points and hot paths.
- Confirmed local dataset is incomplete for real training.
- Confirmed local GPU exists but has 8 GiB VRAM.
- Confirmed system Python lacks training dependencies.
- Confirmed `/home/mithex/.venvs/ml312` exists and should be used as the ML venv.
- Confirmed Colab CLI is installed and no active Colab sessions were present.
- Created tracking files required by the objective.
- Added `benchmarks/training_step_smoke.py` and `benchmarks/eager_training_baseline.py`.
- Added `benchmarks/compile_correctness.py`.
- Ran local CPU synthetic smoke, local CPU small real-audio baseline, and Colab T4 small real-audio eager CUDA baseline.
- Stopped Colab session `f5tts-compile-baseline-t4`; `colab sessions` reported no active sessions after cleanup.
- Repaired local ML venv CUDA by reinstalling to `torch 2.11.0+cu128`, `torchaudio 2.11.0+cu128`, and `torchvision 0.26.0+cu128`.
- Verified after repair: minimal CUDA Conv1d, F5 synthetic CUDA smoke, F5 small real-audio CUDA baseline, and tiny Trainer CUDA compile smoke all pass.
- Fixed the broad compile correctness failure by compiling `CFM._forward_loss_core()` instead of the whole module. Stochastic CFM preprocessing now stays eager.
- Eager-vs-compiled correctness now passes on CPU and local CUDA for loss, cond, pred, gradients, and one AdamW parameter update.
- CUDA Trainer smoke now passes with compiled core active, fallback false, no `_orig_mod.` state keys, and `/tmp/f5tts_trainer_compile_core_smoke_cuda_v2/model_last.pt` written.
- Refactored the DiT training text embedding path to avoid `seq_len.max().item()` inside the compiled core by passing padded `x.shape[1]` plus per-sample valid lengths. The fixed-batch correctness runs no longer emit the previous `Tensor.item()` graph-break warning.
- Ran paired current-code Colab T4 eager/compiled comparison in session `f5tts-compile-paired-t4`, downloaded artifacts, stopped the session, and verified `colab sessions` had no active sessions.
- Added `compile_optimization_analysis.md` after the user explicitly warned not to blindly default to `dynamic=True`. It classifies compile boundaries, refactor-needed areas, eager-only areas, non-compile optimizations, and counterarguments.
- Tightened docs/config comments so `compile.dynamic` is an experiment knob, not a default recommendation.
- Added `benchmarks/benchmark_matrix_runner.py` to run Colab benchmark matrices from `/content/f5tts_matrix_spec.json` with per-case timeouts and partial-artifact preservation.
- Ran Colab L4 session `f5tts-l4-dynamic-matrix`, downloaded all artifacts to `/tmp/f5tts_colab_results/l4_dynamic_matrix/`, stopped the session, and verified `colab sessions` reported no active sessions.
- Added benchmark-only fixed-frame padding support and `benchmarks/fixed_shape_l4_matrix_spec.json`.
- Ran Colab L4 session `f5tts-l4-fixed-shape-matrix`, downloaded all artifacts to `/tmp/f5tts_colab_results/l4_fixed_shape_matrix/`, stopped the session, and verified `colab sessions` reported no active sessions.
- Moved compile fallback inside the deterministic CFM core, including CPU/CUDA RNG restoration before eager retry.
- Added DiT/UNetT eager text-width normalization before the compile boundary and proved it preserves existing embedding behavior.
- Added `tests/test_training_compile.py`; 14 CPU-safe tests now cover defaults, all three backbones, fallback, checkpoint/resume, accumulation metrics, sampling isolation, and `num_workers=0`.
- Reran the fixed-shape L4 matrix after text normalization. Artifacts are under `/tmp/f5tts_colab_results/l4_static_text_normalized/`; static false now has three graphs and no recompile-limit warning.
- Ran paired A100 F5TTS-small BF16 eager/static-compiled training. Artifacts are under `/tmp/f5tts_colab_results/a100_small_bf16_static/`; warmed speedup was `2.10698x`.

## What Worked

- Repo inspection is straightforward from `F5-TTS/`.
- The narrowest working compile target is the deterministic CFM training loss core after eager text/mask/random prep, not the full trainer or whole CFM module.
- The small real-audio eager baseline harness works on CPU and Colab T4.
- Warmed steady-state on the small real-audio T4 path exceeded 2x: eager mean step 0.0392404318s, compiled mean step 0.0176160816s, speedup 2.2275x.
- The current compile boundary is still deliberately narrow: CFM stochastic prep stays eager, the deterministic CFM loss core compiles, and trainer/dataset/checkpoint/sample paths stay eager.
- L4 tiny real-audio dynamic matrix showed `dynamic=default/null` was best among tested compile modes: eager mean step 0.0206601558s; compiled default 0.0152718000s; compiled true 0.0155186346s; compiled false 0.0210852804s with TorchDynamo recompile-limit warnings.
- F5TTS-small fits on L4 and compiled core can complete, but compile warmup is very large: compiled `dynamic=true` warmup steps were 183.19s and 181.89s before one measured step around 0.203s. This is feasibility evidence only, not a speedup claim.
- L4 fixed-audio-shape matrix with `--pad-to-frames 1536` showed default/null and dynamic true both improved warmed tiny throughput, but static false still failed as a policy candidate: eager mean step 0.0166920248s; compiled default 0.0112472860s (1.4841x, about 10,313 break-even steps); compiled true 0.0113972068s (1.4646x, about 17,502 break-even steps); compiled false 0.0170421016s with recompile-limit warnings from variable text length.
- The later text-normalization refactor fixed that static failure. Post-normalization L4: eager `0.0166214668s`; default `0.0109285218s` (`1.5209x`); dynamic true `0.0115034976s` (`1.4449x`); fresh-cache static false `0.0109868852s` (`1.5128x`), all with three graphs and no graph breaks.
- Representative A100 F5TTS-small BF16: eager `0.1267282088s`, compiled static false `0.0601469082s`, `2.10698x`; forward `2.4276x`, backward `2.1217x`; peak allocated memory down 11.42%. Cold break-even is about 3,824 steps.
- Five-step CUDA BF16 correctness passed at `atol=rtol=5e-3`; activation checkpointing and fullgraph one-step CUDA checks passed.

## What Has Not Been Done

- No multi-process DDP/FSDP validation.
- No production sampler/bucketing change; static false still requires stable audio widths supplied by the user's batching policy.
- No compile validation for DiT average text upsampling or flash-attention.
- No A100 `reduce-overhead`/`max-autotune` mode chase; neither is currently a clear low-risk improvement.
- Dataset evidence uses a real 32-sample subset, not a full-corpus epoch.

## Warnings

- Do not edit sampler/dataset/collation before baseline.
- Do not compare compiled results against a weaker baseline.
- Do not present synthetic smoke speedups as real training speedups.
- Do not include generated dependency caches, datasets, checkpoints, logs, or notebooks in git.
- `python` is absent; use `python3` or a venv command.
- Use `/home/mithex/.venvs/ml312` for local smoke work unless the user says otherwise.
- Big GPU benchmark runs must use Colab CLI, not the local RTX 3070.
- Do not allocate Colab compute without a named-session lifecycle plan and cleanup.
- Local CUDA is now usable for smoke checks, but representative/big GPU performance evidence still belongs on Colab per the objective.
- Do not go back to `torch.compile(self.model)` for F5-TTS correctness; that reintroduces stochastic preprocessing inside the compiled region.
- `uv run --python /home/mithex/.venvs/ml312/bin/python` created a repo-local `.venv` instead of using the ML venv. Use `VIRTUAL_ENV=/home/mithex/.venvs/ml312 PATH=/home/mithex/.venvs/ml312/bin:$PATH uv run --active --no-sync ...`; the accidental `.venv` was removed.
- Do not claim short-run speedup: paired T4 compiled warmup total was 149.47103661s vs eager warmup 2.4174570890s, so end-to-end for the small run was much worse compiled.
- Break-even estimate for the exact tiny T4 setup is about 6,931 post-warmup optimizer steps after including compile setup and warmup delta.
- Do not blindly use `--compile-dynamic true`. Test `default/null`, `true`, and when feasible `false`; static/bucketed shapes may be better than dynamic compile, but sampler or bucketing changes need their own baseline and compatibility tests.
- `DiT.TextEmbedding.average_upsample_text_by_mask()` is still not compile-friendly because it uses Python loops and tensor-to-int conversions.
- Do not claim F5TTS-small speedup from the L4 feasibility probes: eager used `warmup=1`, compiled used `warmup=2`, so the measured dataloader positions differ.
- Static false is now compileable when audio widths are fixed because DiT/UNetT text widths are normalized eagerly. It is still not safe to recommend for arbitrary unbucketed audio shapes.
- Do not interpret the A100 2.107x as short-run speedup: four warmups plus ten measured steps took about `257.23s` compiled versus `3.28s` eager.
- The first strict five-step AdamW parameter-allclose protocol failed; final multi-step parameter distances are diagnostic, while one-step math and loss/output trajectories are gates.

## Suggested Next Steps

1. Keep compile and metrics default-off. Recommend compile for long A100-style runs only after considering the measured break-even.
2. Future optional work: multi-process validation and an opt-in bucketed shape policy with sampler/resume compatibility tests.
3. Do not add another compile mode or benchmark without a new pre-registered hypothesis.

## Exact Commands Already Run

```bash
sed -n '1,240p' /home/mithex/.codex/attachments/ac47b788-464d-4f2b-86ba-d73c031d09d2/pasted-text-1.txt
sed -n '241,520p' /home/mithex/.codex/attachments/ac47b788-464d-4f2b-86ba-d73c031d09d2/pasted-text-1.txt
sed -n '521,760p' /home/mithex/.codex/attachments/ac47b788-464d-4f2b-86ba-d73c031d09d2/pasted-text-1.txt
git status --short --branch
git -C F5-TTS status --short --branch
git -C F5-TTS rev-parse --show-toplevel
git -C F5-TTS rev-parse HEAD
git log --oneline -5
git remote -v
python3 --version
python3 - <<'PY'
import importlib.util
for name in ['torch','torchaudio','accelerate','datasets','hydra','wandb','x_transformers','ema_pytorch','vocos']:
    print(name, importlib.util.find_spec(name) is not None)
PY
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader
sed -n '1,520p' /home/mithex/.codex/skills/colab-cli/SKILL.md
colab version
colab sessions
find /home/mithex -maxdepth 4 -type f -name pyvenv.cfg -printf '%h\n'
/home/mithex/.venvs/ml312/bin/python --version
uv pip list --python /home/mithex/.venvs/ml312/bin/python
```
