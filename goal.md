# Goal: torch.compile Training Support

## Objective

Add backward-compatible `torch.compile` support to the F5-TTS training pipeline, preserve baseline correctness, improve real training throughput where feasible, and prove every performance and correctness claim with controlled measurements.

## Constraints

- Existing training behavior must remain backward compatible.
- `torch.compile` must be optional and disabled by default unless evidence later proves a default change is safe.
- Existing Hydra configs, CLI flags, training scripts, checkpoints, logging, sampling, validation, resume flows, dataset loading, samplers, bucketing, and collation must keep working.
- Do not modify sampler, dataset pipeline, collation, bucketing, or major training behavior before baseline measurements exist.
- Do not wrap the full training pipeline blindly. Keep logging, checkpointing, sampling, data loading, optimizer orchestration, and other Python side effects eager unless evidence supports otherwise.
- Keep the repo clean. No hidden state, local paths, benchmark-only shortcuts, undocumented hacks, generated caches, datasets, notebooks, checkpoints, or temporary artifacts in git.

## Success Criteria

- Optional compile controls are available through the repo's existing config or CLI style.
- Defaults reproduce eager behavior.
- Eager and compiled paths match within pre-declared tolerances.
- Measurements include real training throughput, startup cost, steady-state speed, memory impact, loss behavior, and resource notes.
- The implementation reaches at least 2x training speedup if technically feasible, but does not overclaim if evidence shows it is not feasible.
- Optimization continues past 2x only when low risk and evidence-supported.

## Non-Goals

- No compile-only removal of features such as logging, validation, sampling, checkpointing, resume, gradient clipping, or scheduler behavior.
- No hardcoded dataset, GPU, batch shape, sample rate, or local path.
- No broad model or dataset rewrite without measured need.
- No speedup claims from tiny synthetic examples alone.

## Planned Strategy

1. Audit training entry points, model forward/loss path, optimizer/scheduler, AMP/Accelerate behavior, gradient accumulation, data loading, dataset/sampler/collation, checkpoint/resume, logging, validation/sampling, distributed behavior, configs, CLI, and existing tests.
2. Collect eager baseline measurements from the current repo before hot-path changes.
3. Add low-overhead optional metrics that use the same instrumentation for eager and compiled runs.
4. Add conservative compile config plumbing with compile disabled by default.
5. Start with the smallest safe compile target: `CFM._forward_loss_core()` after eager text/mask/noise/time/CFG preparation, while keeping training orchestration eager.
6. Add correctness tests comparing eager vs compiled forward/loss and selected gradients on fixed batches.
7. Benchmark compile knobs one at a time, record graph breaks/recompiles where practical, and keep only changes supported by evidence.

## Current Audit Notes

- Main Hydra training entry: `src/f5_tts/train/train.py`.
- CLI finetune entry: `src/f5_tts/train/finetune_cli.py`.
- Gradio finetune entry has its own training flow setup in `src/f5_tts/train/finetune_gradio.py`.
- Training loop and checkpoint/resume/logging/sampling orchestration: `src/f5_tts/model/trainer.py`.
- Core training loss path: `src/f5_tts/model/cfm.py::CFM.forward`.
- Dataset, dynamic batch sampler, and collation: `src/f5_tts/model/dataset.py`.
- Backbones include DiT, MMDiT, and UNetT. Initial audit focused on DiT and MMDiT because default F5-TTS configs use DiT.
- Compile-hostile sources in current forward path include Python `random()` in `CFM.forward`, text list tokenization inside forward, dynamic sequence lengths, `int(tensor.item())` conversions in DiT text embedding when average upsampling is enabled, cache side effects used for inference, optional activation checkpointing, and data-dependent masks.
- Safe first compile target is likely a prepared tensor forward/loss call with text already tensorized and sampling/cache/logging/checkpointing outside compiled regions.
- Detailed opportunity classification lives in `compile_optimization_analysis.md`. In short: compile the deterministic tensor core first; treat dynamic shape mode, bucketing/padding, activation checkpointing, attention backend changes, mixed precision, and non-DiT backbones as separate measured experiments.
- `compile.dynamic` is an explicit experiment knob, not a default recommendation. Compare `null`, `True`, and `False` against identical eager baselines before recommending a mode.

## Benchmark Policy

Every experiment must be pre-registered in `goal_progress.md` with:

- Hypothesis
- Target code path
- Dataset or data setup
- Command
- Metrics
- Correctness checks
- Acceptance threshold
- Revert criteria

Baseline and optimized commands must be identical except for the explicit variable under test. Report raw numbers, mean, median, p50, p90, p95, min, max, standard deviation, confidence notes, compile startup, steady-state throughput, and memory.

## Correctness Policy

Correctness comparisons must use the same seed, model config, checkpoint or initial weights, data batch where possible, precision, optimizer, scheduler, gradient accumulation, and distributed setup. Compare forward outputs where meaningful, loss, selected gradients, parameter updates, checkpoint save/resume behavior, validation behavior, sampling behavior if touched, and longer smoke-run loss behavior. Tolerances must be declared before results.

Initial tolerance policy:

- CPU float32 smoke: eager/compiled loss absolute and relative tolerance `1e-5` after controlling RNG inputs.
- CUDA float32 smoke: eager/compiled loss absolute tolerance `1e-4`, relative tolerance `1e-4`.
- Mixed precision smoke: declare dtype-specific tolerance per run before measuring; do not loosen after seeing failures without recording justification.

## Anti Reward-Hacking Policy

- Never claim speedup without comparable baseline and optimized measurements.
- Never compare optimized code against a weaker baseline.
- Never silently change batch size, sequence length, precision, dataset, validation frequency, logging frequency, sampler behavior, gradient accumulation, optimizer, scheduler, checkpoint behavior, or measured region.
- Never disable required features only in the compiled run.
- Never hide failed experiments or memory regressions.
- Never delete or weaken tests to make changes pass.
- Never report only the best run when multiple runs exist.
- Never report compile startup excluded speedup without also reporting compile startup cost.
- Never claim 2x, 4x, or 10x without showing the calculation from raw numbers.

## Dataset Compatibility Policy

Dataset, sampler, bucketing, padding, collation, and data movement changes require tests across realistic variations before being kept. Until baseline measurements exist, do not change these paths. If touched later, preserve ordering, randomness, distributed sharding, epoch boundaries, resume semantics, filtering, padding, masks, and memory scalability.

## Acceptance Criteria

- Tracking docs are current: `goal.md`, `goal_progress.md`, `memory.md`, `next_agent_handoff.md`.
- Baseline and compiled results are recorded with exact commands and environment.
- Optional metrics and compile controls are documented.
- Correctness evidence is numeric.
- Compile defaults are conservative and backward compatible.
- Git diff contains only intentional source, test, benchmark, and documentation changes.
- CPU-safe regression tests cover default config/CLI behavior, all supported backbones, stochastic fallback, state-dict cleanliness, checkpoint resume, gradient accumulation metrics, sampling isolation, and zero-worker dataloading.
