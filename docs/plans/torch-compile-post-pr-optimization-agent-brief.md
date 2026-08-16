# Agent Brief: F5-TTS Torch Compile Post-PR Optimization Hunt

> Use this as the authoritative instruction for an autonomous coding/research agent. The goal is to discover, validate, and clearly report precision-preserving performance opportunities on top of the current `torch.compile` PR. Do not integrate anything into the source-of-truth PR branch unless the controller explicitly approves after independent verification.

## 0. Role and Operating Mode

You are a senior PyTorch performance engineer working inside a live repository. Act like a careful scientist, not a benchmark gambler.

Your job is to explore optimization opportunities for this PR, produce evidence, and hand back candidate patches or benchmark scripts. You must make small, meaningful experiments that can run while other agents share the same machine/GPU resources.

Use a hypothesis-driven workflow:

1. Read relevant code first.
2. State specific hypotheses.
3. Design bounded experiments.
4. Run correctness smoke tests before timing.
5. Benchmark with warmup and CUDA synchronization.
6. Report exact commands, logs, metrics, and uncertainty.
7. Do not overclaim. If target is not reached, say so plainly and explain why.

## 1. Repository and PR Context

Source-of-truth worktree:

```text
/media/mithex/NVME 2/Codex Linux/f5-tts-prs/F5-TTS-torch-compile-integration
```

Branch:

```text
torch-compile-upstream-integration
```

Remote:

```text
https://github.com/hcsolakoglu/F5-TTS.git
```

Open PR:

```text
https://github.com/hcsolakoglu/F5-TTS/pull/4
```

Current head at time of this brief:

```text
e04c641 test: keep flow loss accumulation in fp32
```

Known current compile implementation:

- Config is in `src/f5_tts/configs/*.yaml` under `compile:`.
- Trainer passes compile config from `src/f5_tts/train/train.py` into `Trainer`.
- Compile setup is in `src/f5_tts/model/trainer.py::_configure_compile`.
- Actual compile boundary is `src/f5_tts/model/cfm.py::CFM._forward_loss_core_components` through `CFM.compile_training_core`.
- Stochastic preparation remains eager in `CFM._prepare_training_inputs`.
- Current compiled region includes xt interpolation, transformer forward, masked MSE, fp32 loss accumulation, and returns `(loss, loss_sum, denom, cond, pred)`.
- `tests/test_training_compile.py` contains compile API, state_dict cleanliness, eager-vs-compiled parity, global masked mean, DDP scaling probes, fallback behavior, config parsing, and CUDA-conditional tests.

Known current safe/default recommendation:

```yaml
compile:
  enabled: false
  backend: inductor
  mode: default
  fullgraph: false
  dynamic: null
  fallback_to_eager: true
```

Known previous benchmark conclusion from this PR:

- Compile-enabled default has already produced about `1.75x` speedup over compile-disabled eager baseline in relevant measurements.
- The new target is not merely to reproduce that. The stretch target is an additional `50%` speedup over current compile-enabled default, without reducing precision.
- Equivalently, if eager baseline is `1.0x` and current compile default is `1.75x`, target is `>= 2.625x` over eager, or `compiled_default_ms / candidate_ms >= 1.50` in steady state.
- `dynamic=true` previously looked mildly faster on a long variable-shape G4 run but had high cold compile/amortization cost.
- CUDA-graph modes such as `reduce-overhead` and `max-autotune` previously passed correctness but were slower and much more memory-hungry in some measurements.

Treat those as prior evidence, not gospel. You may challenge them with better experiments, but you must produce real data.

## 2. Hard Constraints

Do not violate these.

1. **No precision reduction.**
   - Do not enable lower-precision matmul, TF32, quantization, FP8, lower AMP dtype, or any setting whose main benefit comes from lower numerical precision.
   - Keep precision flags identical across baseline and candidate comparisons.
   - If existing benchmark uses AMP, compare AMP-to-AMP with identical dtype and GradScaler behavior. Do not claim speedup from changing precision.

2. **Do not change data loader, collate, sampler, dataset, or input pipeline for candidate PR code.**
   - You may use synthetic or cached fixed batches for benchmarking.
   - You may discuss shape bucketing/padding as a future direction, but do not implement data loader changes here.

3. **Source-of-truth worktree is read-only for you.**
   - Do not modify `/media/mithex/NVME 2/Codex Linux/f5-tts-prs/F5-TTS-torch-compile-integration` directly, except reading files.
   - Create an isolated worktree for experiments and patches.

4. **Do not commit, push, or rewrite history.**
   - Produce candidate diffs, scripts, and reports only.

5. **Small but meaningful experiments.**
   - Other agents may share the same CPU/GPU. Avoid long training jobs.
   - Prefer smoke-scale model/batch plus enough iterations to separate cold compile, warmup, and steady state.
   - Use a strict time budget. If a run is too slow or compiles indefinitely, stop and report.

6. **No hand-wavy benchmark claims.**
   - Every performance claim needs a command, environment metadata, warmup/measurement protocol, and output.

7. **Correctness before speed.**
   - A faster candidate that fails loss/output/gradient parity is rejected.
   - A candidate that silently falls back to eager is rejected as compile performance evidence.

## 3. Isolated Worktree Setup

Start from the source-of-truth worktree and create your own isolated worktree.

Example, adjust `AGENT_ID` to your assigned name:

```bash
set -euo pipefail
SRC="/media/mithex/NVME 2/Codex Linux/f5-tts-prs/F5-TTS-torch-compile-integration"
ROOT="/media/mithex/NVME 2/Codex Linux/f5-tts-prs/agent-worktrees"
AGENT_ID="replace-with-your-agent-id"
mkdir -p "$ROOT"
cd "$SRC"
git status --short --branch
git worktree add -b "agent/${AGENT_ID}" "$ROOT/${AGENT_ID}" HEAD
cd "$ROOT/${AGENT_ID}"
git status --short --branch
```

If the branch or directory already exists, add a unique suffix. Report the exact path you used.

## 4. Files to Inspect First

Read these before designing experiments:

```text
src/f5_tts/model/cfm.py
src/f5_tts/model/trainer.py
src/f5_tts/train/train.py
tests/test_training_compile.py
src/f5_tts/configs/F5TTS_v1_Base.yaml
src/f5_tts/model/backbones/dit.py or nearby DiT/attention implementation files discovered from imports
pyproject.toml
```

Also inspect relevant utility files when a hypothesis touches them:

```text
src/f5_tts/model/utils.py
src/f5_tts/model/modules.py
src/f5_tts/model/backbones/*
```

Do not invent APIs or config keys. Trace symbols to definitions and callers.

## 5. Primary Objective

Find precision-preserving optimization opportunities that can be layered on top of the current `torch.compile` PR and plausibly improve steady training throughput by at least `50%` over the current compile-enabled default.

Baseline hierarchy for your experiments:

1. `eager`: compile disabled.
2. `compiled_default`: current recommended compile-on profile.
3. `candidate`: your proposed setting/code change on top of compile.

Acceptance target:

```text
candidate steady ms/update <= compiled_default steady ms/update / 1.50
candidate speedup vs eager >= 2.625x
```

If the target is not achieved, still report best candidates and why they fell short.

## 6. Promising Search Areas

Prioritize low-risk, precision-preserving opportunities that are close to torch.compile and easy to review.

**Important correction of intent:** this is primarily a **kernel-launch reduction hunt**, not only a safe public `torch.compile` mode sweep. You are explicitly encouraged to look for obscure, semi-internal, under-documented, or recently added PyTorch/Inductor/Dynamo/Triton knobs that reduce the number of launched CUDA kernels, increase fusion, combine pointwise kernels, enable CUDA graph replay, or avoid recompilation/guard churn. Do web research, inspect installed torch source/configs, inspect upstream PyTorch `main` source, search GitHub issues/PRs, and use your own ML compiler reasoning. Do not restrict yourself to the public `torch.compile(..., mode=...)` API.

The controller specifically wants candidates in the spirit of:

- `torch._inductor.config.coordinate_descent_tuning = True`
- `torch._inductor.config.coordinate_descent_check_all_directions = True`
- `torch._inductor.config.max_autotune_pointwise = True`
- `torch._inductor.config.aggressive_fusion = True`
- `torch._inductor.config.max_fusion_size = ...`
- `torch._inductor.config.max_fusion_buffer_group_pairwise_attempts = ...`
- `torch._inductor.config.score_fusion_memory_threshold = ...`
- `torch._inductor.config.max_pointwise_cat_inputs = ...`
- `torch._inductor.config.force_pointwise_cat = True`
- `torch._inductor.config.combo_kernels = True`
- `torch._inductor.config.benchmark_combo_kernel = True`
- `torch._inductor.config.combo_kernels_autotune = 1 or 2`
- `torch._inductor.config.combo_kernel_max_num_nodes = ...`
- `torch._inductor.config.combo_kernel_allow_mixed_sizes = ...`
- `torch._inductor.config.triton.max_tiles = 32` or other values
- `torch._inductor.config.triton.cudagraphs = True`
- `torch._inductor.config.triton.cudagraph_trees = True`
- `torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = False`
- `torch._inductor.config.triton.cudagraph_capture_sizes = [...]` if applicable
- `torch._inductor.config.triton.cudagraph_min_partition_size = ...`
- `torch._inductor.config.triton.reorder_for_reducing_graph_partitions = True`
- `torch.compiler.cudagraph_mark_step_begin()` inside the benchmark/training iteration when using CUDA graph modes, especially after fixed-shape or bucketed cached batches
- `torch._dynamo.config.cache_size_limit` / `recompile_limit`, `accumulated_cache_size_limit`, `error_on_recompile`, `fail_on_cache_limit_hit`, and related cache/recompile controls
- environment equivalents such as `TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1`, `TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1`, `TORCHINDUCTOR_SPLIT_REDUCTIONS=...`, `TORCH_COMPILE_DEBUG`, `TORCH_LOGS=...`, etc.

Some knobs may be version-specific. For example, `pointwise_compression_threshold` was suggested by the controller but may not exist in the installed PyTorch 2.12 build. If missing, search whether it was renamed, removed, replaced by `max_fusion_size`, `score_fusion_memory_threshold`, `combo_kernels`, `max_pointwise_cat_inputs`, `expand_dimension_for_pointwise_nodes`, or other fusion/compression controls. Do not fail just because a named knob is absent; find the nearest real knob in this torch version and explain the mapping.

Use this installed-environment fact as a starting point, but re-check in your worktree and on the actual GPU environment:

```text
torch 2.12.1+cpu observed in the controller shell
torch._inductor.config.coordinate_descent_tuning = False
torch._inductor.config.max_fusion_size = 64
torch._inductor.config.aggressive_fusion = False
torch._inductor.config.epilogue_fusion = True
torch._inductor.config.shape_padding = True
torch._inductor.config.combo_kernels = False
torch._inductor.config.combo_kernels_autotune = 1
torch._inductor.config.triton.max_tiles = None
torch._inductor.config.triton.cudagraphs = False
torch._inductor.config.triton.cudagraph_trees = True
torch.compiler.cudagraph_mark_step_begin exists
torch._dynamo.config.cache_size_limit = 8
torch._dynamo.config.accumulated_cache_size_limit = 256
```

PyTorch docs say `torch.compiler.cudagraph_mark_step_begin()` indicates a new inference/training iteration is about to begin and helps CUDA Graph Trees free tensors from the prior iteration when heuristics are wrong. Use it as a measured candidate only with CUDA graph/cudagraph-tree modes, and report whether it changes kernel-launch counts, replay behavior, memory, or speed.

### Kernel-launch measurement requirement

In addition to ms/update, measure or estimate kernel launches whenever possible:

- Use `torch.profiler` with CUDA activities on a tiny measured window and count CUDA kernel events.
- Report total CUDA kernel launches per optimizer update, top kernel names, and whether candidate reduced launch count.
- If using CUDA graphs, distinguish real kernel launches during capture/warmup from graph replay behavior during steady measurement.
- If profiler overhead is too high, run a separate tiny profiling pass from timing pass.

Preferred success criterion is not just faster time, but a causal chain:

```text
candidate knob/code -> fewer graph partitions or more fusion/CUDA graph replay -> fewer launches/update -> same precision/correctness -> lower steady ms/update
```

If a candidate is faster without launch reduction, still report it. If it reduces launches but is slower due to occupancy/memory/register pressure, report that too.

### A. Inductor and `torch.compile` options

Investigate only options available in the installed torch version. Use:

```python
import torch
import torch._inductor as inductor
print(torch.__version__)
print(inductor.list_mode_options("default"))
print(inductor.list_mode_options("reduce-overhead"))
print(inductor.list_mode_options("max-autotune"))
print(inductor.list_options())
```

Potential candidates, if supported:

- `mode="default"` vs `mode=None`
- `dynamic=None/True/False`
- `fullgraph=False/True` only if correctness and graph-break behavior are proven
- `options={"shape_padding": True}`
- `options={"triton.cudagraphs": True}` as a bounded experiment, with memory reporting
- `max_autotune` variants only if the compile cost is bounded and no precision is changed
- `epilogue_fusion` only with required supporting options and explicit evidence
- `torch.compiler.set_stance("eager_then_compile")` or `"fail_on_recompile"` as experiments, not untested defaults
- local/remote compile caches only for cold-start measurement, not steady-state speed claims

### B. Dynamic-shape control without data pipeline changes

Investigate whether bounded dynamism can reduce recompiles and improve variable-shape steady state:

- `torch._dynamo.mark_dynamic(tensor, dim, min=..., max=...)` on prepared tensor dimensions, if safe.
- `TORCH_COMPILE_DYNAMIC_SOURCES` or `torch.compiler.config.dynamic_sources` if appropriate.
- Recompile/guard logs with `TORCH_LOGS="recompiles,guards,dynamic,graph_breaks,perf_hints"`.

Do not implement data loader bucketing. You may benchmark fixed vs variable cached synthetic batches to understand compiler behavior.

### C. Compile boundary and regional compilation

Current boundary is `CFM._forward_loss_core_components`. Consider whether another boundary is more efficient without changing public behavior:

- compile only `self.transformer` or a transformer core,
- compile repeated DiT blocks regionally,
- compile a smaller/larger pure tensor function,
- compile optimizer step only as an isolated optional experiment,
- avoid compiling stochastic preparation, logging, dataloader, checkpointing, or scheduler.

Any code-change candidate must preserve:

- `state_dict()` keys,
- forward return shape `(loss, cond, pred)`,
- fallback behavior,
- DDP safety assumptions,
- fp32 loss accumulation safety.

### D. Graph-break and CPU-sync cleanup inside compiled path

Find accidental graph breaks or CPU-GPU syncs inside the compiled path:

- `.item()`
- Python control flow dependent on tensors
- shape conversions that force sync
- unsupported Python loops in hot path
- logging/printing
- list/dict mutation inside compiled region

If you propose a cleanup, prove it with graph-break logs and parity tests.

### E. Attention/backend configuration, if local and easy

Prior evidence suggests SDPA is better than FlashAttention for masked compiled training in this PR. You may re-check attention backend choices only if the experiment is small and precision-preserving.

Do not spend the whole run installing heavy alternate attention packages unless already present.

## 7. Benchmark Best Practices Required

Create or adapt a small benchmark harness if none exists. It should be easy for the controller to rerun.

Minimum benchmark protocol:

1. Print metadata:
   - commit SHA,
   - torch version,
   - CUDA version,
   - cuDNN version,
   - GPU name from `torch.cuda.get_device_name(0)`,
   - Python version,
   - precision flags such as TF32 and matmul precision,
   - compile config and inductor options.
2. Use fresh model instance per combo.
3. Use identical initial weights per combo.
4. Use deterministic cached batches, not a live dataloader.
5. Test at least one fixed-shape regime and one small variable-shape regime if CUDA is available.
6. Separate timing phases:
   - cold first compiled update,
   - prewarm updates,
   - measured steady updates.
7. Use `torch.cuda.synchronize()` around timers on CUDA.
8. Record peak memory:
   - `torch.cuda.max_memory_allocated()`
   - `torch.cuda.max_memory_reserved()`
9. Emit machine-readable JSONL per combo.
10. Use robust statistics:
   - median,
   - mean,
   - p10/p90 or trimmed mean if enough samples.
11. Count/report recompiles, graph breaks, or at least include `TORCH_LOGS` output for a short diagnostic run.
12. Detect and reject fallback-to-eager rows.

Suggested scale:

```text
prewarm: 5-10 optimizer updates
measure: 10-30 optimizer updates
batch/model: small enough to finish quickly, but large enough that compile behavior is meaningful
```

If hardware is CPU-only, still build correctness and harness structure, but mark GPU performance as not measured. Do not infer CUDA speed from CPU.

## 8. Correctness Smoke Tests Required

Before timing each serious candidate, run a quick correctness gate:

- eager vs candidate loss on identical prepared args,
- eager vs candidate `cond` and `pred` shape and finite checks,
- gradient finite checks for representative parameters,
- no state_dict pollution,
- fp32 loss accumulation remains intact,
- no eager fallback active,
- if CUDA and AMP are used, AMP smoke with GradScaler remains finite.

Use existing tests when possible:

```bash
PYTHONPATH=src python3 -m pytest tests/test_training_compile.py -q
```

For targeted work, a narrower test selection is acceptable, but report exactly what passed/skipped.

## 9. Output Format

Your final response must be structured exactly like this:

```markdown
# Agent Result: <agent-id>

## Worktree
- Path:
- Branch:
- Base commit:
- Files changed:

## Executive Summary
- Best candidate:
- Target achieved? yes/no/inconclusive
- Speedup vs eager:
- Speedup vs compiled_default:
- Correctness status:
- Main risk:

## Commands Run
```bash
# exact commands, in order
```

## Environment
| Field | Value |
|---|---|
| torch | |
| CUDA | |
| GPU | |
| Python | |
| commit | |
| precision flags | |

## Correctness Evidence
| Candidate | Tests/probes | Result | Notes |
|---|---|---|---|

## Benchmark Evidence
| Regime | Combo | steady ms/update median | speedup vs eager | speedup vs compiled_default | cold s | amort steps | peak alloc/reserved | fallback? |
|---|---:|---:|---:|---:|---:|---:|---:|---|

## Graph/Recompile Evidence
- graph breaks:
- recompiles:
- guards/dynamic notes:
- perf hints:

## Candidate Patch or Script
- Diff stat:
- Important files:
- How to apply or rerun:

## Decision Matrix
| Candidate | Correctness | Speed | Cold compile | Memory | Risk | Upstream suitability | Verdict |
|---|---|---|---|---|---|---|---|

## Recommendation
- Integrate now / test further / reject:
- Why:
- Next experiment if time allowed:
```

Attach raw JSONL/log paths if generated.

## 10. Rejection Rules

Reject a candidate if any of these are true:

- It reduces precision or changes precision settings relative to baseline.
- It changes data loading/collation/sampling/dataset behavior.
- It only improves cold compile but not steady throughput, unless clearly labeled as cold-start-only.
- It silently falls back to eager.
- It fails parity or gradient finite checks.
- It improves one tiny synthetic shape but obviously worsens variable-shape behavior without disclosure.
- It requires broad refactoring unrelated to compile/benchmarking.
- It is not reproducible from exact commands.

## 11. Suggested Hypotheses to Explore

You do not have to run all of these. Pick a focused subset and do it well.

1. `coordinate_descent_tuning=True` plus `max_autotune_pointwise=True` may find better Triton pointwise/reduction tile choices and reduce the need for many small kernels. Test compile cost separately from steady speed.
2. Fusion-pressure knobs such as `aggressive_fusion=True`, larger `max_fusion_size`, lower/higher `score_fusion_memory_threshold`, `max_fusion_unique_io_buffers`, and pairwise fusion attempts may combine more pointwise/reduction work into fewer kernels.
3. `combo_kernels=True` may combine otherwise separate data-independent kernels into one launch. Explore `benchmark_combo_kernel`, `combo_kernels_autotune`, `combo_kernel_max_num_nodes`, mixed-size allowance, and pointwise-only settings.
4. Triton tiling controls such as `triton.max_tiles`, `triton.prefer_nd_tiling`, `triton.tile_reductions`, `triton.persistent_reductions`, and `triton.cooperative_reductions` may alter launch count or per-kernel efficiency. Test carefully for regressions.
5. CUDA Graph Trees plus explicit `torch.compiler.cudagraph_mark_step_begin()` may reduce repeated launch overhead in fixed-shape or manually bucketed cached-batch benchmarks. This is likely shape-sensitive and memory-sensitive, so report replay behavior and VRAM.
6. `torch._dynamo.config.cache_size_limit` / `recompile_limit` and `accumulated_cache_size_limit` may prevent fallback to eager or excessive recompilation in variable-shape runs. Increasing them is not a speedup by itself; prove it changes cache/recompile behavior and steady throughput.
7. Bounded `mark_dynamic` on frame dimension may reduce recompile churn versus global `dynamic=True`, especially if combined with cache-size controls.
8. `mode=None` and `mode="default"` may differ subtly across torch versions; verify actual mode options and configs.
9. A regional compile boundary inside DiT repeated blocks may reduce launch count or improve fusion compared with compiling the whole CFM loss core.
10. Compiling optimizer step may help launch overhead, but only if correctness and state behavior are clean.
11. Graph-break logs may reveal a cheap `.item()`/Python-control cleanup that unlocks `fullgraph=True`, CUDA graphs, or fewer graph partitions.
12. If a named knob from web/pretraining does not exist in installed torch, search upstream PyTorch source and issues to identify its successor/renamed equivalent, then test the real available knob.

## 12. Final Reminder

The controller will independently verify promising claims before integrating. Optimize for truthful, reproducible evidence. A negative result with clean data is more valuable than an exciting but irreproducible number.
