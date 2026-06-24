# Compile Optimization Analysis

This note classifies F5-TTS training optimization opportunities before choosing
additional compile or batching changes. It is intentionally conservative:
`torch.compile` remains optional and disabled by default until representative
evidence supports a stronger policy.

## Current Compile Boundary

The current implementation compiles `CFM._forward_loss_core()` through
`CFM.compile_training_core()`. `CFM.forward()` still performs text conversion,
length-mask construction, random span/noise/time sampling, and CFG drop
decisions eagerly, then calls the compiled deterministic loss core when enabled.

This boundary is the right first target because it avoids compiling Python RNG,
Python list tokenization, dataloader behavior, optimizer/scheduler orchestration,
logging, checkpointing, sample generation, and Accelerate control flow.

## Directly Compilable Now

- `CFM._forward_loss_core()`: tensor-only interpolation, conditioning mask,
  transformer call, MSE loss, and masked reduction.
- DiT training path with `cache=False`, `cfg_infer=False`, tensor text input,
  and fixed module settings. One-step CUDA correctness also passed with
  activation checkpointing enabled.
- DiT input projection, convolutional position embedding, time embedding,
  transformer blocks, RMS/LayerNorm/AdaLN, feed-forward layers, and output
  projection under the current benchmark settings.
- Basic per-batch loss/backward math for fixed tensor batches, already covered
  by CPU and CUDA eager-vs-compiled correctness checks.

## Needs Refactor Or Separate Validation Before Compile

- `CFM.forward()` stochastic prep: Python `random()` CFG choices, `torch.rand`
  span/time/noise sampling, and text list tokenization are correctness-sensitive.
  Compiling this larger region risks freezing or changing stochastic behavior.
- Dynamic sequence lengths: static compile can specialize well but may recompile
  on new frame/text lengths; `dynamic=True` can reduce recompiles but may reduce
  kernel specialization. Shape bucketing or fixed padding may beat `dynamic=True`
  on long runs, but that touches dataset/sampler policy and needs a baseline plus
  compatibility tests before implementation.
- CFG drop booleans: `drop_audio_cond` and `drop_text` are Python booleans passed
  into the compiled core. They may create separate specializations for the few
  branch combinations. That is acceptable initially, but a future refactor could
  pre-warm expected branches or pass tensor masks if evidence shows recompiles.
- `DiT.TextEmbedding.average_upsample_text_by_mask()`: uses Python loops and
  tensor-to-int conversions. Keep this path out of compile until it is rewritten
  as tensor operations and covered by correctness tests.
- Activation checkpointing: one-step DiT CUDA correctness passed, but its
  memory/throughput tradeoff still needs a dedicated performance run.
- MMDiT and UNetT: CPU-safe compiled-core smoke tests pass. They still need
  dedicated representative performance evidence before broad speed claims.
- Flash-attention paths: `flash_attn` and varlen unpadding can be a real speed
  path, but they add backend/package constraints and need separate baseline,
  correctness, and memory evidence.
- Mixed precision, TF32, and matmul precision: likely important on Colab GPUs,
  but they change numeric behavior and must be compared equally for eager and
  compiled runs with dtype-specific tolerances.

## Should Stay Eager For This Work

- `Trainer.train()` control flow: dataloader iteration, `Accelerator.accumulate`,
  gradient synchronization, optimizer/scheduler calls, grad clipping, EMA,
  progress bars, logging, and metric emission.
- Dataset and dataloader path: `load_from_disk`, `torchaudio.load`, resampling,
  mel extraction, `collate_fn`, worker processes, `pin_memory`, and
  `DynamicBatchSampler` construction.
- Checkpoint save/load/resume and checkpoint rotation.
- Training sample logging, vocoder decode, `CFM.sample()`, ODE integration,
  inference cache writes, and file output.
- Gradio orchestration and subprocess/UI code.

## Non-Compile Optimization Candidates

- Preprocessed mel datasets can remove `torchaudio.load`, resampling, and mel
  computation from the measured training path. This is a dataset-mode decision,
  not a compile result, so compare it separately.
- DataLoader tuning (`num_workers`, `persistent_workers`, `prefetch_factor`,
  `pin_memory`) may matter more than model compile when data wait dominates.
- Duration metadata quality and dynamic-batch construction cost affect startup
  and padding. Any sampler change must preserve ordering, shuffling,
  distributed sharding, and resume semantics.
- Length bucketing or static-shape padding can reduce recompiles and improve
  kernel specialization, but can waste compute and alter batch composition.
- Attention backend selection (`torch` SDPA vs `flash_attn`) may be a larger
  model-side win than compile alone on long sequences.
- Activation checkpointing may enable larger batches or full models at the cost
  of extra recompute; it should be evaluated as memory-throughput tradeoff.

## Dynamic Shape Tradeoffs

Arguments for `dynamic=True`:

- Reduces shape-specialized recompiles on variable-length batches.
- Keeps dataset/sampler semantics unchanged for exploratory runs.
- Useful when a workload has many distinct sequence lengths and compile overhead
  would otherwise dominate.

Arguments against defaulting to `dynamic=True`:

- Dynamic kernels can lose specialization and may be slower once shapes are
  stable or bucketed.
- It can hide the real problem: too many shapes from batching policy or text/audio
  padding variation.
- The observed T4 run had large cold compile cost; avoiding some recompiles does
  not prove better end-to-end time.
- Full/base models on L4/A100 may have different compile/codegen and kernel
  behavior than the tiny T4 run.

Current policy: keep `compile.dynamic=null` in configs, expose explicit
`true/false/default` knobs, and benchmark `null`, `True`, and `False` against
the same baseline before recommending a mode.

Measured refinement:

- Eager DiT/UNetT text-width normalization removes raw text width as a static
  compile guard without changing their existing crop/pad semantics.
- With both audio and text widths fixed on L4, all compile modes used three CFG
  specializations and no graph breaks. Default/null remained slightly fastest
  steady-state (`1.5209x`), while fresh-cache static false was close (`1.5128x`).
- Therefore the refactor makes static false viable for stable-shape workloads;
  it does not justify changing the default or imposing fixed padding.

## Counterarguments To The Current Selective Compile Boundary

- Compiling only the deterministic core may leave Python overhead in
  `CFM.forward()`. Counterpoint: this preserves stochastic semantics and already
  gives a narrow warmed steady-state speedup on T4.
- Compiling a larger region could fuse more work. Counterpoint: text conversion,
  random decisions, mask generation, sampling caches, and branch behavior are
  where correctness risk and graph breaks are concentrated.
- Static shape bucketing could beat dynamic compile. Counterpoint: sampler and
  batching changes are outside the current safe-change envelope until a stronger
  baseline and compatibility matrix exists.
- Tiny T4 evidence may not transfer to full models. Counterpoint: exactly; it is
  only a screening result and should guide, not settle, the next benchmark.

## Completed Representative Matrix

Use Colab CLI for representative GPU work. Prefer a named-session lifecycle,
artifact download, `colab stop`, and `colab sessions` verification.

Completed L4 screening:

- Tiny real-audio `dynamic=default/null` beat `dynamic=true` slightly in warmed
  step time, while `dynamic=false` hit TorchDynamo's recompile limit on variable
  frame lengths.
- F5TTS-small fit on L4 and the compiled path completed, but compile warmup was
  hundreds of seconds for a two-warmup-step probe.
- The first fixed-audio matrix showed that audio padding alone was insufficient:
  static false moved from audio-width to text-width recompiles.
- After eager text normalization, the repeated L4 matrix completed all modes
  with three graphs and no recompile-limit warning. Default/null stayed
  marginally faster than static false, while dynamic true had the highest cold
  cost and slowest compiled steady state.

Completed A100 representative result:

- F5TTS-small, BF16, batch 1, fixed 1536 frames, real audio subset.
- Eager `0.1267282088s/step`; static compiled `0.0601469082s/step`;
  `2.10698x` warmed speedup.
- Cold compile warmup was `253.9039s`; break-even was about 3,824 steps.
- Peak allocated memory fell 11.42%.

No further mode experiment is currently a clear low-risk win. `reduce-overhead`
can add CUDA-graph constraints and memory pressure, while `max-autotune` can add
large compile cost. Future work should prioritize distributed validation or an
opt-in bucketing policy, not default changes or benchmark-only mode chasing.
