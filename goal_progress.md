# torch.compile Goal Progress

## Current Phase

Complete: implementation, representative A100/L4 validation, correctness gates,
cleanup, and handoff are finished.

## Completed Work

- Read the full objective from `/home/mithex/.codex/attachments/ac47b788-464d-4f2b-86ba-d73c031d09d2/pasted-text-1.txt`.
- Verified project root: `/media/mithex/NVME 2/Codex Linux/f5-tts-prs/F5-TTS`.
- Verified git state before changes:
  - Branch: `main`
  - HEAD: `2ae2c9bd9b64dab2cb069c4b97e5e7673c521e01`
  - Status: clean
  - Remote: `https://github.com/hcsolakoglu/F5-TTS.git`
- Audited primary files:
  - `src/f5_tts/train/train.py`
  - `src/f5_tts/train/finetune_cli.py`
  - `src/f5_tts/train/finetune_gradio.py`
  - `src/f5_tts/model/trainer.py`
  - `src/f5_tts/model/cfm.py`
  - `src/f5_tts/model/dataset.py`
  - `src/f5_tts/model/backbones/dit.py`
  - `src/f5_tts/model/backbones/mmdit.py`
  - `src/f5_tts/model/modules.py`
  - `src/f5_tts/model/utils.py`
  - `src/f5_tts/configs/F5TTS_Small.yaml`
  - `pyproject.toml`
- Confirmed no existing test suite or lockfile is present in the checkout.
- Confirmed local dataset files are incomplete for real training: `data/Emilia_ZH_EN_pinyin/` currently contains only `vocab.txt`.
- Confirmed local hardware:
  - CPU: AMD Ryzen 5 5600, 6 cores / 12 threads
  - RAM at audit time: 15 GiB total, 2.6 GiB available
  - GPU: NVIDIA GeForce RTX 3070, 8192 MiB
  - Driver: 595.71.05
  - `nvidia-smi` reported CUDA version: 13.2
- Loaded the Colab CLI skill after user correction because Phase 11 requires the Colab CLI workflow for GPU runs when GPU scale is required.
- Checked Colab CLI:
  - Binary: `/home/mithex/.local/bin/colab`
  - Version: `0.5.11`
  - `colab sessions`: no active sessions found
- Located existing ML venv:
  - `/home/mithex/.venvs/ml312`
  - Python: 3.12.12
  - PyTorch: `2.12.1+cu130`
  - CUDA available locally: yes
  - Local GPU visible to PyTorch: NVIDIA GeForce RTX 3070
  - Installed core packages include `torch`, `torchaudio`, `accelerate`, `datasets`, `bitsandbytes`, `transformers`, `safetensors`, `xformers`
  - Missing F5-TTS-specific packages at check time: `hydra-core`, `wandb`, `x-transformers`, `ema-pytorch`, `vocos`, `torchdiffeq`
- Added optional runtime metrics to `Trainer`, disabled by default.
- Added optional `torch.compile` model-call wrapper to `Trainer`, disabled by default.
- Added Hydra config sections for `metrics` and `compile` to all training YAML configs.
- Added matching finetune CLI flags for metrics and compile controls.
- Documented metrics and compile controls in `src/f5_tts/train/README.md`.
- Added `compile_optimization_analysis.md` to classify directly compilable code, code needing refactor, eager-only boundaries, non-compile optimization candidates, and dynamic-shape tradeoffs.
- Tightened config and README wording so `compile.dynamic=True` is presented as an experiment knob, not a default recommendation.
- Cleaned a dead `DiT.TextEmbedding.forward()` average-upsampling branch that referenced `valid_seq_lens` after proving the current training path passes an integer padded sequence length plus explicit valid lengths.
- Moved lazy runtime fallback inside the deterministic CFM core boundary so eager retry reuses the same prepared tensors and restored PyTorch RNG state.
- Added eager DiT/UNetT text-width normalization before the compiled boundary; fixed-shape static compile now uses three CFG-specialized graphs instead of recompiling on raw text widths.
- Added 14 CPU-safe regression tests covering defaults, CLI/config parsing, DiT/UNetT/MMDiT compile smoke, fallback, loss equivalence, state dicts, checkpoint resume, accumulation-aware metrics, zero-worker dataloading, and sampling isolation.
- Added BF16 and multi-step correctness support to the benchmark harness.
- Completed L4 shape-mode matrices and a paired A100 F5TTS-small BF16 run with downloaded raw artifacts and explicit Colab cleanup.

## Audit Findings

### Safe Initial Compile Targets

- A narrow tensor-only training loss path around `CFM.forward` after text has been tensorized.
- Backbones during training with `cache=False`, no sampling, no checkpoint/logging side effects, and stable batch shapes.
- Optional helper/wrapper for compiled model call invoked from `Trainer` while keeping optimizer, scheduler, grad clipping, logging, EMA, checkpointing, and sampling eager.

### Unsafe Initial Compile Targets

- Full `Trainer.train` loop because it includes dataloader iteration, `Accelerator.accumulate`, logging, progress bars, checkpoint I/O, EMA update, scheduler/optimizer orchestration, and optional sampling/vocoder work.
- `CFM.sample` because it uses ODE integration, inference cache side effects, Python control flow over durations, optional vocoder, and manual seed changes.
- Dataset loading, resampling, mel extraction from raw audio, dynamic batch sampler construction, and collation.
- Checkpoint save/load and resume.

### Compile Risks

- `CFM.forward` uses Python `random()` to choose `drop_audio_cond` and `drop_text`, which creates graph breaks or freezes decisions if compiled naively.
- `CFM.forward` tokenizes Python text lists inside forward; this must stay eager or be moved outside the compiled region.
- `mask_from_start_end_indices` uses `.item()` through `seq_len.max().item()` in `model/utils.py`, creating data-dependent Python control flow.
- The DiT default training path now passes padded `x.shape[1]` plus explicit valid lengths, avoiding the previous tensor `seq_len.max().item()` graph break there. The legacy tensor-sequence fallback and `average_upsample_text_by_mask()` still use Python/tensor-to-int logic and are not compile-friendly.
- Dynamic audio/text lengths can cause recompiles if shapes vary without bucketing or dynamic compile support.
- Activation checkpointing and `fullgraph=True` passed one-step CUDA correctness smoke tests; their performance tradeoffs remain unbenchmarked.
- Inference cache properties in DiT/MMDiT are side-effectful; training uses `cache=False`, so compile must not include sampling cache paths.
- CPU Trainer compile smoke confirmed graph breaks from `seq_len.max().item()` in `mask_from_start_end_indices` and Python `random()` in `CFM.forward` CFG dropout decisions.

### Performance Bottlenecks Outside Compile

- Data loading from audio paths, resampling, and mel spectrogram generation may dominate for raw datasets.
- Dynamic batch sampler setup scans frame lengths and sorts the whole dataset.
- Variable sequence lengths and padding efficiency can strongly affect throughput independently of compile.
- Host RAM is limited on the local machine, which may constrain representative local benchmarking.

### Optimization Taxonomy And Counterarguments

See `compile_optimization_analysis.md` for the current decision record.

- Directly compilable now: `CFM._forward_loss_core()` plus tested DiT, UNetT, and MMDiT tensor training paths; DiT activation checkpointing and BF16 have dedicated correctness evidence.
- Needs refactor or separate validation: larger `CFM.forward()` regions, arbitrary static audio shapes, DiT average text upsampling, flash-attention, distributed multi-process fallback behavior, and alternative compile modes.
- Should stay eager for this work: `Trainer.train()` orchestration, dataset/dataloader/collate/sampler paths, checkpoint/resume, sample logging, vocoder decode, `CFM.sample()`, ODE integration, and Gradio subprocess/UI code.
- Non-compile candidates: preprocessed mel datasets, DataLoader worker/prefetch/pin-memory tuning, duration metadata quality, length bucketing or fixed padding, attention backend selection, and activation-checkpointing memory tradeoffs.
- Counterargument to defaulting `dynamic=True`: it can reduce shape recompiles but may lose specialization and hide a batching-shape problem. Static or `null` compile plus deliberate bucketing/padding may be faster on long runs, but sampler changes require baseline and compatibility tests first.
- Counterargument to compiling only the core: a larger region might reduce Python overhead, but the current boundary preserves stochastic CFG semantics and avoids text tokenization, random mask/noise/time setup, cache side effects, logging, checkpointing, and Accelerate control flow.

## Experiments

The sections below keep the original pre-registered plans together with their results. Any new performance experiment still needs a fresh pre-registration before running.

### Planned Local CUDA Validation: Static Compile After Eager Text Normalization

- Hypothesis: For DiT/UNetT, eagerly cropping/padding integer text tokens to the audio tensor's existing padded sequence length is mathematically equivalent to the backbones' current internal crop/pad behavior and removes variable raw text width as a `dynamic=False` guard source. With benchmark audio frames fixed at 1536, the tiny real-audio path should no longer hit the Dynamo recompile limit.
- Target code path: `CFM.forward()` eager text preparation, compiled `CFM._forward_loss_core()`, DiT text embedding, loss, backward, AdamW, scheduler, and grad clipping.
- Dataset/setup: local 32-sample real-audio dataset at `/home/mithex/work/tts/f5-tts/data/librispeech_asr_custom`; batch size 2, `num_workers=0`, benchmark-only fixed audio padding to 1536 frames, variable source text lengths.
- Command: `PYTHONPATH=src VIRTUAL_ENV=/home/mithex/.venvs/ml312 PATH=/home/mithex/.venvs/ml312/bin:$PATH UV_CACHE_DIR=/home/mithex/work/tts/.uv_cache uv run --active --no-sync python benchmarks/eager_training_baseline.py --dataset-root /home/mithex/work/tts/f5-tts/data/librispeech_asr_custom --dataset-cwd /home/mithex/work/tts/f5-tts --device cuda --warmup-steps 10 --steps 3 --batch-size 2 --num-workers 0 --seed 1234 --model-profile tiny --pad-to-frames 1536 --compile-enabled --compile-dynamic false --output /tmp/f5tts_static_text_normalized_cuda.json`
- Metrics: Dynamo counters, graph breaks, recompile-limit warnings, unique graphs, step/forward/backward time, finite loss/grad norm, and peak CUDA memory. Timings are mechanics context only, not a speedup claim.
- Correctness checks: the CPU regression suite must prove raw vs normalized DiT input embeddings are identical; CPU and CUDA eager-vs-compiled correctness must remain within the existing tolerances.
- Acceptance threshold: run completes, losses and gradients remain finite, no `config.recompile_limit` warning is recorded, and graph count remains consistent with the small finite CFG branch set rather than per-text-width recompilation.
- Revert criteria: any numerical mismatch, eager text-normalization behavior change, compile failure, recompile-limit warning, or material graph growth tied to raw text widths.

Result: passed after one failed implementation attempt.

- First run artifact: `/tmp/f5tts_static_text_normalized_cuda.json`.
- First run result: failed the acceptance threshold. Dynamo still hit `config.recompile_limit (8)` with text width mismatch (`expected 218, actual 201`). Inspection showed the normalization hook had been inserted into `CFM.sample()` instead of training `CFM.forward()`. This attempt is retained as a failed experiment; no performance conclusion is drawn from it.
- Fix: moved the hook to `CFM.forward()` and added a regression test that captures the text tensor crossing the compiled core boundary and asserts shape `(batch, audio_frames)`.
- Corrected artifact: `/tmp/f5tts_static_text_normalized_cuda_v2.json`.
- Corrected result: passed on local RTX 3070. Dynamo counters reported `graph_break={}`, `frames.ok=3`, `frames.total=3`, `stats.unique_graphs=3`, and no `unimplemented` recompile-limit entry.
- Corrected mechanics context: mean step `0.0115166310s`, forward `0.0035880740s`, backward `0.0072484930s`, peak allocated `58,248,704` bytes. These local timings are not a speedup claim.
- Correctness evidence after the fix: 12 CPU-safe regression tests passed; CPU eager-vs-compiled max loss delta remained `2.384185791015625e-07`, max gradient delta `1.4901161193847656e-08`, and max parameter-update delta `2.9103830456733704e-11`.

### Planned Colab Lifecycle: L4 Fixed-Shape Matrix After Text Normalization

- Hypothesis: With both benchmark audio frames and DiT raw text tensor width stabilized at 1536, `compile.dynamic=false` will avoid the prior L4 recompile-limit failure and may outperform default/null or `dynamic=true`. Counterargument: static specialization may still lose because CFG branches require multiple graphs, compile startup is large, or padding waste dominates.
- Target code path: identical tiny real-audio training step for eager, compiled default/null, compiled `dynamic=true`, and compiled `dynamic=false`; only the compile mode changes.
- Dataset/setup: the same 32-sample real-audio dataset used by the prior L4 matrices, batch size 2, `num_workers=0`, seed 1234, benchmark-only `--pad-to-frames 1536`, warmup 11, measured steps 5.
- Session name: `f5tts-l4-static-text-normalized`.
- Hardware request: Colab L4. Do not silently switch to A100/G4; this experiment is a controlled comparison against the prior L4 failure.
- Remote command: `benchmarks/benchmark_matrix_runner.py --spec /content/f5tts_fixed_matrix_spec.json`; the spec writes to `/content/f5tts_fixed_matrix_results`.
- Artifacts to retrieve: `matrix_summary.json` plus all four case JSON files and the session execution log if needed.
- Metrics: step/data wait/H2D/forward/backward/optimizer/scheduler/zero-grad/throughput/padding/loss/grad norm/memory, compile setup/cold warmup, Dynamo graph/recompile counters, environment, and per-case stderr.
- Correctness checks: finite measured loss/grad norm; existing CPU/CUDA numerical equivalence and text-normalization tests remain the direct correctness evidence.
- Acceptance threshold: all four comparable cases complete; static false has no recompile-limit warning; artifacts are downloaded; session is stopped; `colab sessions` confirms cleanup.
- Revert/abort criteria: allocation/auth/package failure, non-L4 hardware, OOM, timeout, compile failure, recompile-limit warning under static false, or missing artifacts. Record partial results without changing hardware or benchmark policy.

Cold-start follow-up:

- The matrix runs default/null, dynamic true, then static false in one VM. Inductor's disk cache is shared across subprocesses, so later-case warmup cannot be assumed cold.
- Hypothesis: static false has materially lower cold compile overhead than dynamic/default once both input widths are fixed, but this must be measured with an unused cache directory.
- Command policy: rerun only the identical static-false case with `TORCHINDUCTOR_CACHE_DIR=/content/inductor_cache_static_false_cold`, write `/content/f5tts_fixed_matrix_results/tiny_l4_fixed_compile_false_cold.json`, and download it before stopping.
- Acceptance threshold: fresh cache path, finite run, no recompile-limit warning, parsed nonzero artifact. Use this artifact for static-false startup/break-even; use the four-way matrix for steady-state comparisons.

Result: passed, with one artifact-transfer rerun.

- First session: `f5tts-l4-static-text-normalized`. All four cases passed, but the CLI reported a successful tar download while creating a zero-byte local file. The session had already been stopped, so console tails were not accepted as authoritative artifacts.
- Recovery session: `f5tts-l4-static-text-normalized-rerun`. The first kernel connection was lost during extraction; one `colab restart-kernel` recovered the same L4 session. The identical matrix then completed, each JSON was downloaded individually, parsed, and checked for nonzero size before shutdown.
- Cleanup: both sessions were stopped. Final `colab sessions` reported no active sessions.
- Hardware: NVIDIA L4, 23034 MiB, driver 580.82.07; Python 3.12.13; PyTorch 2.11.0+cu128; CUDA 12.8.
- Artifacts: `/tmp/f5tts_colab_results/l4_static_text_normalized/`
  - `matrix_summary.json`
  - `tiny_l4_fixed_eager.json`
  - `tiny_l4_fixed_compile_default.json`
  - `tiny_l4_fixed_compile_true.json`
  - `tiny_l4_fixed_compile_false.json`
  - `tiny_l4_fixed_compile_false_cold.json`
  - `session-rerun-log.jsonl`

Steady-state four-way matrix:

- Eager: mean step `0.0166214668s`, forward `0.0076256498s`, backward `0.0077359388s`, warmup total `1.8743208620s`, peak allocated `67,540,480` bytes.
- Default/null: mean step `0.0109285218s`, `1.5209x` speedup; forward `2.0978x`, backward `1.2992x`; compile setup `2.7956324250s`, warmup total `43.5618298350s`, peak allocated `109,382,656` bytes, `unique_graphs=3`, no graph breaks/recompile-limit entry.
- Dynamic true: mean step `0.0115034976s`, `1.4449x` speedup; forward `1.9214x`, backward `1.2808x`; setup `0.9125902640s`, warmup total `88.8821641430s`, peak allocated `109,382,656` bytes, `unique_graphs=3`, no graph breaks/recompile-limit entry.
- Static false in the ordered matrix: mean step `0.0111696510s`, `1.4881x` speedup, `unique_graphs=3`, no graph breaks/recompile-limit entry. Its low warmup/memory values were cache-influenced because it ran after the other compiled cases and are not used as cold-start evidence.

Fresh-cache static-false follow-up:

- Mean step `0.0109868852s`, `1.5128x` versus the matrix eager baseline.
- Setup `0.9140063380s`; warmup total `41.4047858710s`; first warmup step `18.6510603950s`.
- Peak allocated `109,382,656` bytes; peak reserved `155,189,248` bytes; `unique_graphs=3`; no graph breaks or recompile-limit entry.
- Estimated cold break-even: about `7,178` post-warmup steps. Default/null break-even is about `7,814`; dynamic true about `17,179`.

Decision:

- The refactor solved the concrete static-compile failure: fixed audio plus eager DiT text normalization reduced static false from eight graphs and a recompile-limit fallback to three finite CFG-specialized graphs.
- It did not make static false the fastest steady-state policy. Default/null remained slightly faster in the controlled four-way matrix, while fresh-cache static false was within run noise.
- Keep `compile.dynamic=null`, compile disabled, and fallback enabled as conservative defaults. Treat `dynamic=false` as a valid option only when the audio batching policy provides stable shapes; do not infer that arbitrary unbucketed training is now static-shape safe.
- The best L4 warmed speedup remains about `1.52x`, below the 2x target. Cold compile still needs roughly seven thousand steps to amortize, and cold peak allocation is about 62% above eager in this tiny case.

### Planned Correctness Experiment: Eager vs Compiled Tensor Batch

- Hypothesis: The optional compiled model-call path matches eager training math on a fixed tensor batch within the predeclared tolerance when stochastic CFG branch selection is disabled for the comparison.
- Target code path: `CFM.forward`, DiT backbone, loss, backward gradients, AdamW parameter update, with text already tensorized so tokenizer/list preprocessing stays eager.
- Dataset/setup: deterministic synthetic mel tensor, token tensor, and lengths; batch size 2, frames 64, mel dim 16, text length 24, vocab size 64.
- Commands:
  - CPU: `PYTHONPATH=src /home/mithex/.venvs/ml312/bin/python benchmarks/compile_correctness.py --device cpu --backend inductor --batch-size 2 --frames 64 --mel-dim 16 --text-len 24 --vocab-size 64 --seed 1234`
  - CUDA: `PYTHONPATH=src /home/mithex/.venvs/ml312/bin/python benchmarks/compile_correctness.py --device cuda --backend inductor --batch-size 2 --frames 64 --mel-dim 16 --text-len 24 --vocab-size 64 --seed 1234`
- Metrics: numeric max/mean absolute and relative deltas for loss, cond, pred, gradients, and parameters after one optimizer step; eager and compiled timings recorded only as context.
- Correctness checks: allclose for loss, cond, pred, every common gradient tensor, and every common parameter tensor after update.
- Acceptance threshold: CPU `atol=1e-5`, `rtol=1e-5`; CUDA `atol=1e-4`, `rtol=1e-4`.
- Revert criteria: missing or extra parameter/gradient keys, failed allclose under the predeclared tolerance, or correctness harness changing production code semantics.

Follow-up multi-step protocol:

- Extend the same fixed-batch comparison to 5 optimizer steps with a fixed per-step seed sequence and compare the full loss trajectory plus final gradients and parameters.
- Float32 tolerances remain CPU `atol=rtol=1e-5` and CUDA `atol=rtol=1e-4`.
- BF16 CUDA tolerance is predeclared as `atol=rtol=5e-3`, reflecting BF16 mantissa precision while remaining tight enough to catch training-path divergence. Do not change this threshold after observing results without recording a new technical justification.
- Commands:
  - CPU float32: `PYTHONPATH=src VIRTUAL_ENV=/home/mithex/.venvs/ml312 PATH=/home/mithex/.venvs/ml312/bin:$PATH UV_CACHE_DIR=/home/mithex/work/tts/.uv_cache uv run --active --no-sync python benchmarks/compile_correctness.py --device cpu --backend inductor --steps 5 --batch-size 2 --frames 64 --mel-dim 16 --text-len 24 --vocab-size 64 --seed 1234`
  - CUDA BF16: `PYTHONPATH=src VIRTUAL_ENV=/home/mithex/.venvs/ml312 PATH=/home/mithex/.venvs/ml312/bin:$PATH UV_CACHE_DIR=/home/mithex/work/tts/.uv_cache uv run --active --no-sync python benchmarks/compile_correctness.py --device cuda --backend inductor --precision bf16 --steps 5 --batch-size 2 --frames 64 --mel-dim 16 --text-len 24 --vocab-size 64 --seed 1234`

The first 5-step CPU run failed the original all-tensors-at-every-horizon gate:

- Loss trajectory still matched tightly: max absolute delta `3.5762786865234375e-07`.
- Final gradient max absolute delta was `2.2681429982185364e-05`.
- Final parameter max absolute delta was `0.00045956342364661396`.
- Technical interpretation: the one-step gradient/update comparison remains the direct numerical-equivalence invariant. Across repeated AdamW updates, tiny compiled/eager gradient differences around the model's zero-initialized AdaLN/output parameters can be amplified into different adaptive-update directions, so exact final-parameter allclose at the one-step tolerance is not a stable longer-run invariant.
- Protocol correction, recorded after the failure: for multi-step runs, gate the first-step loss/outputs/gradients/parameter update at the original tolerance, gate the full loss trajectory and final loss/outputs at that tolerance, and report final multi-step gradients/parameters as diagnostic distances including relative L2. Do not erase the failed strict result or use the corrected gate to claim identical parameter trajectories.

Corrected multi-step results:

- CPU float32 passed the corrected gate. Loss-trajectory max absolute delta `3.5762786865234375e-07`; first-step gradient max delta `1.4901161193847656e-08`; first-step parameter-update max delta `2.9103830456733704e-11`. Final five-step parameter relative L2 was `0.0001943628` and remains diagnostic-only.
- Local CUDA BF16 passed the predeclared `atol=rtol=5e-3` gate. Loss-trajectory max absolute delta `5.7220458984375e-06`; first-step gradient max delta `0.000244140625`; first-step parameter-update max delta `0.0001999253873`; final five-step parameter relative L2 `0.0002442892`.
- BF16 harness smoke over the real-audio loader also completed on the local RTX 3070 with finite loss and grad norm.
- Activation-checkpointing follow-up is predeclared at the existing one-step CUDA float32 `atol=rtol=1e-4`: run the same tiny correctness harness with `--checkpoint-activations`. This is a compatibility smoke, not a performance claim.
- Fullgraph follow-up is predeclared at the same one-step CUDA float32 tolerance using `--fullgraph`. This validates the exposed strict-graph control; it is not evidence that fullgraph should be the default.

### Planned Colab Lifecycle: A100 F5TTS-Small BF16 Paired Run

- Hypothesis: On a representative F5TTS-small model with BF16 and stable audio/text widths, compiling the deterministic CFM loss core can produce a measurable full-model steady-state speedup. Counterargument: for a compute-heavy transformer, eager SDPA/GEMM kernels may already dominate, leaving little compile benefit while codegen startup remains expensive.
- Target code path: real-audio `CustomDataset`, CFM stochastic prep, full F5TTS-small DiT (`dim=768`, `depth=18`, `heads=12`), BF16 autocast, AdamW fused, LinearLR, grad clipping, and optional compiled `_forward_loss_core()`.
- Dataset/setup: same 32 real WAV samples; batch size 1; `num_workers=0`; seed 1234; benchmark-only `--pad-to-frames 1536`; identical eager and compiled dataloader positions.
- Session name: `f5tts-a100-small-bf16-static`.
- Hardware request: A100. If unavailable, stop and record the allocation failure; do not silently substitute L4/G4.
- Cases:
  - eager: `--model-profile f5tts-small --precision bf16 --pad-to-frames 1536 --warmup-steps 4 --steps 10 --batch-size 1`
  - compiled: identical plus `--compile-enabled --compile-dynamic false`
- Warmup rationale: with Python seed 1234, the first four CFM calls cover the three CFG branch combinations `(audio,text) = (False,False), (True,False), (True,True)`, so the measured window should not discover a new boolean specialization.
- Metrics: step/forward/backward/optimizer timing, samples/frames throughput, loss/grad norm, cold compile setup/warmup, Dynamo counters, and peak memory.
- Correctness evidence: existing one-step float32 and five-step BF16 eager-vs-compiled gates; finite full-model losses/grad norms; compare paired loss ranges without requiring identical multi-step AdamW trajectories.
- Acceptance threshold: both cases complete on the same A100, compiled case has no graph-break/recompile-limit warning, artifacts download and parse before cleanup, and only directly comparable values are used for speedup.
- Abort criteria: A100 allocation failure, OOM, timeout, non-finite loss/grad, compile failure, new graph specialization in the measured window, or missing artifacts.

Result: passed.

- Session: `f5tts-a100-small-bf16-static`.
- Assigned hardware: NVIDIA A100-SXM4-40GB, 40960 MiB, driver 580.82.07; PyTorch 2.11.0+cu128; CUDA 12.8; BF16 supported.
- One kernel connection dropped before the matrix began; session status showed the last execution was still package installation, so one kernel restart was used. The paired matrix then completed.
- Artifacts:
  - `/tmp/f5tts_colab_results/a100_small_bf16_static/matrix_summary.json`
  - `/tmp/f5tts_colab_results/a100_small_bf16_static/f5small_a100_bf16_eager.json`
  - `/tmp/f5tts_colab_results/a100_small_bf16_static/f5small_a100_bf16_compile_false.json`
  - `/tmp/f5tts_colab_results/a100_small_bf16_static/session-log.jsonl`
- Cleanup: files were downloaded and parsed before `colab stop`; final `colab sessions` reported no active sessions.

Comparable steady-state results:

- Eager: mean step `0.1267282088s`, forward `0.0513551858s`, backward `0.0691409754s`, samples/s `7.8951960473`, frames/s `10966.5446814258`, peak allocated `3,521,055,232` bytes, peak reserved `3,602,907,136` bytes.
- Compiled static false: mean step `0.0601469082s`, forward `0.0211546716s`, backward `0.0325878998s`, samples/s `16.6341509834`, frames/s `23112.0065964669`, peak allocated `3,118,811,648` bytes, peak reserved `3,133,145,088` bytes.
- Speedup calculation: `0.1267282088 / 0.0601469082 = 2.10698x`.
- Forward speedup: `2.42760x`; backward speedup: `2.12168x`.
- Peak allocated memory decreased `11.42%`; peak reserved decreased `13.04%`.
- Dynamo counters: `unique_graphs=3`, `graph_break={}`, no `unimplemented` recompile-limit entry.

Cold-start and amortization:

- Eager four-warmup total: `2.0129186560s`.
- Compiled wrapper setup: `2.7219770810s`.
- Compiled warmup steps: `89.9654257590s`, `80.0506849540s`, `0.0600634720s`, `83.8277122610s`; total `253.9038864460s`. The fast third step reused an existing CFG specialization; the other three compiled the expected boolean branches.
- Break-even calculation: `(2.7219770810 + 253.9038864460 - 2.0129186560) / (0.1267282088 - 0.0601469082) = 3,824.09` post-warmup steps.
- Exact short-run wall-clock estimate for four warmups plus ten measured steps: eager `3.2802s`; compiled `257.2273s`. Compile is inappropriate for short fine-tunes despite the warmed 2.107x result.

Correctness/context:

- Both full-model runs had finite losses and grad norms. Eager measured loss mean `8.9218754292`; compiled `8.7774775982`. Eager grad-norm mean `11.5356838226`; compiled `11.3791728973`.
- Separate fixed-batch correctness is stronger evidence than comparing these post-warmup AdamW trajectories: five-step local CUDA BF16 passed the predeclared gate with loss-trajectory max delta `5.7220458984375e-06`.
- Activation-checkpointing one-step CUDA float32 compatibility also passed under `atol=rtol=1e-4`: loss max delta `2.384185791015625e-07`, gradient max delta `8.381903171539307e-09`, parameter-update max delta `3.637978807091713e-11`.
- `fullgraph=True` one-step CUDA float32 compatibility passed under the same tolerance: loss max delta `2.384185791015625e-07`, gradient max delta `7.450580596923828e-09`, parameter-update max delta `2.1827872842550278e-11`.

Decision:

- The 2x objective is technically feasible and demonstrated for a representative F5TTS-small BF16 fixed-shape A100 training path.
- Do not enable compile by default. The cold startup penalty is hundreds of seconds, break-even is about 3.8k steps, and arbitrary variable-length training still requires dynamic/default compile or an explicit stable-shape batching policy.
- Do not claim a general 2.107x across GPUs, models, datasets, or batching policies. L4 tiny results were about 1.52x, and unbucketed static compile previously regressed after hitting recompile limits.

Initial result: failed when the harness compiled the whole `CFM` module. Eager and compiled RNG/stochastic preprocessing diverged, causing loss/cond/gradient/parameter mismatches. This confirmed the compile target was too broad.

Fix: refactored `CFM.forward` so text conversion, mask creation, random span/noise/time sampling, and CFG drop decisions stay eager. Added `CFM._forward_loss_core()`, `compile_training_core()`, and `clear_training_compile()`, and changed `Trainer._configure_compile()` to compile that core on the unwrapped model instead of wrapping the whole Accelerate model. The compiled callable is stored with `object.__setattr__` so it is not registered into the module state dict.

Rerun result: passed on CPU and CUDA through `uv run --active --no-sync` with `/home/mithex/.venvs/ml312`.

- CPU command: `PYTHONPATH=src VIRTUAL_ENV=/home/mithex/.venvs/ml312 PATH=/home/mithex/.venvs/ml312/bin:$PATH UV_CACHE_DIR=/home/mithex/work/tts/.uv_cache uv run --active --no-sync python benchmarks/compile_correctness.py --device cpu --backend inductor --batch-size 2 --frames 64 --mel-dim 16 --text-len 24 --vocab-size 64 --seed 1234`
- CPU environment: Python 3.12.12, `torch 2.11.0+cu128`, CUDA available but device CPU.
- CPU max deltas: loss max_abs 2.384185791015625e-07, cond max_abs 0, pred max_abs 0, gradient max_abs 1.4901161193847656e-08, parameter-after-update max_abs 2.9103830456733704e-11. All checks passed under `atol=1e-5`, `rtol=1e-5`.
- CUDA command: `PYTHONPATH=src VIRTUAL_ENV=/home/mithex/.venvs/ml312 PATH=/home/mithex/.venvs/ml312/bin:$PATH UV_CACHE_DIR=/home/mithex/work/tts/.uv_cache uv run --active --no-sync python benchmarks/compile_correctness.py --device cuda --backend inductor --batch-size 2 --frames 64 --mel-dim 16 --text-len 24 --vocab-size 64 --seed 1234`
- CUDA environment: local RTX 3070, Python 3.12.12, `torch 2.11.0+cu128`, CUDA 12.8, cuDNN 91900.
- CUDA max deltas: loss max_abs 1.1920928955078125e-07, cond max_abs 0, pred max_abs 0, gradient max_abs 7.450580596923828e-09, parameter-after-update max_abs 2.1827872842550278e-11. All checks passed under `atol=1e-4`, `rtol=1e-4`.
- Follow-up: later DiT refactor removed the `TextEmbedding.forward` tensor `seq_len.max().item()` graph break from the current training core by passing padded `x.shape[1]` and explicit valid sequence lengths. Average text upsampling remains outside the compile-friendly path.
- Production Trainer smoke: one-update CUDA `Trainer.train()` smoke passed with compiled core active, fallback false, no `_orig_mod.` state keys, and `/tmp/f5tts_trainer_compile_core_smoke_cuda_v2/model_last.pt` written. The first throwaway smoke attempts failed because the synthetic dataset used `num_workers=0` with the trainer's existing `persistent_workers=True`, then used `mel` instead of the repo collate_fn's required `mel_spec` key.

### Planned Baseline Experiment: Local Smoke

- Hypothesis: A tiny local synthetic/preprocessed batch can validate imports, model construction, training-step mechanics, and metrics plumbing without making performance claims.
- Target code path: `CFM.forward` plus one eager optimizer step through the trainer-equivalent path.
- Dataset/setup: synthetic mel tensors and token tensors; no dataset/sampler performance claim.
- Command: `PYTHONPATH=src /home/mithex/.venvs/ml312/bin/python benchmarks/training_step_smoke.py --device cuda --warmup-steps 2 --steps 5 --batch-size 2 --frames 64 --mel-dim 16 --text-len 24 --vocab-size 64 --seed 1234`
- Metrics: step time, forward time, backward time, optimizer time, zero grad time, peak GPU memory if CUDA is available.
- Correctness: loss finite; gradients finite for selected parameters.
- Acceptance threshold: smoke completes without changing default behavior.
- Revert criteria: any required source change before baseline or failure caused by benchmark-only special casing.

Result: failed before measurement. The local CUDA run raised `RuntimeError: CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed ... CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` in Conv1d. This is an environment failure, not an F5-TTS performance result. No speed claim.

Repair result: fixed by reinstalling the ML venv PyTorch stack to CUDA 12.8 (`torch 2.11.0+cu128`, `torchaudio 2.11.0+cu128`, `torchvision 0.26.0+cu128`). Minimal CUDA Conv1d now passes.

Rerun result after repair: passed on local RTX 3070, compile disabled, mechanics-only. Mean step time 0.0071005184s, forward 0.0032375054s, backward 0.0034182518s, mean loss 2.0968930721, peak allocated 21,635,072 bytes.

### Planned Baseline Experiment: Local CPU Smoke

- Hypothesis: The same tiny synthetic/preprocessed batch can validate imports, model construction, training-step mechanics, and metrics collection on CPU after the local CUDA environment failure.
- Target code path: `CFM.forward` plus one eager optimizer step through the trainer-equivalent path.
- Dataset/setup: synthetic mel tensors and token tensors; no dataset/sampler performance claim.
- Command: `PYTHONPATH=src /home/mithex/.venvs/ml312/bin/python benchmarks/training_step_smoke.py --device cpu --warmup-steps 1 --steps 3 --batch-size 2 --frames 64 --mel-dim 16 --text-len 24 --vocab-size 64 --seed 1234`
- Metrics: step time, forward time, backward time, optimizer time, zero grad time.
- Correctness: loss finite; gradients finite for selected parameters.
- Acceptance threshold: smoke completes without changing default behavior.
- Revert criteria: any benchmark-only special casing or source behavior change.

Result: passed as mechanics-only smoke, compile disabled. This is not a real training throughput result.

- Environment: `/home/mithex/.venvs/ml312/bin/python`, Python 3.12.12, `torch 2.12.1+cu130`, device CPU, CUDA available but not used.
- Measured steps: 3 after 1 warmup.
- Step time seconds: mean 0.0078316560, median 0.0074018300, p90 0.0087069980, p95 0.0087069980, min 0.0073861400, max 0.0087069980, stdev 0.0007581090.
- Forward time seconds: mean 0.0031183700, median 0.0028195270, min 0.0027835460, max 0.0037520370.
- Backward time seconds: mean 0.0032756867, median 0.0031911450, min 0.0031852850, max 0.0034506300.
- Optimizer time seconds: mean 0.0013994620, median 0.0013884480.
- Loss: mean 1.9374804099, median 1.9408851862, min 1.8831588030, max 1.9883972406.
- Grad norm: mean 0.6032531857, median 0.6105772257, min 0.5500909090, max 0.6490914226.

### Planned Baseline Experiment: Real/Representative Training on Colab

- Hypothesis: Eager training throughput on a real or representative F5-TTS dataset establishes the comparison floor for compile work.
- Target code path: normal training command with compile disabled.
- Dataset/setup: real F5-TTS dataset or representative subset, documented before use.
- Command: pending dataset availability, Colab session plan, and environment setup.
- Metrics: full baseline requirements from `goal.md`.
- Correctness: loss behavior and checkpoint/resume smoke when feasible.
- Acceptance threshold: enough steady-state measured steps for a credible baseline.
- Revert criteria: dataset or command changes that make the baseline weaker than optimized runs.

### Planned Colab Lifecycle: F5TTS-Small Real-Audio Feasibility Probe

- Hypothesis: The shipped F5TTS-small DiT architecture can run on the same 32-sample real-audio `CustomDataset` path on Colab T4 with batch size 1, but compile overhead and/or memory may be high enough that a full paired benchmark needs a larger backend or a longer run plan.
- Target code path: `benchmarks/eager_training_baseline.py`, `CustomDataset`, `collate_fn`, CFM eager prep, DiT architecture matching `F5TTS_Small.yaml`, AdamW fused, LinearLR scheduler, grad clipping. Dataset/sampler/collation behavior remains unchanged.
- Dataset/setup: `/home/mithex/work/tts/f5-tts/data/librispeech_asr_custom` packed under `/content/F5-TTS/data/librispeech_asr_custom`, 32 WAV samples, real audio lengths, `num_workers=0`, shuffle true, drop_last false.
- Session name: `f5tts-small-probe-t4`
- Hardware request: Colab GPU T4. This is a bounded feasibility probe, not the final full-model benchmark.
- Eager command: `python benchmarks/eager_training_baseline.py --dataset-root /content/F5-TTS/data/librispeech_asr_custom --dataset-cwd /content/F5-TTS --device cuda --warmup-steps 1 --steps 1 --batch-size 1 --num-workers 0 --seed 1234 --model-profile f5tts-small --output /content/f5tts_small_eager_t4_probe.json`
- Conditional compiled command, only if eager fits: `python benchmarks/eager_training_baseline.py --dataset-root /content/F5-TTS/data/librispeech_asr_custom --dataset-cwd /content/F5-TTS --device cuda --warmup-steps 2 --steps 1 --batch-size 1 --num-workers 0 --seed 1234 --model-profile f5tts-small --compile-enabled --compile-dynamic true --output /content/f5tts_small_compiled_t4_probe.json`
- Metrics: same benchmark metrics plus peak memory and compile setup/cold-start data.
- Correctness checks: finite loss and grad norm; compile correctness for fixed batches is already covered separately, but this probe should not be used for broad correctness claims.
- Acceptance threshold: eager probe completes and reports memory/time; compiled probe is attempted only if eager memory is comfortably below T4 capacity and command timeout can bound cost.
- Revert/abort criteria: OOM, timeout, package/install failure, CUDA unavailable, compile failure, or unacceptable runtime. Record failure and stop the Colab session instead of retrying blindly.

### Planned Colab Lifecycle: Small Real-Audio Eager CUDA Baseline

- Session name: `f5tts-compile-baseline-t4`
- Hardware request: Colab GPU T4 for initial low-cost CUDA baseline. This is not the final big-GPU benchmark.
- Payload: current `F5-TTS/` checkout plus `/home/mithex/work/tts/f5-tts/data/librispeech_asr_custom` packed under `F5-TTS/data/librispeech_asr_custom`.
- Remote setup: extract payload under `/content`, install only missing Python runtime dependencies, verify `torch`, CUDA, and GPU identity.
- Remote command: run `benchmarks/eager_training_baseline.py` with `--device cuda --warmup-steps 3 --steps 10 --batch-size 2 --num-workers 0 --seed 1234`.
- Artifacts to retrieve: `/content/f5tts_eager_t4_baseline.json` and optional Colab log if needed.
- Cleanup: run `colab stop -s f5tts-compile-baseline-t4`, then `colab sessions` to verify no session remains.
- Revert/abort criteria: allocation/auth failure, package install failure, CUDA unavailable, or benchmark exception. Record failure instead of retrying blindly.

Result: passed. Session was stopped and `colab sessions` reported no active sessions.

- Assigned runtime: Colab T4, session `f5tts-compile-baseline-t4`.
- Remote environment: Python 3.12.13, `torch 2.11.0+cu128`, CUDA 12.8, GPU Tesla T4, GPU memory 15,637,086,208 bytes.
- Remote setup installed missing packages: `hydra-core`, `x-transformers`, `ema-pytorch`, `vocos`, `torchdiffeq`, `cached_path`, `pypinyin`, `rjieba`, `unidecode`.
- Artifact downloaded to `/tmp/f5tts_colab_results/f5tts_eager_t4_baseline.json`.
- Dataset: 32 rows, duration min 2.96s, max 16.085s, mean 13.929375s, batch size 2, num_workers 0, shuffle true, drop_last false.
- Model: tiny DiT profile, mel_dim 100, byte tokenizer, AdamW fused, constant LinearLR, float32.
- Measured steps: 10 after 3 warmup.
- Step time seconds: mean 0.0359838037, median 0.0357717145, p90 0.0370595030, p95 0.0380412150, min 0.0341683700, max 0.0380412150, stdev 0.0010873890.
- Data wait seconds: mean 0.0535758820, median 0.0506966005, p95 0.0721496860.
- Forward time seconds: mean 0.0120697025, median 0.0120313075, p95 0.0124953220.
- Backward time seconds: mean 0.0230796207, median 0.0229773800, p95 0.0248402810.
- Optimizer time seconds: mean 0.0004744801, median 0.0004784800.
- Samples/s: mean 55.6259409588, median 55.9145445021.
- Frames/s: mean 72911.4816720684, median 75079.7773688015.
- Padding ratio: mean 0.0888046797, median 0.0447258335, p95 0.4019746121.
- Mean frames per batch: mean 1312.4, median 1363.0.
- Loss: mean 10.6461351395, median 10.3457980156, min 8.8871555328, max 12.8232412338.
- Grad norm: mean 3.3602608681, median 3.2928441763, min 2.6704962254, max 4.1034879684.
- Peak GPU allocated memory: 66,918,400 bytes.
- Peak GPU reserved memory: 81,788,928 bytes.
- Scope: small real-audio CUDA eager baseline only; not a final representative full-model benchmark.

### Planned Colab Lifecycle: Small Real-Audio Paired Eager/Compiled CUDA Comparison

- Hypothesis: The deterministic compiled CFM training core is correct on the real-audio benchmark path, but speedup may be limited by shape specializations/recompiles, tiny model size, T4 compile overhead, and data wait. The run is a small comparison against the existing eager T4 baseline, not final full-model proof.
- Target code path: `benchmarks/eager_training_baseline.py` over `CustomDataset`, `collate_fn`, CFM eager prep, compiled `_forward_loss_core`, DiT backbone, AdamW fused, LinearLR scheduler, grad clipping.
- Dataset/setup: same packed payload style as the eager T4 baseline; `/home/mithex/work/tts/f5-tts/data/librispeech_asr_custom` copied into `F5-TTS/data/librispeech_asr_custom`, 32 WAV samples, batch size 2, `num_workers=0`, shuffle true, drop_last false.
- Session name: `f5tts-compile-paired-t4`
- Hardware request: Colab GPU T4. This keeps hardware comparable to `/tmp/f5tts_colab_results/f5tts_eager_t4_baseline.json`.
- Eager command: `python benchmarks/eager_training_baseline.py --dataset-root /content/F5-TTS/data/librispeech_asr_custom --dataset-cwd /content/F5-TTS --device cuda --warmup-steps 11 --steps 5 --batch-size 2 --num-workers 0 --seed 1234 --output /content/f5tts_eager_t4_paired.json`
- Compiled command: `python benchmarks/eager_training_baseline.py --dataset-root /content/F5-TTS/data/librispeech_asr_custom --dataset-cwd /content/F5-TTS --device cuda --warmup-steps 11 --steps 5 --batch-size 2 --num-workers 0 --seed 1234 --compile-enabled --compile-dynamic true --output /content/f5tts_compiled_t4_paired.json`
- Metrics: same step/data wait/host-to-device/forward/backward/optimizer/scheduler/zero-grad/samples/s/frames/s/padding/loss/grad-norm/memory metrics as eager baseline, plus compile target, setup time, and first warmup step/forward time as lazy compile cold-start context.
- Correctness checks: finite loss/grad norm for all measured steps, same model/data/config shape as eager baseline, and current local CPU/CUDA eager-vs-compiled numeric correctness evidence remains the detailed equivalence proof for fixed batches.
- Acceptance threshold: run completes, artifact is retrieved, Colab session is stopped, and results are compared without claiming speedup unless measured steady-state metrics support it.
- Revert/abort criteria: Colab allocation/auth failure, package install failure, CUDA unavailable, benchmark exception, compile failure, or memory blowup. Record failure instead of retrying blindly.

Local screening before Colab:

- Default dynamic compile on CPU real-audio smoke still recompiled heavily after the graph-break fix: measured step mean 10.2721098865s over 2 steps.
- `--compile-dynamic true` on CPU real-audio smoke showed cold compile and one shape compile, then a steady second measured step at 0.0744382100s, close to the earlier eager CPU baseline.
- Local CUDA smoke with `--compile-dynamic true`, 3 warmups, and 3 measured steps showed the same pattern: first measured step recompiled at 16.7988668540s, then steady measured steps at 0.0102710500s and 0.0111714980s. This was smoke/mechanics only, not a performance claim.

Result: passed as a small paired T4 comparison. Session `f5tts-compile-paired-t4` was stopped and `colab sessions` reported no active sessions.

- Artifacts:
  - Eager: `/tmp/f5tts_colab_results/f5tts_eager_t4_paired.json`
  - Compiled: `/tmp/f5tts_colab_results/f5tts_compiled_t4_paired.json`
- Remote environment: Python 3.12.13, `torch 2.11.0+cu128`, CUDA 12.8, GPU Tesla T4.
- Paired eager command: `python benchmarks/eager_training_baseline.py --dataset-root /content/F5-TTS/data/librispeech_asr_custom --dataset-cwd /content/F5-TTS --device cuda --warmup-steps 11 --steps 5 --batch-size 2 --num-workers 0 --seed 1234 --output /content/f5tts_eager_t4_paired.json`
- Paired compiled command: `python benchmarks/eager_training_baseline.py --dataset-root /content/F5-TTS/data/librispeech_asr_custom --dataset-cwd /content/F5-TTS --device cuda --warmup-steps 11 --steps 5 --batch-size 2 --num-workers 0 --seed 1234 --compile-enabled --compile-dynamic true --output /content/f5tts_compiled_t4_paired.json`
- Eager steady-state measured steps: mean step 0.0392404318s, forward 0.0141599246s, backward 0.0234454230s, samples/s 51.0564117672, frames/s 65978.1395374444, loss mean 11.1748369217, grad norm mean 3.1748567104, peak allocated 66,918,400 bytes, peak reserved 81,788,928 bytes.
- Compiled steady-state measured steps: mean step 0.0176160816s, forward 0.0067113142s, backward 0.0100827606s, samples/s 113.9034347794, frames/s 147077.7529036386, loss mean 11.2383642197, grad norm mean 3.2559156418, peak allocated 63,289,344 bytes, peak reserved 75,497,472 bytes.
- Steady-state speedup calculation: step time `0.0392404318 / 0.0176160816 = 2.2275x`; forward time `0.0141599246 / 0.0067113142 = 2.1099x`; backward time `0.0234454230 / 0.0100827606 = 2.3253x`; samples/s `113.9034347794 / 51.0564117672 = 2.2309x`; frames/s `147077.7529036386 / 65978.1395374444 = 2.2292x`.
- Compile overhead: `compile_training_core()` setup time 2.8252335490s. Lazy compile/codegen dominated warmup: compiled warmup step times were `[66.478, 36.848, 0.018, 46.000, 0.018, 0.017, 0.018, 0.017, 0.018, 0.018, 0.022]` seconds. Compiled total warmup was 149.47103661s vs eager total warmup 2.4174570890s. End-to-end including warmup for this small run was 149.5591170180s compiled vs 2.6136592480s eager, so compile is not beneficial for very short jobs.
- Break-even estimate for this exact tiny T4 setup: including compile setup and warmup delta, overhead is about 149.87881307s and steady measured step saving is about 0.0216243502s, so compile needs roughly 6,931 post-warmup optimizer steps to break even.
- Decision: keep the selective compiled core default-off. The objective's 2x target is met only for warmed steady-state on this tiny T4 real-audio path; broader/full-model and longer-run evidence is still required before any stronger claim or default change.

### Compile/metrics implementation smoke

- Command: `PYTHONPATH=src /home/mithex/.venvs/ml312/bin/python -m py_compile src/f5_tts/model/trainer.py src/f5_tts/train/train.py src/f5_tts/train/finetune_cli.py benchmarks/training_step_smoke.py benchmarks/eager_training_baseline.py`
- Result: passed.
- Command: parse every YAML config with OmegaConf and assert `metrics.enabled=False`, `compile.enabled=False`, `compile.target=model`, `compile.fallback_to_eager=True`.
- Result: passed.
- Command: `PYTHONPATH=src /home/mithex/.venvs/ml312/bin/python src/f5_tts/train/finetune_cli.py --help | rg -n "metrics_|compile_|usage"`
- Result: passed; CLI help includes metrics and compile flags.
- Command: instantiate `Trainer(..., compile_enabled=True, metrics_enabled=True)` and assert unwrapped model state dict keys do not start with `_orig_mod.`.
- Result: passed. Compile wrapper was active and state dict keys remained backward compatible.
- Command: tiny `Trainer.train()` CPU smoke with `CUDA_VISIBLE_DEVICES=`, `compile_enabled=True`, metrics enabled, synthetic dataset, checkpoint path under `/tmp/f5tts_trainer_compile_smoke_cpu`.
- Result: passed. Saved `/tmp/f5tts_trainer_compile_smoke_cpu/model_last.pt`; compile wrapper stayed active; fallback did not trigger.
- Graph break evidence from CPU compile smoke: TorchDynamo warned on `Tensor.item()` in `mask_from_start_end_indices` and Python `random()` in `CFM.forward`.
- Failed attempt: the same Trainer smoke without hiding CUDA hit the known local `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`; fallback to eager also failed because eager Conv1d hits the same local CUDA environment issue.
- Rerun after CUDA repair: tiny `Trainer.train()` CUDA smoke with `compile_enabled=True`, metrics enabled, synthetic dataset, checkpoint path under `/tmp/f5tts_trainer_compile_smoke_cuda`.
- Result: passed. Saved `/tmp/f5tts_trainer_compile_smoke_cuda/model_last.pt`; device CUDA; compile wrapper stayed active; fallback did not trigger.

### Verification After Optimization Taxonomy Pass

- Command: `git diff --check`
- Result: passed.
- Command: `PYTHONPATH=src VIRTUAL_ENV=/home/mithex/.venvs/ml312 PATH=/home/mithex/.venvs/ml312/bin:$PATH UV_CACHE_DIR=/home/mithex/work/tts/.uv_cache uv run --active --no-sync python -m py_compile src/f5_tts/model/backbones/dit.py src/f5_tts/model/cfm.py src/f5_tts/model/trainer.py src/f5_tts/train/train.py src/f5_tts/train/finetune_cli.py benchmarks/eager_training_baseline.py benchmarks/compile_correctness.py`
- Result: passed.
- Command: parse all six YAML configs with OmegaConf and assert metrics and compile remain disabled by default, `compile.target=model`, `compile.dynamic=null`, and fallback enabled.
- Result: passed.
- CPU correctness command: `PYTHONPATH=src VIRTUAL_ENV=/home/mithex/.venvs/ml312 PATH=/home/mithex/.venvs/ml312/bin:$PATH UV_CACHE_DIR=/home/mithex/work/tts/.uv_cache uv run --active --no-sync python benchmarks/compile_correctness.py --device cpu --backend inductor --batch-size 2 --frames 64 --mel-dim 16 --text-len 24 --vocab-size 64 --seed 1234`
- CPU result: passed. Max deltas remained loss `2.384185791015625e-07`, cond `0`, pred `0`, gradient `1.4901161193847656e-08`, parameter update `2.9103830456733704e-11`.
- CUDA correctness command: `PYTHONPATH=src VIRTUAL_ENV=/home/mithex/.venvs/ml312 PATH=/home/mithex/.venvs/ml312/bin:$PATH UV_CACHE_DIR=/home/mithex/work/tts/.uv_cache uv run --active --no-sync python benchmarks/compile_correctness.py --device cuda --backend inductor --batch-size 2 --frames 64 --mel-dim 16 --text-len 24 --vocab-size 64 --seed 1234`
- CUDA result: passed on local RTX 3070. Max deltas remained loss `1.1920928955078125e-07`, cond `0`, pred `0`, gradient `7.450580596923828e-09`, parameter update `2.1827872842550278e-11`; peak allocated `26976256` bytes.

### Planned Colab Lifecycle: L4 Dynamic-Mode Matrix And F5TTS-Small Probe

- Hypothesis: On a larger Colab GPU, the tiny real-audio path can distinguish `compile.dynamic=default/null`, `true`, and `false` without changing dataset, sampler, batch size, precision, or measured region. `dynamic=True` may reduce shape-specialized recompiles, but static/default modes may be faster if the shape set is small enough or specialization wins.
- Secondary hypothesis: F5TTS-small real-audio eager training should fit on L4 with batch size 1, and a bounded compiled `dynamic=True` probe can reveal whether compile setup is feasible enough to justify a longer F5TTS-small or A100 run.
- Target code path: `benchmarks/eager_training_baseline.py` over `CustomDataset`, `collate_fn`, CFM eager prep, optional compiled `_forward_loss_core`, DiT backbone, AdamW fused, LinearLR scheduler, and grad clipping. Dataset/sampler/collation behavior remains unchanged.
- Dataset/setup: `/home/mithex/work/tts/f5-tts/data/librispeech_asr_custom` packed under `/content/data/librispeech_asr_custom`, 32 WAV samples, real audio lengths, `num_workers=0`, shuffle true, drop_last false. Audio paths remain relative to `/content`.
- Session name: `f5tts-l4-dynamic-matrix`
- Hardware request: Colab GPU L4. If L4 allocation fails due availability/quota, stop and record the failure rather than silently changing hardware.
- Payload: current dirty checkout plus `.git` metadata and the 32-sample dataset, uploaded as `/content/f5tts_l4_payload.tar.gz` and extracted under `/content`.
- Matrix runner: `benchmarks/benchmark_matrix_runner.py` with spec `/content/f5tts_matrix_spec.json`, output directory `/content/f5tts_matrix_results`.
- Tiny eager command args: `--dataset-root /content/data/librispeech_asr_custom --dataset-cwd /content --device cuda --warmup-steps 11 --steps 5 --batch-size 2 --num-workers 0 --seed 1234 --model-profile tiny`
- Tiny compiled dynamic variants: same args plus `--compile-enabled --compile-dynamic default`, `--compile-enabled --compile-dynamic true`, and `--compile-enabled --compile-dynamic false`; each case timeout `900s`.
- F5TTS-small eager probe args: `--dataset-root /content/data/librispeech_asr_custom --dataset-cwd /content --device cuda --warmup-steps 1 --steps 1 --batch-size 1 --num-workers 0 --seed 1234 --model-profile f5tts-small`; timeout `900s`.
- F5TTS-small compiled feasibility args: same F5TTS-small args with `--warmup-steps 2 --steps 1 --compile-enabled --compile-dynamic true`; timeout `1200s`.
- Metrics: same step/data wait/host-to-device/forward/backward/optimizer/scheduler/zero-grad/samples/s/frames/s/padding/loss/grad-norm/memory metrics as earlier runs, plus compile target, setup time, first warmup step/forward time, per-case timeout/failure status, and raw JSON artifacts.
- Correctness checks: finite loss and grad norm for every completed measured step; current CPU/CUDA fixed-batch eager-vs-compiled correctness remains the numeric equivalence check for the compiled core.
- Acceptance threshold: retrieve `matrix_summary.json` and all completed case artifacts, stop the Colab session, and compare only completed comparable cases. Do not claim F5TTS-small speedup from a one-step feasibility probe.
- Revert/abort criteria: Colab allocation/auth failure, missing CUDA/L4, package install failure, benchmark exception, compile failure, OOM, timeout, or memory blowup. Record partial results and cleanup instead of retrying blindly or changing hardware.

Result: passed. Session `f5tts-l4-dynamic-matrix` was stopped and `colab sessions` reported no active sessions.

- Assigned runtime: Colab L4, `nvidia-smi` reported `NVIDIA L4, 23034 MiB, driver 580.82.07`.
- Remote environment: Python 3.12.13, `torch 2.11.0+cu128`, CUDA 12.8.
- Remote setup installed missing packages with `colab install` via uv: `hydra-core`, `x-transformers`, `ema-pytorch`, `vocos`, `torchdiffeq`, `cached_path`, `pypinyin`, `rjieba`, `unidecode`.
- Local artifacts:
  - `/tmp/f5tts_colab_results/l4_dynamic_matrix/matrix_summary.json`
  - `/tmp/f5tts_colab_results/l4_dynamic_matrix/tiny_l4_eager.json`
  - `/tmp/f5tts_colab_results/l4_dynamic_matrix/tiny_l4_compile_default.json`
  - `/tmp/f5tts_colab_results/l4_dynamic_matrix/tiny_l4_compile_true.json`
  - `/tmp/f5tts_colab_results/l4_dynamic_matrix/tiny_l4_compile_false.json`
  - `/tmp/f5tts_colab_results/l4_dynamic_matrix/f5small_l4_eager_probe.json`
  - `/tmp/f5tts_colab_results/l4_dynamic_matrix/f5small_l4_compile_true_probe.json`
- CLI note: the session log showed a keep-alive 403 warning, but upload, execution, download, and explicit stop all succeeded.

Tiny real-audio dynamic-mode comparison, same dataset/seed/batch/warmup/steps:

- Eager: mean step `0.0206601558s`, forward `0.0093535128s`, backward `0.0104544800s`, samples/s `96.8198644924`, frames/s `125475.1731280127`, peak allocated `66,756,608` bytes.
- Compiled `dynamic=default/null`: mean step `0.0152718000s`, forward `0.0058397646s`, backward `0.0085751442s`, samples/s `130.9941671347`, frames/s `169477.0676657413`, peak allocated `109,424,128` bytes, compile setup `2.7999131280s`, warmup total `97.3897459790s`.
- Compiled `dynamic=true`: mean step `0.0155186346s`, forward `0.0061314034s`, backward `0.0085302650s`, samples/s `128.9201958034`, frames/s `166891.0565609467`, peak allocated `107,087,872` bytes, compile setup `0.9088126630s`, warmup total `110.1440379440s`.
- Compiled `dynamic=false`: mean step `0.0210852804s`, forward `0.0096193964s`, backward `0.0106105314s`, samples/s `94.8541007313`, frames/s `122864.0409858261`, peak allocated `109,278,208` bytes, compile setup `0.9151850530s`, warmup total `121.7671251480s`. TorchDynamo hit `config.recompile_limit (8)` due frame-length shape mismatch (`x0` expected 1328, actual 1418).
- Tiny steady-state calculations:
  - `dynamic=default/null` step speedup: `0.0206601558 / 0.0152718000 = 1.3528x`; forward `1.6017x`; backward `1.2192x`.
  - `dynamic=true` step speedup: `1.3313x`; forward `1.5255x`; backward `1.2256x`.
  - `dynamic=false` step ratio: `0.9798x`, a slight regression with recompile-limit warnings.
  - Including compile setup and warmup, `dynamic=default/null` needs about `18,183` post-warmup optimizer steps to break even in this tiny L4 setup; `dynamic=true` needs about `21,169`; `dynamic=false` does not break even because measured steady-state was slower than eager.
- Decision: do not default to `dynamic=true`. In this L4 tiny real-audio matrix, `dynamic=default/null` was slightly faster than `dynamic=true`, while `dynamic=false` was unsafe for unbucketed variable lengths.

F5TTS-small L4 feasibility probes:

- Eager probe (`warmup=1`, `steps=1`, batch size 1): completed. One measured step `0.2069568230s`, forward `0.0572819480s`, backward `0.1275938150s`, frames/s `7054.6115795395`, peak allocated `3,548,358,144` bytes, peak reserved `3,936,354,304` bytes.
- Compiled `dynamic=true` feasibility probe (`warmup=2`, `steps=1`, batch size 1): completed. Compile setup `0.8962355920s`; warmup steps were `183.1935012990s` and `181.8901644390s`; one measured step `0.2028218360s`, forward `0.0523815240s`, backward `0.1284051380s`, peak allocated `3,351,776,256` bytes, peak reserved `3,657,433,088` bytes.
- The F5TTS-small eager and compiled measured steps are not a fair speed comparison because their warmup counts intentionally differ and therefore measured different dataloader positions. The supported conclusion is only that F5TTS-small fits on L4 and the compiled path can complete, but compile warmup is about `365s` for two warmup steps and must not be ignored.
- Decision: do not run A100 or longer F5TTS-small compile benchmarks until a better shape strategy or a clearly justified long-run amortization plan is pre-registered. A matching F5TTS-small eager run with the same warmup count is needed before any F5TTS-small speedup claim.

### Local Fixed-Frame Compileability Validation

- Change under test: benchmark-only `--pad-to-frames` support in `benchmarks/eager_training_baseline.py`, optional fixed output length for `mask_from_frac_lengths()`, and a tensor masked loss reduction in `CFM._forward_loss_core()` instead of boolean-indexing `loss[rand_span_mask]`.
- Production behavior: no dataset, sampler, collate, or config defaults were changed. Padding is only an explicit benchmark argument.
- CPU correctness command after the loss rewrite: `PYTHONPATH=src VIRTUAL_ENV=/home/mithex/.venvs/ml312 PATH=/home/mithex/.venvs/ml312/bin:$PATH UV_CACHE_DIR=/home/mithex/work/tts/.uv_cache uv run --active --no-sync python benchmarks/compile_correctness.py --device cpu --backend inductor --batch-size 2 --frames 64 --mel-dim 16 --text-len 24 --vocab-size 64 --seed 1234`
- CPU result: passed. Max deltas: loss `2.384185791015625e-07`, cond `0`, pred `0`, gradient `1.4901161193847656e-08`, parameter update `2.9103830456733704e-11`.
- CUDA correctness command after the loss rewrite: `PYTHONPATH=src VIRTUAL_ENV=/home/mithex/.venvs/ml312 PATH=/home/mithex/.venvs/ml312/bin:$PATH UV_CACHE_DIR=/home/mithex/work/tts/.uv_cache uv run --active --no-sync python benchmarks/compile_correctness.py --device cuda --backend inductor --batch-size 2 --frames 64 --mel-dim 16 --text-len 24 --vocab-size 64 --seed 1234`
- CUDA result: passed on local RTX 3070. Max deltas: loss `2.384185791015625e-07`, cond `0`, pred `0`, gradient `7.450580596923828e-09`, parameter update `2.1827872842550278e-11`.
- Fixed-frame static compile smoke command: `PYTHONPATH=src VIRTUAL_ENV=/home/mithex/.venvs/ml312 PATH=/home/mithex/.venvs/ml312/bin:$PATH UV_CACHE_DIR=/home/mithex/work/tts/.uv_cache uv run --active --no-sync python benchmarks/eager_training_baseline.py --dataset-root /home/mithex/work/tts/f5-tts/data/librispeech_asr_custom --dataset-cwd /home/mithex/work/tts/f5-tts --device cuda --warmup-steps 1 --steps 1 --batch-size 2 --num-workers 0 --seed 1234 --model-profile tiny --pad-to-frames 1536 --compile-enabled --compile-dynamic false --output /tmp/f5tts_padded_compile_false_smoke_after_loss.json`
- Fixed-frame static compile smoke result: passed. `dynamo_counters.graph_break` was empty and `stats.unique_graphs` was `2`; before the loss rewrite, the same path showed an `aten.nonzero.default` graph break from boolean loss indexing. The measured single step is not a throughput claim because it still includes local compile/codegen cost.

### Planned Colab Lifecycle: L4 Fixed-Shape Static/Dynamic Matrix

- Hypothesis: If real-audio batches are padded to a fixed benchmark frame length, `compile.dynamic=false` may regain static specialization and avoid the recompile-limit failure seen in the unpadded L4 matrix. Counterargument: fixed padding wastes compute, may erase any static-kernel advantage, and should not be generalized to default training without a sampler/bucketing design.
- Target code path: same as the L4 dynamic-mode matrix, with `CustomDataset`, `collate_fn`, CFM eager prep, optional compiled `_forward_loss_core`, DiT backbone, AdamW fused, LinearLR scheduler, and grad clipping.
- Shape policy: benchmark-only `--pad-to-frames 1536`. Keep `mel_lengths` unchanged so loss/masks operate on real valid frames; record both original batch padding ratio and `compute_padding_ratio` introduced by fixed-frame padding.
- Dataset/setup: `/home/mithex/work/tts/f5-tts/data/librispeech_asr_custom` packed under `/content/data/librispeech_asr_custom`, 32 WAV samples, real audio lengths, `num_workers=0`, shuffle true, drop_last false. Audio paths remain relative to `/content`.
- Session name: `f5tts-l4-fixed-shape-matrix`
- Hardware request: Colab GPU L4. If L4 allocation fails due availability/quota, stop and record the failure rather than silently changing hardware. A100/G4 remain later options if the fixed-shape L4 evidence justifies a larger or second-backend run.
- Payload: current dirty checkout plus `.git` metadata and the 32-sample dataset, uploaded as `/content/f5tts_l4_fixed_payload.tar.gz` and extracted under `/content`.
- Matrix runner: `benchmarks/benchmark_matrix_runner.py` with spec `/content/f5tts_fixed_matrix_spec.json`, output directory `/content/f5tts_fixed_matrix_results`.
- Cases: tiny eager fixed-shape baseline, tiny compiled `dynamic=default/null`, tiny compiled `dynamic=true`, and tiny compiled `dynamic=false`; each uses `--dataset-root /content/data/librispeech_asr_custom --dataset-cwd /content --device cuda --warmup-steps 11 --steps 5 --batch-size 2 --num-workers 0 --seed 1234 --model-profile tiny --pad-to-frames 1536`.
- Per-case timeout: `900s`. Do not add F5TTS-small or A100 in the same session; this run is only for the shape-policy question.
- Metrics: step/data wait/host-to-device/forward/backward/optimizer/scheduler/zero-grad/samples/s/frames/s/padding/compute-padding/input-frames/loss/grad-norm/memory, compile setup, first warmup step/forward time, per-case timeout/failure status, stderr, and `dynamo_counters`.
- Correctness checks: finite loss and grad norm for every completed measured step; CPU/CUDA fixed-batch correctness remains the numeric equivalence check for the compiled core.
- Acceptance threshold: retrieve `matrix_summary.json` and all completed case artifacts, stop the Colab session, verify `colab sessions` shows no active sessions, and compare only completed comparable cases.
- Revert/abort criteria: Colab allocation/auth failure, missing CUDA/L4, package install failure, benchmark exception, compile failure, OOM, timeout, `pad_to_frames` too short for a batch, memory blowup, or static compile still showing shape recompile-limit behavior.

Result: passed with an important negative result for static compile. Session `f5tts-l4-fixed-shape-matrix` was stopped and `colab sessions` reported no active sessions.

- Assigned runtime: Colab L4, `nvidia-smi` reported `NVIDIA L4, 23034 MiB, driver 580.82.07`.
- Remote environment: Python 3.12.13, `torch 2.11.0+cu128`, CUDA 12.8.
- Remote setup installed missing packages with `colab install` via uv: `hydra-core`, `x-transformers`, `ema-pytorch`, `vocos`, `torchdiffeq`, `cached_path`, `pypinyin`, `rjieba`, `unidecode`.
- Local artifacts:
  - `/tmp/f5tts_colab_results/l4_fixed_shape_matrix/matrix_summary.json`
  - `/tmp/f5tts_colab_results/l4_fixed_shape_matrix/tiny_l4_fixed_eager.json`
  - `/tmp/f5tts_colab_results/l4_fixed_shape_matrix/tiny_l4_fixed_compile_default.json`
  - `/tmp/f5tts_colab_results/l4_fixed_shape_matrix/tiny_l4_fixed_compile_true.json`
  - `/tmp/f5tts_colab_results/l4_fixed_shape_matrix/tiny_l4_fixed_compile_false.json`
- Fixed-frame eager baseline: mean step `0.0166920248s`, forward `0.0075236672s`, backward `0.0078840414s`, samples/s `119.8293639009`, frames/s `155233.7731779289`, original padding ratio `0.0752428907`, compute padding ratio after fixed-frame padding `0.1567057292`, peak allocated `67,520,000` bytes.
- Fixed-frame compiled `dynamic=default/null`: mean step `0.0112472860s`, forward `0.0036368708s`, backward `0.0062732980s`, samples/s `178.0562881855`, frames/s `230267.4557943645`, peak allocated `109,361,664` bytes, compile setup `2.8626828600s`, warmup total `55.3273592040s`, `dynamo_counters.graph_break={}`, `unique_graphs=4`.
- Fixed-frame compiled `dynamic=true`: mean step `0.0113972068s`, forward `0.0040159092s`, backward `0.0060332864s`, samples/s `175.4987396783`, frames/s `227222.0432090666`, peak allocated `109,361,664` bytes, compile setup `0.9370556240s`, warmup total `93.7744505440s`, `dynamo_counters.graph_break={}`, `unique_graphs=3`.
- Fixed-frame compiled `dynamic=false`: mean step `0.0170421016s`, forward `0.0077301346s`, backward `0.0078455468s`, samples/s `121.0924262879`, frames/s `157927.0559556002`, peak allocated `68,044,288` bytes, compile setup `0.9243271250s`, warmup total `60.7343910750s`, `dynamo_counters.graph_break={}`, `unique_graphs=8`.
- TorchDynamo still hit `config.recompile_limit (8)` for `dynamic=false`, now due text length (`text` size mismatch, expected 218, actual 201) rather than audio frame length. Fixed audio padding alone is therefore insufficient for static compile on this real-audio path.
- Fixed-shape steady-state calculations:
  - `dynamic=default/null` step speedup: `0.0166920248 / 0.0112472860 = 1.4841x`; forward `2.0687x`; backward `1.2568x`; estimated break-even about `10,313` post-warmup optimizer steps.
  - `dynamic=true` step speedup: `1.4646x`; forward `1.8735x`; backward `1.3068x`; estimated break-even about `17,502` post-warmup optimizer steps.
  - `dynamic=false` step ratio: `0.9795x`, a slight regression with recompile-limit behavior; no break-even because measured steady-state was slower than eager.
- Decision: keep `compile.dynamic=null` as the default config value and keep compile default-off. Do not recommend `dynamic=false` unless both audio and text shapes are stabilized, and do not default to `dynamic=true` because `default/null` remained slightly faster in this controlled fixed-frame L4 run.

### Planned Baseline Experiment: Small Real-Audio CPU Eager Baseline

- Hypothesis: A small real-audio `CustomDataset` baseline can validate the eager F5-TTS model/dataset/loss/optimizer measurement harness before moving the same style of run to Colab.
- Target code path: `CustomDataset`, `collate_fn`, `CFM.forward`, DiT backbone, AdamW, LinearLR scheduler, grad clipping.
- Dataset/setup: `/home/mithex/work/tts/f5-tts/data/librispeech_asr_custom`, 32 WAV samples, durations 2.96s to 16.085s, mean 13.929375s. Audio paths are relative to `/home/mithex/work/tts/f5-tts`.
- Command: `PYTHONPATH=src /home/mithex/.venvs/ml312/bin/python benchmarks/eager_training_baseline.py --dataset-root /home/mithex/work/tts/f5-tts/data/librispeech_asr_custom --dataset-cwd /home/mithex/work/tts/f5-tts --device cpu --warmup-steps 1 --steps 3 --batch-size 2 --num-workers 0 --seed 1234`
- Metrics: data wait, host-to-device transfer, forward, backward, optimizer, scheduler, zero grad, step time, samples/s, frames/s, padding ratio, sequence lengths, loss, grad norm.
- Correctness: loss finite; gradients finite for selected parameters.
- Acceptance threshold: completes without source behavior changes and records raw timings.
- Revert criteria: benchmark-only changes to dataset semantics or hidden local path assumptions in source code.

Result: passed as constrained CPU eager baseline, compile disabled. This is small real-audio evidence for the benchmark harness, not the final representative GPU throughput claim.

- Environment: `/home/mithex/.venvs/ml312/bin/python`, Python 3.12.12, `torch 2.12.1+cu130`, device CPU, CUDA available but not used.
- Dataset: 32 rows, duration min 2.96s, max 16.085s, mean 13.929375s, batch size 2, num_workers 0, shuffle true, drop_last false.
- Model: tiny DiT profile, mel_dim 100, byte tokenizer, AdamW, constant LinearLR, float32.
- Measured steps: 3 after 1 warmup.
- Step time seconds: mean 0.0822970077, median 0.0840628500, p90 0.0843916570, p95 0.0843916570, min 0.0784365160, max 0.0843916570, stdev 0.0033473236.
- Data wait seconds: mean 0.0206160697, median 0.0211454200.
- Forward time seconds: mean 0.0366346687, median 0.0366547060.
- Backward time seconds: mean 0.0432397367, median 0.0440006850.
- Optimizer time seconds: mean 0.0023309540, median 0.0021147130.
- Samples/s: mean 24.3296918221, median 23.7917224985.
- Frames/s: mean 33021.9047545951, median 32977.1934678747.
- Padding ratio: mean 0.0561093353, median 0.0578014184.
- Mean frames per batch: mean 1358.0, median 1354.0.
- Loss: mean 11.2203989029, median 11.2568712234, min 9.6663846970, max 12.7379407883.
- Grad norm: mean 3.5562676589, median 3.6789166927, min 3.1863710880, max 3.8035151958.

Local CUDA rerun after repair: passed on local RTX 3070, compile disabled, still small real-audio evidence only. Mean step time 0.0128200773s, data wait 0.0215115647s, forward 0.0052646943s, backward 0.0070405900s, samples/s 156.0550523584, frames/s 211868.3740718555, mean loss 11.1765152613, peak allocated 64,451,584 bytes.

## Remaining Limitations

- No multi-process DDP/FSDP run was available; the implementation keeps compile inside the wrapped CFM call, but distributed fallback behavior is only reasoned about, not empirically validated.
- Static false still requires stable audio shapes. The repo's default dynamic sampler was not modified; production bucketing/padding remains a separate opt-in design problem.
- DiT average text upsampling and flash-attention paths were not compiled or benchmarked.
- `reduce-overhead` and `max-autotune` were not promoted or benchmarked on A100 because their extra compile/memory cost is not a clear low-risk improvement over the validated default/static paths.
- The representative dataset is a 32-sample real-audio subset, not a full-corpus epoch benchmark. Model compute is representative; data-pipeline scale is not.

## Final Validation

- `PYTHONDONTWRITEBYTECODE=1 /home/mithex/.venvs/ml312/bin/python -m unittest discover -s tests -v`: 14 tests passed.
- Final CPU FP32 one-step eager/compiled correctness: passed; max loss delta `2.384185791015625e-07`, max gradient delta `1.4901161193847656e-08`, max parameter-update delta `2.9103830456733704e-11`.
- Final local CUDA BF16 five-step correctness with `dynamic=false`: passed at predeclared `atol=rtol=5e-3`; loss-trajectory max delta `5.7220458984375e-06`, final parameter relative L2 diagnostic `0.0002442892`.
- Representative A100 F5TTS-small BF16 static compile: `2.10698x` warmed speedup, about `3,824` steps to amortize cold compile, and 11.42% lower peak allocated memory.
- L4 controlled matrix: default/null `1.5209x`, dynamic true `1.4449x`, fresh-cache static false `1.5128x`; all used three graphs after eager text normalization.
- All Colab sessions were stopped and `colab sessions` reported no active sessions.

## Next Actions

No work is required for the current deliverable. Keep compile and metrics default-off.
Future optional work is multi-process validation and an opt-in shape-bucketing
policy tested against sampler, sharding, and resume semantics.

## Evidence Links

- Objective attachment: `/home/mithex/.codex/attachments/ac47b788-464d-4f2b-86ba-d73c031d09d2/pasted-text-1.txt`
- Repo root: `/media/mithex/NVME 2/Codex Linux/f5-tts-prs/F5-TTS`
- HEAD: `2ae2c9bd9b64dab2cb069c4b97e5e7673c521e01`
- Synthetic smoke utility: `benchmarks/training_step_smoke.py`
- Small real-audio eager baseline utility: `benchmarks/eager_training_baseline.py`
- Colab T4 eager baseline artifact: `/tmp/f5tts_colab_results/f5tts_eager_t4_baseline.json`
- L4 post-normalization artifacts: `/tmp/f5tts_colab_results/l4_static_text_normalized/`
- A100 F5TTS-small BF16 artifacts: `/tmp/f5tts_colab_results/a100_small_bf16_static/`
