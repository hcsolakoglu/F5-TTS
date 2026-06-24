# Project Memory: torch.compile Training Support

## Durable Facts

- Work started on 2026-06-23 in `/media/mithex/NVME 2/Codex Linux/f5-tts-prs/F5-TTS`.
- Starting branch: `main`.
- Starting commit: `2ae2c9bd9b64dab2cb069c4b97e5e7673c521e01`.
- Starting repo status inside `F5-TTS/`: clean.
- Outer workspace `/media/mithex/NVME 2/Codex Linux/f5-tts-prs` appears to be inside a larger parent git tree where `F5-TTS/` is untracked. Treat `F5-TTS/` as the project git root.
- Remote: `https://github.com/hcsolakoglu/F5-TTS.git`.
- Python command available on host: `python3`; `python` is not on PATH.
- System Python at start has no F5-TTS training dependencies installed.
- Existing ML venv found at `/home/mithex/.venvs/ml312`.
- `/home/mithex/.venvs/ml312` has Python 3.12.12, local CUDA available, and sees the local NVIDIA GeForce RTX 3070.
- Local CUDA was repaired on 2026-06-23 by reinstalling the venv PyTorch stack to `torch 2.11.0+cu128`, `torchaudio 2.11.0+cu128`, and `torchvision 0.26.0+cu128`; PyTorch now reports CUDA 12.8 and cuDNN 91900.
- `/home/mithex/.venvs/ml312` is missing these F5-TTS imports at audit time: `hydra-core`, `wandb`, `x-transformers`, `ema-pytorch`, `vocos`, `torchdiffeq`.
- Colab CLI is installed at `/home/mithex/.local/bin/colab`, version `0.5.11`; `colab sessions` reported no active sessions.
- Local GPU at start: NVIDIA GeForce RTX 3070, 8192 MiB, driver 595.71.05, CUDA version reported by `nvidia-smi` 13.2.
- Local dataset at start is incomplete for training: `data/Emilia_ZH_EN_pinyin/` only contains `vocab.txt`.

## Important Decisions

- Keep `torch.compile` disabled by default.
- Do not change dataset, sampler, bucketing, or collation before baseline measurements.
- Do not compile the whole training loop.
- Use a narrow compiled model/loss region first, after eager preprocessing and text tensorization.
- Keep sampling, logging, checkpointing, EMA, scheduler, optimizer, grad clipping, dataloading, and dataset transforms eager.
- Record failed experiments and negative results instead of hiding them.
- Compile support now compiles `CFM._forward_loss_core()` through `CFM.compile_training_core()` instead of wrapping the whole `CFM`/Accelerate module. Stochastic prep stays eager, while `self.model` remains the source of checkpoint state, EMA, optimizer parameters, and `accelerator.unwrap_model`, avoiding `_orig_mod.` state dict pollution.
- `compile.dynamic` must remain an explicit experiment knob. Do not default to `dynamic=True`; compare `null`, `True`, and `False` under the same dataset, shape policy, precision, and measured warmup before recommending a setting.
- `compile_optimization_analysis.md` is the current taxonomy for directly compilable code, refactor-needed code, eager-only boundaries, non-compile optimization candidates, and counterarguments.
- Runtime fallback belongs inside `CFM._run_training_core()`, not around `CFM.forward()`. It reuses prepared tensors and restores CPU/CUDA RNG state before eager retry because transformer dropout consumes PyTorch RNG.
- DiT and UNetT raw integer text can be eagerly cropped/padded to the audio tensor's padded width without changing their existing semantics. This removes raw text width as a separate compiled-core guard; it does not stabilize audio shapes.
- Trainer metrics aggregate complete optimizer updates across gradient-accumulation microbatches. `data_wait_time` includes Accelerate device dispatch; production H2D time is not isolated.

## Known Caveats

- `CFM.forward` uses Python `random()`, which is unsafe to compile naively if the CFG-drop decisions should remain semantically eager/random per call.
- Text list tokenization inside `CFM.forward` should not be part of the compiled graph.
- Dynamic sequence length handling can create guard failures and recompiles.
- `DiT.TextEmbedding.average_upsample_text_by_mask` contains Python loops and tensor-to-int conversions.
- Activation checkpointing and `fullgraph=True` passed one-step CUDA correctness checks; their performance tradeoffs remain unmeasured.
- Real performance conclusions require GPU measurements on a realistic dataset or representative subset, not only synthetic smoke tests.
- Big GPU performance runs should use Colab CLI with a named session, explicit lifecycle plan, artifact retrieval, `colab stop`, and `colab sessions` verification.
- The DiT training path was refactored so `get_input_embed()` passes padded `x.shape[1]` plus per-sample valid lengths into `TextEmbedding.forward()`. This avoids the `seq_len.max().item()` graph break for the compiled training core while preserving valid-position masking.
- `DiT.TextEmbedding.average_upsample_text_by_mask()` still has Python loops and tensor-to-int conversions; keep average-upsample mode out of compile until it is rewritten and tested.
- Multi-process distributed fallback has not been empirically tested.
- Static false is only safe when audio shapes are stabilized by the user's batching policy. No production sampler/bucketing default was changed.

## Failed Attempts

- Local CUDA smoke with `/home/mithex/.venvs/ml312` failed before measurement: `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` in Conv1d during `CFM.forward`. Treat local CUDA in this venv as unreliable until repaired; use CPU for mechanics and Colab CLI for representative GPU performance.
- Trainer compile smoke without hiding CUDA also failed from the same local CUDNN mismatch; fallback to eager could not help because eager Conv1d fails in that environment too.
- Repaired: minimal CUDA Conv1d, F5 synthetic CUDA smoke, F5 small real-audio CUDA baseline, and tiny Trainer CUDA compile smoke all pass after reinstalling the venv PyTorch stack to CUDA 12.8.
- Eager-vs-compiled correctness initially failed when the harness compiled the whole `CFM` module. Root cause: CFM stochastic preprocessing stayed inside the compiled region, so eager and compiled runs diverged under the same seed. Fix was to compile only `_forward_loss_core()` after eager random span/noise/time/drop decisions.
- The first text-normalization implementation was accidentally inserted in `CFM.sample()` rather than `CFM.forward()`; local static compile still hit text-width recompiles. A CFM-boundary regression test prevents recurrence after the hook was moved.
- The first L4 post-normalization artifact download claimed success but produced a zero-byte tar after the session was stopped. The matrix was rerun and each JSON was downloaded and parsed individually before cleanup.
- A five-step CPU protocol requiring final AdamW gradients/parameters to remain allclose at one-step tolerance failed despite a `3.58e-7` loss-trajectory delta. The failure is retained. Longer-run final gradient/parameter distances are diagnostics; first-step math and the multi-step loss/output trajectory remain gates.

## Benchmark Results

- Local CPU synthetic smoke passed with compile disabled. Command: `PYTHONPATH=src /home/mithex/.venvs/ml312/bin/python benchmarks/training_step_smoke.py --device cpu --warmup-steps 1 --steps 3 --batch-size 2 --frames 64 --mel-dim 16 --text-len 24 --vocab-size 64 --seed 1234`.
- CPU smoke scope: mechanics only, not real training throughput. Mean step time 0.0078316560s, mean forward 0.0031183700s, mean backward 0.0032756867s, mean loss 1.9374804099, mean grad norm 0.6032531857.
- After CUDA repair, local synthetic CUDA smoke passed on RTX 3070 with compile disabled. Mean step time 0.0071005184s, forward 0.0032375054s, backward 0.0034182518s, mean loss 2.0968930721, peak allocated 21,635,072 bytes.
- Small real-audio CPU eager baseline passed on `/home/mithex/work/tts/f5-tts/data/librispeech_asr_custom` with 32 WAV samples. Scope: constrained CPU harness baseline, not final representative GPU throughput. Mean step time 0.0822970077s, data wait 0.0206160697s, forward 0.0366346687s, backward 0.0432397367s, samples/s 24.3296918221, frames/s 33021.9047545951, mean loss 11.2203989029, mean grad norm 3.5562676589.
- After CUDA repair, local small real-audio CUDA baseline passed on RTX 3070 with compile disabled. Mean step time 0.0128200773s, data wait 0.0215115647s, forward 0.0052646943s, backward 0.0070405900s, samples/s 156.0550523584, frames/s 211868.3740718555, mean loss 11.1765152613, peak allocated 64,451,584 bytes.
- Colab T4 small real-audio CUDA eager baseline passed. Artifact: `/tmp/f5tts_colab_results/f5tts_eager_t4_baseline.json`. Environment: Python 3.12.13, `torch 2.11.0+cu128`, Tesla T4. Mean step time 0.0359838037s, data wait 0.0535758820s, forward 0.0120697025s, backward 0.0230796207s, samples/s 55.6259409588, frames/s 72911.4816720684, mean loss 10.6461351395, mean grad norm 3.3602608681, peak allocated 66,918,400 bytes, peak reserved 81,788,928 bytes. Colab session `f5tts-compile-baseline-t4` was stopped and `colab sessions` reported no active sessions.
- Optional Trainer metrics and compile config plumbing are implemented and default off. Syntax checks, YAML config parse checks, CLI help checks, Trainer constructor checks, state dict key checks, and a CPU Trainer compile smoke passed.
- After CUDA repair, tiny Trainer CUDA compile smoke passed. Compile wrapper stayed active, fallback did not trigger, and `/tmp/f5tts_trainer_compile_smoke_cuda/model_last.pt` was written.
- After the deterministic-core refactor, eager-vs-compiled correctness passed on CPU and local CUDA using `uv run --active --no-sync` with `/home/mithex/.venvs/ml312`. CPU max deltas: loss 2.384185791015625e-07, cond 0, pred 0, gradient 1.4901161193847656e-08, parameter update 2.9103830456733704e-11. CUDA max deltas: loss 1.1920928955078125e-07, cond 0, pred 0, gradient 7.450580596923828e-09, parameter update 2.1827872842550278e-11.
- Production CUDA Trainer smoke after the deterministic-core refactor passed with compiled core active, fallback false, no `_orig_mod.` state keys, and `/tmp/f5tts_trainer_compile_core_smoke_cuda_v2/model_last.pt` written. A throwaway smoke attempt with `num_workers=0` failed because the existing trainer sets `persistent_workers=True`; another failed because the synthetic dataset used `mel` instead of `mel_spec` for the repo collate function.
- After the DiT text-length refactor, fixed-batch correctness still passed on CPU and local CUDA with the same max deltas as above, and the previous `Tensor.item()` graph-break warning disappeared from those correctness runs.
- After adding `compile_optimization_analysis.md`, tightening dynamic-shape wording, and cleaning a dead DiT average-upsampling branch, `git diff --check`, `py_compile`, YAML default assertions, and CPU/CUDA fixed-batch compile correctness passed. CPU max deltas: loss 2.384185791015625e-07, gradient 1.4901161193847656e-08, parameter update 2.9103830456733704e-11. CUDA max deltas: loss 1.1920928955078125e-07, gradient 7.450580596923828e-09, parameter update 2.1827872842550278e-11.
- Added `benchmarks/benchmark_matrix_runner.py` so Colab sessions can run a JSON-defined benchmark matrix with per-case timeouts and preserve partial artifacts.
- Paired Colab T4 current-code comparison passed and was cleaned up. Artifacts: `/tmp/f5tts_colab_results/f5tts_eager_t4_paired.json` and `/tmp/f5tts_colab_results/f5tts_compiled_t4_paired.json`. Session `f5tts-compile-paired-t4` was stopped and `colab sessions` reported no active sessions.
- Paired T4 steady-state result with warmup excluded: eager mean step 0.0392404318s, compiled mean step 0.0176160816s, speedup 2.2275x. Forward speedup 2.1099x, backward speedup 2.3253x, samples/s speedup 2.2309x, frames/s speedup 2.2292x. GPU peak allocation was slightly lower compiled (63,289,344 bytes) than eager (66,918,400 bytes).
- Paired T4 compile overhead is large: setup 2.8252335490s, compiled warmup total 149.47103661s vs eager warmup total 2.4174570890s. End-to-end including warmup for this small run was 149.5591170180s compiled vs 2.6136592480s eager. Do not present compile as beneficial for short jobs.
- Break-even estimate for that exact tiny T4 setup is about 6,931 post-warmup optimizer steps: `(compiled warmup + setup - eager warmup) / (eager mean step - compiled mean step)`.
- This statement was superseded by the later paired A100 F5TTS-small BF16 result. Short-run and default-on claims remain unsupported.
- The next Colab matrix should use L4 for a common modern-GPU signal, A100 for full/base-model compile-amortization evidence, and G4 only if useful or if L4/A100 allocation is unavailable. Every run needs a named session, artifact download, `colab stop`, and `colab sessions` verification.
- Colab L4 dynamic-mode matrix completed and was cleaned up. Artifacts are under `/tmp/f5tts_colab_results/l4_dynamic_matrix/`. Tiny real-audio results: eager mean step 0.0206601558s; compiled `dynamic=default/null` 0.0152718000s (1.3528x warmed step speedup, about 18,183 post-warmup break-even steps); compiled `dynamic=true` 0.0155186346s (1.3313x, about 21,169 break-even steps); compiled `dynamic=false` 0.0210852804s and hit TorchDynamo recompile limit from frame-length shape changes. Do not default to `dynamic=True`; on this L4 tiny matrix, `default/null` was slightly better and static false was worse.
- F5TTS-small L4 probes completed. Eager batch-1 one-step probe fit with peak allocated 3.55GB and measured step 0.2069568230s. Compiled `dynamic=true` also fit with peak allocated 3.35GB, but two warmup steps took 183.19s and 181.89s; the one measured step was 0.2028218360s. Because eager used `warmup=1` and compiled used `warmup=2`, this is only compile feasibility evidence, not a speedup comparison.
- Colab L4 fixed-frame matrix completed and was cleaned up. Artifacts are under `/tmp/f5tts_colab_results/l4_fixed_shape_matrix/`. The benchmark-only `--pad-to-frames 1536` path preserved real `mel_lengths` and recorded compute padding. Tiny fixed-shape results: eager mean step 0.0166920248s; compiled `dynamic=default/null` 0.0112472860s (1.4841x warmed step speedup, about 10,313 post-warmup break-even steps); compiled `dynamic=true` 0.0113972068s (1.4646x, about 17,502 break-even steps); compiled `dynamic=false` 0.0170421016s and still hit TorchDynamo recompile limit, now from variable text length rather than audio frame length. Fixed audio padding alone is not enough for static compile; text shape bucketing/padding must be addressed before retrying `dynamic=false`.

- Post-normalization L4 artifacts are under `/tmp/f5tts_colab_results/l4_static_text_normalized/`. Fixed audio plus eager text normalization reduced all compiled modes to three graphs with no recompile-limit entry. Eager `0.0166214668s`; default/null `0.0109285218s` (`1.5209x`); dynamic true `0.0115034976s` (`1.4449x`); static false fresh-cache `0.0109868852s` (`1.5128x`). Cold break-even was about 7,814 default, 17,179 dynamic true, and 7,178 static false.
- Representative A100 F5TTS-small BF16 artifacts are under `/tmp/f5tts_colab_results/a100_small_bf16_static/`. Eager mean step `0.1267282088s`; compiled static false `0.0601469082s`; warmed speedup `2.10698x`. Forward `2.4276x`, backward `2.1217x`. Peak allocated memory fell 11.42%. Three cold CFG graph compilations produced `253.9039s` warmup and a roughly 3,824-step break-even. Short-run total was `257.23s` compiled versus `3.28s` eager.
- Five-step CUDA BF16 correctness passed at predeclared `atol=rtol=5e-3`: loss-trajectory max delta `5.72e-6`, final parameter relative L2 `2.44e-4`. Activation-checkpointing and fullgraph one-step CUDA checks also passed.

## Where Work Stopped

- Implementation, representative A100 validation, L4 shape analysis, BF16 correctness, checkpoint/resume, sampling isolation, and CPU-safe regression coverage are complete.
- Compile and metrics remain disabled by default; `compile.dynamic` remains `null`.
- Final validation passed: 14 CPU-safe tests, CPU FP32 one-step correctness, and local CUDA BF16 five-step static correctness on the finalized source.
- Repository cleanup, diff audit, and handoff wording are complete. No Colab sessions are active.
