# Colab L4 real trainer torch.compile A/B

Generated: 2026-06-28

## Scope

- Commit: `e04c641e6cc38f280d1987e58e853313c3cc44db`
- Branch: `torch-compile-upstream-integration`
- GPU requested/assigned: `L4` / `NVIDIA L4`
- Torch: `2.11.0+cu128`
- Dataset: `openslr/librispeech_asr`, config `clean`, split `train.100`, streaming
- Samples: 1000
- Audio hours per epoch: 2.0577500868
- Epochs / updates: 2 epochs, 130 updates
- Unique batch shapes: 65
- Unique frame lengths: 594
- Trainer/model: real F5-TTS `Trainer`, `DynamicBatchSampler`, dataset collate path, CFM + DiT depth=18 dim=768 heads=12, 158,056,804 params
- Precision: fp32, TF32 disabled, no AMP/quantization
- Candidates: `eager`, `compiled_default`
- Local raw log: `benchmarks/logs/colab_l4_real_ab_v14_full.local.log`

## Correctness probe

`compiled_default` one-batch parity against eager:

- `loss_abs_diff`: 0.0
- `pred_max_abs_diff`: 0.0
- `grad_max_abs_diff`: 2.9802322387695312e-08
- compile state: enabled true, fallback false

## Timing and memory

| candidate | wall s | mean update s | median update s | p90 update s | first update s | last update s | wall speedup vs eager | median speedup vs eager | max alloc MB | max reserved MB | unique graphs | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| eager | 182.381 | 1.403 | 1.515 | 1.607 | 1.944 | 1.508 | 1.000x | 1.000x | 14788 | 17710 | - | false |
| compiled_default | 1088.757 | 8.375 | 1.202 | 1.286 | 92.717 | 1.169 | 0.168x | 1.261x | 13042 | 16600 | 8 | false |

## Derived metrics

- Median step saved by compile: 0.313252 s/update, about 313 ms/update.
- Compiled wall extra vs eager: 906.376 s.
- Observed break-even from this run: about 2893 updates.
- Eager processed about 4.1155 audio-hours in 182.38 s, roughly 81.2x realtime audio throughput.
- Compiled processed about 4.1155 audio-hours in 1088.76 s, roughly 13.6x realtime audio throughput.
- Compile reduced reported CUDA max allocated memory by about 1.75 GB and reserved memory by about 1.11 GB, but lost heavily on wall time.

## Interpretation

This L4 run confirms the split:

- `compiled_default` is faster in steady median update time: 1.202 s vs 1.515 s, 1.26x.
- `compiled_default` is much slower in full wall time: 1088.76 s vs 182.38 s, 0.168x wall speed.
- The first compiled update alone cost 92.7 s, and Dynamo produced 8 unique graphs on 65 unique batch shapes. The mean update time is therefore dominated by compile/recompile spikes even though median/p90 warmed steps are faster.
- For this real variable-shape L4 workload, compile does not amortize in 130 updates. Observed break-even is roughly 2893 updates under these exact conditions.

## Notes

- Colab CLI was updated from 0.5.11 to 0.6.0 after a first L4 run was pruned by keepalive failure. The successful run ended cleanly and `colab sessions` reported no active sessions afterward.
- Earlier G4/Blackwell result showing eager wall around 29.6 s is not representative of this L4 run. The qualitative pattern is consistent, however: compiled median step improves, wall time regresses because of compile/recompile overhead.
