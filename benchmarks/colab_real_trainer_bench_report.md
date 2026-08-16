# Colab G4 real trainer torch.compile benchmark
Generated: 2026-06-27T20:46:26.330668

## Workload
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- Torch: 2.11.0+cu128
- Dataset: openslr/librispeech_asr/clean split=train.100 streaming, 1000 samples, 2.058 audio hours, 76 speakers
- Real trainer: F5-TTS DiT depth=18 dim=768 heads=12, 158,056,804 params, DynamicBatchSampler frame batches
- Epochs/updates: 2 epochs, 130 updates, 65 unique batch shapes; first shapes: [[24, 213, 44], [24, 230, 51], [24, 260, 48], [24, 280, 50]] ...
- Precision: fp32, TF32 disabled, no AMP/quantization ({'amp_or_quantization': 'none', 'float32_matmul_precision': 'highest', 'tf32_cudnn': False, 'tf32_matmul': False})

## Correctness probe
- compiled_default: loss_abs_diff=0.0, pred_max_abs_diff=0.0, grad_max_abs_diff=5.960464477539063e-08, fallback=False
- reduce_overhead_markstep: loss_abs_diff=0.0, pred_max_abs_diff=0.0, grad_max_abs_diff=5.960464477539063e-08, fallback=False
- combo_cudagraphs_markstep: loss_abs_diff=0.0, pred_max_abs_diff=0.0, grad_max_abs_diff=5.960464477539063e-08, fallback=False

## Timing summary
| candidate | median update s | median speedup vs eager | mean update s | wall s | wall speedup vs eager | max reserved MB | unique graphs | recompile limit | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| eager | 0.241979 | 1.000x | 0.228014 | 29.642 | 1.000x | 17788 | - | False | False |
| compiled_default | 0.209801 | 1.153x | 2.910281 | 378.336 | 0.078x | 15482 | 8 | True | False |
| compiled_mark_dynamic | 0.238047 | 1.017x | 3.184766 | 414.020 | 0.072x | 17000 | 8 | True | False |
| reduce_overhead_markstep | 0.218786 | 1.106x | 2.867695 | 372.800 | 0.080x | 15060 | 8 | False | False |
| combo_cudagraphs_markstep | 0.225578 | 1.073x | 3.922852 | 509.971 | 0.058x | 15964 | 8 | False | False |

## Decision
- Do not integrate cudagraph/combo/markstep profile into PR for real variable-length F5-TTS training yet. It is correctness-safe in the probe, but not production-speed-safe on the real dynamic-batch workload.
- Eager wall time for the 130-update trainer section was 29.64s. All compile candidates spent 372-510s wall time due compilation/recompilation overhead, despite small steady-state median improvements.
- Root cause evidence: 65 unique batch shapes from real DynamicBatchSampler, 8 Dynamo unique graphs, and recompile-limit warnings on default/mark_dynamic. The synthetic repeated-shape wins do not transfer to this workload without bucketing/shape-control or deeper compiler-safe model changes.
- Safe PR action: keep existing default-off compile support only. Do not add advanced cudagraph/combo defaults. If adding options/markstep fields, document as experimental/fixed-shape only, not validated production speed path for variable real data.
