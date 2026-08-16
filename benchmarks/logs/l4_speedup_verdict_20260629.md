# F5-TTS L4 speedup verdict partial/final evidence
## Completed result rows
| candidate | wall_s | wall_vs_eager | median_s | median_speedup | first_s | graphs | fallback | target | alloc_mb |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| eager | 92.269 | 1.000 | 1.534 | 1.000 | 1.988 | None | False | None | 14788.3 |
| compiled_dit_blocks_dynamic | 110.849 | 0.832 | 1.255 | 1.223 | 24.308 | 2 | False | dit_blocks | 12838.7 |
| compiled_dit_blocks_fullgraph_dynamic | 111.049 | 0.831 | 1.259 | 1.218 | 24.119 | 2 | False | dit_blocks | 12838.7 |

## Partial/rejected long-compile rows
- no_autotune: results=0, correctness=1, last_heartbeat=958.2, log=/media/mithex/NVME 2/Codex Linux/f5-tts-prs/agent-worktrees/fullgraph-speedup-glm1/benchmarks/logs/agent_no_autotune_l4_v2.20260629_182037.log
  - correctness compiled_branchless_dynamic_no_autotune_pw: loss_abs_diff=0.0, pred_max_abs_diff=0.0, grad_max_abs_diff=2.9802322387695312e-08, fallback=False, target=cfm_loss_core
- reduce_mark: results=0, correctness=1, last_heartbeat=810.5, log=/media/mithex/NVME 2/Codex Linux/f5-tts-prs/agent-worktrees/fullgraph-speedup-glm1/benchmarks/logs/agent_reduce_mark_l4_v2.20260629_183745.log
  - correctness compiled_branchless_reduce_overhead_markstep: loss_abs_diff=0.0, pred_max_abs_diff=0.0, grad_max_abs_diff=2.9802322387695312e-08, fallback=False, target=cfm_loss_core

## Decision
- Keep compile opt-in; do not default fullgraph/reduce-overhead/no-autotune full-core.
- Best validated performance profile remains regional `dit_blocks` for DiT/F5-TTS: steady median improves and VRAM drops, but short 65-update wall still loses to eager.
- `dit_blocks + fullgraph` adds no benefit over non-fullgraph.
- Full-core `cfm_loss_core` variants (`fullgraph`, `no_autotune_pw`, `reduce_overhead_markstep`) are rejected as wall-time defaults because they spend many multiples of eager epoch time before producing a Trainer result.
