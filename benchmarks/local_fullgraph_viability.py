#!/usr/bin/env python3
"""Local GPU viability experiment for F5-TTS torch.compile candidates.

Hard-to-game: real CFM/DiT branchless-CFG code path, real collate_fn, real backward,
real optimizer.step+scheduler, real masks, ragged variable-length batches (mimics
LibriSpeech 65-unique-shape pressure). Fresh subprocess per candidate (resets Dynamo).
Correctness gate vs eager on identical batch (loss/grad parity, finite loss, state_dict
keys clean, fallback inactive) BEFORE timing. Captures graph_breaks, unique_graphs,
recompiles, first/median/p90/wall, VRAM, fallback.

NOTE: tiny DiT on a small GPU. Timing is small-scale and NON-decisive for L4 wall-time;
viability/graph-break evidence IS decisive.
"""
from __future__ import annotations
import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

CANDIDATES: dict[str, dict[str, Any]] = {
    "eager": {"compile": False},
    "cfm_fullgraph_autodyn": {"compile": True, "target": "cfm_loss_core", "fullgraph": True, "dynamic": None},
    "cfm_fullgraph_dynamic": {"compile": True, "target": "cfm_loss_core", "fullgraph": True, "dynamic": True},
    "cfm_nofullgraph_dynamic": {"compile": True, "target": "cfm_loss_core", "fullgraph": False, "dynamic": True},
    "dit_blocks_fullgraph": {"compile": True, "target": "dit_blocks", "fullgraph": True, "dynamic": None},
    "dit_blocks_nofullgraph_dynamic": {"compile": True, "target": "dit_blocks", "fullgraph": False, "dynamic": True},
    "cfm_reduce_overhead_markstep": {"compile": True, "target": "cfm_loss_core", "fullgraph": True, "dynamic": None, "mode": "reduce-overhead", "mark_step": True},
    "cfm_combo_cudagraphs": {
        "compile": True, "target": "cfm_loss_core", "fullgraph": True, "dynamic": None, "mark_step": True,
        "options": {"triton.cudagraphs": True, "combo_kernels": True, "benchmark_combo_kernel": True, "combo_kernels_autotune": 2, "combo_kernel_allow_mixed_sizes": 2},
    },
}


def _build_model(vocab_size, vocab_char_map, device):
    import torch
    from f5_tts.model import CFM, DiT
    cfg = dict(dim=96, depth=6, heads=4, ff_mult=2, text_dim=128, text_mask_padding=True,
               qk_norm=None, conv_layers=2, pe_attn_head=None, attn_backend="torch",
               attn_mask_enabled=False, checkpoint_activations=False)
    mel_kw = dict(n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=80,
                  target_sample_rate=24000, mel_spec_type="vocos")
    m = CFM(transformer=DiT(**cfg, text_num_embeds=vocab_size, mel_dim=80),
            mel_spec_kwargs=mel_kw, vocab_char_map=vocab_char_map,
            audio_drop_prob=0.3, cond_drop_prob=0.2)
    return m.to(device), cfg, mel_kw


class SynthDataset:
    def __init__(self, n=64, n_mel=80, base_len=40, seed=0):
        import torch
        import string as _string
        g = torch.Generator().manual_seed(seed)
        self.mels, self.texts = [], []
        chars = list(_string.ascii_lowercase + " ")
        for i in range(n):
            L = max(8, base_len + int(20 * (torch.rand(1, generator=g).item() * n // 8)))
            self.mels.append(torch.randn(n_mel, L, generator=g) * 0.5)
            tl = 4 + int(torch.rand(1, generator=g).item() * 12)
            self.texts.append("".join(chars[int(torch.randint(0, len(chars), (1,), generator=g).item())] for _ in range(tl)))
    def __len__(self): return len(self.mels)
    def get_frame_len(self, i): return self.mels[i].shape[-1]
    def __getitem__(self, i): return {"mel_spec": self.mels[i], "text": self.texts[i]}


def worker(candidate, out_dir, n_updates, n_shapes):
    import torch
    import random as _random
    from torch.utils.data import SequentialSampler
    from f5_tts.model.dataset import DynamicBatchSampler, collate_fn
    from f5_tts.model.utils import get_tokenizer
    import torch._dynamo as dynamo

    cfg_c = CANDIDATES[candidate]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")
    dynamo.reset(); dynamo.utils.counters.clear()

    vocab_char_map, vocab_size = get_tokenizer("", "byte")
    ds = SynthDataset(n=n_shapes, n_mel=80)
    sampler = SequentialSampler(ds)
    bs = DynamicBatchSampler(sampler, 32*4, max_samples=32, random_seed=0, drop_residual=False)
    batches = list(bs.batches)
    pre = []
    shape_set = set()
    for b in batches:
        items = [ds[i] for i in b]
        c = collate_fn(items)
        mel = c["mel"].permute(0, 2, 1).to(device)
        lens = c["mel_lengths"].to(device)
        text = c["text"]  # list of str; CFM tokenizes internally
        tlen = int(c["text_lengths"].amax().item())
        pre.append((mel, lens, text))
        shape_set.add((mel.shape[0], mel.shape[1], tlen))
    shapes = sorted(shape_set)

    model, _, _ = _build_model(vocab_size, vocab_char_map, device)
    opt = torch.optim.AdamW(model.parameters(), lr=7.5e-5, fused=torch.cuda.is_available())
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)

    gate_mel, gate_lens, gate_text = pre[0]

    def run_eager(mel, lens, text):
        m2, _, _ = _build_model(vocab_size, vocab_char_map, device)
        m2.load_state_dict(model.state_dict())
        m2.train()
        o2 = torch.optim.AdamW(m2.parameters(), lr=7.5e-5, fused=torch.cuda.is_available())
        o2.zero_grad(set_to_none=True)
        _random.seed(777); torch.manual_seed(777)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(777)
        loss = m2(mel, text, lens=lens)[0]
        loss.backward()
        grads = [p.grad.detach().clone() if p.grad is not None else None for p in m2.parameters()]
        return loss.detach(), grads, list(m2.state_dict().keys())

    if cfg_c["compile"]:
        ck = dict(backend="inductor", fullgraph=cfg_c.get("fullgraph", False), dynamic=cfg_c.get("dynamic"))
        ck = {k: v for k, v in ck.items() if v is not None}
        try:
            if cfg_c.get("options"):
                # Do NOT pass mode together with options (task constraint).
                model.compile_training_core(target=cfg_c["target"], runtime_fallback=False, options=cfg_c["options"], **ck)
            else:
                kw = dict(ck)
                if cfg_c.get("mode"):
                    kw["mode"] = cfg_c["mode"]
                model.compile_training_core(target=cfg_c["target"], runtime_fallback=False, **kw)
        except Exception as e:
            res = {"candidate": candidate, "setup_ok": False, "error": "setup:" + repr(e)}
            print("RESULT_JSON=" + json.dumps(res, default=str))
            return
        if cfg_c.get("mark_step"):
            ms = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
            orig = model.forward
            def fwd(*a, **k):
                ms()
                return orig(*a, **k)
            model.forward = fwd

    needs_loss_clone = bool(cfg_c.get("mark_step") or cfg_c.get("options") or cfg_c.get("mode") == "reduce-overhead")

    model.train()
    opt.zero_grad(set_to_none=True)
    _random.seed(777); torch.manual_seed(777)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(777)
    try:
        c_loss = model(gate_mel, gate_text, lens=gate_lens)[0]
        if needs_loss_clone:
            c_loss = c_loss.clone()
        c_loss.backward()
    except Exception as e:
        res = {"candidate": candidate, "setup_ok": True, "correctness": False, "error": "gate_run:" + repr(e),
               "compile_state": getattr(model, "training_compile_state", {})}
        print("RESULT_JSON=" + json.dumps(res, default=str))
        return
    c_grads = [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]
    c_keys = list(model.state_dict().keys())
    e_loss, e_grads, e_keys = run_eager(gate_mel, gate_lens, gate_text)
    finite = bool(torch.isfinite(c_loss).all().item())
    loss_close = bool(torch.allclose(c_loss.detach(), e_loss, atol=1e-4, rtol=1e-4))
    grad_close = all(g is None and eg is None or (g is not None and eg is not None and torch.allclose(g, eg, atol=1e-3, rtol=1e-3))
                     for g, eg in zip(c_grads, e_grads))
    keys_clean = all("_orig_mod" not in k for k in c_keys) and c_keys == e_keys
    fallback = bool(getattr(model, "_compile_fallback_active", False))
    correctness = finite and loss_close and grad_close and keys_clean and not fallback

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    records = []; err = None
    t0 = time.perf_counter(); bi = 0
    try:
        for u in range(n_updates):
            mel, lens, text = pre[bi % len(pre)]
            opt.zero_grad(set_to_none=True)
            s = time.perf_counter()
            loss = model(mel, text, lens=lens)[0]
            if needs_loss_clone:
                loss = loss.clone()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            records.append(time.perf_counter() - s)
            bi += 1
    except Exception as e:
        err = repr(e)
    wall = time.perf_counter() - t0
    counters = {k: dict(v) for k, v in dynamo.utils.counters.items()}
    gb = counters.get("graph_break", {})
    stats = counters.get("stats", {})
    res = {
        "candidate": candidate, "setup_ok": True, "error": err, "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "correctness": correctness, "finite_loss": finite, "loss_close": loss_close,
        "grad_close": grad_close, "keys_clean": keys_clean, "fallback": fallback,
        "gate_loss_compiled": float(c_loss.detach()), "gate_loss_eager": float(e_loss),
        "unique_batch_shapes": len(shapes), "n_updates": n_updates,
        "first_update_s": records[0] if records else None,
        "median_update_s": statistics.median(records) if records else None,
        "p90_update_s": (statistics.quantiles(records, n=10)[8] if len(records) >= 10 else None),
        "mean_update_s": (sum(records) / len(records)) if records else None,
        "wall_s": wall,
        "max_memory_allocated_mb": (torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None),
        "max_memory_reserved_mb": (torch.cuda.max_memory_reserved() / 1024**2 if torch.cuda.is_available() else None),
        "graph_break_counters": gb, "unique_graphs": stats.get("unique_graphs"),
        "recompiles": stats.get("recompiles"), "all_counters": counters,
        "requires_loss_clone": needs_loss_clone, "compile_state": getattr(model, "training_compile_state", {}), "candidate_config": cfg_c,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{candidate}.json").write_text(json.dumps(res, indent=2, sort_keys=True, default=str))
    print("RESULT_JSON=" + json.dumps(res, sort_keys=True, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--candidate", default=None)
    ap.add_argument("--candidates", nargs="*", default=list(CANDIDATES))
    ap.add_argument("--updates", type=int, default=30)
    ap.add_argument("--shapes", type=int, default=64)
    ap.add_argument("--out", default=str(REPO / "benchmarks" / "logs" / "local_fullgraph_viability"))
    args = ap.parse_args()
    out = Path(args.out)
    if args.worker:
        worker(args.candidate, out, args.updates, args.shapes)
        return
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    me = str(Path(__file__).resolve())
    results = []
    for c in args.candidates:
        print(f"\n=== {c} ===", flush=True)
        env = dict(os.environ); env.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            p = subprocess.run([py, me, "--worker", "--candidate", c, "--out", str(out),
                                "--updates", str(args.updates), "--shapes", str(args.shapes)],
                               env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
        except subprocess.TimeoutExpired as e:
            print(f"TIMEOUT: {e}", flush=True); continue
        print(p.stdout[-2500:], flush=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT_JSON=")), None)
        if line:
            results.append(json.loads(line[len("RESULT_JSON="):]))
    (out / "summary.json").write_text(json.dumps(results, indent=2, sort_keys=True, default=str))
    cols = ["candidate", "correct", "graph_breaks", "unique_graphs", "recompiles", "first_s", "median_s", "wall_s", "vram_mb", "error"]
    print("\n" + " | ".join(cols))
    for r in results:
        gb = r.get("graph_break_counters", {})
        gb_n = sum(gb.values()) if isinstance(gb, dict) else (gb or 0)
        vram = r.get("max_memory_allocated_mb")
        row = [str(r.get("candidate")), str(r.get("correctness")), str(gb_n),
               str(r.get("unique_graphs")), str(r.get("recompiles")),
               f"{r['first_update_s']:.3f}" if r.get("first_update_s") else "-",
               f"{r['median_update_s']:.4f}" if r.get("median_update_s") else "-",
               f"{r['wall_s']:.2f}" if r.get("wall_s") else "-",
               f"{vram:.0f}" if vram else "-", (str(r.get("error"))[:30] if r.get("error") else "-")]
        print(" | ".join(row))


if __name__ == "__main__":
    main()
