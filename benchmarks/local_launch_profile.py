#!/usr/bin/env python3
"""torch.profiler CUDA launch-count pass: eager vs fullgraph vs reduce-overhead.
Tiny DiT, real CFM code path, 5 steady updates profiled (skip first warmup/compile).
Reports launches/update and top kernels per candidate."""
from __future__ import annotations
import sys, json, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from f5_tts.model import CFM, DiT
from f5_tts.model.dataset import collate_fn
from f5_tts.model.utils import get_tokenizer
import random as _random

def build(device):
    cfg = dict(dim=96, depth=6, heads=4, ff_mult=2, text_dim=128, text_mask_padding=True,
               qk_norm=None, conv_layers=2, pe_attn_head=None, attn_backend="torch",
               attn_mask_enabled=False, checkpoint_activations=False)
    mel_kw = dict(n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=80,
                  target_sample_rate=24000, mel_spec_type="vocos")
    vcm, vs = get_tokenizer("", "byte")
    m = CFM(transformer=DiT(**cfg, text_num_embeds=vs, mel_dim=80), mel_spec_kwargs=mel_kw,
            vocab_char_map=vcm, audio_drop_prob=0.3, cond_drop_prob=0.2).to(device)
    return m, vcm, vs

def mkbatches(device, n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    import string as st
    chars = list(st.ascii_lowercase + " ")
    batches = []
    for i in range(n):
        bsz = 1 + int(torch.rand(1, generator=g).item() * 3)
        items = []
        for _ in range(bsz):
            L = max(8, 30 + int(torch.rand(1, generator=g).item() * 40))
            mel = torch.randn(80, L, generator=g) * 0.5
            tl = 4 + int(torch.rand(1, generator=g).item() * 12)
            txt = "".join(chars[int(torch.randint(0, len(chars), (1,), generator=g).item())] for _ in range(tl))
            items.append({"mel_spec": mel, "text": txt})
        c = collate_fn(items)
        mel = c["mel"].permute(0, 2, 1).to(device)
        lens = c["mel_lengths"].to(device)
        batches.append((mel, c["text"], lens))
    return batches

def run(name, device, n_profile=5):
    model, vcm, vs = build(device)
    opt = torch.optim.AdamW(model.parameters(), lr=7.5e-5, fused=torch.cuda.is_available())
    bs = mkbatches(device)
    needs_clone = name != "eager"
    if name == "fullgraph":
        model.compile_training_core(target="cfm_loss_core", backend="inductor", fullgraph=True, runtime_fallback=False)
    elif name == "reduce_overhead":
        model.compile_training_core(target="cfm_loss_core", backend="inductor", fullgraph=True, mode="reduce-overhead", runtime_fallback=False)
        ms = torch.compiler.cudagraph_mark_step_begin
        orig = model.forward
        def fwd(*a, **k):
            ms(); return orig(*a, **k)
        model.forward = fwd
    # warmup (compile) - run through all batches once
    for mel, txt, lens in bs:
        opt.zero_grad(set_to_none=True)
        _random.seed(0); torch.manual_seed(0)
        loss = model(mel, txt, lens=lens)[0]
        if needs_clone: loss = loss.clone()
        loss.backward(); opt.step()
        if torch.cuda.is_available(): torch.cuda.synchronize()
    # profiled steady updates
    launches = []
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=False) as prof:
        for i in range(n_profile):
            mel, txt, lens = bs[i % len(bs)]
            opt.zero_grad(set_to_none=True)
            _random.seed(0); torch.manual_seed(0)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            loss = model(mel, txt, lens=lens)[0]
            if needs_clone: loss = loss.clone()
            loss.backward(); opt.step()
            if torch.cuda.is_available(): torch.cuda.synchronize()
    events = prof.key_averages()
    total_cuda = sum(e.count for e in events)
    per_upd = total_cuda / n_profile
    # top kernels by cuda time
    cuda_events = sorted([e for e in events if e.count > 0], key=lambda e: e.self_device_time_total, reverse=True)[:8]
    top = [{"key": e.key[:40], "calls": e.count, "us_total": int(e.self_device_time_total)} for e in cuda_events]
    res = {"candidate": name, "launches_total": total_cuda, "launches_per_update": per_upd, "top_kernels": top}
    print(json.dumps(res, indent=2))
    return res

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    out = []
    for name in ["eager", "fullgraph", "reduce_overhead"]:
        print(f"\n=== {name} ===")
        out.append(run(name, device))
    Path("benchmarks/logs/local_fullgraph_viability/launch_profile.json").write_text(json.dumps(out, indent=2))
