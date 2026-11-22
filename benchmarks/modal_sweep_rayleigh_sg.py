"""
Modal Parallel Ablation: Stop-Gradient Rayleigh (`rayleigh_sg`) vs `rayleigh` vs `edges+pw`
===========================================================================================
Directly tests the mathematical fix for Rayleigh quotient anti-decay (norm/outlier inflation):
  E_D^{rayleigh_sg}(W) = sum(Delta^2) / (2 * sg(||W||_F^2) + eps)
whose backward gradient is pure scale-invariant Neumann Laplacian heat diffusion:
  grad_W = -Delta W / ||W||_F^2  (ZERO anti-decay, ZERO L2 shrinkage, ZERO FFN starvation).

Runs 5 parallel A10G containers (2,000 steps, seed=42, ~3 min wall-clock) comparing:
  1. rayleigh_sg_0.5
  2. rayleigh_sg_1.0
  3. rayleigh_sg_2.0
  4. rayleigh_sg_anneal2  (cooldown_ratio=0.15)
  5. edges_pw_30.0        (normalization="edges", layer_weighting="param_weighted", lam=30.0)
"""

import json
import math
import os
import random
import time
from typing import Any, Dict, List

import modal

app = modal.App("dreg-rayleigh-sg-ablation")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "datasets", "tokenizers>=0.13.0", "numpy")
    .add_local_python_source("dreg")
)

volume = modal.Volume.from_name("dreg-checkpoints")
CHECKPOINT_DIR = "/root/checkpoints"


@app.function(image=image, gpu="A10G", volumes={CHECKPOINT_DIR: volume}, timeout=3600)
def run_ablation(
    name: str,
    normalization: str,
    topo_lambda: float,
    anneal: bool,
    max_steps: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    import copy

    import numpy as np
    import torch
    import torch.optim as optim
    from datasets import load_dataset
    from tokenizers import Tokenizer

    from dreg import (
        Base3TritQuantizer,
        DirichletLoss,
        StepCooldownScheduler,
        TopographicConfig,
        TopographicTransformer,
        dct2d,
        dirichlet_energy_2d,
        idct2d,
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")

    tok = Tokenizer.from_file(os.path.join(CHECKPOINT_DIR, "tinystories_bpe_4096.json"))
    cfg = TopographicConfig(
        vocab_size=tok.get_vocab_size(),
        d_model=256,
        n_heads=4,
        n_layers=8,
        ffn_dim=1024,
        max_seq_len=256,
        topo_lambda=topo_lambda,
        topo_normalization=normalization,
        topo_layer_weighting="param_weighted",
    )
    model = TopographicTransformer(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    topo_loss_fn = DirichletLoss(
        weight_decay=topo_lambda,
        normalization=normalization,
        layer_weighting="param_weighted",
    )
    sched = (
        StepCooldownScheduler(topo_loss_fn, total_steps=max_steps, cooldown_ratio=0.15, min_weight_decay=0.0)
        if anneal
        else None
    )

    train_stream = load_dataset("roneneldan/TinyStories", split="train", streaming=True).shuffle(seed=seed, buffer_size=2048)
    val_stream = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    def gen(stream):
        buf = []
        for s in stream:
            buf.extend(tok.encode(s["text"]).ids)
            while len(buf) >= 32 * 257:
                c = buf[: 32 * 257]
                buf = buf[32 * 257 :]
                t = torch.tensor(c, dtype=torch.long).view(32, 257)
                yield t[:, :-1], t[:, 1:]

    val_iter = iter(gen(val_stream))
    val_batches = [next(val_iter) for _ in range(25)]

    def eval_ppl(net) -> float:
        net.eval()
        tl, tt = 0.0, 0
        with torch.no_grad():
            for x, y in val_batches:
                x, y = x.to(device), y.to(device)
                _, loss = net(x, y)
                tl += loss.item() * y.numel()
                tt += y.numel()
        net.train()
        return math.exp(min(tl / max(tt, 1), 20.0))

    train_iter = iter(gen(train_stream))
    t0 = time.time()
    model.train()
    for step in range(1, max_steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(gen(train_stream))
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, ce = model(x, y)
        if sched is not None:
            cur_lam = sched.step()
            loss = ce + topo_loss_fn(model) if cur_lam > 0 else ce
        elif topo_lambda > 0:
            loss = ce + topo_loss_fn(model)
        else:
            loss = ce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    train_s = time.time() - t0
    model.eval()
    fp32 = eval_ppl(model)

    qh = Base3TritQuantizer(0.15, 0.40, 1.00)
    attn_low, ffn_low, ray_list, crest_list, rms_list = [], [], [], [], []
    with torch.no_grad():
        for p in model.get_linear_projections():
            w = p.weight.detach()
            rms = w.pow(2).mean().sqrt().item()
            rms_list.append(rms)
            crest_list.append(w.abs().max().item() / (rms + 1e-12))
            ray_list.append(dirichlet_energy_2d(w, normalization="rayleigh").item())
            s = dct2d(w)
            rho = qh.compute_radial_grid(*w.shape, w.device)
            lp = ((s[rho <= 0.40] ** 2).sum() / (s ** 2).sum()).item() * 100.0
            (attn_low if w.shape[0] == w.shape[1] else ffn_low).append(lp)

    # Spatial INT4 sanity check (was 5057 PPL under anti-decay rayleigh!)
    m_i4 = copy.deepcopy(model)
    with torch.no_grad():
        for p in m_i4.get_linear_projections():
            w = p.weight.detach()
            sc = w.abs().max() / 7.0 + 1e-8
            p.weight.copy_(torch.clamp(torch.round(w / sc), -8, 7) * sc)
    sp_int4 = eval_ppl(m_i4)

    regimes = [
        ("Ultra-0.43b", (0.08, 0.22, 0.50)),
        ("Sub1-0.91b", (0.10, 0.30, 0.75)),
        ("Calib-1.64b", (0.15, 0.40, 1.00)),
        ("Mid-2.18b", (0.20, 0.55, 1.15)),
        ("HiFi-2.79b", (0.30, 0.70, 1.30)),
    ]
    pareto = {}
    for rn, (r0, r1, r2) in regimes:
        qm = copy.deepcopy(model)
        q = Base3TritQuantizer(r0, r1, r2, band2_mode="lloyd_max", band01_mode="mse_clip")
        with torch.no_grad():
            for p in qm.get_linear_projections():
                s = dct2d(p.weight.detach())
                pk = q.quantize_matrix(s)
                p.weight.copy_(idct2d(q.dequantize_matrix(pk, device=device)))
        pareto[rn] = round(eval_ppl(qm), 2)

    res = {
        "name": name,
        "normalization": normalization,
        "topo_lambda": topo_lambda,
        "anneal": anneal,
        "train_time_s": round(train_s, 1),
        "fp32_ppl": round(fp32, 2),
        "mean_rayleigh": round(float(np.mean(ray_list)), 4),
        "max_rms": round(float(np.max(rms_list)), 4),
        "max_w_crest": round(float(np.max(crest_list)), 2),
        "attn_low_freq_pct": round(float(np.mean(attn_low)), 2),
        "ffn_low_freq_pct": round(float(np.mean(ffn_low)), 2),
        "spatial_int4_ppl": round(sp_int4, 2),
        "pareto": pareto,
    }
    print(
        f"[+] {name:<20} | FP32={fp32:5.2f} | INT4={sp_int4:6.2f} | crest={res['max_w_crest']:4.1f} | "
        f"Attn={res['attn_low_freq_pct']:5.1f}% FFN={res['ffn_low_freq_pct']:5.1f}% | "
        f"0.43b={pareto['Ultra-0.43b']:6.2f} | 0.91b={pareto['Sub1-0.91b']:6.2f} | "
        f"1.64b={pareto['Calib-1.64b']:5.2f} | 2.18b={pareto['Mid-2.18b']:5.2f}"
    )
    return res


@app.local_entrypoint()
def main(steps: int = 2000, seed: int = 42):
    jobs = [
        ("std", "rayleigh_sg", 0.0, False, steps, seed),
        ("rayleigh_sg_0.5", "rayleigh_sg", 0.5, False, steps, seed),
        ("rayleigh_sg_1.0", "rayleigh_sg", 1.0, False, steps, seed),
        ("rayleigh_sg_2.0", "rayleigh_sg", 2.0, False, steps, seed),
        ("rayleigh_sg_anneal2", "rayleigh_sg", 2.0, True, steps, seed),
        ("edges_pw_30", "edges", 30.0, False, steps, seed),
    ]
    print(f"[*] Launching {len(jobs)} parallel rayleigh_sg ablation runs ({steps} steps, seed={seed})...")
    results: List[Dict[str, Any]] = list(run_ablation.starmap(jobs))

    print("\n" + "=" * 130)
    print(f" STOP-GRADIENT RAYLEIGH (`rayleigh_sg`) ABLATION ({steps} steps, seed={seed}, mse_clip + lloyd_max)")
    print("=" * 130)
    print(
        f"{'Variant':<20} | {'FP32':>5} | {'INT4':>6} | {'Crest':>5} | {'Attn':>6} | {'FFN':>6} | "
        f"{'0.43b':>7} | {'0.91b':>7} | {'1.64b':>6} | {'2.18b':>6} | {'2.79b':>6}"
    )
    print("-" * 130)
    for r in results:
        p = r["pareto"]
        print(
            f"{r['name']:<20} | {r['fp32_ppl']:>5.2f} | {r['spatial_int4_ppl']:>6.2f} | {r['max_w_crest']:>5.1f} | "
            f"{r['attn_low_freq_pct']:>5.1f}% | {r['ffn_low_freq_pct']:>5.1f}% | "
            f"{p['Ultra-0.43b']:>7.2f} | {p['Sub1-0.91b']:>7.2f} | {p['Calib-1.64b']:>6.2f} | "
            f"{p['Mid-2.18b']:>6.2f} | {p['HiFi-2.79b']:>6.2f}"
        )
    print("=" * 130)

    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", f"rayleigh_sg_ablation_{steps}s_s{seed}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved {out}")
