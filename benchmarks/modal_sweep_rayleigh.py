"""
Modal Parallel Calibration Sweep: Rayleigh Scale-Invariant & Param-Weighted Dirichlet
=====================================================================================
Tests Phase 2 algorithmic upgrades on TinyStories 10M (A10G GPUs in parallel):
  1. Scale-invariant Rayleigh normalization: E_D(W) = sum(Delta^2) / (2 * ||W||_F^2)
  2. Parameter-weighted layer aggregation: eliminates 4x FFN gradient attenuation
  3. 1D Lloyd-Max MSE-optimal Band-2 Base-3 Trit quantization

Measures across lambdas (in parallel):
  - Dense FP32 Val PPL (25 validation batches = 204,800 tokens)
  - Separate Attention (256x256) vs FFN (256x1024) low-frequency DCT energy E(rho <= 0.40)
  - Quantized Val PPL across 4 bitrates: 0.430, 0.914 (sub-1.0 bpp!), 1.638, and 2.181 bpp
"""

import json
import math
import os
import random
import time
from typing import Any, Dict, List

import modal

app = modal.App("dreg-rayleigh-sweep")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "datasets",
        "tokenizers>=0.13.0",
        "numpy",
    )
    .add_local_python_source("dreg")
)

volume = modal.Volume.from_name("dreg-checkpoints")
CHECKPOINT_DIR = "/root/checkpoints"


@app.function(
    image=image,
    gpu="A10G",
    volumes={CHECKPOINT_DIR: volume},
    timeout=3600,
)
def train_and_eval_rayleigh(
    topo_lambda: float,
    max_steps: int = 2000,
    seed: int = 42,
    lr: float = 5e-4,
    batch_size: int = 32,
    seq_len: int = 256,
    save_checkpoint: bool = True,
) -> Dict[str, Any]:
    import copy

    import numpy as np
    import torch
    import torch.optim as optim
    from datasets import load_dataset
    from tokenizers import Tokenizer

    from dreg import (
        Base3TritQuantizer,
        TopographicConfig,
        TopographicTransformer,
        dct2d,
        dirichlet_energy_2d,
        idct2d,
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] [lambda={topo_lambda}, steps={max_steps}, seed={seed}] Device: {device}")

    tok_path = os.path.join(CHECKPOINT_DIR, "tinystories_bpe_4096.json")
    tokenizer = Tokenizer.from_file(tok_path)

    cfg = TopographicConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=256,
        n_heads=4,
        n_layers=8,
        ffn_dim=1024,
        max_seq_len=seq_len,
        topo_lambda=topo_lambda,
        topo_normalization="rayleigh",
        topo_layer_weighting="param_weighted",
    )
    model = TopographicTransformer(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_stream = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    val_stream = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    def data_generator(stream):
        token_buffer = []
        for sample in stream:
            enc = tokenizer.encode(sample["text"])
            token_buffer.extend(enc.ids)
            while len(token_buffer) >= batch_size * (seq_len + 1):
                chunk = token_buffer[: batch_size * (seq_len + 1)]
                token_buffer = token_buffer[batch_size * (seq_len + 1) :]
                t = torch.tensor(chunk, dtype=torch.long).view(batch_size, seq_len + 1)
                yield t[:, :-1], t[:, 1:]

    val_batches = []
    val_iter = iter(data_generator(val_stream))
    for _ in range(25):
        val_batches.append(next(val_iter))

    def eval_ppl(net) -> float:
        net.eval()
        tot_loss, tot_tok = 0.0, 0
        with torch.no_grad():
            for x, y in val_batches:
                x, y = x.to(device), y.to(device)
                _, loss = net(x, y)
                tot_loss += loss.item() * y.numel()
                tot_tok += y.numel()
        net.train()
        return math.exp(min(tot_loss / max(tot_tok, 1), 20.0))

    train_iter = iter(data_generator(train_stream))
    t0 = time.time()
    model.train()
    eval_every = max(100, max_steps // 10)
    for step in range(1, max_steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(data_generator(train_stream))
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, ce = model(x, y)
        loss = ce + model.topographic_loss() if topo_lambda > 0.0 else ce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % eval_every == 0 or step == max_steps:
            vp = eval_ppl(model)
            print(f"  [lam={topo_lambda}] step {step:5d}/{max_steps} | CE={ce.item():.4f} | Val PPL={vp:.2f}")

    train_s = time.time() - t0
    fp32_ppl = eval_ppl(model)

    # Measure Attention vs FFN spectral compaction (rho <= 0.40) and roughness
    q_helper = Base3TritQuantizer(0.15, 0.40, 1.00)
    attn_low, ffn_low, ed_edges_list, rayleigh_list = [], [], [], []
    with torch.no_grad():
        for p in model.get_linear_projections():
            w = p.weight.detach()
            ed_edges_list.append(dirichlet_energy_2d(w, normalization="edges").item())
            rayleigh_list.append(dirichlet_energy_2d(w, normalization="rayleigh").item())
            s = dct2d(w)
            rho = q_helper.compute_radial_grid(*w.shape, w.device)
            low_pct = ((s[rho <= 0.40] ** 2).sum() / (s ** 2).sum()).item() * 100.0
            if w.shape[0] == w.shape[1]:
                attn_low.append(low_pct)
            else:
                ffn_low.append(low_pct)

    # Evaluate across 5 quantization bitrates (including sub-1.0 bpp!)
    regimes = [
        ("Ultra-0.43b", (0.08, 0.22, 0.50)),
        ("Sub1-0.91b",  (0.10, 0.30, 0.75)),
        ("Calib-1.64b", (0.15, 0.40, 1.00)),
        ("Mid-2.18b",   (0.20, 0.55, 1.15)),
        ("HiFi-2.79b",  (0.30, 0.70, 1.30)),
    ]
    quant_results = {}
    for rname, (r0, r1, r2) in regimes:
        qm = copy.deepcopy(model)
        quantizer = Base3TritQuantizer(r0=r0, r1=r1, r2=r2, band2_mode="lloyd_max")
        tot_bits, tot_w = 0.0, 0
        with torch.no_grad():
            for p in qm.get_linear_projections():
                s = dct2d(p.weight.detach())
                pk = quantizer.quantize_matrix(s)
                tot_bits += pk["bpp"] * s.numel()
                tot_w += s.numel()
                p.weight.copy_(idct2d(quantizer.dequantize_matrix(pk, device=device)))
        bpp = tot_bits / max(tot_w, 1)
        qppl = eval_ppl(qm)
        quant_results[rname] = {"bpp": round(bpp, 3), "ppl": round(qppl, 2)}

    if save_checkpoint:
        tag = f"rayleigh_pw_lam{topo_lambda}_s{seed}_{max_steps}s" if topo_lambda > 0 else f"standard_s{seed}_{max_steps}s"
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"tinystories_10M_{tag}.pt")
        torch.save(model.state_dict(), ckpt_path)
        volume.commit()

    res = {
        "topo_lambda": topo_lambda,
        "max_steps": max_steps,
        "seed": seed,
        "train_time_s": round(train_s, 1),
        "fp32_ppl": round(fp32_ppl, 2),
        "mean_ed_edges": round(float(np.mean(ed_edges_list)), 6),
        "mean_rayleigh": round(float(np.mean(rayleigh_list)), 4),
        "attn_low_freq_pct": round(float(np.mean(attn_low)), 2),
        "ffn_low_freq_pct": round(float(np.mean(ffn_low)), 2),
        "quantized": quant_results,
    }
    print(f"[+] Done [lam={topo_lambda}]: {json.dumps(res)}")
    return res


@app.local_entrypoint()
def main(steps: int = 2000, seed: int = 42, lambdas: str = "0.0,0.25,0.5,1.0,2.0"):
    lam_list: List[float] = [float(x.strip()) for x in lambdas.split(",")]
    print(f"[*] Launching parallel Rayleigh+param_weighted sweep on Modal A10G: lambdas={lam_list}, steps={steps}, seed={seed}")

    results = list(
        train_and_eval_rayleigh.map(
            lam_list,
            kwargs={"max_steps": steps, "seed": seed},
        )
    )

    print("\n" + "=" * 112)
    print(f" RAYLEIGH + PARAM-WEIGHTED + LLOYD-MAX SWEEP (TinyStories 10M, {steps} steps, seed={seed})")
    print("=" * 112)
    print(
        f"{'Lambda':>7} | {'FP32':>6} | {'Rayleigh':>8} | {'Attn E<.4':>9} | {'FFN E<.4':>9} | "
        f"{'0.43b PPL':>9} | {'0.91b PPL':>9} | {'1.64b PPL':>9} | {'2.18b PPL':>9} | {'2.79b PPL':>9}"
    )
    print("-" * 112)
    for r in results:
        q = r["quantized"]
        print(
            f"{r['topo_lambda']:>7.2f} | {r['fp32_ppl']:>6.2f} | {r['mean_rayleigh']:>8.4f} | "
            f"{r['attn_low_freq_pct']:>8.1f}% | {r['ffn_low_freq_pct']:>8.1f}% | "
            f"{q['Ultra-0.43b']['ppl']:>9.2f} | {q['Sub1-0.91b']['ppl']:>9.2f} | "
            f"{q['Calib-1.64b']['ppl']:>9.2f} | {q['Mid-2.18b']['ppl']:>9.2f} | {q['HiFi-2.79b']['ppl']:>9.2f}"
        )
    print("=" * 112)

    out_local = os.path.join("results", f"rayleigh_sweep_{steps}s_s{seed}.json")
    os.makedirs("results", exist_ok=True)
    with open(out_local, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved local JSON report to {out_local}")
