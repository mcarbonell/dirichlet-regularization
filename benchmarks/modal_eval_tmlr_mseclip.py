"""
Re-evaluate all 12 converged 10,000-step TMLR checkpoints (4 conditions x 3 seeds)
already saved in Modal volume `dreg-checkpoints` with the upgraded Base-3 TritQ
quantizer (band01_mode="mse_clip" + band2_mode="lloyd_max").

Produces:
  - Complete N=3 multi-seed mean +/- std 6-point Pareto frontier (0.430 - 3.774 bpp)
  - Direct 3-step Quantizer Ablation at 1.638 bpp:
      (1) minmax + legacy Band-2
      (2) minmax + lloyd_max Band-2
      (3) mse_clip Band-0/1 + lloyd_max Band-2  (zero extra bits, .tritq v1 compatible)
  - End-to-end on-disk .tritq round-trip PPL verification (seed 42)
  - Saves results/tmlr_10000s_mseclip_multiseed_results.json
"""

import dataclasses
import json
import math
import os
from typing import Any, Dict, List

import modal

app = modal.App("dreg-tmlr-eval-mseclip")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "datasets", "tokenizers>=0.13.0", "numpy")
    .add_local_python_source("dreg")
)

volume = modal.Volume.from_name("dreg-checkpoints")
CHECKPOINT_DIR = "/root/checkpoints"


@app.function(image=image, gpu="A10G", volumes={CHECKPOINT_DIR: volume}, timeout=1200)
def eval_all_saved_checkpoints() -> Dict[str, Any]:
    import copy

    import numpy as np
    import torch
    from datasets import load_dataset
    from tokenizers import Tokenizer

    from dreg import (
        Base3TritQuantizer,
        TopographicConfig,
        TopographicTransformer,
        TritQFormat,
        dct2d,
        dirichlet_energy_2d,
        idct2d,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(os.path.join(CHECKPOINT_DIR, "tinystories_bpe_4096.json"))
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
        return math.exp(min(tl / max(tt, 1), 20.0))

    def eval_quant(model, quantizer: Base3TritQuantizer):
        qm = copy.deepcopy(model)
        tot_bits, tot_w = 0.0, 0
        with torch.no_grad():
            for p in qm.get_linear_projections():
                s = dct2d(p.weight.detach())
                pk = quantizer.quantize_matrix(s)
                tot_bits += pk["bpp"] * s.numel()
                tot_w += s.numel()
                p.weight.copy_(idct2d(quantizer.dequantize_matrix(pk, device=device)))
        return tot_bits / max(tot_w, 1), eval_ppl(qm)

    cfg = TopographicConfig(
        vocab_size=tok.get_vocab_size(),
        d_model=256,
        n_heads=4,
        n_layers=8,
        ffn_dim=1024,
        max_seq_len=256,
    )

    conditions = ["std", "rayleigh_lam1", "rayleigh_lam2", "rayleigh_anneal2"]
    seeds = [42, 100, 2024]
    regimes = [
        ("Ultra-0.43b", (0.08, 0.22, 0.50)),
        ("Sub1-0.91b", (0.10, 0.30, 0.75)),
        ("Calib-1.64b", (0.15, 0.40, 1.00)),
        ("Mid-2.18b", (0.20, 0.55, 1.15)),
        ("HiFi-2.79b", (0.30, 0.70, 1.30)),
        ("Cons-3.77b", (0.45, 0.90, 1.41)),
    ]

    runs: List[Dict[str, Any]] = []
    for cond in conditions:
        for seed in seeds:
            ckpt = os.path.join(CHECKPOINT_DIR, f"tinystories_10M_tmlr_{cond}_s{seed}_10000s.pt")
            model = TopographicTransformer(cfg).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()
            fp32 = eval_ppl(model)

            # Compaction & crest diagnostics
            qh = Base3TritQuantizer(0.15, 0.40, 1.00)
            attn_low, ffn_low, ray_list, crest_list = [], [], [], []
            with torch.no_grad():
                for p in model.get_linear_projections():
                    w = p.weight.detach()
                    rms = w.pow(2).mean().sqrt().item()
                    crest_list.append(w.abs().max().item() / (rms + 1e-12))
                    ray_list.append(dirichlet_energy_2d(w, normalization="rayleigh").item())
                    s = dct2d(w)
                    rho = qh.compute_radial_grid(*w.shape, w.device)
                    lp = ((s[rho <= 0.40] ** 2).sum() / (s ** 2).sum()).item() * 100.0
                    (attn_low if w.shape[0] == w.shape[1] else ffn_low).append(lp)

            # 6-point Pareto with mse_clip + lloyd_max
            pareto = {}
            for rname, (r0, r1, r2) in regimes:
                q = Base3TritQuantizer(r0, r1, r2, band2_mode="lloyd_max", band01_mode="mse_clip")
                bpp, ppl = eval_quant(model, q)
                pareto[rname] = {"bpp": round(bpp, 3), "ppl": round(ppl, 3)}

            # 3-step Quantizer Ablation at 1.638 bpp
            _, ppl_legacy = eval_quant(model, Base3TritQuantizer(0.15, 0.40, 1.00, band2_mode="legacy", band01_mode="minmax"))
            _, ppl_lloyd = eval_quant(model, Base3TritQuantizer(0.15, 0.40, 1.00, band2_mode="lloyd_max", band01_mode="minmax"))
            ppl_full = pareto["Calib-1.64b"]["ppl"]

            # True .tritq round-trip verification (seed 42)
            rt_ppl = None
            rt_bytes = None
            if seed == 42:
                tmp_path = f"/tmp/{cond}_s42_mseclip.tritq"
                q_cal = Base3TritQuantizer(0.15, 0.40, 1.00, band2_mode="lloyd_max", band01_mode="mse_clip")
                TritQFormat.save(tmp_path, model.state_dict(), dataclasses.asdict(cfg), quantizer=q_cal)
                rt_bytes = os.path.getsize(tmp_path)
                _, rec_w = TritQFormat.load(tmp_path, device=device)
                m_rt = TopographicTransformer(cfg).to(device)
                m_rt.load_state_dict(rec_w)
                rt_ppl = round(eval_ppl(m_rt), 3)

            row = {
                "condition": cond,
                "seed": seed,
                "fp32_ppl": round(fp32, 3),
                "mean_rayleigh": round(float(np.mean(ray_list)), 4),
                "max_w_crest": round(float(np.max(crest_list)), 2),
                "attn_low_freq_pct": round(float(np.mean(attn_low)), 2),
                "ffn_low_freq_pct": round(float(np.mean(ffn_low)), 2),
                "pareto": pareto,
                "ablation_1.64b": {
                    "minmax_legacy": round(ppl_legacy, 3),
                    "minmax_lloydmax": round(ppl_lloyd, 3),
                    "mseclip_lloydmax": round(ppl_full, 3),
                },
                "tritq_roundtrip_ppl": rt_ppl,
                "tritq_disk_bytes": rt_bytes,
            }
            runs.append(row)
            print(
                f"  [{cond:<16} s={seed}] FP32={fp32:.2f} | "
                f"0.43b={pareto['Ultra-0.43b']['ppl']:7.2f} | "
                f"0.91b={pareto['Sub1-0.91b']['ppl']:6.2f} | "
                f"1.64b={ppl_full:5.2f} (leg={ppl_legacy:6.2f}, lm={ppl_lloyd:5.2f}) | "
                f"2.18b={pareto['Mid-2.18b']['ppl']:5.2f} | "
                f"2.79b={pareto['HiFi-2.79b']['ppl']:5.2f}"
            )

    summary = {}
    for cond in conditions:
        cr = [r for r in runs if r["condition"] == cond]

        def _agg(vals):
            return {"mean": round(float(np.mean(vals)), 3), "std": round(float(np.std(vals, ddof=1)), 3)}

        summary[cond] = {
            "fp32_ppl": _agg([r["fp32_ppl"] for r in cr]),
            "attn_low_freq_pct": _agg([r["attn_low_freq_pct"] for r in cr]),
            "ffn_low_freq_pct": _agg([r["ffn_low_freq_pct"] for r in cr]),
            "max_w_crest": _agg([r["max_w_crest"] for r in cr]),
            "pareto": {
                rn: {"bpp": cr[0]["pareto"][rn]["bpp"], **_agg([r["pareto"][rn]["ppl"] for r in cr])}
                for rn in cr[0]["pareto"]
            },
            "ablation_1.64b": {
                k: _agg([r["ablation_1.64b"][k] for r in cr])
                for k in cr[0]["ablation_1.64b"]
            },
        }
    return {"summary": summary, "runs": runs}


@app.local_entrypoint()
def main():
    res = eval_all_saved_checkpoints.remote()
    s = res["summary"]
    print("\n" + "=" * 126)
    print(" TMLR 10k MULTI-SEED PARETO WITH MSE-CLIP + LLOYD-MAX (Zero Extra Bits, N=3 seeds)")
    print("=" * 126)
    print(
        f"{'Condition':<18} | {'FP32':>11} | {'0.43b PPL':>15} | {'0.91b PPL':>13} | "
        f"{'1.64b PPL':>13} | {'2.18b PPL':>13} | {'2.79b PPL':>11} | {'3.77b PPL':>11}"
    )
    print("-" * 126)
    for cond, d in s.items():
        p = d["pareto"]
        print(
            f"{cond:<18} | "
            f"{d['fp32_ppl']['mean']:>5.2f}+/-{d['fp32_ppl']['std']:<4.2f} | "
            f"{p['Ultra-0.43b']['mean']:>7.1f}+/-{p['Ultra-0.43b']['std']:<5.1f} | "
            f"{p['Sub1-0.91b']['mean']:>6.2f}+/-{p['Sub1-0.91b']['std']:<5.2f} | "
            f"{p['Calib-1.64b']['mean']:>6.2f}+/-{p['Calib-1.64b']['std']:<4.2f} | "
            f"{p['Mid-2.18b']['mean']:>6.2f}+/-{p['Mid-2.18b']['std']:<4.2f} | "
            f"{p['HiFi-2.79b']['mean']:>5.2f}+/-{p['HiFi-2.79b']['std']:<4.2f} | "
            f"{p['Cons-3.77b']['mean']:>5.2f}+/-{p['Cons-3.77b']['std']:<4.2f}"
        )
    print("=" * 126)

    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", "tmlr_10000s_mseclip_multiseed_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"[+] Saved {out}")
