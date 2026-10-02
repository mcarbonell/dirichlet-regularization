"""
Empirical Pareto Frontier Evaluation on TinyStories 10M (Modal GPU)
===================================================================
Evaluates the Rate-Distortion tradeoff (bpp vs Validation Perplexity)
on the pre-trained 10,000-step TinyStories models stored in Modal Volume:
  - Standard Model:     tinystories_10M_standard_standard_10000s.pt
  - Topographic Model:  tinystories_10M_topographic_lambda30.0_10000s.pt

Sweeps across 6 radial cutoff regimes:
  1. Ultra-Aggressive (~0.47 bpp / 68x)
  2. Aggressive       (~0.68 bpp / 47x)
  3. Calibrated TritQ (~0.95 bpp / 34x)
  4. Mid-Band         (~1.25 bpp / 25x)
  5. High-Fidelity    (~1.65 bpp / 19x)
  6. Conservative     (~2.10 bpp / 15x)

Outputs:
  - Formatted Markdown / LaTeX comparative table
  - JSON file saved to volume for paper plotting
"""

import copy
import json
import math
import os
from typing import Any, Dict, List

import modal

app = modal.App("dreg-eval-pareto")

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
    timeout=600,
)
def evaluate_pareto_frontier() -> List[Dict[str, Any]]:
    import torch
    from datasets import load_dataset
    from tokenizers import Tokenizer

    from dreg.model import TopographicConfig, TopographicTransformer
    from dreg.quantization import Base3TritQuantizer
    from dreg.spectral import dct2d, idct2d
    from dreg.topology import dirichlet_energy_2d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Pareto Frontier Benchmark on: {device} ({torch.cuda.get_device_name(0)})")

    # 1. Load Tokenizer & Validation Dataset
    tok_path = os.path.join(CHECKPOINT_DIR, "tinystories_bpe_4096.json")
    tokenizer = Tokenizer.from_file(tok_path)
    val_stream = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    def data_generator(stream, batch_sz=32, seq_l=256):
        token_buffer = []
        for sample in stream:
            enc = tokenizer.encode(sample["text"])
            token_buffer.extend(enc.ids)
            while len(token_buffer) >= batch_sz * (seq_l + 1):
                chunk = token_buffer[: batch_sz * (seq_l + 1)]
                token_buffer = token_buffer[batch_sz * (seq_l + 1) :]
                t = torch.tensor(chunk, dtype=torch.long).view(batch_sz, seq_l + 1)
                yield t[:, :-1], t[:, 1:]

    val_batches = []
    val_iter = iter(data_generator(val_stream, batch_sz=32, seq_l=256))
    for _ in range(25):
        val_batches.append(next(val_iter))
    print(f"[+] Loaded {len(val_batches)} validation batches (204,800 tokens)")

    # 2. Load 10M Models
    cfg = TopographicConfig(
        vocab_size=4096,
        d_model=256,
        n_heads=4,
        n_layers=8,
        ffn_dim=1024,
        max_seq_len=256,
        topo_lambda=0.0,
    )

    std_model = TopographicTransformer(cfg).to(device)
    topo_model = TopographicTransformer(cfg).to(device)

    std_ckpt_path = os.path.join(CHECKPOINT_DIR, "tinystories_10M_standard_standard_10000s.pt")
    topo_ckpt_path = os.path.join(CHECKPOINT_DIR, "tinystories_10M_topographic_lambda30.0_10000s.pt")

    std_model.load_state_dict(torch.load(std_ckpt_path, map_location=device))
    topo_model.load_state_dict(torch.load(topo_ckpt_path, map_location=device))
    std_model.eval()
    topo_model.eval()

    def eval_ppl(m: TopographicTransformer) -> float:
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for x, y in val_batches:
                x, y = x.to(device), y.to(device)
                _, loss = m(x, y)
                total_loss += loss.item() * y.numel()
                total_tokens += y.numel()
        avg_loss = total_loss / max(total_tokens, 1)
        return math.exp(min(avg_loss, 20.0))

    # Base FP32 Perplexity and Roughness
    std_fp32_ppl = eval_ppl(std_model)
    topo_fp32_ppl = eval_ppl(topo_model)

    ed_std = float(torch.stack([dirichlet_energy_2d(p.weight, normalization="edges") for p in std_model.get_linear_projections()]).mean().item())
    ed_topo = float(torch.stack([dirichlet_energy_2d(p.weight, normalization="edges") for p in topo_model.get_linear_projections()]).mean().item())

    print("\n" + "=" * 90)
    print(" BASELINE MODEL METRICS (FP32)")
    print("=" * 90)
    print(f"[*] Standard Model:     FP32 PPL = {std_fp32_ppl:.2f} | Mean E_D = {ed_std:.6f}")
    print(f"[*] Topographic Model:  FP32 PPL = {topo_fp32_ppl:.2f} | Mean E_D = {ed_topo:.6f} (Roughness reduction: -{(1 - ed_topo / ed_std) * 100:.1f}%)")
    print("=" * 90)

    # 3. Radial threshold configurations across rate spectrum
    regimes = [
        {"name": "Ultra-Aggressive", "r0": 0.08, "r1": 0.22, "r2": 0.50},
        {"name": "Aggressive",       "r0": 0.10, "r1": 0.30, "r2": 0.75},
        {"name": "Calibrated TritQ", "r0": 0.15, "r1": 0.40, "r2": 1.00},
        {"name": "Mid-Band",         "r0": 0.20, "r1": 0.55, "r2": 1.15},
        {"name": "High-Fidelity",    "r0": 0.30, "r1": 0.70, "r2": 1.30},
        {"name": "Conservative",     "r0": 0.45, "r1": 0.90, "r2": 1.41},
    ]

    def quantize_and_measure(base_model, r0, r1, r2):
        m = copy.deepcopy(base_model)
        quantizer = Base3TritQuantizer(r0=r0, r1=r1, r2=r2)
        total_comp_bytes = 0
        total_orig_bytes = 0

        with torch.no_grad():
            for p in m.get_linear_projections():
                w = p.weight.detach()
                total_orig_bytes += w.numel() * 4
                s = dct2d(w)
                packed = quantizer.quantize_matrix(s)
                comp_bytes = (
                    packed["band0"]["data"].nbytes
                    + packed["band1"]["data"].nbytes
                    + packed["band2"]["data"].nbytes
                )
                total_comp_bytes += comp_bytes
                rec_s = quantizer.dequantize_matrix(packed, device=device)
                p.weight.copy_(idct2d(rec_s))

        bpp = (total_comp_bytes * 8.0) / max(total_orig_bytes / 4, 1)
        ratio = total_orig_bytes / max(total_comp_bytes, 1)
        ppl = eval_ppl(m)
        return ppl, bpp, ratio

    results = []
    print(f"\n{'Regime':<18} | {'Radii (r0,r1,r2)':<16} | {'bpp':>6} | {'Ratio':>6} | {'Std PPL':>8} | {'Topo PPL':>8} | {'Gain':>8}")
    print("-" * 85)

    for reg in regimes:
        r0, r1, r2 = reg["r0"], reg["r1"], reg["r2"]
        radii_str = f"({r0:.2f},{r1:.2f},{r2:.2f})"

        ppl_std, bpp_std, ratio_std = quantize_and_measure(std_model, r0, r1, r2)
        ppl_topo, bpp_topo, ratio_topo = quantize_and_measure(topo_model, r0, r1, r2)
        gain = ppl_std - ppl_topo

        row = {
            "regime": reg["name"],
            "r0": r0,
            "r1": r1,
            "r2": r2,
            "bpp": round(bpp_topo, 3),
            "ratio": round(ratio_topo, 1),
            "std_ppl": round(ppl_std, 2),
            "topo_ppl": round(ppl_topo, 2),
            "gain": round(gain, 2),
            "delta_std": round(ppl_std - std_fp32_ppl, 2),
            "delta_topo": round(ppl_topo - topo_fp32_ppl, 2),
        }
        results.append(row)

        print(f"{reg['name']:<18} | {radii_str:<16} | {bpp_topo:>5.3f} | {ratio_topo:>5.1f}x | {ppl_std:>8.2f} | {ppl_topo:>8.2f} | {gain:>+8.2f}")

    print("-" * 85)

    # Save results to modal volume
    out_path = os.path.join(CHECKPOINT_DIR, "pareto_curve_tinystories_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "std_fp32_ppl": std_fp32_ppl,
            "topo_fp32_ppl": topo_fp32_ppl,
            "ed_std": ed_std,
            "ed_topo": ed_topo,
            "sweep": results,
        }, f, indent=2)
    volume.commit()
    print(f"[+] Saved Pareto sweep results to {out_path}")

    return results


@app.local_entrypoint()
def main():
    results = evaluate_pareto_frontier.remote()
    print("\n[+] Pareto Evaluation Finished!")
    print(json.dumps(results, indent=2))
