"""
Pareto Quantization Curve Benchmark (bpp vs PPL)
================================================
Generates the complete Pareto frontier of rate-distortion tradeoff:
evaluates bits per parameter (bpp) vs Validation Perplexity across
a systematic sweep of radial spectral thresholds for both Topographic
(dreg) and Standard models.

Covers:
  - Ultra-low bitrates (~0.47 bpp / 68x) up to ~2.20 bpp
  - Exact bpp calculation for each radial threshold setting
  - Generates comparative tabular data and JSON report
  - Optionally plots the empirical Pareto curves to PNG
"""

import argparse
import copy
import json
import math
import os
from typing import Any, Dict, List, Tuple

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dreg import (
    Base3TritQuantizer,
    TopographicConfig,
    TopographicTransformer,
    dct2d,
    dirichlet_energy_2d,
    idct2d,
)


def create_synthetic_corpus(vocab_size: int, seq_len: int, num_samples: int, seed: int = 42) -> torch.Tensor:
    """Generates synthetic sequential data with local Markov dependencies."""
    rng = np.random.RandomState(seed)
    tokens = np.zeros((num_samples, seq_len), dtype=np.int64)
    tokens[:, 0] = rng.randint(0, vocab_size, size=num_samples)
    for t in range(1, seq_len):
        prob = rng.rand(num_samples)
        structured = (tokens[:, t - 1] * 7 + 13) % vocab_size
        random_tok = rng.randint(0, vocab_size, size=num_samples)
        tokens[:, t] = np.where(prob > 0.35, structured, random_tok)
    return torch.tensor(tokens, dtype=torch.long)


def evaluate_ppl(
    model: TopographicTransformer,
    data: torch.Tensor,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Evaluates cross-entropy loss and returns perplexity over dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size].to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            _, loss = model(inputs, targets)
            num_tok = targets.numel()
            total_loss += loss.item() * num_tok
            total_tokens += num_tok

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(min(avg_loss, 20.0))


def quantize_model_with_radii(
    model: TopographicTransformer,
    r0: float,
    r1: float,
    r2: float,
) -> Tuple[TopographicTransformer, float, float]:
    """
    Quantizes all linear projections in model using custom radial thresholds.
    Returns: (quantized_model, measured_bpp, compression_ratio)
    """
    q_model = copy.deepcopy(model)
    quantizer = Base3TritQuantizer(r0=r0, r1=r1, r2=r2)
    total_orig_bytes = 0
    total_comp_bytes = 0

    for m in q_model.get_linear_projections():
        orig_w = m.weight.data
        total_orig_bytes += orig_w.numel() * 4  # FP32 bytes

        spec = dct2d(orig_w)
        packed = quantizer.quantize_matrix(spec)
        rec_spec = quantizer.dequantize_matrix(packed, device=orig_w.device)
        rec_w = idct2d(rec_spec)
        m.weight.data.copy_(rec_w)

        comp_bytes = (
            packed["band0"]["data"].nbytes
            + packed["band1"]["data"].nbytes
            + packed["band2"]["data"].nbytes
        )
        total_comp_bytes += comp_bytes

    bpp = (total_comp_bytes * 8.0) / max(total_orig_bytes / 4, 1)
    ratio = total_orig_bytes / max(total_comp_bytes, 1)
    return q_model, bpp, ratio


def train_quick_pair(
    steps: int = 150,
    topo_lambda: float = 15.0,
    vocab_size: int = 256,
    seq_len: int = 64,
    device: torch.device = torch.device("cpu"),
) -> Tuple[TopographicTransformer, TopographicTransformer, torch.Tensor]:
    """Trains a standard model and a topographic model for Pareto curve evaluation."""
    torch.manual_seed(42)
    train_data = create_synthetic_corpus(vocab_size, seq_len, 1200, seed=42)
    val_data = create_synthetic_corpus(vocab_size, seq_len, 300, seed=999)

    cfg = TopographicConfig(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        ffn_dim=256,
        max_seq_len=seq_len,
        topo_lambda=0.0,
    )

    # 1. Train Standard Model
    print(f"[*] Training Standard Model ({steps} steps)...")
    model_std = TopographicTransformer(cfg).to(device)
    opt_std = torch.optim.AdamW(model_std.parameters(), lr=1e-3, weight_decay=1e-4)
    model_std.train()
    idx = 0
    for _ in range(steps):
        if idx + 32 > len(train_data):
            idx = 0
        batch = train_data[idx : idx + 32].to(device)
        idx += 32
        opt_std.zero_grad()
        _, loss = model_std(batch[:, :-1], batch[:, 1:])
        loss.backward()
        opt_std.step()

    # 2. Train Topographic Model
    print(f"[*] Training Topographic Model (lambda={topo_lambda}, {steps} steps)...")
    torch.manual_seed(42)
    model_topo = TopographicTransformer(cfg).to(device)
    opt_topo = torch.optim.AdamW(model_topo.parameters(), lr=1e-3, weight_decay=1e-4)
    model_topo.train()
    idx = 0
    for _ in range(steps):
        if idx + 32 > len(train_data):
            idx = 0
        batch = train_data[idx : idx + 32].to(device)
        idx += 32
        opt_topo.zero_grad()
        _, ce = model_topo(batch[:, :-1], batch[:, 1:])
        linears = model_topo.get_linear_projections()
        topo_penalty = topo_lambda * torch.stack(
            [dirichlet_energy_2d(m.weight, normalization="edges") for m in linears]
        ).mean()
        loss = ce + topo_penalty
        loss.backward()
        opt_topo.step()

    return model_std, model_topo, val_data


def main():
    parser = argparse.ArgumentParser(description="Full Pareto Curve (bpp vs PPL) Benchmark")
    parser.add_argument("--steps", type=int, default=150, help="Training steps if training from scratch")
    parser.add_argument("--topo-lambda", type=float, default=15.0, help="Dirichlet regularization strength")
    parser.add_argument("--plot-png", type=str, default="pareto_curve.png", help="Path to save Pareto plot")
    parser.add_argument("--output-json", type=str, default="pareto_curve_results.json", help="Path to save results JSON")
    args = parser.parse_args()

    print("=" * 85)
    print(" EMPIRICAL PARETO CURVE SWEEP: COMPRESSION RATE (BPP) VS PERPLEXITY (PPL)")
    print("=" * 85)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")

    # Train or load reference pair
    model_std, model_topo, val_data = train_quick_pair(
        steps=args.steps,
        topo_lambda=args.topo_lambda,
        device=device,
    )

    std_fp32_ppl = evaluate_ppl(model_std, val_data, device=device)
    topo_fp32_ppl = evaluate_ppl(model_topo, val_data, device=device)

    ed_std = float(
        np.mean([dirichlet_energy_2d(m.weight, normalization="edges").item() for m in model_std.get_linear_projections()])
    )
    ed_topo = float(
        np.mean([dirichlet_energy_2d(m.weight, normalization="edges").item() for m in model_topo.get_linear_projections()])
    )

    print("\n--- Baseline Dense Models (FP32) ---")
    print(f"[*] Standard Baseline:   FP32 PPL = {std_fp32_ppl:.2f} | E_D = {ed_std:.6f}")
    print(f"[*] Topographic (Ours):  FP32 PPL = {topo_fp32_ppl:.2f} | E_D = {ed_topo:.6f} (-{(1-ed_topo/ed_std)*100:.1f}%)")

    # Define radial threshold sweep configurations
    configs = [
        {"name": "Ultra-Extreme (68x)", "r0": 0.04, "r1": 0.10, "r2": 0.25},
        {"name": "Aggressive (48x)", "r0": 0.05, "r1": 0.14, "r2": 0.32},
        {"name": "Standard (.tritq)", "r0": 0.06, "r1": 0.20, "r2": 0.45},
        {"name": "Mid-Compression", "r0": 0.08, "r1": 0.25, "r2": 0.55},
        {"name": "High-Fidelity", "r0": 0.10, "r1": 0.32, "r2": 0.68},
        {"name": "Conservative", "r0": 0.15, "r1": 0.45, "r2": 0.85},
    ]

    print("\n--- Sweeping Rate-Distortion Frontier across 6 Quantization Regimes ---")
    curve_data: List[Dict[str, Any]] = []

    for cfg in configs:
        r0, r1, r2 = cfg["r0"], cfg["r1"], cfg["r2"]
        print(f"  > Evaluating {cfg['name']:<22} (r0={r0:.2f}, r1={r1:.2f}, r2={r2:.2f})...", end="", flush=True)

        # Quantize Standard
        q_std, bpp_std, ratio_std = quantize_model_with_radii(model_std, r0, r1, r2)
        ppl_std = evaluate_ppl(q_std, val_data, device=device)

        # Quantize Topographic
        q_topo, bpp_topo, ratio_topo = quantize_model_with_radii(model_topo, r0, r1, r2)
        ppl_topo = evaluate_ppl(q_topo, val_data, device=device)

        gain = ppl_std - ppl_topo
        print(f" Rate: {bpp_topo:.3f} bpp ({ratio_topo:.1f}x) | Std PPL: {ppl_std:.2f} | Topo PPL: {ppl_topo:.2f} (Advantage: -{gain:.2f})")

        curve_data.append({
            "regime": cfg["name"],
            "r0": r0,
            "r1": r1,
            "r2": r2,
            "bpp_std": bpp_std,
            "bpp_topo": bpp_topo,
            "ratio_std": ratio_std,
            "ratio_topo": ratio_topo,
            "ppl_std": ppl_std,
            "ppl_topo": ppl_topo,
            "gain_ppl": gain,
        })

    # Print Summary Table
    print("\n" + "=" * 98)
    print(" EMPIRICAL PARETO FRONTIER: RATE (BPP) VS PERPLEXITY (PPL)")
    print("=" * 98)
    print(f"{'Quantization Regime':<24} | {'Radii (r0, r1, r2)':<20} | {'Effective bpp':<14} | {'Std PPL':<10} | {'Topo PPL':<10} | {'Advantage'}")
    print("-" * 98)
    for row in curve_data:
        radii_str = f"({row['r0']:.2f}, {row['r1']:.2f}, {row['r2']:.2f})"
        rate_str = f"{row['bpp_topo']:.3f} bpp ({row['ratio_topo']:.0f}x)"
        print(f"{row['regime']:<24} | {radii_str:<20} | {rate_str:<14} | {row['ppl_std']:<10.2f} | {row['ppl_topo']:<10.2f} | -{row['gain_ppl']:.2f} PPL")
    print("-" * 98)
    print(f"{'Dense Uncompressed FP32':<24} | {'---':<20} | {'32.000 bpp (1x)':<14} | {std_fp32_ppl:<10.2f} | {topo_fp32_ppl:<10.2f} | -{std_fp32_ppl - topo_fp32_ppl:.2f} PPL")
    print("=" * 98)

    # Plot Pareto curve with Matplotlib
    try:
        bpps = [row["bpp_topo"] for row in curve_data]
        ppls_std = [row["ppl_std"] for row in curve_data]
        ppls_topo = [row["ppl_topo"] for row in curve_data]

        plt.figure(figsize=(9, 5))
        plt.plot(bpps, ppls_std, marker="o", color="tab:red", lw=2.2, label="Standard AdamW Baseline")
        plt.plot(bpps, ppls_topo, marker="s", color="tab:blue", lw=2.5, label="Dirichlet Regularized (dreg)")

        plt.xlabel("Quantization Bitrate (bits per parameter / bpp)", fontsize=11)
        plt.ylabel("Validation Perplexity (PPL)", fontsize=11)
        plt.title("Empirical Rate-Distortion Pareto Frontier: bpp vs Perplexity", fontsize=12, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right", fontsize=10)
        plt.tight_layout()
        plt.savefig(args.plot_png, dpi=200)
        plt.close()
        print(f"[+] Saved Pareto plot figure to: {os.path.abspath(args.plot_png)}")
    except Exception as e:
        print(f"[!] Warning: Could not save plot ({e})")

    # Save JSON results
    output_dict = {
        "dense_fp32": {
            "standard_ppl": std_fp32_ppl,
            "topographic_ppl": topo_fp32_ppl,
            "roughness_ed_std": ed_std,
            "roughness_ed_topo": ed_topo,
        },
        "pareto_curve": curve_data,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2)
    print(f"[+] Saved Pareto curve JSON to: {os.path.abspath(args.output_json)}\n")


if __name__ == "__main__":
    main()
