"""
Multi-Seed Sweep & Statistical Robustness Benchmark
====================================================
Evaluates Topographic Regularization (dreg) vs Standard Baseline across multiple
independent random seeds (>= 3 seeds) with optional Dirichlet Annealing.

Measures:
  1. Dense FP32 Validation Perplexity
  2. Mean Weight Dirichlet Roughness (E_D)
  3. Quantized Validation Perplexity (0.945 bpp Base-3 TritQ)
  4. Relative Degradation (Delta PPL)

Outputs:
  - Mean +/- Std statistical summary table
  - JSON results report for publication and audit tracking
"""

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim

from dreg import (
    Base3TritQuantizer,
    CosineDirichletScheduler,
    DirichletLoss,
    StepCooldownScheduler,
    TopographicConfig,
    TopographicTransformer,
    dct2d,
    dirichlet_energy_2d,
    idct2d,
)


def set_seed(seed: int) -> None:
    """Sets random seeds across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_synthetic_corpus(
    vocab_size: int,
    seq_len: int,
    num_samples: int,
    seed: int = 42,
) -> torch.Tensor:
    """
    Creates structured Markovian sequential data with token dependencies
    to simulate language modeling without external network downloads.
    """
    rng = np.random.RandomState(seed)
    tokens = np.zeros((num_samples, seq_len), dtype=np.int64)

    # First token uniform
    tokens[:, 0] = rng.randint(0, vocab_size, size=num_samples)
    for t in range(1, seq_len):
        # 65% chance of following a structured Markov transition, 35% random
        prob = rng.rand(num_samples)
        structured = (tokens[:, t - 1] * 7 + 13) % vocab_size
        random_tok = rng.randint(0, vocab_size, size=num_samples)
        tokens[:, t] = np.where(prob > 0.35, structured, random_tok)

    return torch.tensor(tokens, dtype=torch.long)


def evaluate_perplexity(
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


def compute_model_roughness(model: TopographicTransformer) -> float:
    """Computes mean Dirichlet energy (E_D) over all 2D linear weight matrices."""
    linears = model.get_linear_projections()
    energies = [dirichlet_energy_2d(m.weight, normalization="edges").item() for m in linears]
    return float(np.mean(energies)) if energies else 0.0


def quantize_and_evaluate(
    model: TopographicTransformer,
    val_data: torch.Tensor,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Quantizes model weights in-place to Sub-1.0 bpp Base-3 Trit format,
    evaluates validation PPL, and restores original FP32 weights.
    Returns: (quantized_ppl, bpp)
    """
    quantizer = Base3TritQuantizer()
    saved_weights = {}
    total_orig_bytes = 0
    total_comp_bytes = 0

    # 1. Quantize linear projections via 2D-DCT + Base-3 Trit packing
    for m in model.get_linear_projections():
        orig_w = m.weight.data
        total_orig_bytes += orig_w.numel() * 4  # FP32 bytes
        saved_weights[id(m)] = orig_w.clone()

        spec = dct2d(orig_w)
        packed = quantizer.quantize_matrix(spec)
        rec_spec = quantizer.dequantize_matrix(packed, device=orig_w.device)
        rec_w = idct2d(rec_spec)
        m.weight.data.copy_(rec_w)

        comp_bytes = packed["band0"]["data"].nbytes + packed["band1"]["data"].nbytes + packed["band2"]["data"].nbytes
        total_comp_bytes += comp_bytes

    # 2. Evaluate under quantized weights
    quant_ppl = evaluate_perplexity(model, val_data, device=device)
    bpp = (total_comp_bytes * 8.0) / max(total_orig_bytes / 4, 1)

    # 3. Restore FP32 weights
    for m in model.get_linear_projections():
        if id(m) in saved_weights:
            m.weight.data.copy_(saved_weights[id(m)])

    return quant_ppl, bpp


def train_single_run(
    seed: int,
    cfg: TopographicConfig,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    is_topographic: bool,
    topo_lambda: float,
    annealing: str,
    cooldown_ratio: float,
    max_steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> Dict[str, float]:
    """Executes a single training run with the given seed and hyperparameters."""
    set_seed(seed)

    model = TopographicTransformer(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Set up Dirichlet loss & scheduler
    topo_loss_fn = DirichletLoss(weight_decay=topo_lambda, normalization="edges") if is_topographic else None
    scheduler = None

    if is_topographic and topo_loss_fn is not None:
        if annealing == "step_cooldown":
            scheduler = StepCooldownScheduler(
                loss_fn=topo_loss_fn,
                total_steps=max_steps,
                cooldown_ratio=cooldown_ratio,
            )
        elif annealing == "cosine":
            scheduler = CosineDirichletScheduler(
                loss_fn=topo_loss_fn,
                total_steps=max_steps,
            )

    model.train()
    step = 0
    num_samples = len(train_data)
    idx = 0

    while step < max_steps:
        # Mini-batch slice
        if idx + batch_size > num_samples:
            idx = 0
        batch = train_data[idx : idx + batch_size].to(device)
        idx += batch_size

        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        optimizer.zero_grad()
        _, ce_loss = model(inputs, targets)

        loss = ce_loss
        if is_topographic and topo_loss_fn is not None:
            topo_penalty = topo_loss_fn(model.modules())
            loss = loss + topo_penalty

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        step += 1

    # Evaluations
    fp32_ppl = evaluate_perplexity(model, val_data, device=device)
    roughness = compute_model_roughness(model)
    quant_ppl, bpp = quantize_and_evaluate(model, val_data, device=device)
    delta_ppl = quant_ppl - fp32_ppl

    return {
        "seed": seed,
        "fp32_ppl": fp32_ppl,
        "roughness_ed": roughness,
        "quant_ppl": quant_ppl,
        "delta_ppl": delta_ppl,
        "bpp": bpp,
    }


def compute_stats(runs: List[Dict[str, float]], key: str) -> Tuple[float, float]:
    """Computes mean and sample standard deviation for a given metric key."""
    values = [r[key] for r in runs]
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean_val, std_val


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Robustness & Annealing Benchmark")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2024], help="Random seeds to evaluate")
    parser.add_argument("--steps", type=int, default=300, help="Training steps per seed")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--topo-lambda", type=float, default=0.02, help="Topographic Dirichlet regularization weight")
    parser.add_argument(
        "--annealing",
        type=str,
        default="none",
        choices=["none", "step_cooldown", "cosine"],
        help="Dirichlet annealing schedule",
    )
    parser.add_argument("--cooldown-ratio", type=float, default=0.20, help="Terminal cooldown step ratio")
    parser.add_argument("--output-json", type=str, default="multiseed_results.json", help="Path to save JSON summary")
    args = parser.parse_args()

    print("=" * 80)
    print(" MULTI-SEED STATISTICAL ROBUSTNESS BENCHMARK")
    print(f" Seeds: {args.seeds} | Steps: {args.steps} | Annealing: {args.annealing}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")

    # Generate synthetic structured dataset
    vocab_size = 256
    seq_len = 64
    print("[*] Generating dataset (Markovian structured sequence corpus)...")
    train_data = create_synthetic_corpus(vocab_size, seq_len, num_samples=1200, seed=1)
    val_data = create_synthetic_corpus(vocab_size, seq_len, num_samples=300, seed=999)

    cfg = TopographicConfig(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        ffn_dim=256,
        max_seq_len=seq_len,
        topo_lambda=0.0,
    )

    # 1. Evaluate Standard Baseline across seeds
    print(f"\n--- [1/2] Training Standard Baseline (lambda=0.0) across {len(args.seeds)} seeds ---")
    standard_runs = []
    t_start = time.perf_counter()
    for seed in args.seeds:
        print(f"  > Running Standard Baseline (Seed {seed})...", end="", flush=True)
        res = train_single_run(
            seed=seed,
            cfg=cfg,
            train_data=train_data,
            val_data=val_data,
            is_topographic=False,
            topo_lambda=0.0,
            annealing="none",
            cooldown_ratio=0.0,
            max_steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
        standard_runs.append(res)
        print(f" FP32 PPL: {res['fp32_ppl']:.2f} | Quant PPL: {res['quant_ppl']:.2f} | E_D: {res['roughness_ed']:.6f}")

    # 2. Evaluate Topographic Model across seeds
    label = f"Topographic (lambda={args.topo_lambda}" + (f", {args.annealing})" if args.annealing != "none" else ")")
    print(f"\n--- [2/2] Training {label} across {len(args.seeds)} seeds ---")
    topo_runs = []
    for seed in args.seeds:
        print(f"  > Running Topographic Model (Seed {seed})...", end="", flush=True)
        res = train_single_run(
            seed=seed,
            cfg=cfg,
            train_data=train_data,
            val_data=val_data,
            is_topographic=True,
            topo_lambda=args.topo_lambda,
            annealing=args.annealing,
            cooldown_ratio=args.cooldown_ratio,
            max_steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
        topo_runs.append(res)
        print(f" FP32 PPL: {res['fp32_ppl']:.2f} | Quant PPL: {res['quant_ppl']:.2f} | E_D: {res['roughness_ed']:.6f}")

    total_time = time.perf_counter() - t_start

    # Compute Statistical Metrics (Mean +/- Std)
    std_fp32_m, std_fp32_s = compute_stats(standard_runs, "fp32_ppl")
    std_ed_m, std_ed_s = compute_stats(standard_runs, "roughness_ed")
    std_q_m, std_q_s = compute_stats(standard_runs, "quant_ppl")
    std_d_m, std_d_s = compute_stats(standard_runs, "delta_ppl")

    top_fp32_m, top_fp32_s = compute_stats(topo_runs, "fp32_ppl")
    top_ed_m, top_ed_s = compute_stats(topo_runs, "roughness_ed")
    top_q_m, top_q_s = compute_stats(topo_runs, "quant_ppl")
    top_d_m, top_d_s = compute_stats(topo_runs, "delta_ppl")

    roughness_reduction = (1.0 - (top_ed_m / (std_ed_m + 1e-12))) * 100.0
    quant_gain = std_q_m - top_q_m

    print("\n" + "=" * 90)
    print(f" STATISTICAL BENCHMARK SUMMARY ({len(args.seeds)} SEEDS: {args.seeds})")
    print("=" * 90)
    print(f"{'Metric':<32} | {'Standard Baseline':<25} | {'Topographic (Ours)':<25}")
    print("-" * 90)
    print(f"{'FP32 Val Perplexity':<32} | {std_fp32_m:.2f} +/- {std_fp32_s:.2f}{'':<13} | {top_fp32_m:.2f} +/- {top_fp32_s:.2f}")
    print(f"{'Weight Roughness (E_D)':<32} | {std_ed_m:.6f} +/- {std_ed_s:.6f}{'':<5} | {top_ed_m:.6f} +/- {top_ed_s:.6f}")
    print(f"{'Quantized PPL (0.945 bpp)':<32} | {std_q_m:.2f} +/- {std_q_s:.2f}{'':<13} | {top_q_m:.2f} +/- {top_q_s:.2f}")
    print(f"{'PPL Degradation (Delta)':<32} | +{std_d_m:.2f} +/- {std_d_s:.2f}{'':<12} | +{top_d_m:.2f} +/- {top_d_s:.2f}")
    print("-" * 90)
    print(f"[*] Statistical Weight Roughness Reduction: {roughness_reduction:.1f}%")
    print(f"[*] Topographic Quantization Advantage:     -{quant_gain:.2f} PPL under 0.945 bpp")
    print(f"[*] Total Benchmark Duration:               {total_time:.2f} seconds")
    print("=" * 90)

    # Save structured results
    out_payload = {
        "seeds": args.seeds,
        "steps": args.steps,
        "annealing": args.annealing,
        "topo_lambda": args.topo_lambda,
        "summary": {
            "standard": {
                "fp32_ppl": {"mean": std_fp32_m, "std": std_fp32_s},
                "roughness_ed": {"mean": std_ed_m, "std": std_ed_s},
                "quant_ppl": {"mean": std_q_m, "std": std_q_s},
                "delta_ppl": {"mean": std_d_m, "std": std_d_s},
            },
            "topographic": {
                "fp32_ppl": {"mean": top_fp32_m, "std": top_fp32_s},
                "roughness_ed": {"mean": top_ed_m, "std": top_ed_s},
                "quant_ppl": {"mean": top_q_m, "std": top_q_s},
                "delta_ppl": {"mean": top_d_m, "std": top_d_s},
            },
            "relative_gain": {
                "roughness_reduction_pct": roughness_reduction,
                "quant_ppl_advantage": quant_gain,
            },
        },
        "raw_runs": {
            "standard": standard_runs,
            "topographic": topo_runs,
        },
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"[+] Saved statistical results JSON to: {os.path.abspath(args.output_json)}\n")


if __name__ == "__main__":
    main()
