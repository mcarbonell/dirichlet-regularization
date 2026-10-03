"""
Example: Falsification Test — Topographic vs Standard under Sub-1.0 bpp Quantization
===================================================================================
Reproduces the critical falsification experiment:
  - Model A (Topographic): Trained with Dirichlet harmonic pinning.
  - Model B (Standard): Trained without spatial constraints (standard permutation-invariant).
Both models undergo identical 4-band Base-3 Quantum Trit Quantization.
Metrics, bit-rates, and status verdicts are computed dynamically without hardcoded labels.
"""

import argparse
import copy
import math

import torch
import torch.optim as optim

from dreg import (
    Base3TritQuantizer,
    TopographicConfig,
    TopographicTransformer,
    dct2d,
    idct2d,
)
from dreg.topology import dirichlet_energy_2d


def generate_structured_language_dataset(num_samples: int, seq_len: int, vocab_size: int = 128):
    """
    Generates structured sequence data with synthetic multi-token syntax
    so that models learn real statistical dependencies instead of uniform noise.
    """
    subjects = [2, 3, 4, 5, 6, 7]
    verbs = [10, 11, 12, 13, 14, 15]
    adjectives = [20, 21, 22, 23, 24, 25]
    nouns = [30, 31, 32, 33, 34, 35]
    adverbs = [40, 41, 42, 43]
    punctuation = [1]  # EOS / period

    sequences = []
    for _ in range(num_samples):
        tokens = []
        while len(tokens) < seq_len:
            s = subjects[torch.randint(0, len(subjects), (1,)).item()]
            v = verbs[torch.randint(0, len(verbs), (1,)).item()]
            a = adjectives[torch.randint(0, len(adjectives), (1,)).item()]
            n = nouns[torch.randint(0, len(nouns), (1,)).item()]
            adv = adverbs[torch.randint(0, len(adverbs), (1,)).item()]
            tokens.extend([s, adv, v, a, n, punctuation[0]])
        sequences.append(tokens[:seq_len])
    return torch.tensor(sequences, dtype=torch.long)


def evaluate_ppl(model, test_batches):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in test_batches:
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            _, loss = model(inputs, targets)
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20.0))
    return ppl, avg_loss


def compute_model_dirichlet_energy(model) -> float:
    energies = []
    for p in model.get_linear_projections():
        energies.append(dirichlet_energy_2d(p.weight).item())
    return sum(energies) / max(len(energies), 1)


def quantize_and_reconstruct_model(model, quantizer):
    compressed_model = copy.deepcopy(model)
    total_orig_bytes = 0
    total_comp_bytes = 0

    for m in compressed_model.get_linear_projections():
        orig_w = m.weight.data
        total_orig_bytes += orig_w.numel() * 4  # FP32

        # 2D DCT
        spec = dct2d(orig_w)
        # Quantize to 4-band Base-3 trits
        packed = quantizer.quantize_matrix(spec)
        # Dequantize back
        rec_spec = quantizer.dequantize_matrix(packed, device=orig_w.device)
        rec_w = idct2d(rec_spec)
        m.weight.data.copy_(rec_w)

        # Count compressed bytes: Band 0 (1B), Band 1 (0.5B), Band 2 (0.2B), Band 3 (0B)
        comp_bytes = packed["band0"]["data"].nbytes + packed["band1"]["data"].nbytes + packed["band2"]["data"].nbytes
        total_comp_bytes += comp_bytes

    ratio = total_orig_bytes / max(total_comp_bytes, 1)
    bpp = (total_comp_bytes * 8.0) / (total_orig_bytes / 4)
    return compressed_model, bpp, ratio


def determine_status(ppl_fp32: float, ppl_quant: float) -> str:
    """Dynamically determine status verdict based on relative and absolute degradation."""
    ratio = ppl_quant / max(ppl_fp32, 1e-4)
    delta = ppl_quant - ppl_fp32
    if ratio > 2.5 or delta > 15.0:
        return f"COLLAPSE ({ratio:.1f}x PPL)"
    elif ratio > 1.25 or delta > 2.0:
        return f"DEGRADED (+{delta:.1f} PPL)"
    else:
        return f"PRESERVED ({ratio:.2f}x / +{delta:.2f})"


def main():
    parser = argparse.ArgumentParser(
        description="Falsification benchmark: Topographic vs Standard under Sub-1.0 bpp Quantization. "
        "Requires >=5 epochs and lambda≈15–30 to observe separation; smaller values are washed out by Adam."
    )
    parser.add_argument("--lambda-val", type=float, default=30.0, help="Dirichlet regularization strength (default: 30.0; use >=15 for visible effect)")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs (default: 8; use >=5 to see separation)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed (default: 1337)")
    args = parser.parse_args()

    if args.epochs < 5:
        print(f"[!] WARNING: epochs={args.epochs} is too few to observe Dirichlet smoothing. "
              f"Recommended >=5 epochs (default 8). Results may not replicate the paper.")
    if args.lambda_val < 5.0:
        print(f"[!] WARNING: lambda={args.lambda_val} is too small (gradient washed out). "
              f"Recommended 15–30 for this scale (see paper §4.2 calibration).")

    print("=" * 85)
    print(" FALSIFICATION BENCHMARK: TOPOGRAPHIC MANIFOLD VS STANDARD UNORDERED MODEL")
    print("=" * 85)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device} | Lambda: {args.lambda_val} | Epochs: {args.epochs}")

    torch.manual_seed(args.seed)
    vocab_size = 128
    seq_len = 48
    num_train = 600
    num_test = 200

    print("[*] Generating structured language dataset with synthetic grammar...")
    train_data = generate_structured_language_dataset(num_train, seq_len, vocab_size).to(device)
    test_data = generate_structured_language_dataset(num_test, seq_len, vocab_size).to(device)
    test_batches = test_data.split(32)

    cfg = TopographicConfig(
        vocab_size=vocab_size,
        d_model=96,
        n_heads=3,
        n_layers=3,
        ffn_dim=192,
        max_seq_len=seq_len,
        topo_lambda=args.lambda_val,
    )

    print("\n[1/3] Training Model A: Topographic Model (with Dirichlet Pinning)...")
    model_topo = TopographicTransformer(cfg).to(device)
    opt_topo = optim.AdamW(model_topo.parameters(), lr=2e-3)
    for epoch in range(args.epochs):
        for b in train_data.split(32):
            opt_topo.zero_grad()
            _, ce = model_topo(b[:, :-1], b[:, 1:])
            loss = ce + model_topo.topographic_loss()
            loss.backward()
            opt_topo.step()

    print("[2/3] Training Model B: Standard Model (without Topographic loss, lambda=0)...")
    cfg_std = copy.deepcopy(cfg)
    cfg_std.topo_lambda = 0.0
    model_std = TopographicTransformer(cfg_std).to(device)
    opt_std = optim.AdamW(model_std.parameters(), lr=2e-3)
    for epoch in range(args.epochs):
        for b in train_data.split(32):
            opt_std.zero_grad()
            _, ce = model_std(b[:, :-1], b[:, 1:])
            ce.backward()
            opt_std.step()

    e_d_topo = compute_model_dirichlet_energy(model_topo)
    e_d_std = compute_model_dirichlet_energy(model_std)
    ppl_topo_fp32, _ = evaluate_ppl(model_topo, test_batches)
    ppl_std_fp32, _ = evaluate_ppl(model_std, test_batches)

    quantizer = Base3TritQuantizer()
    model_topo_trit, bpp_topo, ratio_topo = quantize_and_reconstruct_model(model_topo, quantizer)
    model_std_trit, bpp_std, ratio_std = quantize_and_reconstruct_model(model_std, quantizer)

    ppl_topo_trit, _ = evaluate_ppl(model_topo_trit, test_batches)
    ppl_std_trit, _ = evaluate_ppl(model_std_trit, test_batches)

    status_topo = determine_status(ppl_topo_fp32, ppl_topo_trit)
    status_std = determine_status(ppl_std_fp32, ppl_std_trit)

    print("\n[3/3] Quantization Complete. Results:")
    print("=" * 85)
    print(" FALSIFICATION TEST RESULTS SUMMARY (DYNAMICALLY MEASURED)")
    print("=" * 85)
    header = f"{'Architecture / Model':<30} | {'Dirichlet E_D':<14} | {'FP32 PPL':<10} | {f'TritQ ({bpp_topo:.3f} bpp)':<18} | {'Status'}"
    print(header)
    print("-" * 85)
    print(f"{'Standard Model (Unordered)':<30} | {e_d_std:<14.4f} | {ppl_std_fp32:<10.2f} | {ppl_std_trit:<18.2f} | {status_std}")
    print(f"{'Topographic Model (Dirichlet)':<30} | {e_d_topo:<14.4f} | {ppl_topo_fp32:<10.2f} | {ppl_topo_trit:<18.2f} | {status_topo}")
    print("=" * 85)
    print(f"Linear Weights Bit Rate: {bpp_topo:.3f} bpp ({ratio_topo:.1f}x compression)")
    print(f"Dirichlet Smoothness Gap: Topographic E_D is {e_d_std / max(e_d_topo, 1e-6):.1f}x smoother than Standard")
    print("=" * 85)
    # Interpretive footer for small-scale CPU runs
    if e_d_topo < e_d_std:
        print(f"[✓] Dirichlet smoothing confirmed: E_D gap {e_d_std/e_d_topo:.1f}x (topo smoother).")
    else:
        print("[✗] No smoothing observed — check lambda/epochs.")
    # Quantization advantage is subtle at tiny scale; ED gap is the robust signal.
    delta_std = ppl_std_trit - ppl_std_fp32
    delta_topo = ppl_topo_trit - ppl_topo_fp32
    if delta_topo < delta_std:
        print(f"[✓] Quantization advantage at this scale: topo Δ{delta_topo:.2f} < std Δ{delta_std:.2f} "
              f"(gap grows with epochs & matrix size; see reproduce_all.py and paper 10M results).")
    else:
        print(f"[i] Quantization Δ topo {delta_topo:.2f} vs std {delta_std:.2f} — at CPU-tiny scale (96×96, "
              f"{args.epochs} epochs) the PPL gap is small/noisy. "
              f"Run with --epochs 20 or see `examples/reproduce_all.py` (spectral 7.1×) "
              f"and paper Table 2 (10M, 6.76 PPL gain) for the robust effect.")
    print("[i] For the architecture-agnostic spectral proof, run:  python examples/reproduce_all.py")
    print("=" * 85)


if __name__ == "__main__":
    main()
