"""
Modal Benchmark: 2D Bipartite TSP Permutation + Base-3 TritQ on GPT-2 (124M)
=============================================================================
Bridges 'spec-rama' (2D Bipartite TSP Weight Reordering) with
'dirichlet-regularization' (Base-3 Trit Spectral Quantization).

Investigates:
  Can 2D Bipartite TSP Channel Permutations from 'spec-rama' induce
  sufficient spatial smoothness in pre-trained GPT-2 weights to prevent
  the spectral quantization collapse observed in native coordinate order?

Experimental Arms:
  1. Base Pre-trained GPT-2 FP32 (Unquantized Reference)
  2. Native Order + Base-3 TritQ (Direct Spectral Quantization without reordering)
  3. Spec-RAMA 2D TSP Permuted + Base-3 TritQ (Calibrated radii: r0=0.15, r1=0.40, r2=1.00)
  4. Spec-RAMA 2D TSP Permuted + Base-3 TritQ (High-Fidelity radii: r0=0.30, r1=0.70, r2=1.30)
  5. Spec-RAMA 2D TSP Permuted + Base-3 TritQ (Conservative radii: r0=0.45, r1=0.90, r2=1.41)

Outputs:
  - Total Variation (TV) & Dirichlet Energy (E_D) reduction via 2D TSP
  - Validation Perplexity across all regimes
  - JSON results report saved to Modal Volume
"""

import copy
import json
import math
import os
import time
from typing import Any, Dict, List, Tuple

import modal

app = modal.App("dreg-specrama-gpt2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets",
        "numpy",
        "tokenizers",
        "tqdm",
    )
    .add_local_python_source("dreg")
)

volume = modal.Volume.from_name("dreg-checkpoints")
CHECKPOINT_DIR = "/root/checkpoints"


# ---------------------------------------------------------------------------
# Spec-RAMA 2D Bipartite TSP Implementation
# ---------------------------------------------------------------------------
def compute_greedy_tsp_1d(matrix, axis: int = 0):
    import torch
    data = matrix.t().detach() if axis == 1 else matrix.detach()
    num_vecs = data.shape[0]

    if num_vecs <= 2:
        return torch.arange(num_vecs, dtype=torch.long, device=matrix.device)

    dist_matrix = torch.cdist(data.float(), data.float())
    visited = torch.zeros(num_vecs, dtype=torch.bool, device=matrix.device)
    path = torch.zeros(num_vecs, dtype=torch.long, device=matrix.device)

    curr = 0
    path[0] = curr
    visited[curr] = True
    dist_matrix.fill_diagonal_(float("inf"))

    for i in range(1, num_vecs):
        dists = dist_matrix[curr].clone()
        dists[visited] = float("inf")
        nxt = torch.argmin(dists)
        path[i] = nxt
        visited[nxt] = True
        curr = nxt

    return path


def compute_total_variation_2d(matrix):
    import torch
    tv_rows = torch.abs(matrix[1:, :] - matrix[:-1, :]).sum()
    tv_cols = torch.abs(matrix[:, 1:] - matrix[:, :-1]).sum()
    return (tv_rows + tv_cols).item()


def compute_bipartite_2d_permutations(weight, max_iters: int = 4, tol: float = 1e-3):
    import torch
    out_dim, in_dim = weight.shape
    row_perm = torch.arange(out_dim, dtype=torch.long, device=weight.device)
    col_perm = torch.arange(in_dim, dtype=torch.long, device=weight.device)

    w_curr = weight.detach().clone()
    prev_tv = compute_total_variation_2d(w_curr)

    for iter_idx in range(max_iters):
        r_p = compute_greedy_tsp_1d(w_curr, axis=0)
        row_perm = row_perm[r_p]
        w_curr = w_curr[r_p, :]

        c_p = compute_greedy_tsp_1d(w_curr, axis=1)
        col_perm = col_perm[c_p]
        w_curr = w_curr[:, c_p]

        curr_tv = compute_total_variation_2d(w_curr)
        rel_improvement = (prev_tv - curr_tv) / (prev_tv + 1e-8)
        if rel_improvement < tol and iter_idx >= 1:
            break
        prev_tv = curr_tv

    return row_perm, col_perm


# ---------------------------------------------------------------------------
# Modal Remote Benchmark
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={CHECKPOINT_DIR: volume},
)
def run_tsp_quantization_benchmark() -> Dict[str, Any]:
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    from dreg.quantization import Base3TritQuantizer
    from dreg.spectral import dct2d, idct2d
    from dreg.topology import dirichlet_energy_2d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Spec-RAMA + Dirichlet Benchmark on: {device} ({torch.cuda.get_device_name(0)})")

    # 1. Load Pretrained GPT-2
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    base_model.eval()

    def get_dense_weights(m: GPT2LMHeadModel) -> List[Tuple[str, torch.Tensor]]:
        weights = []
        for name, module in m.transformer.h.named_modules():
            if hasattr(module, "weight") and module.weight is not None and module.weight.dim() == 2:
                weights.append((name, module.weight))
        return weights

    dense_weights = get_dense_weights(base_model)
    print(f"[+] Loaded {model_name} with {len(dense_weights)} dense projections")

    # 2. Prepare Validation Data (TinyStories stream)
    raw_val = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    val_tokens_list = []
    seq_l = 128
    for item in raw_val:
        enc = tokenizer.encode(item["text"])
        val_tokens_list.extend(enc)
        if len(val_tokens_list) >= 200 * seq_l:
            break
    n_chunks = len(val_tokens_list) // seq_l
    val_tokens = torch.tensor(val_tokens_list[: n_chunks * seq_l], dtype=torch.long).view(n_chunks, seq_l)
    print(f"[+] Cached validation set: {len(val_tokens)} sequences ({len(val_tokens) * seq_l:,} tokens)")

    def eval_ppl(m: GPT2LMHeadModel) -> float:
        m.eval()
        total_loss = 0.0
        total_tok = 0
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(val_tokens), batch_size):
                batch = val_tokens[i : i + batch_size].to(device)
                outputs = m(batch, labels=batch)
                n = batch.numel()
                total_loss += outputs.loss.item() * n
                total_tok += n
        avg_loss = total_loss / max(total_tok, 1)
        return math.exp(min(avg_loss, 20.0))

    # Base FP32 Perplexity
    base_ppl = eval_ppl(base_model)
    raw_tv_list = [compute_total_variation_2d(w.data) for _, w in dense_weights]
    raw_ed_list = [dirichlet_energy_2d(w.data, normalization="edges").item() for _, w in dense_weights]
    mean_raw_tv = float(np.mean(raw_tv_list))
    mean_raw_ed = float(np.mean(raw_ed_list))
    print(f"[*] Base FP32 GPT-2: PPL = {base_ppl:.2f} | Mean TV = {mean_raw_tv:.1f} | Mean E_D = {mean_raw_ed:.6f}")

    # 3. Compute 2D TSP Permutations on All 48 Dense Layers
    print("\n[*] Computing 2D Bipartite TSP Permutations on all 48 dense projections (GPU accelerated)...")
    tsp_start = time.time()
    permutations = {}
    perm_tv_list = []
    perm_ed_list = []

    for name, w_tensor in dense_weights:
        w_orig = w_tensor.data
        r_p, c_p = compute_bipartite_2d_permutations(w_orig, max_iters=3, tol=1e-3)
        w_perm = w_orig[r_p, :][:, c_p]

        tv_p = compute_total_variation_2d(w_perm)
        ed_p = dirichlet_energy_2d(w_perm, normalization="edges").item()
        perm_tv_list.append(tv_p)
        perm_ed_list.append(ed_p)

        permutations[name] = (r_p, c_p)

    tsp_time = time.time() - tsp_start
    mean_perm_tv = float(np.mean(perm_tv_list))
    mean_perm_ed = float(np.mean(perm_ed_list))
    tv_red = (1.0 - mean_perm_tv / mean_raw_tv) * 100.0
    ed_red = (1.0 - mean_perm_ed / mean_raw_ed) * 100.0

    print(f"[+] 2D Bipartite TSP Permutations computed in {tsp_time:.2f}s")
    print(f"[*] Mean Total Variation: {mean_raw_tv:.1f} -> {mean_perm_tv:.1f} (-{tv_red:.1f}%)")
    print(f"[*] Mean Dirichlet Energy: {mean_raw_ed:.6f} -> {mean_perm_ed:.6f} (-{ed_red:.1f}%)")

    # 4. Quantization Helper with optional permutation
    def quantize_gpt2_variant(use_tsp: bool, r0: float, r1: float, r2: float):
        m = copy.deepcopy(base_model)
        quantizer = Base3TritQuantizer(r0=r0, r1=r1, r2=r2)
        total_orig_bytes = 0
        total_comp_bytes = 0

        with torch.no_grad():
            for name, w_tensor in get_dense_weights(m):
                w_orig = w_tensor.data
                total_orig_bytes += w_orig.numel() * 4

                if use_tsp:
                    r_p, c_p = permutations[name]
                    w_work = w_orig[r_p, :][:, c_p]
                else:
                    w_work = w_orig

                # 2D DCT
                s = dct2d(w_work)
                packed = quantizer.quantize_matrix(s)
                comp_bytes = (
                    packed["band0"]["data"].nbytes
                    + packed["band1"]["data"].nbytes
                    + packed["band2"]["data"].nbytes
                )
                total_comp_bytes += comp_bytes

                # Inverse DCT
                rec_s = quantizer.dequantize_matrix(packed, device=device)
                rec_w = idct2d(rec_s)

                if use_tsp:
                    # Invert permutations
                    inv_r = torch.argsort(r_p)
                    inv_c = torch.argsort(c_p)
                    rec_w_orig = rec_w[inv_r, :][:, inv_c]
                    w_tensor.data.copy_(rec_w_orig)
                else:
                    w_tensor.data.copy_(rec_w)

        bpp = (total_comp_bytes * 8.0) / max(total_orig_bytes / 4, 1)
        ppl = eval_ppl(m)
        return ppl, bpp

    # 5. Comparative Evaluation Across Regimes
    regimes = [
        {"name": "Calibrated TritQ", "r0": 0.15, "r1": 0.40, "r2": 1.00},
        {"name": "Mid-Band",         "r0": 0.20, "r1": 0.55, "r2": 1.15},
        {"name": "High-Fidelity",    "r0": 0.30, "r1": 0.70, "r2": 1.30},
        {"name": "Conservative",     "r0": 0.45, "r1": 0.90, "r2": 1.41},
    ]

    print("\n" + "=" * 96)
    print(" GPT-2 (124M): NATIVE COORDINATES VS SPEC-RAMA 2D TSP PERMUTATION UNDER TRITQ")
    print("=" * 96)
    print(f"| {'Regime':<18} | {'Rate (bpp)':>10} | {'Native PPL':>12} | {'Spec-RAMA TSP PPL':>18} | {'Advantage (Δ)':>14} |")
    print("|:-------------------|:----------:|:------------:|:------------------:|:--------------:|")

    results_table = []
    for reg in regimes:
        r0, r1, r2 = reg["r0"], reg["r1"], reg["r2"]
        ppl_nat, bpp_nat = quantize_gpt2_variant(use_tsp=False, r0=r0, r1=r1, r2=r2)
        ppl_tsp, bpp_tsp = quantize_gpt2_variant(use_tsp=True, r0=r0, r1=r1, r2=r2)
        advantage = ppl_nat - ppl_tsp

        row = {
            "regime": reg["name"],
            "r0": r0,
            "r1": r1,
            "r2": r2,
            "bpp": round(bpp_tsp, 3),
            "native_ppl": round(ppl_nat, 2),
            "tsp_ppl": round(ppl_tsp, 2),
            "advantage": round(advantage, 2),
        }
        results_table.append(row)
        print(f"| {reg['name']:<18} | {bpp_tsp:>9.3f}b | {ppl_nat:>12.2f} | {ppl_tsp:>18.2f} | {advantage:>+14.2f} |")

    print("=" * 96)

    output_data = {
        "model": "gpt2-124m",
        "base_fp32_ppl": base_ppl,
        "mean_raw_tv": mean_raw_tv,
        "mean_perm_tv": mean_perm_tv,
        "tv_reduction_pct": tv_red,
        "mean_raw_ed": mean_raw_ed,
        "mean_perm_ed": mean_perm_ed,
        "ed_reduction_pct": ed_red,
        "tsp_time_s": tsp_time,
        "regimes": results_table,
    }

    out_file = os.path.join(CHECKPOINT_DIR, "gpt2_specrama_tsp_tritq_results.json")
    with open(out_file, "w") as f:
        json.dump(output_data, f, indent=2)
    volume.commit()
    print(f"[+] Saved results to {out_file}")

    return output_data


@app.local_entrypoint()
def main():
    results = run_tsp_quantization_benchmark.remote()
    print("\n[+] Benchmark finished successfully!")
    print(json.dumps(results, indent=2))
