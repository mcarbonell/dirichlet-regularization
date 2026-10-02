"""
Modal GPU Fine-Tuning & Quantization Benchmark: GPT-2 (124M)
============================================================
Evaluates Dirichlet Topological Regularization on standard pre-trained LLM:
  - Architecture: HuggingFace GPT-2 (124M params, 12 layers, 768 hidden, 12 heads)
  - Projections regularized: 48 2D matrices (c_attn, c_proj, c_fc, c_proj)
  - Compares:
      1. Pre-trained Base GPT-2 (FP32 vs Base-3 TritQ)
      2. Standard Fine-Tuned GPT-2 (lambda = 0.0)
      3. Dirichlet Fine-Tuned GPT-2 (lambda = 1e-3 with StepCooldown)

Outputs:
  - Validation Perplexity before & after Base-3 TritQ
  - Harmonic Roughness (E_D) reduction across all 48 dense projections
  - Checkpoint and results JSON saved to Modal Volume
"""

import copy
import json
import math
import os
import time
from typing import Any, Dict, List, Tuple

import modal

app = modal.App("dreg-gpt2-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets",
        "tokenizers",
        "numpy",
        "tqdm",
    )
    .add_local_python_source("dreg")
)

volume = modal.Volume.from_name("dreg-checkpoints")
CHECKPOINT_DIR = "/root/checkpoints"


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={CHECKPOINT_DIR: volume},
)
def run_gpt2_benchmark(
    steps: int = 300,
    batch_size: int = 8,
    lr: float = 5e-5,
    topo_lambda: float = 1e-3,
    cooldown_ratio: float = 0.85,
) -> Dict[str, Any]:
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    from dreg.quantization import Base3TritQuantizer
    from dreg.spectral import dct2d, idct2d
    from dreg.topology import StepCooldownScheduler, dirichlet_energy_2d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running GPT-2 (124M) Benchmark on: {device} ({torch.cuda.get_device_name(0)})")

    # 1. Load Pretrained GPT-2 & Tokenizer
    model_name = "gpt2"
    print(f"[*] Loading pretrained {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"[+] Loaded {model_name} ({total_params:,} parameters)")

    def get_dense_weights(m: GPT2LMHeadModel) -> List[Tuple[str, torch.Tensor]]:
        weights = []
        for name, module in m.transformer.h.named_modules():
            if hasattr(module, "weight") and module.weight is not None and module.weight.dim() == 2:
                weights.append((name, module.weight))
        return weights

    dense_weights = get_dense_weights(base_model)
    print(f"[+] Identified {len(dense_weights)} dense 2D weight projections across 12 transformer blocks")

    def compute_roughness(m: GPT2LMHeadModel) -> float:
        weights = get_dense_weights(m)
        energies = [dirichlet_energy_2d(w, normalization="edges").item() for _, w in weights]
        return float(np.mean(energies))

    # 2. Prepare Data (TinyStories streaming)
    print("[*] Preparing evaluation and fine-tuning dataset...")
    raw_train = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    raw_val = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    def extract_tokens(stream, num_samples=1000, seq_l=128):
        token_list = []
        for i, item in enumerate(stream):
            enc = tokenizer.encode(item["text"])
            token_list.extend(enc)
            if len(token_list) >= num_samples * seq_l:
                break
        n_chunks = len(token_list) // seq_l
        t = torch.tensor(token_list[: n_chunks * seq_l], dtype=torch.long).view(n_chunks, seq_l)
        return t

    train_tokens = extract_tokens(raw_train, num_samples=1200, seq_l=128)
    val_tokens = extract_tokens(raw_val, num_samples=200, seq_l=128)
    print(f"[+] Prepared {len(train_tokens)} training sequences and {len(val_tokens)} validation sequences")

    def eval_ppl(m: GPT2LMHeadModel) -> float:
        m.eval()
        total_loss = 0.0
        total_tok = 0
        with torch.no_grad():
            for i in range(0, len(val_tokens), batch_size):
                batch = val_tokens[i : i + batch_size].to(device)
                outputs = m(batch, labels=batch)
                n = batch.numel()
                total_loss += outputs.loss.item() * n
                total_tok += n
        avg_loss = total_loss / max(total_tok, 1)
        return math.exp(min(avg_loss, 20.0))

    def quantize_gpt2(m: GPT2LMHeadModel, r0=0.15, r1=0.40, r2=1.00):
        q = copy.deepcopy(m)
        quantizer = Base3TritQuantizer(r0=r0, r1=r1, r2=r2)
        total_orig_bytes = 0
        total_comp_bytes = 0

        with torch.no_grad():
            for _, w_tensor in get_dense_weights(q):
                orig_w = w_tensor.data
                total_orig_bytes += orig_w.numel() * 4
                s = dct2d(orig_w)
                packed = quantizer.quantize_matrix(s)
                comp_bytes = (
                    packed["band0"]["data"].nbytes
                    + packed["band1"]["data"].nbytes
                    + packed["band2"]["data"].nbytes
                )
                total_comp_bytes += comp_bytes
                rec_s = quantizer.dequantize_matrix(packed, device=device)
                w_tensor.data.copy_(idct2d(rec_s))

        bpp = (total_comp_bytes * 8.0) / max(total_orig_bytes / 4, 1)
        return q, bpp

    # 3. Evaluate Base Pretrained GPT-2
    print("\n--- Evaluating Base Pretrained GPT-2 ---")
    base_fp32_ppl = eval_ppl(base_model)
    base_ed = compute_roughness(base_model)
    print(f"[*] Base Pretrained GPT-2:  FP32 PPL = {base_fp32_ppl:.2f} | Mean E_D = {base_ed:.6f}")

    base_q, base_bpp = quantize_gpt2(base_model)
    base_q_ppl = eval_ppl(base_q)
    print(f"[*] Base Pretrained TritQ:  {base_bpp:.3f} bpp PPL = {base_q_ppl:.2f} (Δ {base_q_ppl - base_fp32_ppl:+.2f})")

    # 4. Fine-Tuning Function
    def finetune(is_topo: bool, lam: float) -> Tuple[GPT2LMHeadModel, Dict[str, Any]]:
        m = copy.deepcopy(base_model)
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
        if is_topo:
            from dreg.topology import DirichletLoss
            topo_loss_fn = DirichletLoss(weight_decay=lam, normalization="edges")
            sched = StepCooldownScheduler(loss_fn=topo_loss_fn, total_steps=steps, cooldown_ratio=cooldown_ratio)
        else:
            sched = None

        tag = "Topographic (Dirichlet)" if is_topo else "Standard"
        print(f"\n[*] Starting Fine-Tuning: {tag} ({steps} steps, lr={lr}, lambda={lam})...")
        start_t = time.time()
        idx = 0

        for step in range(1, steps + 1):
            if idx + batch_size > len(train_tokens):
                idx = 0
            batch = train_tokens[idx : idx + batch_size].to(device)
            idx += batch_size

            opt.zero_grad()
            outputs = m(batch, labels=batch)
            ce_loss = outputs.loss

            if is_topo and sched is not None:
                cur_lam = sched.step()
                if cur_lam > 0:
                    weights = get_dense_weights(m)
                    raw_ed = torch.stack([dirichlet_energy_2d(w, normalization="edges") for _, w in weights]).mean()
                    loss = ce_loss + cur_lam * raw_ed
                else:
                    loss = ce_loss
            else:
                loss = ce_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()

            if step % 50 == 0 or step == steps:
                print(f"  Step {step:4d}/{steps} | Loss: {ce_loss.item():.4f}")

        elapsed = time.time() - start_t
        print(f"[+] Completed {tag} fine-tuning in {elapsed:.1f}s")
        return m, {"time_s": elapsed}

    # 5. Fine-Tune Standard
    std_ft_model, _ = finetune(is_topo=False, lam=0.0)
    std_ft_ppl = eval_ppl(std_ft_model)
    std_ft_ed = compute_roughness(std_ft_model)
    std_ft_q, std_ft_bpp = quantize_gpt2(std_ft_model)
    std_ft_q_ppl = eval_ppl(std_ft_q)

    # 6. Fine-Tune Topographic
    topo_ft_model, _ = finetune(is_topo=True, lam=topo_lambda)
    topo_ft_ppl = eval_ppl(topo_ft_model)
    topo_ft_ed = compute_roughness(topo_ft_model)
    topo_ft_q, topo_ft_bpp = quantize_gpt2(topo_ft_model)
    topo_ft_q_ppl = eval_ppl(topo_ft_q)

    print("\n" + "=" * 90)
    print(" GPT-2 (124M) SOTA QUANTIZATION & TOPOGRAPHIC COMPARISON RESULTS")
    print("=" * 90)
    print(f"| Model Variant                | FP32 PPL | Mean E_D   | TritQ ({topo_ft_bpp:.2f} bpp) PPL | Degradation (Δ) |")
    print("|:-----------------------------|:--------:|:----------:|:--------------------:|:---------------:|")
    print(f"| Base Pretrained GPT-2        | {base_fp32_ppl:>8.2f} | {base_ed:>10.6f} | {base_q_ppl:>20.2f} | {base_q_ppl - base_fp32_ppl:>+15.2f} |")
    print(f"| Standard Fine-Tuned          | {std_ft_ppl:>8.2f} | {std_ft_ed:>10.6f} | {std_ft_q_ppl:>20.2f} | {std_ft_q_ppl - std_ft_ppl:>+15.2f} |")
    print(f"| Dirichlet Fine-Tuned (Ours)  | {topo_ft_ppl:>8.2f} | {topo_ft_ed:>10.6f} | {topo_ft_q_ppl:>20.2f} | {topo_ft_q_ppl - topo_ft_ppl:>+15.2f} |")
    print("=" * 90)

    # Save results to modal volume
    out_file = os.path.join(CHECKPOINT_DIR, "gpt2_dirichlet_benchmark_results.json")
    results = {
        "model": "gpt2-124m",
        "steps": steps,
        "lr": lr,
        "topo_lambda": topo_lambda,
        "base_pretrained": {
            "fp32_ppl": base_fp32_ppl,
            "ed": base_ed,
            "tritq_ppl": base_q_ppl,
            "delta_ppl": base_q_ppl - base_fp32_ppl,
        },
        "standard_finetuned": {
            "fp32_ppl": std_ft_ppl,
            "ed": std_ft_ed,
            "tritq_ppl": std_ft_q_ppl,
            "delta_ppl": std_ft_q_ppl - std_ft_ppl,
        },
        "dirichlet_finetuned": {
            "fp32_ppl": topo_ft_ppl,
            "ed": topo_ft_ed,
            "tritq_ppl": topo_ft_q_ppl,
            "delta_ppl": topo_ft_q_ppl - topo_ft_ppl,
        },
    }
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print(f"[+] Saved GPT-2 benchmark results to {out_file}")

    return results


@app.local_entrypoint()
def main(steps: int = 300, topo_lambda: float = 1e-3):
    results = run_gpt2_benchmark.remote(steps=steps, topo_lambda=topo_lambda)
    print("\n[+] GPT-2 Run Finished!")
    print(json.dumps(results, indent=2))
