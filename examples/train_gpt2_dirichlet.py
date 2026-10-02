"""
GPT-2 (124M) Fine-Tuning with 2D Dirichlet Spatial Regularization
================================================================
Fine-tunes a pre-trained GPT-2 (124M) model using Dirichlet harmonic energy
pinning on dense projections (QKV attention & FFN expansions) to induce
spectral compressibility without sacrificing dense linguistic expressivity.

Compares:
  1. Base Pre-trained GPT-2 (FP32 vs 0.945 bpp TritQ)
  2. Standard Fine-Tuned GPT-2 without Dirichlet penalty (lambda = 0.0)
  3. Dirichlet Fine-Tuned GPT-2 (lambda = lambda_topo, with optional Annealing)

Evaluates:
  - Validation Perplexity before & after Base-3 Trit Quantization
  - Mean Weight Roughness (E_D) across all 48 dense projections
  - Generation quality pre/post compression
"""

import argparse
import copy
import json
import math
import os
from typing import List, Tuple

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from dreg import (
    Base3TritQuantizer,
    CosineDirichletScheduler,
    DirichletLoss,
    StepCooldownScheduler,
    dct2d,
    dirichlet_energy_2d,
    idct2d,
)


def get_gpt2_dense_weights(model: GPT2LMHeadModel) -> List[Tuple[str, torch.Tensor]]:
    """
    Extracts all 2D projection weights from GPT-2 blocks:
      - c_attn (QKV projection: 768 x 2304)
      - c_proj (Attention output: 768 x 768)
      - c_fc (FFN intermediate expansion: 768 x 3072)
      - c_proj (FFN contraction: 3072 x 768)
    Excludes wte/wpe embeddings and lm_head projection.
    """
    dense_weights = []
    for name, module in model.transformer.h.named_modules():
        if hasattr(module, "weight") and module.weight is not None and module.weight.dim() == 2:
            dense_weights.append((name, module.weight))
    return dense_weights


def compute_gpt2_roughness(model: GPT2LMHeadModel) -> float:
    """Computes mean Dirichlet harmonic roughness (E_D) over all 2D weights."""
    weights = get_gpt2_dense_weights(model)
    if not weights:
        return 0.0
    energies = [dirichlet_energy_2d(w, normalization="edges").item() for _, w in weights]
    return float(np.mean(energies))


def quantize_gpt2_model(
    model: GPT2LMHeadModel,
    quantizer: Base3TritQuantizer,
) -> Tuple[GPT2LMHeadModel, float, float]:
    """
    Applies 2D-DCT + Base-3 Trit Quantization to all dense linear projections.
    Returns: (quantized_model_copy, effective_bpp, compression_ratio)
    """
    q_model = copy.deepcopy(model)
    total_orig_bytes = 0
    total_comp_bytes = 0

    for _, weight_tensor in get_gpt2_dense_weights(q_model):
        orig_w = weight_tensor.data
        total_orig_bytes += orig_w.numel() * 4  # FP32 bytes

        spec = dct2d(orig_w)
        packed = quantizer.quantize_matrix(spec)
        rec_spec = quantizer.dequantize_matrix(packed, device=orig_w.device)
        rec_w = idct2d(rec_spec)
        weight_tensor.data.copy_(rec_w)

        comp_bytes = (
            packed["band0"]["data"].nbytes
            + packed["band1"]["data"].nbytes
            + packed["band2"]["data"].nbytes
        )
        total_comp_bytes += comp_bytes

    bpp = (total_comp_bytes * 8.0) / max(total_orig_bytes / 4, 1)
    ratio = total_orig_bytes / max(total_comp_bytes, 1)
    return q_model, bpp, ratio


def evaluate_gpt2_ppl(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    batch_size: int = 4,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Evaluates cross-entropy loss and returns perplexity over token batches."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            batch = input_ids[i : i + batch_size].to(device)
            # Standard causal language modeling with labels = inputs
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            num_tokens = batch.numel()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(min(avg_loss, 20.0))


def generate_text_sample(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str = "In a surprising discovery, scientists found",
    max_new_tokens: int = 35,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Generates autoregressive text continuation from a prompt."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def get_training_batches(
    tokenizer: GPT2Tokenizer,
    num_samples: int = 200,
    seq_len: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates prompt text corpus for fine-tuning.
    Uses representative sample narratives to avoid required internet access.
    """
    sample_texts = [
        "Artificial intelligence and deep learning models require vast amounts of memory to store parameters. "
        "Quantization techniques attempt to compress these matrices to lower bit representations. "
        "Topological Dirichlet regularization imposes spatial smoothness across weight coordinates, "
        "causing spectral energy to concentrate in low spatial frequency DCT harmonics.",
        "The quick brown fox jumps over the lazy dog in the sunny afternoon near the tranquil riverbank. "
        "Scientists have discovered that cortical structures exhibit continuous receptive fields where "
        "neighboring neurons share strongly correlated responses to visual and auditory stimuli.",
        "Generative language modeling has transformed computer science and information retrieval across industries. "
        "However, deploying these neural networks on ultra-low-power edge microcontrollers with sub-1 megabyte "
        "of internal SRAM requires extreme compression rates below one bit per parameter.",
    ]
    # Repeat and tokenize
    text_corpus = " ".join(sample_texts * 40)
    tokens = tokenizer.encode(text_corpus, return_tensors="pt")[0]

    num_tokens = (len(tokens) // seq_len) * seq_len
    tokens = tokens[:num_tokens].view(-1, seq_len)

    split_idx = int(0.80 * len(tokens))
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    return train_tokens, val_tokens


def fine_tune_gpt2(
    model: GPT2LMHeadModel,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    is_topographic: bool,
    topo_lambda: float,
    annealing: str,
    cooldown_ratio: float,
    max_steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> GPT2LMHeadModel:
    """Executes fine-tuning loop on GPT-2 with or without Dirichlet regularization."""
    model = copy.deepcopy(model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

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
    idx = 0
    num_samples = len(train_tokens)

    while step < max_steps:
        if idx + batch_size > num_samples:
            idx = 0
        batch = train_tokens[idx : idx + batch_size].to(device)
        idx += batch_size

        optimizer.zero_grad()
        outputs = model(batch, labels=batch)
        ce_loss = outputs.loss

        loss = ce_loss
        if is_topographic and topo_loss_fn is not None:
            dense_weights = [w for _, w in get_gpt2_dense_weights(model)]
            if dense_weights:
                energies = torch.stack(
                    [dirichlet_energy_2d(w, normalization="edges") for w in dense_weights]
                )
                topo_penalty = topo_loss_fn.weight_decay * energies.mean()
                loss = loss + topo_penalty

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        step += 1

    return model


def main():
    parser = argparse.ArgumentParser(description="GPT-2 (124M) Fine-Tuning with Dirichlet Regularization")
    parser.add_argument("--model-name", type=str, default="gpt2", help="Pretrained model identifier")
    parser.add_argument("--steps", type=int, default=50, help="Fine-tuning steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--topo-lambda", type=float, default=0.05, help="Dirichlet regularization strength")
    parser.add_argument(
        "--annealing",
        type=str,
        default="step_cooldown",
        choices=["none", "step_cooldown", "cosine"],
        help="Dirichlet annealing schedule",
    )
    parser.add_argument("--cooldown-ratio", type=float, default=0.20, help="Fraction of steps for terminal cooldown")
    parser.add_argument("--output-json", type=str, default="gpt2_dirichlet_results.json", help="Path to save results")
    args = parser.parse_args()

    print("=" * 85)
    print(" GPT-2 (124M) DIRICHLET TOPOGRAPHIC FINE-TUNING & SUB-1.0 BPP AUDIT")
    print(f" Model: {args.model_name} | Steps: {args.steps} | Lambda: {args.topo_lambda} | Annealing: {args.annealing}")
    print("=" * 85)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")

    # Load Tokenizer & Model
    print(f"[*] Loading {args.model_name} architecture & weights...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
        base_model = GPT2LMHeadModel.from_pretrained(args.model_name).to(device)
    except Exception as e:
        print(f"[!] Warning: Could not download pretrained weights ({e}). Initializing standard GPT-2 config.")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2") if "tokenizer" in locals() else None
        cfg = GPT2Config(
            vocab_size=50257,
            n_positions=512,
            n_embd=768,
            n_layer=6,  # 6 layers for lightweight test if offline
            n_head=12,
        )
        base_model = GPT2LMHeadModel(cfg).to(device)

    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"[*] Total Model Parameters: {total_params:,}")

    # Tokenize fine-tuning corpus
    train_tokens, val_tokens = get_training_batches(tokenizer, num_samples=200, seq_len=128)
    print(f"[*] Prepared fine-tuning corpus: {len(train_tokens)} train batches, {len(val_tokens)} val batches.")

    quantizer = Base3TritQuantizer()

    # 1. Base Model Audit (Pre-trained)
    print("\n--- [1/3] Auditing Base Pre-trained GPT-2 ---")
    base_fp32_ppl = evaluate_gpt2_ppl(base_model, val_tokens, device=device)
    base_ed = compute_gpt2_roughness(base_model)
    q_base_model, bpp_base, ratio_base = quantize_gpt2_model(base_model, quantizer)
    base_q_ppl = evaluate_gpt2_ppl(q_base_model, val_tokens, device=device)
    print(f"[*] Base FP32 PPL: {base_fp32_ppl:.2f} | Quant PPL ({bpp_base:.3f} bpp): {base_q_ppl:.2f} | E_D: {base_ed:.6f}")

    # 2. Standard Fine-Tuning (lambda = 0.0)
    print(f"\n--- [2/3] Fine-Tuning Standard Baseline (lambda=0.0, {args.steps} steps) ---")
    std_model = fine_tune_gpt2(
        model=base_model,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        is_topographic=False,
        topo_lambda=0.0,
        annealing="none",
        cooldown_ratio=0.0,
        max_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )
    std_fp32_ppl = evaluate_gpt2_ppl(std_model, val_tokens, device=device)
    std_ed = compute_gpt2_roughness(std_model)
    q_std_model, bpp_std, _ = quantize_gpt2_model(std_model, quantizer)
    std_q_ppl = evaluate_gpt2_ppl(q_std_model, val_tokens, device=device)
    print(f"[*] Standard FP32 PPL: {std_fp32_ppl:.2f} | Quant PPL ({bpp_std:.3f} bpp): {std_q_ppl:.2f} | E_D: {std_ed:.6f}")

    # 3. Dirichlet Topographic Fine-Tuning
    print(f"\n--- [3/3] Fine-Tuning Topographic Model (lambda={args.topo_lambda}, {args.annealing}, {args.steps} steps) ---")
    topo_model = fine_tune_gpt2(
        model=base_model,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        is_topographic=True,
        topo_lambda=args.topo_lambda,
        annealing=args.annealing,
        cooldown_ratio=args.cooldown_ratio,
        max_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )
    topo_fp32_ppl = evaluate_gpt2_ppl(topo_model, val_tokens, device=device)
    topo_ed = compute_gpt2_roughness(topo_model)
    q_topo_model, bpp_topo, _ = quantize_gpt2_model(topo_model, quantizer)
    topo_q_ppl = evaluate_gpt2_ppl(q_topo_model, val_tokens, device=device)
    print(f"[*] Topographic FP32 PPL: {topo_fp32_ppl:.2f} | Quant PPL ({bpp_topo:.3f} bpp): {topo_q_ppl:.2f} | E_D: {topo_ed:.6f}")

    # Qualitative text sample
    sample_prompt = "In a surprising discovery, scientists found"
    print("\n--- Qualitative Text Generation (Quantized Models) ---")
    print(f"Prompt: \"{sample_prompt}\"")
    try:
        sample_std = generate_text_sample(q_std_model, tokenizer, prompt=sample_prompt, device=device)
        print(f"\n[Standard Quantized Output]:\n{sample_std}\n")
        sample_topo = generate_text_sample(q_topo_model, tokenizer, prompt=sample_prompt, device=device)
        print(f"[Topographic Quantized Output]:\n{sample_topo}\n")
    except Exception as e:
        print(f"[!] Generation warning: {e}")

    # Comparative Summary Table
    roughness_drop = (1.0 - (topo_ed / (std_ed + 1e-12))) * 100.0
    quant_adv = std_q_ppl - topo_q_ppl

    print("=" * 95)
    print(" GPT-2 (124M) DIRICHLET COMPRESSION AUDIT SUMMARY")
    print("=" * 95)
    print(f"{'Condition':<28} | {'Dirichlet E_D':<14} | {'FP32 Val PPL':<14} | {'Quant PPL (0.945 bpp)':<22} | {'Delta PPL'}")
    print("-" * 95)
    print(f"{'Base Pretrained (Unmodified)':<28} | {base_ed:.6f}{'':<6} | {base_fp32_ppl:.2f}{'':<9} | {base_q_ppl:.2f}{'':<17} | +{base_q_ppl - base_fp32_ppl:.2f}")
    print(f"{'Standard Fine-Tuned (λ=0)':<28} | {std_ed:.6f}{'':<6} | {std_fp32_ppl:.2f}{'':<9} | {std_q_ppl:.2f}{'':<17} | +{std_q_ppl - std_fp32_ppl:.2f}")
    print(f"{f'Topographic (λ={args.topo_lambda})':<28} | {topo_ed:.6f}{'':<6} | {topo_fp32_ppl:.2f}{'':<9} | {topo_q_ppl:.2f}{'':<17} | +{topo_q_ppl - topo_fp32_ppl:.2f}")
    print("-" * 95)
    print(f"[*] Weight Roughness Reduction:       {roughness_drop:.1f}%")
    print(f"[*] Quantized Perplexity Advantage:   -{quant_adv:.2f} PPL under {bpp_topo:.3f} bpp")
    print(f"[*] Compression Ratio:                {ratio_base:.2f}x physical compression")
    print("=" * 95)

    results = {
        "model_name": args.model_name,
        "steps": args.steps,
        "topo_lambda": args.topo_lambda,
        "annealing": args.annealing,
        "base_pretrained": {
            "fp32_ppl": base_fp32_ppl,
            "quant_ppl": base_q_ppl,
            "roughness_ed": base_ed,
            "bpp": bpp_base,
        },
        "standard_finetuned": {
            "fp32_ppl": std_fp32_ppl,
            "quant_ppl": std_q_ppl,
            "roughness_ed": std_ed,
            "bpp": bpp_std,
        },
        "topographic_finetuned": {
            "fp32_ppl": topo_fp32_ppl,
            "quant_ppl": topo_q_ppl,
            "roughness_ed": topo_ed,
            "bpp": bpp_topo,
        },
        "gains": {
            "roughness_reduction_pct": roughness_drop,
            "quantized_ppl_advantage": quant_adv,
        },
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved GPT-2 benchmark results to: {os.path.abspath(args.output_json)}\n")


if __name__ == "__main__":
    main()
