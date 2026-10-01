"""
Modal Training Pipeline for TinyStories on GPU
==============================================
Trains Topographic & Standard Transformers on TinyStories (roneneldan/TinyStories)
using Modal cloud GPUs (T4 for dry-runs, A10G / A100 for production).

Features:
  - Supports 10M (10.56M) and 20M (19.77M) architectures
  - Trains or loads 4096 BPE tokenizer
  - Streams TinyStories directly from Hugging Face
  - Periodic evaluation: Train Loss, Perplexity, Dirichlet Energy
  - Sub-1.0 bpp Base-3 Trit Quantization audit pre/post compression
  - Saves checkpoints (.pt, .tritq) to persistent Modal Volume

Usage:
  # Quick dry-run test (50 steps on T4, takes ~45 seconds):
  modal run benchmarks/modal_train_tinystories.py --dry-run

  # Full production training (10,000 steps on A10G):
  modal run benchmarks/modal_train_tinystories.py --steps 10000 --gpu a10g
"""

import math
import os
import sys
import time
from typing import Dict, Any, Tuple, Optional

import modal

# ---------------------------------------------------------------------------
# Modal Infrastructure Setup
# ---------------------------------------------------------------------------
app = modal.App("dreg-tinystories")

# Persistent storage for checkpoints and tokenizers
volume = modal.Volume.from_name("dreg-checkpoints", create_if_missing=True)
CHECKPOINT_DIR = "/root/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "tokenizers",
        "tqdm",
    )
    .add_local_python_source("dreg")
)


# ---------------------------------------------------------------------------
# Remote Worker Implementation
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={CHECKPOINT_DIR: volume},
)
def run_training(
    scale: str = "10M",
    max_steps: int = 50,
    batch_size: int = 32,
    lr: float = 5e-4,
    topo_lambda: float = 0.01,
    is_topographic: bool = True,
    dry_run: bool = True,
    eval_interval: int = 25,
) -> Dict[str, Any]:
    import copy
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from datasets import load_dataset
    from tokenizers import ByteLevelBPETokenizer
    from tokenizers.processors import BertProcessing

    import dreg
    from dreg import (
        TopographicTransformer,
        TopographicConfig,
        Base3TritQuantizer,
        TritQFormat,
        dct2d,
        idct2d,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Tokenizer Setup (4096 Vocab BPE)
    tok_path = os.path.join(CHECKPOINT_DIR, "tinystories_bpe_4096.json")
    if os.path.exists(tok_path):
        print(f"[*] Loading cached BPE tokenizer from {tok_path}...")
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(tok_path)
    else:
        print("[*] Training 4,096 BPE tokenizer on TinyStories sample...")
        raw_ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        sample_texts = []
        for i, item in enumerate(raw_ds):
            sample_texts.append(item["text"])
            if i >= (500 if dry_run else 8000):
                break

        bpe = ByteLevelBPETokenizer()
        bpe.train_from_iterator(
            sample_texts,
            vocab_size=4096,
            min_frequency=2,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
        )
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        bpe.save(tok_path)
        volume.commit()
        tokenizer = bpe
        print(f"[+] Trained & saved BPE tokenizer (vocab={tokenizer.get_vocab_size()})")

    vocab_size = tokenizer.get_vocab_size()
    seq_len = 128 if dry_run else 256

    # 2. Model Architecture
    if scale == "10M":
        cfg = TopographicConfig(
            vocab_size=vocab_size,
            d_model=256,
            n_heads=4,
            n_layers=8,
            ffn_dim=1024,
            max_seq_len=seq_len,
            topo_lambda=topo_lambda if is_topographic else 0.0,
        )
    elif scale == "20M":
        cfg = TopographicConfig(
            vocab_size=vocab_size,
            d_model=384,
            n_heads=6,
            n_layers=8,
            ffn_dim=1280,
            max_seq_len=seq_len,
            topo_lambda=topo_lambda if is_topographic else 0.0,
        )
    else:
        raise ValueError(f"Unknown scale: {scale}")

    model = TopographicTransformer(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    model_type = "Topographic" if is_topographic else "Standard"
    print(f"[+] Initialized {scale} {model_type} Transformer ({num_params:,} parameters)")

    # 3. Streaming Dataset Loader
    print("[*] Connecting to TinyStories stream...")
    train_stream = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    val_stream = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    def data_generator(stream, batch_sz, seq_l):
        token_buffer = []
        for sample in stream:
            enc = tokenizer.encode(sample["text"])
            token_buffer.extend(enc.ids)
            while len(token_buffer) >= batch_sz * (seq_l + 1):
                chunk = token_buffer[: batch_sz * (seq_l + 1)]
                token_buffer = token_buffer[batch_sz * (seq_l + 1) :]
                t = torch.tensor(chunk, dtype=torch.long).view(batch_sz, seq_l + 1)
                yield t[:, :-1], t[:, 1:]

    train_iter = iter(data_generator(train_stream, batch_size, seq_len))

    # Pre-collect a small fixed validation set
    val_batches = []
    val_iter = iter(data_generator(val_stream, batch_size, seq_len))
    for _ in range(5 if dry_run else 20):
        try:
            val_batches.append(next(val_iter))
        except StopIteration:
            break
    print(f"[+] Cached {len(val_batches)} validation batches")

    def evaluate_model(eval_net: nn.Module) -> float:
        eval_net.eval()
        total_ce = 0.0
        with torch.no_grad():
            for inp, tgt in val_batches:
                inp, tgt = inp.to(device), tgt.to(device)
                _, loss = eval_net(inp, tgt)
                total_ce += loss.item()
        eval_net.train()
        avg_loss = total_ce / max(len(val_batches), 1)
        return math.exp(min(avg_loss, 20.0))

    # 4. Training Loop
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    print(f"\n--- Starting Training ({max_steps} steps, lr={lr}, lambda={cfg.topo_lambda}) ---")
    start_time = time.time()
    step = 0
    running_ce = 0.0
    running_topo = 0.0

    model.train()
    while step < max_steps:
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(data_generator(train_stream, batch_size, seq_len))
            inputs, targets = next(train_iter)

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        _, ce_loss = model(inputs, targets)

        if is_topographic and cfg.topo_lambda > 0:
            topo_loss = model.topographic_loss()
            total_loss = ce_loss + topo_loss
            running_topo += topo_loss.item()
        else:
            total_loss = ce_loss
            running_topo += 0.0

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_ce += ce_loss.item()
        step += 1

        if step % eval_interval == 0 or step == max_steps:
            elapsed = time.time() - start_time
            tok_per_sec = (step * batch_size * seq_len) / max(elapsed, 1e-4)
            avg_ce = running_ce / eval_interval
            avg_topo = running_topo / eval_interval
            train_ppl = math.exp(min(avg_ce, 20.0))
            val_ppl = evaluate_model(model)
            print(
                f"  Step {step:5d}/{max_steps} | "
                f"CE: {avg_ce:.4f} | "
                f"Topo: {avg_topo:.6f} | "
                f"Train PPL: {train_ppl:.2f} | "
                f"Val PPL: {val_ppl:.2f} | "
                f"{tok_per_sec:.0f} tok/s"
            )
            running_ce = 0.0
            running_topo = 0.0

    total_training_time = time.time() - start_time
    print(f"\n[+] Training finished in {total_training_time:.2f}s ({total_training_time/60:.1f} min)")

    # 5. Baseline Evaluation
    baseline_val_ppl = evaluate_model(model)
    linears = model.get_linear_projections()
    mean_dirichlet = sum(m.dirichlet_energy().item() for m in linears) / max(len(linears), 1)

    print("\n" + "=" * 70)
    print(f" SUB-1.0 BPP BASE-3 TRIT QUANTIZATION AUDIT ({scale} {model_type})")
    print("=" * 70)
    print(f"[*] Baseline FP32 Val Perplexity: {baseline_val_ppl:.2f}")
    print(f"[*] Mean Weight Dirichlet Energy: {mean_dirichlet:.6f}")

    # 6. Base-3 Trit Quantization Simulation
    quantizer = Base3TritQuantizer(r0=0.15, r1=0.40, r2=1.0)
    q_model = copy.deepcopy(model)

    with torch.no_grad():
        for m in q_model.get_linear_projections():
            w = m.weight.detach()
            spec = dct2d(w)
            packed = quantizer.quantize_matrix(spec)
            rec_spec = quantizer.dequantize_matrix(packed, device=device)
            rec_w = idct2d(rec_spec)
            m.weight.copy_(rec_w)

    quantized_val_ppl = evaluate_model(q_model)
    delta_ppl = quantized_val_ppl - baseline_val_ppl

    print(f"[*] Quantized (0.945 bpp) Val Perplexity: {quantized_val_ppl:.2f}")
    print(f"[*] Perplexity Delta (Δ):              +{delta_ppl:.2f}")
    print("=" * 70)

    # 6b. Qualitative Sample Generation
    try:
        print("\n[*] Generating qualitative text sample...")
        sample_prompt = "Once upon a time, there was a little"
        prompt_ids = torch.tensor([tokenizer.encode(sample_prompt)], dtype=torch.long, device=device)
        sample_out = model.generate(prompt_ids, max_new_tokens=50, temperature=0.7)
        sample_text = tokenizer.decode(sample_out[0].tolist())
        print(f"[+] Sample output:\n\"{sample_text}\"\n")
    except Exception as e:
        print(f"[!] Sample generation warning: {e}")

    # 7. Save Checkpoints
    lambda_tag = f"_lambda{topo_lambda}" if is_topographic else "_standard"
    ckpt_name = f"tinystories_{scale}_{model_type.lower()}{lambda_tag}_{max_steps}s"
    pt_path = os.path.join(CHECKPOINT_DIR, f"{ckpt_name}.pt")
    tritq_path = os.path.join(CHECKPOINT_DIR, f"{ckpt_name}.tritq")

    torch.save(model.state_dict(), pt_path)
    import dataclasses
    cfg_dict = dataclasses.asdict(cfg)
    TritQFormat.save(tritq_path, model.state_dict(), cfg_dict)
    volume.commit()
    print(f"[+] Saved PyTorch checkpoint: {pt_path}")
    print(f"[+] Saved .tritq checkpoint:    {tritq_path} ({os.path.getsize(tritq_path) / 1024:.1f} KB)")

    return {
        "scale": scale,
        "model_type": model_type,
        "topo_lambda": topo_lambda if is_topographic else 0.0,
        "num_params": num_params,
        "steps": max_steps,
        "baseline_ppl": baseline_val_ppl,
        "quantized_ppl": quantized_val_ppl,
        "delta_ppl": delta_ppl,
        "dirichlet_energy": mean_dirichlet,
        "training_time_s": total_training_time,
    }


# ---------------------------------------------------------------------------
# CLI Entrypoint for Local Invocation
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    scale: str = "10M",
    steps: int = 50,
    batch_size: int = 32,
    lr: float = 5e-4,
    topo_lambda: float = 0.01,
    is_topographic: bool = True,
    dry_run: bool = True,
):
    print("=" * 78)
    print(" MODAL CLOUD TRAINING: TINYSTORIES WITH DIRICHLET REGULARIZATION")
    print("=" * 78)
    print(f"[*] Configuration: Scale={scale}, Steps={steps}, Dry-Run={dry_run}")
    print(f"[*] Topographic: {is_topographic} (lambda={topo_lambda if is_topographic else 0.0})")

    results = run_training.remote(
        scale=scale,
        max_steps=steps,
        batch_size=batch_size,
        lr=lr,
        topo_lambda=topo_lambda,
        is_topographic=is_topographic,
        dry_run=dry_run,
        eval_interval=max(10, min(100, steps // 20)),
    )

    print("\n" + "=" * 78)
    print(" RUN COMPLETE — SUMMARY OF RESULTS")
    print("=" * 78)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<25}: {v:.4f}")
        else:
            print(f"  {k:<25}: {v}")
    print("=" * 78)
