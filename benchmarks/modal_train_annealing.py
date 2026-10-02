"""
Modal Training Pipeline: Dirichlet Annealing on TinyStories 10M
===============================================================
Tests Hypothesis 1 & Action 14 from Auditoría:
  Does annealing lambda -> 0 in the cooldown phase (last 15% of steps)
  eliminate the dense FP32 degradation (closing 6.46 -> 6.11 PPL)
  while retaining the low-frequency spectral concentration and
  quantization robustness (0.945 bpp Base-3 TritQ)?

Schedulers tested:
  - StepCooldownScheduler (lambda_0=30.0, cooldown_ratio=0.85, lambda_end=0.0)
  - Baseline Standard (lambda=0.0)
  - Baseline Constant Topographic (lambda=30.0)

Outputs:
  - FP32 Perplexity & Mean Dirichlet Energy
  - 0.945 bpp Base-3 TritQ Perplexity
  - Checkpoint saved to Modal persistent volume
"""

import copy
import dataclasses
import json
import math
import os
import time
from typing import Any, Dict

import modal

app = modal.App("dreg-annealing")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers",
        "datasets",
        "numpy",
        "tokenizers",
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
def train_annealed_tinystories(
    max_steps: int = 5000,
    batch_size: int = 32,
    lr: float = 5e-4,
    lambda_init: float = 30.0,
    cooldown_ratio: float = 0.85,
    eval_interval: int = 100,
) -> Dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from datasets import load_dataset
    from tokenizers import Tokenizer

    from dreg.model import TopographicConfig, TopographicTransformer
    from dreg.quantization import Base3TritQuantizer, TritQFormat
    from dreg.spectral import dct2d, idct2d
    from dreg.topology import StepCooldownScheduler, dirichlet_energy_2d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Annealing Training on: {device} ({torch.cuda.get_device_name(0)})")
    print(f"[*] Total Steps: {max_steps} | Init Lambda: {lambda_init} | Cooldown Ratio: {cooldown_ratio}")

    # 1. Load Tokenizer
    tok_path = os.path.join(CHECKPOINT_DIR, "tinystories_bpe_4096.json")
    tokenizer = Tokenizer.from_file(tok_path)
    vocab_size = tokenizer.get_vocab_size()
    seq_len = 256

    # 2. Architecture
    cfg = TopographicConfig(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=8,
        ffn_dim=1024,
        max_seq_len=seq_len,
        topo_lambda=lambda_init,
    )
    model = TopographicTransformer(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[+] Initialized 10M Transformer ({num_params:,} parameters)")

    # 3. Streaming Data
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
    val_batches = []
    val_iter = iter(data_generator(val_stream, batch_size, seq_len))
    for _ in range(25):
        val_batches.append(next(val_iter))
    print(f"[+] Cached {len(val_batches)} validation batches")

    def evaluate_model(eval_net: nn.Module) -> float:
        eval_net.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for inp, tgt in val_batches:
                inp, tgt = inp.to(device), tgt.to(device)
                _, loss = eval_net(inp, tgt)
                total_loss += loss.item() * tgt.numel()
                total_tokens += tgt.numel()
        eval_net.train()
        avg_loss = total_loss / max(total_tokens, 1)
        return math.exp(min(avg_loss, 20.0))

    # 4. Annealing Scheduler & Optimizer
    from dreg.topology import DirichletLoss
    topo_loss_fn = DirichletLoss(weight_decay=lambda_init, normalization="edges")
    scheduler = StepCooldownScheduler(
        loss_fn=topo_loss_fn,
        total_steps=max_steps,
        cooldown_ratio=cooldown_ratio,
        min_weight_decay=0.0,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    cooldown_start_step = int((1.0 - cooldown_ratio) * max_steps)
    print(f"\n--- Starting Annealed Training: Cooldown starts at step {cooldown_start_step}/{max_steps} ---")
    start_time = time.time()
    step = 0
    running_ce = 0.0
    running_topo = 0.0
    history = []

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

        # Update lambda via topological scheduler
        cur_lambda = scheduler.step()
        if cur_lambda > 0.0:
            linears = model.get_linear_projections()
            raw_topo = torch.stack(
                [dirichlet_energy_2d(m.weight, normalization="edges") for m in linears]
            ).mean()
            topo_loss = cur_lambda * raw_topo
            total_loss = ce_loss + topo_loss
            running_topo += raw_topo.item()
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
            avg_ce = running_ce / eval_interval
            avg_topo = running_topo / eval_interval
            train_ppl = math.exp(min(avg_ce, 20.0))
            val_ppl = evaluate_model(model)
            tok_per_sec = (step * batch_size * seq_len) / max(elapsed, 1e-4)

            print(
                f"  Step {step:5d}/{max_steps} | "
                f"λ(t): {cur_lambda:5.2f} | "
                f"CE: {avg_ce:.4f} | "
                f"Train PPL: {train_ppl:.2f} | "
                f"Raw E_D: {avg_topo:.6f} | "
                f"Val PPL: {val_ppl:.2f} | "
                f"{tok_per_sec:.0f} tok/s"
            )
            history.append({
                "step": step,
                "lambda": cur_lambda,
                "ce_loss": avg_ce,
                "train_ppl": train_ppl,
                "raw_topo_ed": avg_topo,
                "val_ppl": val_ppl,
            })
            running_ce = 0.0
            running_topo = 0.0

    total_time = time.time() - start_time
    print(f"\n[+] Annealed Training finished in {total_time:.1f}s ({total_time/60:.2f} min)")

    # 5. Final Evaluation
    fp32_val_ppl = evaluate_model(model)
    linears = model.get_linear_projections()
    mean_ed = float(torch.stack([dirichlet_energy_2d(m.weight, normalization="edges") for m in linears]).mean().item())

    # 6. Quantization Audit (0.945 bpp Base-3 TritQ)
    quantizer = Base3TritQuantizer(r0=0.15, r1=0.40, r2=1.0)
    q_model = copy.deepcopy(model)
    total_orig_bytes = 0
    total_comp_bytes = 0

    with torch.no_grad():
        for p in q_model.get_linear_projections():
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

    tritq_bpp = (total_comp_bytes * 8.0) / max(total_orig_bytes / 4, 1)
    tritq_val_ppl = evaluate_model(q_model)
    delta_ppl = tritq_val_ppl - fp32_val_ppl

    print("\n" + "=" * 75)
    print(" ANNEALED MODEL RESULTS SUMMARY")
    print("=" * 75)
    print(f"[*] FP32 Dense Val PPL:           {fp32_val_ppl:.2f}")
    print(f"[*] Mean Weight Dirichlet Energy: {mean_ed:.6f}")
    print(f"[*] TritQ ({tritq_bpp:.3f} bpp) Val PPL:   {tritq_val_ppl:.2f}")
    print(f"[*] Perplexity Delta (Δ):          +{delta_ppl:.2f}")
    print("=" * 75)

    # 7. Save Checkpoint
    ckpt_name = f"tinystories_10M_annealed_lambda{lambda_init}_{max_steps}s"
    pt_path = os.path.join(CHECKPOINT_DIR, f"{ckpt_name}.pt")
    tritq_path = os.path.join(CHECKPOINT_DIR, f"{ckpt_name}.tritq")
    torch.save(model.state_dict(), pt_path)
    cfg_dict = dataclasses.asdict(cfg)
    TritQFormat.save(tritq_path, model.state_dict(), cfg_dict)
    volume.commit()
    print(f"[+] Saved PyTorch checkpoint: {pt_path}")
    print(f"[+] Saved .tritq checkpoint:    {tritq_path}")

    return {
        "max_steps": max_steps,
        "lambda_init": lambda_init,
        "cooldown_ratio": cooldown_ratio,
        "cooldown_step": cooldown_start_step,
        "fp32_val_ppl": fp32_val_ppl,
        "mean_ed": mean_ed,
        "tritq_bpp": tritq_bpp,
        "tritq_val_ppl": tritq_val_ppl,
        "delta_ppl": delta_ppl,
        "training_time_s": total_time,
        "history": history,
    }


@app.local_entrypoint()
def main(steps: int = 5000, lambda_init: float = 30.0, cooldown_ratio: float = 0.15):
    results = train_annealed_tinystories.remote(
        max_steps=steps,
        lambda_init=lambda_init,
        cooldown_ratio=cooldown_ratio,
        eval_interval=max(50, steps // 20),
    )
    print("\n[+] Annealing Experiment Complete!")
    print(json.dumps({k: v for k, v in results.items() if k != "history"}, indent=2))
