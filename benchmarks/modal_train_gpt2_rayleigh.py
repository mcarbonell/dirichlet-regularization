"""
Modal GPU Benchmark: Pre-Trained GPT-2 (124M) Rayleigh + Param-Weighted Adaptation
==================================================================================
Runs 5 parallel NVIDIA A10G containers on pre-trained OpenAI GPT-2 (124M, 48 dense
2D Conv1D projections = 84.9M weights) to test whether properly calibrated scale-invariant
Rayleigh Dirichlet regularization + parameter-weighted aggregation can topographically
reorganize an already-pretrained 124M LLM for extreme Base-3 TritQ compression.

Conditions (all 1,000 steps, batch_size=8, seq_len=256, lr=2e-4 cosine, seed=42):
  1. std_ft           : Standard fine-tuning control (lambda = 0.0)
  2. rayleigh_0.5     : Rayleigh + param_weighted (lambda = 0.5)
  3. rayleigh_1.0     : Rayleigh + param_weighted (lambda = 1.0)
  4. rayleigh_2.0     : Rayleigh + param_weighted (lambda = 2.0)
  5. rayleigh_anneal2 : Rayleigh + param_weighted (lambda = 2.0 -> 0.0, cooldown_ratio=0.15)
Also evaluates Base Pretrained GPT-2 (0 steps) on the exact same validation split.
"""

import json
import math
import os
import random
import time
from typing import Any, Dict, List, Tuple

import modal

app = modal.App("dreg-gpt2-rayleigh")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets",
        "tokenizers",
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
    timeout=3600,
)
def run_gpt2_condition(
    condition: str,
    topo_lambda: float,
    anneal: bool,
    steps: int = 1000,
    batch_size: int = 8,
    seq_len: int = 256,
    lr: float = 2e-4,
    cooldown_ratio: float = 0.15,
    seed: int = 42,
    eval_base: bool = False,
    normalization: str = "rayleigh_sg",
) -> Dict[str, Any]:
    import copy

    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    from dreg import (
        Base3TritQuantizer,
        DirichletLoss,
        StepCooldownScheduler,
        dct2d,
        dirichlet_energy_2d,
        idct2d,
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] [GPT-2 {condition}] Device: {device} | steps={steps}, lr={lr}, lambda={topo_lambda}, anneal={anneal}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    def get_dense_modules(m: GPT2LMHeadModel) -> List[Tuple[str, torch.nn.Module]]:
        mods = []
        for name, mod in m.transformer.h.named_modules():
            w = getattr(mod, "weight", None)
            if w is not None and isinstance(w, torch.Tensor) and w.dim() == 2:
                mods.append((name, mod))
        return mods

    dense_mods = get_dense_modules(model)
    assert len(dense_mods) == 48, f"Expected 48 GPT-2 dense projections, got {len(dense_mods)}"

    raw_train = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    raw_val = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    def stream_chunks(stream):
        buf: List[int] = []
        for item in stream:
            buf.extend(tokenizer.encode(item["text"]))
            while len(buf) >= batch_size * seq_len:
                chunk = buf[: batch_size * seq_len]
                buf = buf[batch_size * seq_len :]
                yield torch.tensor(chunk, dtype=torch.long).view(batch_size, seq_len)

    val_batches = []
    val_iter = iter(stream_chunks(raw_val))
    for _ in range(25):  # 25 * 8 * 256 = 51,200 validation tokens
        val_batches.append(next(val_iter))

    def eval_ppl(net: GPT2LMHeadModel) -> float:
        net.eval()
        tot_loss, tot_tok = 0.0, 0
        with torch.no_grad():
            for batch in val_batches:
                batch = batch.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = net(batch, labels=batch)
                n = batch.numel()
                tot_loss += out.loss.float().item() * n
                tot_tok += n
        net.train()
        return math.exp(min(tot_loss / max(tot_tok, 1), 20.0))

    def audit_model(net: GPT2LMHeadModel) -> Dict[str, Any]:
        net.eval()
        fp32_ppl = eval_ppl(net)

        q_helper = Base3TritQuantizer(0.15, 0.40, 1.00)
        attn_low, ffn_low, ed_edges_list, rayleigh_list = [], [], [], []
        with torch.no_grad():
            for name, mod in get_dense_modules(net):
                w = mod.weight.detach().float()
                ed_edges_list.append(dirichlet_energy_2d(w, normalization="edges").item())
                rayleigh_list.append(dirichlet_energy_2d(w, normalization="rayleigh").item())
                s = dct2d(w)
                rho = q_helper.compute_radial_grid(*w.shape, w.device)
                low_pct = ((s[rho <= 0.40] ** 2).sum() / (s ** 2).sum()).item() * 100.0
                if "attn" in name:
                    attn_low.append(low_pct)
                else:
                    ffn_low.append(low_pct)

        regimes = [
            ("Sub1-0.91b", (0.10, 0.30, 0.75)),
            ("Calib-1.64b", (0.15, 0.40, 1.00)),
            ("Mid-2.18b", (0.20, 0.55, 1.15)),
            ("HiFi-2.79b", (0.30, 0.70, 1.30)),
            ("Cons-3.77b", (0.45, 0.90, 1.41)),
        ]
        pareto = {}
        for rname, (r0, r1, r2) in regimes:
            qm = copy.deepcopy(net)
            quantizer = Base3TritQuantizer(r0=r0, r1=r1, r2=r2, band2_mode="lloyd_max")
            tot_bits, tot_w = 0.0, 0
            with torch.no_grad():
                for _, mod in get_dense_modules(qm):
                    s = dct2d(mod.weight.detach().float())
                    pk = quantizer.quantize_matrix(s)
                    tot_bits += pk["bpp"] * s.numel()
                    tot_w += s.numel()
                    mod.weight.data.copy_(idct2d(quantizer.dequantize_matrix(pk, device=device)))
            bpp = tot_bits / max(tot_w, 1)
            pareto[rname] = {"bpp": round(bpp, 3), "ppl": round(eval_ppl(qm), 3)}

        # Spatial baselines
        def _eval_sp_int4() -> float:
            qm = copy.deepcopy(net)
            with torch.no_grad():
                for _, mod in get_dense_modules(qm):
                    w = mod.weight.detach().float()
                    sc = w.abs().max() / 7.0 + 1e-8
                    mod.weight.data.copy_(torch.clamp(torch.round(w / sc), -8, 7) * sc)
            return eval_ppl(qm)

        def _eval_sp_int2() -> float:
            qm = copy.deepcopy(net)
            with torch.no_grad():
                for _, mod in get_dense_modules(qm):
                    w = mod.weight.detach().float()
                    sc = w.abs().mean() * 1.25 + 1e-8
                    mod.weight.data.copy_(torch.clamp(torch.round(w / sc), -1, 1) * sc)
            return eval_ppl(qm)

        def _eval_sp_int2_g64() -> float:
            qm = copy.deepcopy(net)
            with torch.no_grad():
                for _, mod in get_dense_modules(qm):
                    w = mod.weight.detach().float()
                    orig = w.shape
                    wg = w.reshape(-1, 64)
                    sc = wg.abs().mean(dim=1, keepdim=True) * 1.25 + 1e-8
                    mod.weight.data.copy_((torch.clamp(torch.round(wg / sc), -1, 1) * sc).reshape(orig))
            return eval_ppl(qm)

        return {
            "fp32_ppl": round(fp32_ppl, 3),
            "mean_ed_edges": round(float(np.mean(ed_edges_list)), 6),
            "mean_rayleigh": round(float(np.mean(rayleigh_list)), 4),
            "attn_low_freq_pct": round(float(np.mean(attn_low)), 2),
            "ffn_low_freq_pct": round(float(np.mean(ffn_low)), 2),
            "pareto": pareto,
            "baselines": {
                "spatial_int4_4.00b": round(_eval_sp_int4(), 3),
                "spatial_int2_2.00b": round(_eval_sp_int2(), 3),
                "spatial_int2_g64_2.25b": round(_eval_sp_int2_g64(), 3),
            },
        }

    base_audit = audit_model(model) if eval_base else None
    if base_audit is not None:
        print(
            f"[+] [Base Pretrained GPT-2] FP32={base_audit['fp32_ppl']:.2f} | "
            f"Rayleigh={base_audit['mean_rayleigh']:.4f} | "
            f"Attn={base_audit['attn_low_freq_pct']:.1f}% FFN={base_audit['ffn_low_freq_pct']:.1f}% | "
            f"1.64b={base_audit['pareto']['Calib-1.64b']['ppl']:.2f} | "
            f"2.18b={base_audit['pareto']['Mid-2.18b']['ppl']:.2f}"
        )

    # Fine-tune with warmup + cosine LR schedule
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup_steps = min(50, steps // 10)

    def lr_lambda(cur_step: int) -> float:
        if cur_step < warmup_steps:
            return float(cur_step + 1) / float(max(1, warmup_steps))
        prog = float(cur_step - warmup_steps) / float(max(1, steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    topo_loss_fn = DirichletLoss(
        weight_decay=topo_lambda,
        normalization=normalization,
        layer_weighting="param_weighted",
    )
    topo_sched = (
        StepCooldownScheduler(
            loss_fn=topo_loss_fn,
            total_steps=steps,
            cooldown_ratio=cooldown_ratio,
            initial_weight_decay=topo_lambda,
            min_weight_decay=0.0,
        )
        if anneal and topo_lambda > 0.0
        else None
    )

    dense_mod_list = [mod for _, mod in dense_mods]
    train_iter = iter(stream_chunks(raw_train))
    t0 = time.time()
    eval_every = max(100, steps // 5)

    for step in range(1, steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(stream_chunks(raw_train))
            batch = next(train_iter)
        batch = batch.to(device)

        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch, labels=batch)
            ce_loss = out.loss.float()

        if topo_sched is not None:
            cur_lam = topo_sched.step()
            loss = ce_loss + topo_loss_fn(dense_mod_list) if cur_lam > 0.0 else ce_loss
        elif topo_lambda > 0.0:
            cur_lam = topo_lambda
            loss = ce_loss + topo_loss_fn(dense_mod_list)
        else:
            cur_lam = 0.0
            loss = ce_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        lr_sched.step()

        if step % eval_every == 0 or step == steps:
            vp = eval_ppl(model)
            print(f"  [GPT-2 {condition}] step {step:4d}/{steps} | lam={cur_lam:.2f} | CE={ce_loss.item():.4f} | Val PPL={vp:.2f}")

    train_s = time.time() - t0
    final_audit = audit_model(model)
    print(
        f"[+] [GPT-2 {condition}] DONE in {train_s:.1f}s | FP32={final_audit['fp32_ppl']:.2f} | "
        f"Rayleigh={final_audit['mean_rayleigh']:.4f} | "
        f"Attn={final_audit['attn_low_freq_pct']:.1f}% FFN={final_audit['ffn_low_freq_pct']:.1f}% | "
        f"1.64b={final_audit['pareto']['Calib-1.64b']['ppl']:.2f} | "
        f"2.18b={final_audit['pareto']['Mid-2.18b']['ppl']:.2f} | "
        f"g64-INT2={final_audit['baselines']['spatial_int2_g64_2.25b']:.2f}"
    )

    return {
        "condition": condition,
        "topo_lambda": topo_lambda,
        "anneal": anneal,
        "steps": steps,
        "lr": lr,
        "train_time_s": round(train_s, 1),
        "base_pretrained": base_audit,
        **final_audit,
    }


@app.local_entrypoint()
def main(
    steps: int = 1500,
    lr: float = 3e-4,
    lambdas: str = "0.0,25.0,75.0,200.0,500.0",
    anneal_lambda: float = 200.0,
    normalization: str = "rayleigh_sg",
):
    lam_list = [float(x.strip()) for x in lambdas.split(",")]
    jobs = []
    for i, lam in enumerate(lam_list):
        name = "std_ft" if lam == 0.0 else f"{normalization}_{lam:g}"
        jobs.append((name, lam, False, steps, 8, 256, lr, 0.15, 42, i == 0, normalization))
    if anneal_lambda > 0.0:
        jobs.append((f"{normalization}_anneal_{anneal_lambda:g}", anneal_lambda, True, steps, 8, 256, lr, 0.15, 42, False, normalization))

    print(
        f"[*] Launching {len(jobs)} parallel GPT-2 (124M) {normalization} adaptation runs on Modal A10G "
        f"(steps={steps}, lr={lr}, mse_clip + lloyd_max)..."
    )
    results: List[Dict[str, Any]] = list(run_gpt2_condition.starmap(jobs))

    base_res = results[0].pop("base_pretrained", None)
    for r in results[1:]:
        r.pop("base_pretrained", None)

    print("\n" + "=" * 124)
    print(f" GPT-2 (124M) {normalization.upper()} + PARAM-WEIGHTED + MSE-CLIP + LLOYD-MAX ({steps} steps, lr={lr})")
    print("=" * 124)
    print(
        f"{'Variant':<24} | {'FP32':>6} | {'Rayleigh':>8} | {'Attn<.4':>7} | {'FFN<.4':>7} | "
        f"{'0.91b':>9} | {'1.64b':>9} | {'2.18b':>9} | {'2.79b':>7} | {'3.77b':>7}"
    )
    print("-" * 124)
    rows = ([{"condition": "base_pretrained", **base_res}] if base_res else []) + results
    for r in rows:
        p = r["pareto"]
        print(
            f"{r['condition']:<24} | {r['fp32_ppl']:>6.2f} | {r['mean_rayleigh']:>8.4f} | "
            f"{r['attn_low_freq_pct']:>6.1f}% | {r['ffn_low_freq_pct']:>6.1f}% | "
            f"{p['Sub1-0.91b']['ppl']:>9.2f} | {p['Calib-1.64b']['ppl']:>9.2f} | "
            f"{p['Mid-2.18b']['ppl']:>9.2f} | {p['HiFi-2.79b']['ppl']:>7.2f} | "
            f"{p['Cons-3.77b']['ppl']:>7.2f}"
        )
    print("=" * 124)

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"gpt2_124m_{normalization}_{steps}s_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "gpt2-124m",
                "normalization": normalization,
                "steps": steps,
                "lr": lr,
                "base_pretrained": base_res,
                "runs": results,
            },
            f,
            indent=2,
        )
    print(f"[+] Saved GPT-2 results to {out_path}")
