"""
Modal TMLR Multi-Seed Experimental Suite: TinyStories 10M (10,000 steps, N=3 seeds)
===================================================================================
Runs 12 independent NVIDIA A10G GPU containers in parallel covering:
  - 4 controlled conditions (all 10,000 steps, identical hyperparameters & eval set):
      1. std               : Standard AdamW baseline (lambda = 0.0)
      2. rayleigh_lam1     : Scale-invariant Rayleigh + param_weighted (lambda = 1.0)
      3. rayleigh_lam2     : Scale-invariant Rayleigh + param_weighted (lambda = 2.0)
      4. rayleigh_anneal2  : Controlled 10k Annealing (lambda = 2.0 -> 0.0, cooldown_ratio = 0.15)
  - 3 random seeds: [42, 100, 2024] for TMLR-grade mean +/- std reporting.

Inside each container, evaluates:
  - Dense FP32 Validation PPL (25 batches = 204,800 tokens, identical across all runs)
  - Roughness (E_D edges, Rayleigh quotient) & separate Attention vs FFN E(rho <= 0.40)
  - Full 6-point Base-3 TritQ Lloyd-Max Pareto frontier (0.430 to 3.774 bpp)
  - Direct Band-2 ablation (Legacy heuristic vs Lloyd-Max at 1.638 bpp)
  - Full SOTA PTQ baselines (Spatial INT4, Spatial INT2, Group INT2 g=64, INT8 SVD, Spectral INT4)
  - True on-disk .tritq round-trip verification (seed 42)
"""

import dataclasses
import json
import math
import os
import random
import time
from typing import Any, Dict, List

import modal

app = modal.App("dreg-tmlr-10k-suite")

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


def _commit_with_retry(vol: modal.Volume, retries: int = 5) -> None:
    for attempt in range(retries):
        try:
            vol.commit()
            return
        except Exception as exc:
            if attempt == retries - 1:
                print(f"[!] Volume commit warning after {retries} attempts: {exc}")
            else:
                time.sleep(2.0 * (attempt + 1))


@app.function(
    image=image,
    gpu="A10G",
    volumes={CHECKPOINT_DIR: volume},
    timeout=7200,
)
def train_and_eval_tmlr(
    condition: str,
    topo_lambda: float,
    anneal: bool,
    seed: int,
    max_steps: int = 10000,
    lr: float = 5e-4,
    batch_size: int = 32,
    seq_len: int = 256,
    cooldown_ratio: float = 0.15,
) -> Dict[str, Any]:
    import copy

    import numpy as np
    import torch
    import torch.optim as optim
    from datasets import load_dataset
    from tokenizers import Tokenizer

    from dreg import (
        Base3TritQuantizer,
        DirichletLoss,
        StepCooldownScheduler,
        TopographicConfig,
        TopographicTransformer,
        TritQFormat,
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
    tag = f"{condition}_s{seed}"
    print(f"[*] [{tag}] Starting {max_steps}-step run on {device} (lambda={topo_lambda}, anneal={anneal})")

    tok_path = os.path.join(CHECKPOINT_DIR, "tinystories_bpe_4096.json")
    tokenizer = Tokenizer.from_file(tok_path)

    cfg = TopographicConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=256,
        n_heads=4,
        n_layers=8,
        ffn_dim=1024,
        max_seq_len=seq_len,
        topo_lambda=topo_lambda,
        topo_normalization="rayleigh",
        topo_layer_weighting="param_weighted",
    )
    model = TopographicTransformer(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    topo_loss_fn = DirichletLoss(
        weight_decay=topo_lambda,
        normalization="rayleigh",
        layer_weighting="param_weighted",
    )
    scheduler = (
        StepCooldownScheduler(
            loss_fn=topo_loss_fn,
            total_steps=max_steps,
            cooldown_ratio=cooldown_ratio,
            initial_weight_decay=topo_lambda,
            min_weight_decay=0.0,
        )
        if anneal and topo_lambda > 0.0
        else None
    )

    train_stream = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    train_stream = train_stream.shuffle(seed=seed, buffer_size=2048)
    val_stream = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    def data_generator(stream):
        token_buffer = []
        for sample in stream:
            enc = tokenizer.encode(sample["text"])
            token_buffer.extend(enc.ids)
            while len(token_buffer) >= batch_size * (seq_len + 1):
                chunk = token_buffer[: batch_size * (seq_len + 1)]
                token_buffer = token_buffer[batch_size * (seq_len + 1) :]
                t = torch.tensor(chunk, dtype=torch.long).view(batch_size, seq_len + 1)
                yield t[:, :-1], t[:, 1:]

    val_batches = []
    val_iter = iter(data_generator(val_stream))
    for _ in range(25):
        val_batches.append(next(val_iter))

    def eval_ppl(net) -> float:
        net.eval()
        tot_loss, tot_tok = 0.0, 0
        with torch.no_grad():
            for x, y in val_batches:
                x, y = x.to(device), y.to(device)
                _, loss = net(x, y)
                tot_loss += loss.item() * y.numel()
                tot_tok += y.numel()
        net.train()
        return math.exp(min(tot_loss / max(tot_tok, 1), 20.0))

    train_iter = iter(data_generator(train_stream))
    t0 = time.time()
    model.train()
    eval_every = max(500, max_steps // 10)
    history = []

    for step in range(1, max_steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(data_generator(train_stream))
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        _, ce = model(x, y)

        if scheduler is not None:
            cur_lam = scheduler.step()
            loss = ce + topo_loss_fn(model) if cur_lam > 0.0 else ce
        elif topo_lambda > 0.0:
            cur_lam = topo_lambda
            loss = ce + topo_loss_fn(model)
        else:
            cur_lam = 0.0
            loss = ce

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_every == 0 or step == max_steps:
            vp = eval_ppl(model)
            history.append({"step": step, "lambda": cur_lam, "ce": round(ce.item(), 4), "val_ppl": round(vp, 3)})
            print(f"  [{tag}] step {step:5d}/{max_steps} | lam={cur_lam:.2f} | CE={ce.item():.4f} | Val PPL={vp:.2f}")

    train_s = time.time() - t0
    model.eval()
    fp32_ppl = eval_ppl(model)

    # 1. Spectral compaction & roughness audit
    q_helper = Base3TritQuantizer(0.15, 0.40, 1.00)
    attn_low, ffn_low, ed_edges_list, rayleigh_list = [], [], [], []
    with torch.no_grad():
        for p in model.get_linear_projections():
            w = p.weight.detach()
            ed_edges_list.append(dirichlet_energy_2d(w, normalization="edges").item())
            rayleigh_list.append(dirichlet_energy_2d(w, normalization="rayleigh").item())
            s = dct2d(w)
            rho = q_helper.compute_radial_grid(*w.shape, w.device)
            low_pct = ((s[rho <= 0.40] ** 2).sum() / (s ** 2).sum()).item() * 100.0
            if w.shape[0] == w.shape[1]:
                attn_low.append(low_pct)
            else:
                ffn_low.append(low_pct)

    # 2. Full 6-point Pareto Frontier (Lloyd-Max Band-2)
    regimes = [
        ("Ultra-0.43b", (0.08, 0.22, 0.50)),
        ("Sub1-0.91b", (0.10, 0.30, 0.75)),
        ("Calib-1.64b", (0.15, 0.40, 1.00)),
        ("Mid-2.18b", (0.20, 0.55, 1.15)),
        ("HiFi-2.79b", (0.30, 0.70, 1.30)),
        ("Cons-3.77b", (0.45, 0.90, 1.41)),
    ]
    pareto_results: Dict[str, Dict[str, float]] = {}
    for rname, (r0, r1, r2) in regimes:
        qm = copy.deepcopy(model)
        quantizer = Base3TritQuantizer(r0=r0, r1=r1, r2=r2, band2_mode="lloyd_max")
        tot_bits, tot_w = 0.0, 0
        with torch.no_grad():
            for p in qm.get_linear_projections():
                s = dct2d(p.weight.detach())
                pk = quantizer.quantize_matrix(s)
                tot_bits += pk["bpp"] * s.numel()
                tot_w += s.numel()
                p.weight.copy_(idct2d(quantizer.dequantize_matrix(pk, device=device)))
        bpp = tot_bits / max(tot_w, 1)
        qppl = eval_ppl(qm)
        pareto_results[rname] = {"bpp": round(bpp, 3), "ppl": round(qppl, 3)}

    # 3. Direct Band-2 Ablation: Legacy heuristic at 1.638 bpp
    qm_leg = copy.deepcopy(model)
    q_legacy = Base3TritQuantizer(r0=0.15, r1=0.40, r2=1.00, band2_mode="legacy")
    with torch.no_grad():
        for p in qm_leg.get_linear_projections():
            s = dct2d(p.weight.detach())
            pk = q_legacy.quantize_matrix(s)
            p.weight.copy_(idct2d(q_legacy.dequantize_matrix(pk, device=device)))
    legacy_164_ppl = eval_ppl(qm_leg)

    # 4. SOTA PTQ Baselines on the exact same model
    def _eval_spatial_int4() -> float:
        m = copy.deepcopy(model)
        with torch.no_grad():
            for p in m.get_linear_projections():
                w = p.weight.detach()
                scale = w.abs().max() / 7.0 + 1e-8
                p.weight.copy_(torch.clamp(torch.round(w / scale), -8, 7) * scale)
        return eval_ppl(m)

    def _eval_spatial_int2() -> float:
        m = copy.deepcopy(model)
        with torch.no_grad():
            for p in m.get_linear_projections():
                w = p.weight.detach()
                scale = w.abs().mean() * 1.25 + 1e-8
                p.weight.copy_(torch.clamp(torch.round(w / scale), -1, 1) * scale)
        return eval_ppl(m)

    def _eval_spatial_int2_g64() -> float:
        m = copy.deepcopy(model)
        with torch.no_grad():
            for p in m.get_linear_projections():
                w = p.weight.detach()
                orig = w.shape
                wg = w.reshape(-1, 64)
                scale = wg.abs().mean(dim=1, keepdim=True) * 1.25 + 1e-8
                p.weight.copy_((torch.clamp(torch.round(wg / scale), -1, 1) * scale).reshape(orig))
        return eval_ppl(m)

    def _eval_svd_int8(target_bpp: float = 1.64) -> float:
        m = copy.deepcopy(model)
        with torch.no_grad():
            for p in m.get_linear_projections():
                w = p.weight.detach()
                rows, cols = w.shape
                rank = max(1, min(min(rows, cols), int(math.floor((target_bpp / 8.0) * rows * cols / (rows + cols)))))
                u, sv, vh = torch.linalg.svd(w, full_matrices=False)
                u_k = u[:, :rank] * sv[:rank].sqrt().unsqueeze(0)
                v_k = sv[:rank].sqrt().unsqueeze(1) * vh[:rank, :]
                su = u_k.abs().max() / 127.0 + 1e-8
                sv_s = v_k.abs().max() / 127.0 + 1e-8
                u_q = torch.clamp(torch.round(u_k / su), -127, 127) * su
                v_q = torch.clamp(torch.round(v_k / sv_s), -127, 127) * sv_s
                p.weight.copy_(u_q @ v_q)
        return eval_ppl(m)

    def _eval_spectral_int4() -> float:
        m = copy.deepcopy(model)
        with torch.no_grad():
            for p in m.get_linear_projections():
                s = dct2d(p.weight.detach())
                scale = s.abs().max() / 7.0 + 1e-8
                p.weight.copy_(idct2d(torch.clamp(torch.round(s / scale), -8, 7) * scale))
        return eval_ppl(m)

    baselines = {
        "spatial_int4_4.00b": round(_eval_spatial_int4(), 3),
        "spatial_int2_2.00b": round(_eval_spatial_int2(), 3),
        "spatial_int2_g64_2.25b": round(_eval_spatial_int2_g64(), 3),
        "svd_int8_1.64b": round(_eval_svd_int8(1.64), 3),
        "spectral_int4_4.00b": round(_eval_spectral_int4(), 3),
        "tritq_legacy_1.64b": round(legacy_164_ppl, 3),
    }

    # 5. Save checkpoint and (for seed=42) verify true on-disk .tritq round-trip
    ckpt_base = os.path.join(CHECKPOINT_DIR, f"tinystories_10M_tmlr_{tag}_{max_steps}s")
    torch.save(model.state_dict(), f"{ckpt_base}.pt")

    tritq_disk_bytes = None
    tritq_roundtrip_ppl = None
    if seed == 42:
        q_calib = Base3TritQuantizer(r0=0.15, r1=0.40, r2=1.00, band2_mode="lloyd_max")
        tritq_path = f"{ckpt_base}.tritq"
        TritQFormat.save(tritq_path, model.state_dict(), dataclasses.asdict(cfg), quantizer=q_calib)
        tritq_disk_bytes = os.path.getsize(tritq_path)
        _, rec_weights = TritQFormat.load(tritq_path, device=device)
        m_rt = TopographicTransformer(cfg).to(device)
        m_rt.load_state_dict(rec_weights)
        tritq_roundtrip_ppl = round(eval_ppl(m_rt), 3)

    res = {
        "condition": condition,
        "topo_lambda": topo_lambda,
        "anneal": anneal,
        "cooldown_ratio": cooldown_ratio if anneal else 0.0,
        "seed": seed,
        "max_steps": max_steps,
        "train_time_s": round(train_s, 1),
        "fp32_ppl": round(fp32_ppl, 3),
        "mean_ed_edges": round(float(np.mean(ed_edges_list)), 6),
        "mean_rayleigh": round(float(np.mean(rayleigh_list)), 4),
        "attn_low_freq_pct": round(float(np.mean(attn_low)), 2),
        "ffn_low_freq_pct": round(float(np.mean(ffn_low)), 2),
        "pareto": pareto_results,
        "baselines": baselines,
        "tritq_disk_bytes": tritq_disk_bytes,
        "tritq_roundtrip_ppl": tritq_roundtrip_ppl,
        "history": history,
    }

    with open(f"{ckpt_base}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    _commit_with_retry(volume)

    print(
        f"[+] [{tag}] DONE in {train_s/60:.1f}m | FP32={fp32_ppl:.2f} | "
        f"Attn={res['attn_low_freq_pct']:.1f}% FFN={res['ffn_low_freq_pct']:.1f}% | "
        f"0.91b={pareto_results['Sub1-0.91b']['ppl']:.2f} | "
        f"1.64b={pareto_results['Calib-1.64b']['ppl']:.2f} | "
        f"2.18b={pareto_results['Mid-2.18b']['ppl']:.2f} | "
        f"g64-INT2={baselines['spatial_int2_g64_2.25b']:.2f}"
    )
    return res


@app.local_entrypoint()
def main(steps: int = 10000, seeds: str = "42,100,2024"):
    import numpy as np

    seed_list = [int(s.strip()) for s in seeds.split(",")]
    conditions = [
        ("std", 0.0, False),
        ("rayleigh_lam1", 1.0, False),
        ("rayleigh_lam2", 2.0, False),
        ("rayleigh_anneal2", 2.0, True),
    ]

    jobs = []
    for cond_name, lam, anneal in conditions:
        for s in seed_list:
            jobs.append((cond_name, lam, anneal, s, steps))

    print(f"[*] Launching {len(jobs)} parallel A10G runs ({len(conditions)} conditions x {len(seed_list)} seeds, {steps} steps)...")
    raw_results: List[Dict[str, Any]] = list(train_and_eval_tmlr.starmap(jobs))

    # Aggregate mean +/- std across seeds per condition
    summary: Dict[str, Any] = {}
    for cond_name, _, _ in conditions:
        runs = [r for r in raw_results if r["condition"] == cond_name]
        if not runs:
            continue

        def _agg(vals: List[float]) -> Dict[str, float]:
            return {"mean": round(float(np.mean(vals)), 3), "std": round(float(np.std(vals, ddof=1 if len(vals) > 1 else 0)), 3)}

        pareto_agg = {}
        for rname in runs[0]["pareto"]:
            pareto_agg[rname] = {
                "bpp": runs[0]["pareto"][rname]["bpp"],
                **_agg([r["pareto"][rname]["ppl"] for r in runs]),
            }

        base_agg = {}
        for bname in runs[0]["baselines"]:
            base_agg[bname] = _agg([r["baselines"][bname] for r in runs])

        summary[cond_name] = {
            "n_seeds": len(runs),
            "fp32_ppl": _agg([r["fp32_ppl"] for r in runs]),
            "mean_ed_edges": _agg([r["mean_ed_edges"] for r in runs]),
            "mean_rayleigh": _agg([r["mean_rayleigh"] for r in runs]),
            "attn_low_freq_pct": _agg([r["attn_low_freq_pct"] for r in runs]),
            "ffn_low_freq_pct": _agg([r["ffn_low_freq_pct"] for r in runs]),
            "pareto": pareto_agg,
            "baselines": base_agg,
        }

    print("\n" + "=" * 124)
    print(f" TMLR MULTI-SEED SUMMARY (TinyStories 10M, {steps} steps, seeds={seed_list})")
    print("=" * 124)
    print(
        f"{'Condition':<18} | {'FP32 PPL':>13} | {'Attn E<.4':>11} | {'FFN E<.4':>11} | "
        f"{'0.91b PPL':>13} | {'1.64b PPL':>13} | {'2.18b PPL':>13} | {'g64 INT2':>13}"
    )
    print("-" * 124)
    for cond_name, _, _ in conditions:
        s = summary[cond_name]
        print(
            f"{cond_name:<18} | "
            f"{s['fp32_ppl']['mean']:>6.2f}+/-{s['fp32_ppl']['std']:<4.2f} | "
            f"{s['attn_low_freq_pct']['mean']:>5.1f}+/-{s['attn_low_freq_pct']['std']:<3.1f} | "
            f"{s['ffn_low_freq_pct']['mean']:>5.1f}+/-{s['ffn_low_freq_pct']['std']:<3.1f} | "
            f"{s['pareto']['Sub1-0.91b']['mean']:>6.2f}+/-{s['pareto']['Sub1-0.91b']['std']:<4.2f} | "
            f"{s['pareto']['Calib-1.64b']['mean']:>6.2f}+/-{s['pareto']['Calib-1.64b']['std']:<4.2f} | "
            f"{s['pareto']['Mid-2.18b']['mean']:>6.2f}+/-{s['pareto']['Mid-2.18b']['std']:<4.2f} | "
            f"{s['baselines']['spatial_int2_g64_2.25b']['mean']:>6.2f}+/-{s['baselines']['spatial_int2_g64_2.25b']['std']:<4.2f}"
        )
    print("=" * 124)

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"tmlr_{steps}s_multiseed_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"steps": steps, "seeds": seed_list, "summary": summary, "runs": raw_results}, f, indent=2)
    print(f"[+] Saved complete TMLR multi-seed results to {out_path}")
