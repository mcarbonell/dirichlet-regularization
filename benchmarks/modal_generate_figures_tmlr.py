"""
Generate Publication-Quality TMLR Figures (300 DPI) & Qualitative 1.64-bpp Generations
======================================================================================
Loads the converged 10,000-step checkpoints (seed=42) from Modal volume `dreg-checkpoints`:
  - Standard (std_s42)
  - Legacy Dirichlet (topographic_lambda30.0_10000s, numel + uniform) [for FFN starvation comparison]
  - Rayleigh Annealed (rayleigh_anneal2_s42, rayleigh + param_weighted, cooldown_ratio=0.15)

Generates:
  - Figure 1: Spatial weight heatmaps (FFN gate_proj + Attention q_proj) for Standard vs Rayleigh Annealed
  - Figure 2: 2D-DCT cumulative spectral energy (FFN 256x1024 & Attn 256x256) showing Standard vs
              Legacy numel+uniform (FFN starvation!) vs Rayleigh+param_weighted (91.4% Attn / 94.1% FFN!)
  - Figure 3 (generated locally from verified JSON): 2-panel Rayleigh calibration + 10k N=3 Pareto frontier
  - Qualitative TinyStories continuations at FP32 and 1.638 bpp (Base-3 TritQ mse_clip + lloyd_max)
"""

import json
import os

import modal

app = modal.App("dreg-generate-figures-tmlr")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "matplotlib", "numpy", "tokenizers>=0.13.0")
    .add_local_python_source("dreg")
)

volume = modal.Volume.from_name("dreg-checkpoints")
CHECKPOINT_DIR = "/root/checkpoints"


@app.function(image=image, volumes={CHECKPOINT_DIR: volume}, timeout=600)
def generate_remote_figures_and_samples():
    import copy
    import io

    import matplotlib
    import numpy as np
    import torch

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tokenizers import Tokenizer

    from dreg import Base3TritQuantizer, TopographicConfig, TopographicTransformer, dct2d, idct2d

    tok = Tokenizer.from_file(os.path.join(CHECKPOINT_DIR, "tinystories_bpe_4096.json"))
    cfg = TopographicConfig(
        vocab_size=tok.get_vocab_size(),
        d_model=256,
        n_heads=4,
        n_layers=8,
        ffn_dim=1024,
        max_seq_len=256,
    )

    std_m = TopographicTransformer(cfg)
    std_m.load_state_dict(
        torch.load(os.path.join(CHECKPOINT_DIR, "tinystories_10M_tmlr_std_s42_10000s.pt"), map_location="cpu")
    )
    std_m.eval()

    ray_m = TopographicTransformer(cfg)
    ray_m.load_state_dict(
        torch.load(
            os.path.join(CHECKPOINT_DIR, "tinystories_10M_tmlr_rayleigh_anneal2_s42_10000s.pt"),
            map_location="cpu",
        )
    )
    ray_m.eval()

    leg_path = os.path.join(CHECKPOINT_DIR, "tinystories_10M_topographic_lambda30.0_10000s.pt")
    leg_m = None
    if os.path.exists(leg_path):
        leg_m = TopographicTransformer(cfg)
        leg_m.load_state_dict(torch.load(leg_path, map_location="cpu"))
        leg_m.eval()

    # Quantize both std_m and ray_m to 1.638 bpp (mse_clip + lloyd_max) for qualitative samples
    def quantize_copy(m):
        qm = copy.deepcopy(m)
        q = Base3TritQuantizer(r0=0.15, r1=0.40, r2=1.00, band2_mode="lloyd_max", band01_mode="mse_clip")
        with torch.no_grad():
            for p in qm.get_linear_projections():
                pk = q.quantize_matrix(dct2d(p.weight.detach()))
                p.weight.copy_(idct2d(q.dequantize_matrix(pk, device=torch.device("cpu"))))
        qm.eval()
        return qm

    std_q = quantize_copy(std_m)
    ray_q = quantize_copy(ray_m)

    prompts = [
        "Once upon a time, there was a little girl named Lily.",
        "One day, Tim went to the park with his red ball.",
    ]
    generations = {}
    for prompt in prompts:
        ids = torch.tensor([tok.encode(prompt).ids], dtype=torch.long)
        with torch.no_grad():
            torch.manual_seed(0)
            g_std_fp32 = tok.decode(std_m.generate(ids, max_new_tokens=50, temperature=0.0)[0].tolist())
            g_ray_fp32 = tok.decode(ray_m.generate(ids, max_new_tokens=50, temperature=0.0)[0].tolist())
            g_std_q = tok.decode(std_q.generate(ids, max_new_tokens=50, temperature=0.0)[0].tolist())
            g_ray_q = tok.decode(ray_q.generate(ids, max_new_tokens=50, temperature=0.0)[0].tolist())
        generations[prompt] = {
            "std_fp32": g_std_fp32,
            "rayleigh_anneal_fp32": g_ray_fp32,
            "std_1.64bpp": g_std_q,
            "rayleigh_anneal_1.64bpp": g_ray_q,
        }

    # Extract Layer 3 representative matrices
    w_std_ffn = std_m.blocks[2].mlp.gate_proj.weight.detach().float()
    w_ray_ffn = ray_m.blocks[2].mlp.gate_proj.weight.detach().float()
    w_std_attn = std_m.blocks[2].attn.q_proj.weight.detach().float()
    w_ray_attn = ray_m.blocks[2].attn.q_proj.weight.detach().float()

    # Figure 1: Spatial weight heatmaps (shared symmetric percentile color scale per row for fair visual contrast)
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))

    v_ffn_std = float(np.percentile(np.abs(w_std_ffn[:128, :128].numpy()), 99))
    v_ffn_ray = float(np.percentile(np.abs(w_ray_ffn[:128, :128].numpy()), 99))
    im0 = axes[0, 0].imshow(w_std_ffn[:128, :128].numpy(), cmap="RdBu_r", vmin=-v_ffn_std, vmax=v_ffn_std, aspect="auto")
    axes[0, 0].set_title("Standard AdamW — FFN Gate (128x128 crop)\nUnstructured White-Noise Geometry", fontweight="bold")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(w_ray_ffn[:128, :128].numpy(), cmap="RdBu_r", vmin=-v_ffn_ray, vmax=v_ffn_ray, aspect="auto")
    axes[0, 1].set_title(
        "Dirichlet Rayleigh Annealed — FFN Gate (128x128 crop)\nEmergent Smooth Cortical Manifold (94.1% E<=0.40)",
        fontweight="bold",
        color="#00695C",
    )
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    v_att_std = float(np.percentile(np.abs(w_std_attn.numpy()), 99))
    v_att_ray = float(np.percentile(np.abs(w_ray_attn.numpy()), 99))
    im2 = axes[1, 0].imshow(w_std_attn.numpy(), cmap="RdBu_r", vmin=-v_att_std, vmax=v_att_std, aspect="auto")
    axes[1, 0].set_title("Standard AdamW — Attention W_q (256x256)\nUnstructured White-Noise Geometry", fontweight="bold")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(w_ray_attn.numpy(), cmap="RdBu_r", vmin=-v_att_ray, vmax=v_att_ray, aspect="auto")
    axes[1, 1].set_title(
        "Dirichlet Rayleigh Annealed — Attention W_q (256x256)\nEmergent Smooth Cortical Manifold (91.4% E<=0.40)",
        fontweight="bold",
        color="#00695C",
    )
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Emergent 2D Cortical Weight Manifolds in TinyStories-10M (10,000 steps)", fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    b1 = io.BytesIO()
    fig.savefig(b1, format="png", dpi=300)
    plt.close(fig)

    # Figure 2: Mean 2D-DCT Cumulative Spectral Energy across ALL FFN vs ALL Attention layers!
    def mean_cum_curve(model, is_attn: bool):
        grid = np.linspace(0.0, np.sqrt(2.0), 300)
        curves = []
        with torch.no_grad():
            for p in model.get_linear_projections():
                w = p.weight.detach().float()
                if (w.shape[0] == w.shape[1]) != is_attn:
                    continue
                s = dct2d(w)
                m, n = s.shape
                u = torch.arange(m, dtype=torch.float32).unsqueeze(1) / float(m)
                v = torch.arange(n, dtype=torch.float32).unsqueeze(0) / float(n)
                rho = torch.sqrt(u.pow(2) + v.pow(2)).numpy().ravel()
                e = (s.pow(2)).numpy().ravel()
                idx = np.argsort(rho)
                rho_s = rho[idx]
                cum = np.cumsum(e[idx]) / max(e.sum(), 1e-12) * 100.0
                curves.append(np.interp(grid, rho_s, cum))
        return grid, np.mean(curves, axis=0), np.std(curves, axis=0)

    g, e_std_ffn, s_std_ffn = mean_cum_curve(std_m, is_attn=False)
    _, e_ray_ffn, s_ray_ffn = mean_cum_curve(ray_m, is_attn=False)
    _, e_std_att, s_std_att = mean_cum_curve(std_m, is_attn=True)
    _, e_ray_att, s_ray_att = mean_cum_curve(ray_m, is_attn=True)
    if leg_m is not None:
        _, e_leg_ffn, _ = mean_cum_curve(leg_m, is_attn=False)
        _, e_leg_att, _ = mean_cum_curve(leg_m, is_attn=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)

    ax1.plot(g, e_ray_ffn, label="Rayleigh + Param-Weighted (Ours, 94.1% @ 0.40)", color="#00796B", lw=2.6)
    ax1.fill_between(g, np.clip(e_ray_ffn - s_ray_ffn, 0, 100), np.clip(e_ray_ffn + s_ray_ffn, 0, 100), color="#00796B", alpha=0.15)
    if leg_m is not None:
        e40_leg_ffn = float(np.interp(0.40, g, e_leg_ffn))
        ax1.plot(g, e_leg_ffn, label=f"Legacy numel+uniform (FFN starved, {e40_leg_ffn:.1f}%)", color="#7B1FA2", lw=2.0, ls="-.")
    ax1.plot(g, e_std_ffn, label="Standard AdamW (12.9% @ 0.40)", color="#E65100", lw=2.0, ls="--")
    ax1.axvline(0.40, color="#455A64", ls=":", lw=1.4, label="Band 0+1 Cutoff (r1=0.40)")
    ax1.set_title("FFN Projections (256x1024 / 1024x256, 24 layers)", fontweight="bold")
    ax1.set_xlabel("Normalized 2D-DCT Radial Frequency (rho)")
    ax1.set_ylabel("Cumulative Spectral Energy (%)")
    ax1.set_xlim(0.0, 1.42)
    ax1.set_ylim(0, 103)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=8.8)

    ax2.plot(g, e_ray_att, label="Rayleigh + Param-Weighted (Ours, 91.4% @ 0.40)", color="#00796B", lw=2.6)
    ax2.fill_between(g, np.clip(e_ray_att - s_ray_att, 0, 100), np.clip(e_ray_att + s_ray_att, 0, 100), color="#00796B", alpha=0.15)
    if leg_m is not None:
        e40_leg_att = float(np.interp(0.40, g, e_leg_att))
        ax2.plot(g, e_leg_att, label=f"Legacy numel+uniform ({e40_leg_att:.1f}%)", color="#7B1FA2", lw=2.0, ls="-.")
    ax2.plot(g, e_std_att, label="Standard AdamW (12.5% @ 0.40)", color="#E65100", lw=2.0, ls="--")
    ax2.axvline(0.40, color="#455A64", ls=":", lw=1.4, label="Band 0+1 Cutoff (r1=0.40)")
    ax2.set_title("Attention Projections (256x256, 32 layers)", fontweight="bold")
    ax2.set_xlabel("Normalized 2D-DCT Radial Frequency (rho)")
    ax2.set_xlim(0.0, 1.42)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", fontsize=8.8)

    fig.suptitle("Layer-Averaged 2D-DCT Spectral Compaction (Eliminating FFN Starvation)", fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    b2 = io.BytesIO()
    fig.savefig(b2, format="png", dpi=300)
    plt.close(fig)

    return {
        "fig1": b1.getvalue(),
        "fig2": b2.getvalue(),
        "generations": generations,
    }


def generate_local_fig3():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with open("results/rayleigh_sweep_2000s_s42.json", encoding="utf-8") as f:
        sweep = json.load(f)
    with open("results/tmlr_10000s_mseclip_multiseed_results.json", encoding="utf-8") as f:
        ms_mc = json.load(f)["summary"]
    with open("results/tmlr_10000s_multiseed_results.json", encoding="utf-8") as f:
        ms_base = json.load(f)["summary"]

    plt.rcParams.update({"font.size": 9.8})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9))

    # Panel (a): 2k Rayleigh Calibration Sweep (balanced Attn/FFN compaction vs 1.64 & 0.91 bpp PPL)
    lams = [r["topo_lambda"] for r in sweep]
    x_idx = np.arange(len(lams))
    attn_e = [r["attn_low_freq_pct"] for r in sweep]
    ffn_e = [r["ffn_low_freq_pct"] for r in sweep]
    ppl_164 = [r["quantized"]["Calib-1.64b"]["ppl"] for r in sweep]
    ppl_091 = [r["quantized"]["Sub1-0.91b"]["ppl"] for r in sweep]

    l1 = ax1.plot(x_idx, attn_e, marker="o", color="#00796B", lw=2.3, label="Attn E(rho<=0.40) [%]")
    l2 = ax1.plot(x_idx, ffn_e, marker="^", color="#26A69A", lw=2.0, ls="--", label="FFN E(rho<=0.40) [%]")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels([f"{v:g}" for v in lams])
    ax1.set_xlabel("Rayleigh Coupling Strength (lambda)")
    ax1.set_ylabel("Spectral Compaction E(rho <= 0.40) [%]", color="#00695C")
    ax1.tick_params(axis="y", labelcolor="#00695C")
    ax1.set_ylim(0, 108)
    ax1.grid(True, alpha=0.3)

    ax1r = ax1.twinx()
    l3 = ax1r.plot(x_idx, ppl_164, marker="s", color="#C62828", lw=2.3, label="1.638 bpp PPL (19.5x)")
    l4 = ax1r.plot(x_idx, ppl_091, marker="D", color="#EF6C00", lw=1.9, ls=":", label="0.914 bpp PPL (35.0x)")
    ax1r.set_ylabel("Quantized Validation PPL (log scale)", color="#B71C1C")
    ax1r.set_yscale("log")
    ax1r.set_yticks([20, 40, 80, 160])
    ax1r.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1r.set_ylim(16, 240)
    ax1r.tick_params(axis="y", labelcolor="#B71C1C")

    lines = l1 + l2 + l3 + l4
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="center", bbox_to_anchor=(0.48, 0.48), fontsize=8.2, framealpha=0.95)
    ax1.set_title("(a) 2k Calibration: Balanced Compaction & Compression", fontweight="bold")

    # Panel (b): 10k Multi-Seed (N=3) Rate-Distortion Frontier with SOTA PTQ baselines
    regimes = ["Sub1-0.91b", "Calib-1.64b", "Mid-2.18b", "HiFi-2.79b", "Cons-3.77b"]
    bpps = [ms_mc["std"]["pareto"][r]["bpp"] for r in regimes]

    def _plot_cond(cond_key, label, color, marker, ls="-"):
        means = [ms_mc[cond_key]["pareto"][r]["mean"] for r in regimes]
        stds = [ms_mc[cond_key]["pareto"][r]["std"] for r in regimes]
        ax2.errorbar(bpps, means, yerr=stds, marker=marker, color=color, lw=2.3, ls=ls, capsize=3.5, label=label)

    _plot_cond("std", "Standard + TritQ (N=3)", "#E65100", "o", "--")
    _plot_cond("rayleigh_lam1", "Rayleigh lam=1.0 + TritQ (N=3)", "#1565C0", "^", "-.")
    _plot_cond("rayleigh_anneal2", "Rayleigh Annealed 2->0 + TritQ (Ours, N=3)", "#00796B", "s", "-")

    # SOTA PTQ baseline scatter points (evaluated on Standard 10k N=3)
    b_std = ms_base["std"]["baselines"]
    ax2.errorbar(
        [1.64], [b_std["svd_int8_1.64b"]["mean"]], yerr=[b_std["svd_int8_1.64b"]["std"]],
        fmt="X", color="#6A1B9A", ms=8.5, capsize=3, label="SVD INT8 (1.64 bpp, 555.6)",
    )
    ax2.errorbar(
        [2.00], [b_std["spatial_int2_2.00b"]["mean"]], yerr=[b_std["spatial_int2_2.00b"]["std"]],
        fmt="P", color="#AD1457", ms=8.5, capsize=3, label="Spatial INT2 RTN (2.00 bpp, 47.4)",
    )
    ax2.errorbar(
        [2.25], [b_std["spatial_int2_g64_2.25b"]["mean"]], yerr=[b_std["spatial_int2_g64_2.25b"]["std"]],
        fmt="*", color="#283593", ms=10.5, capsize=3, label="Spatial Group INT2 g=64 (2.25 bpp, 36.9)",
    )
    ax2.errorbar(
        [4.00], [b_std["spatial_int4_4.00b"]["mean"]], yerr=[b_std["spatial_int4_4.00b"]["std"]],
        fmt="v", color="#37474F", ms=7.5, capsize=3, label="Spatial INT4 RTN (4.00 bpp, 7.0)",
    )

    ax2.set_yscale("log")
    ax2.set_xlabel("Effective Quantized Bit-Rate (bpp)")
    ax2.set_ylabel("Validation Perplexity (log scale, mean +/- std)")
    ax2.set_xlim(0.75, 4.20)
    ax2.set_ylim(5.5, 1600)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="upper right", fontsize=7.6, framealpha=0.92)
    ax2.set_title("(b) 10k Multi-Seed (N=3) Rate-Distortion vs SOTA PTQ", fontweight="bold")

    fig.suptitle("Dirichlet Regularization Rate-Distortion Frontier (TinyStories-10M)", fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    for d in ("docs/figures", "paper/figures"):
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, "fig3_pareto_quantization.png"), dpi=300)
    plt.close(fig)


@app.local_entrypoint()
def main():
    res = generate_remote_figures_and_samples.remote()
    for d in ("docs/figures", "paper/figures"):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "fig1_weight_heatmaps.png"), "wb") as f:
            f.write(res["fig1"])
        with open(os.path.join(d, "fig2_dct_energy_spectra.png"), "wb") as f:
            f.write(res["fig2"])
    generate_local_fig3()

    os.makedirs("results", exist_ok=True)
    with open("results/qualitative_10k_generations.json", "w", encoding="utf-8") as f:
        json.dump(res["generations"], f, indent=2)

    print("\n[+] Regenerated 300-DPI Figures 1, 2, 3 in both paper/figures/ and docs/figures/")
    print("\n[+] Qualitative Greedy Generations (FP32 vs 1.638 bpp Base-3 TritQ):")
    for p, g in res["generations"].items():
        print(f"\nPrompt: {p}")
        for k, txt in g.items():
            print(f"  [{k:<23}]: {txt}")
