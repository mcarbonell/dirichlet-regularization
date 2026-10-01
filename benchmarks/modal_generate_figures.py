"""
Generate Publication Visualizations & Text Samples from Modal Checkpoints
========================================================================
Runs on Modal (or locally if checkpoints exist) to:
  1. Load converged 10,000-step models (Standard vs Topographic).
  2. Compute 2D-DCT energy distribution curves.
  3. Plot side-by-side weight heatmaps and spectral compaction.
  4. Generate qualitative story continuations.
  5. Save figures to `docs/figures/`.
"""

import os
import modal

app = modal.App("dreg-generate-figures")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "matplotlib",
        "numpy",
        "tokenizers>=0.13.0",
    )
    .add_local_python_source("dreg")
)

volume = modal.Volume.from_name("dreg-checkpoints")
CHECKPOINT_DIR = "/root/checkpoints"


@app.function(
    image=image,
    volumes={CHECKPOINT_DIR: volume},
    timeout=600,
)
def generate_all_artifacts():
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tokenizers import Tokenizer
    from dreg.model import TopographicTransformer, TopographicConfig
    from dreg.spectral import dct2d, idct2d
    from dreg.topology import dirichlet_energy_2d

    print("==============================================================================")
    print(" GENERATING PUBLICATION ARTIFACTS & VISUALIZATIONS")
    print("==============================================================================")

    # 1. Load Tokenizer
    tok_path = os.path.join(CHECKPOINT_DIR, "tinystories_bpe_4096.json")
    tokenizer = Tokenizer.from_file(tok_path)
    print(f"[+] Loaded tokenizer from {tok_path}")

    # 2. Config & Models (Matching 10M Architecture)
    cfg_std = TopographicConfig(
        vocab_size=4096,
        d_model=256,
        n_heads=4,
        n_layers=8,
        ffn_dim=1024,
        max_seq_len=256,
        topo_lambda=0.0,
    )
    cfg_topo = TopographicConfig(
        vocab_size=4096,
        d_model=256,
        n_heads=4,
        n_layers=8,
        ffn_dim=1024,
        max_seq_len=256,
        topo_lambda=30.0,
    )

    std_model = TopographicTransformer(cfg_std)
    topo_model = TopographicTransformer(cfg_topo)

    std_ckpt_path = os.path.join(CHECKPOINT_DIR, "tinystories_10M_standard_standard_10000s.pt")
    topo_ckpt_path = os.path.join(CHECKPOINT_DIR, "tinystories_10M_topographic_lambda30.0_10000s.pt")

    std_model.load_state_dict(torch.load(std_ckpt_path, map_location="cpu"))
    topo_model.load_state_dict(torch.load(topo_ckpt_path, map_location="cpu"))
    std_model.eval()
    topo_model.eval()
    print("[+] Successfully loaded 10,000-step converged models.")

    # 3. Qualitative Text Generation
    prompts = [
        "Once upon a time, there was a little girl named Lily.",
        "One day, Tim went to the park with his red ball.",
    ]
    generations = {}
    print("\n--- Qualitative Story Generation ---")
    for p in prompts:
        enc = tokenizer.encode(p)
        ids = torch.tensor([enc.ids if hasattr(enc, "ids") else enc], dtype=torch.long)
        
        with torch.no_grad():
            out_std = std_model.generate(ids, max_new_tokens=60, temperature=0.7)
            out_topo = topo_model.generate(ids, max_new_tokens=60, temperature=0.7)

        text_std = tokenizer.decode(out_std[0].tolist())
        text_topo = tokenizer.decode(out_topo[0].tolist())
        generations[p] = {"standard": text_std, "topographic": text_topo}
        print(f"\n[Prompt]: {p}")
        print(f"  [Standard]:    {text_std}")
        print(f"  [Topographic]: {text_topo}")

    # 4. Extract Representative Weight Matrices (Layer 3 FFN up-proj & Attn q-proj)
    w_std_ffn = std_model.blocks[2].mlp.gate_proj.weight.detach().float()
    w_topo_ffn = topo_model.blocks[2].mlp.gate_proj.weight.detach().float()

    w_std_attn = std_model.blocks[2].attn.q_proj.weight.detach().float()
    w_topo_attn = topo_model.blocks[2].attn.q_proj.weight.detach().float()

    # 5. Figure 1: Spatial Weight Heatmaps (Texture & Smoothness)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # Sub-slice [128x128] for visual clarity
    im0 = axes[0, 0].imshow(w_std_ffn[:128, :128].numpy(), cmap="viridis", aspect="auto")
    axes[0, 0].set_title("Standard MLP FFN (Unstructured / Noise)", fontsize=11, fontweight="bold")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(w_topo_ffn[:128, :128].numpy(), cmap="viridis", aspect="auto")
    axes[0, 1].set_title("Dirichlet Regularized FFN (Cortical Manifold)", fontsize=11, fontweight="bold", color="darkgreen")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(w_std_attn.numpy(), cmap="magma", aspect="auto")
    axes[1, 0].set_title("Standard Attention W_q (Noise)", fontsize=11, fontweight="bold")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(w_topo_attn.numpy(), cmap="magma", aspect="auto")
    axes[1, 1].set_title("Dirichlet Attention W_q (Smooth)", fontsize=11, fontweight="bold", color="darkgreen")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Figure 1: Emergent Topographic Weight Manifolds in 10M Transformer", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig1_bytes = _fig_to_bytes(fig)
    plt.close(fig)

    # 6. Figure 2: 2D-DCT Cumulative Spectral Energy
    def get_cum_energy(w):
        s = dct2d(w)
        m, n = s.shape
        u = torch.linspace(0, 1, m)
        v = torch.linspace(0, 1, n)
        gu, gv = torch.meshgrid(u, v, indexing="ij")
        rho = torch.sqrt(gu**2 + gv**2).numpy().flatten()
        energy = (s**2).numpy().flatten()
        idx = np.argsort(rho)
        rho_sorted = rho[idx]
        energy_sorted = energy[idx]
        cum_energy = np.cumsum(energy_sorted)
        cum_energy = cum_energy / max(cum_energy[-1], 1e-12) * 100.0
        return rho_sorted, cum_energy

    r_std_ffn, e_std_ffn = get_cum_energy(w_std_ffn)
    r_topo_ffn, e_topo_ffn = get_cum_energy(w_topo_ffn)
    r_std_attn, e_std_attn = get_cum_energy(w_std_attn)
    r_topo_attn, e_topo_attn = get_cum_energy(w_topo_attn)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # FFN
    sub = 500
    ax1.plot(r_topo_ffn[::sub], e_topo_ffn[::sub], label="Topographic (Dirichlet)", color="teal", lw=2.5)
    ax1.plot(r_std_ffn[::sub], e_std_ffn[::sub], label="Standard (AdamW)", color="darkorange", lw=2, ls="--")
    ax1.axvline(x=0.40, color="gray", ls=":", alpha=0.7, label="Quant Cutoff (r=0.40)")
    ax1.set_xlabel("Normalized 2D-DCT Radius (Frequency)", fontsize=10)
    ax1.set_ylabel("Cumulative Spectral Energy (%)", fontsize=10)
    ax1.set_title("FFN Projection Weights (256x1024)", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Attention
    ax2.plot(r_topo_attn[::sub], e_topo_attn[::sub], label="Topographic (Dirichlet)", color="teal", lw=2.5)
    ax2.plot(r_std_attn[::sub], e_std_attn[::sub], label="Standard (AdamW)", color="darkorange", lw=2, ls="--")
    ax2.axvline(x=0.40, color="gray", ls=":", alpha=0.7, label="Quant Cutoff (r=0.40)")
    ax2.set_xlabel("Normalized 2D-DCT Radius (Frequency)", fontsize=10)
    ax2.set_ylabel("Cumulative Spectral Energy (%)", fontsize=10)
    ax2.set_title("Attention Q-Projection (256x256)", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle("Figure 2: 2D-DCT Spectral Energy Compaction (Topographic vs Standard)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig2_bytes = _fig_to_bytes(fig)
    plt.close(fig)

    # 7. Figure 3: Empirical Calibration & Quantization Pareto Curve
    lambdas = [0.0, 0.01, 5.0, 15.0, 30.0]
    ed_vals = [0.004225, 0.004251, 0.003382, 0.002427, 0.001924]
    quant_ppl_2k = [46.60, 47.95, 44.80, 43.26, 39.80]
    fp32_ppl_2k = [10.48, 10.44, 10.62, 10.81, 10.98]

    fig, ax_p1 = plt.subplots(figsize=(8, 5))
    color = "tab:blue"
    ax_p1.set_xlabel(r"Dirichlet Regularization Strength ($\lambda_{\mathrm{topo}}$)", fontsize=11)
    ax_p1.set_ylabel("Weight Dirichlet Energy ($E_D$)", color=color, fontsize=11)
    line1 = ax_p1.plot(lambdas, ed_vals, marker="o", color=color, lw=2.5, label="Dirichlet Energy $E_D$ (Roughness)")
    ax_p1.tick_params(axis="y", labelcolor=color)
    ax_p1.grid(True, alpha=0.3)

    ax_p2 = ax_p1.twinx()
    color2 = "tab:red"
    ax_p2.set_ylabel("Quantized Perplexity (0.945 bpp)", color=color2, fontsize=11)
    line2 = ax_p2.plot(lambdas, quant_ppl_2k, marker="s", color=color2, lw=2.5, ls="--", label="Quantized PPL (0.945 bpp)")
    ax_p2.tick_params(axis="y", labelcolor=color2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_p1.legend(lines, labels, loc="center right")
    plt.title("Figure 3: Dirichlet Regularization Pareto Tradeoff Curve", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig3_bytes = _fig_to_bytes(fig)
    plt.close(fig)

    return {
        "fig1": fig1_bytes,
        "fig2": fig2_bytes,
        "fig3": fig3_bytes,
        "generations": generations,
    }


def _fig_to_bytes(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    buf.seek(0)
    return buf.read()


@app.local_entrypoint()
def main():
    os.makedirs("docs/figures", exist_ok=True)
    res = generate_all_artifacts.remote()

    with open("docs/figures/fig1_weight_heatmaps.png", "wb") as f:
        f.write(res["fig1"])
    with open("docs/figures/fig2_dct_energy_spectra.png", "wb") as f:
        f.write(res["fig2"])
    with open("docs/figures/fig3_pareto_quantization.png", "wb") as f:
        f.write(res["fig3"])

    print("\n[+] Successfully saved publication figures to docs/figures/:")
    print("    - docs/figures/fig1_weight_heatmaps.png")
    print("    - docs/figures/fig2_dct_energy_spectra.png")
    print("    - docs/figures/fig3_pareto_quantization.png")

    print("\n[+] Qualitative Text Generations:")
    for p, gen in res["generations"].items():
        print(f"\nPrompt: '{p}'")
        print(f"  Standard:    {gen['standard']}")
        print(f"  Topographic: {gen['topographic']}")
