"""
Script to generate clean, high-contrast, publication-grade weight heatmaps.
Produces:
  1. docs/figures/fig1_weight_heatmaps.png (and paper/figures/fig1_weight_heatmaps.png)
  2. Clear side-by-side comparison using diverging colormap (coolwarm/RdBu_r)
     demonstrating white noise (standard AdamW) vs smooth cortical manifolds (Dirichlet).
"""

import os
import sys

import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure root is on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from dreg import dirichlet_energy_2d  # noqa: E402


class TopoMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256, bias=False)
        self.fc2 = nn.Linear(256, 128, bias=False)
        self.fc3 = nn.Linear(128, 10, bias=False)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)


def train_single_model(use_dirichlet: bool, epsilon: float = 2e-3, epochs: int = 3, seed: int = 42):
    torch.manual_seed(seed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    data_dir = os.path.join(ROOT_DIR, "data")
    train_ds = datasets.MNIST(data_dir, train=True, download=False, transform=transform)
    loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    model = TopoMLP()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in loader:
            bx = bx.view(bx.size(0), -1)
            logits = model(bx)
            task_loss = crit(logits, by)

            if use_dirichlet:
                # Unattenuated Dirichlet penalty on fc1 and fc2
                diff_r1 = model.fc1.weight[1:, :] - model.fc1.weight[:-1, :]
                diff_c1 = model.fc1.weight[:, 1:] - model.fc1.weight[:, :-1]
                reg1 = epsilon * 0.5 * (diff_r1.pow(2).sum() + diff_c1.pow(2).sum())

                diff_r2 = model.fc2.weight[1:, :] - model.fc2.weight[:-1, :]
                diff_c2 = model.fc2.weight[:, 1:] - model.fc2.weight[:, :-1]
                reg2 = epsilon * 0.5 * (diff_r2.pow(2).sum() + diff_c2.pow(2).sum())

                total_loss = task_loss + reg1 + reg2
            else:
                total_loss = task_loss

            opt.zero_grad()
            total_loss.backward()
            opt.step()

    return model


def main():
    print("[*] Training Standard AdamW Baseline (epsilon=0.0)...")
    m_std = train_single_model(use_dirichlet=False, epochs=3, seed=42)

    print("[*] Training Dirichlet-Regularized Model (epsilon=2e-3)...")
    m_topo = train_single_model(use_dirichlet=True, epsilon=2e-3, epochs=3, seed=42)

    w1_std = m_std.fc1.weight.detach().cpu()
    w1_topo = m_topo.fc1.weight.detach().cpu()
    w2_std = m_std.fc2.weight.detach().cpu()
    w2_topo = m_topo.fc2.weight.detach().cpu()

    ed_std = dirichlet_energy_2d(w1_std, normalization="edges").item()
    ed_topo = dirichlet_energy_2d(w1_topo, normalization="edges").item()
    print(f"[+] Dirichlet Energy W1: Standard = {ed_std:.6f} | Topo = {ed_topo:.6f} (-{(1 - ed_topo/ed_std)*100:.1f}%)")

    # Receptive patch size
    patch_size = 64
    p_std_1 = w1_std[:patch_size, :patch_size].numpy()
    p_topo_1 = w1_topo[:patch_size, :patch_size].numpy()

    p_std_2 = w2_std[:patch_size, :patch_size].numpy()
    p_topo_2 = w2_topo[:patch_size, :patch_size].numpy()

    # Create 2x2 comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), dpi=200)

    # Styling settings
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # Row 1: Layer 1 (W1)
    vmax1 = max(np.percentile(np.abs(p_std_1), 99), np.percentile(np.abs(p_topo_1), 99))
    im00 = axes[0, 0].imshow(p_std_1, cmap="coolwarm", vmin=-vmax1, vmax=vmax1, aspect="equal", interpolation="nearest")
    axes[0, 0].set_title(f"Standard AdamW: Layer 1\n(White Noise, $E_D={ed_std:.4f}$)", fontsize=12, fontweight="bold", color="#8B0000")
    axes[0, 0].set_xlabel(f"Input Features [0..{patch_size})", fontsize=10)
    axes[0, 0].set_ylabel(f"Neuron Units [0..{patch_size})", fontsize=10)
    cb00 = plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)
    cb00.ax.tick_params(labelsize=9)

    im01 = axes[0, 1].imshow(p_topo_1, cmap="coolwarm", vmin=-vmax1, vmax=vmax1, aspect="equal", interpolation="nearest")
    axes[0, 1].set_title(f"Dirichlet Regularized: Layer 1\n(Cortical Manifold, $E_D={ed_topo:.4f}$ [-{(1 - ed_topo/ed_std)*100:.0f}%])", fontsize=12, fontweight="bold", color="#005A32")
    axes[0, 1].set_xlabel(f"Input Features [0..{patch_size})", fontsize=10)
    axes[0, 1].set_ylabel(f"Neuron Units [0..{patch_size})", fontsize=10)
    cb01 = plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cb01.ax.tick_params(labelsize=9)

    # Row 2: Layer 2 (W2)
    vmax2 = max(np.percentile(np.abs(p_std_2), 99), np.percentile(np.abs(p_topo_2), 99))
    im10 = axes[1, 0].imshow(p_std_2, cmap="coolwarm", vmin=-vmax2, vmax=vmax2, aspect="equal", interpolation="nearest")
    axes[1, 0].set_title("Standard AdamW: Layer 2\n(Uncorrelated Stochastic Weights)", fontsize=12, fontweight="bold", color="#8B0000")
    axes[1, 0].set_xlabel(f"Hidden Dim [0..{patch_size})", fontsize=10)
    axes[1, 0].set_ylabel(f"Neuron Units [0..{patch_size})", fontsize=10)
    cb10 = plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cb10.ax.tick_params(labelsize=9)

    im11 = axes[1, 1].imshow(p_topo_2, cmap="coolwarm", vmin=-vmax2, vmax=vmax2, aspect="equal", interpolation="nearest")
    axes[1, 1].set_title("Dirichlet Regularized: Layer 2\n(Emergent Continuous Topography)", fontsize=12, fontweight="bold", color="#005A32")
    axes[1, 1].set_xlabel(f"Hidden Dim [0..{patch_size})", fontsize=10)
    axes[1, 1].set_ylabel(f"Neuron Units [0..{patch_size})", fontsize=10)
    cb11 = plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cb11.ax.tick_params(labelsize=9)

    plt.suptitle("Figure 1: Spatial Weight Structures — Standard AdamW vs Dirichlet Regularization", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_docs = os.path.join(ROOT_DIR, "docs", "figures", "fig1_weight_heatmaps.png")
    out_paper = os.path.join(ROOT_DIR, "paper", "figures", "fig1_weight_heatmaps.png")
    os.makedirs(os.path.dirname(out_docs), exist_ok=True)
    os.makedirs(os.path.dirname(out_paper), exist_ok=True)

    fig.savefig(out_docs, dpi=200, bbox_inches="tight")
    fig.savefig(out_paper, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[+] Saved updated Figure 1 to:\n  - {out_docs}\n  - {out_paper}")


if __name__ == "__main__":
    main()
