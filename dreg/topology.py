"""
Topological Dirichlet Regularization Module
===========================================
Implements 2D cortical manifold embedding and Dirichlet harmonic pinning
to eliminate permutation invariance and concentrate energy into low spatial frequencies.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional, Iterable


def get_grid_dimensions(n: int) -> Tuple[int, int]:
    """
    Finds the integer factor pair (H, W) such that H * W = n
    with the aspect ratio closest to 1.0 (square-like 2D cortical lattice).
    """
    sqrt_n = int(math.isqrt(n))
    for h in range(sqrt_n, 0, -1):
        if n % h == 0:
            return h, n // h
    return 1, n


def dirichlet_energy_2d(
    weight: torch.Tensor,
    grid_shape: Optional[Tuple[int, int]] = None,
    normalization: str = "numel",
) -> torch.Tensor:
    """
    Computes the 2D Dirichlet Harmonic Energy of a weight matrix:
        E(W) = 0.5 * Tr(W^T L W) = 0.5 * sum_{(u,v) in E} ||w_u - w_v||^2

    Args:
        weight: Tensor of shape (out_features, in_features)
        grid_shape: Optional (H, W) tuple to reshape either dimension into a 2D sheet.
                    If None, uses the natural (out_features, in_features) grid.
        normalization: Normalization mode for the energy scalar:
            - 'numel' (default): Normalizes by total elements (M * N). Backward-compatible.
            - 'edges': Normalizes by the exact number of grid edges ((M-1)*N + M*(N-1)),
              providing scale-invariant roughness estimation across varying matrix shapes.
            - 'none': Raw sum of squared differences without division.

    Returns:
        Scalar torch.Tensor representing the normalized Dirichlet energy.
    """
    if grid_shape is not None:
        h, w = grid_shape
        # Embed rows or columns onto 2D cortical sheet
        if weight.shape[0] == h * w:
            sheet = weight.view(h, w, -1)
        elif weight.shape[1] == h * w:
            sheet = weight.view(-1, h, w).permute(1, 2, 0)
        else:
            sheet = weight
    else:
        sheet = weight

    if sheet.dim() == 2:
        diff_h = sheet[1:, :] - sheet[:-1, :]
        diff_w = sheet[:, 1:] - sheet[:, :-1]
        sum_sq = diff_h.pow(2).sum() + diff_w.pow(2).sum()
        h, w = sheet.shape
        num_edges = (h - 1) * w + h * (w - 1)
    elif sheet.dim() == 3:
        diff_h = sheet[1:, :, :] - sheet[:-1, :, :]
        diff_w = sheet[:, 1:, :] - sheet[:, :-1, :]
        sum_sq = diff_h.pow(2).sum() + diff_w.pow(2).sum()
        h, w, c = sheet.shape
        num_edges = ((h - 1) * w + h * (w - 1)) * c
    else:
        raise ValueError(f"Unsupported weight tensor shape for Dirichlet energy: {sheet.shape}")

    if normalization == "numel":
        return sum_sq / (sheet.numel() + 1e-8)
    elif normalization == "edges":
        return sum_sq / (max(num_edges, 1) + 1e-8)
    elif normalization == "none":
        return sum_sq
    else:
        raise ValueError(f"Unknown normalization mode: {normalization}. Expected 'numel', 'edges', or 'none'.")


class DirichletLoss(nn.Module):
    """
    PyTorch loss wrapper for Dirichlet harmonic pinning.
    Traverses linear projections and accumulates topological gradient penalty.
    """
    def __init__(self, weight_decay: float = 0.01, normalization: str = "numel"):
        super().__init__()
        self.weight_decay = weight_decay
        self.normalization = normalization

    def forward(self, modules: Iterable[nn.Module]) -> torch.Tensor:
        total_energy = None
        count = 0
        for mod in modules:
            if hasattr(mod, "weight") and mod.weight is not None and mod.weight.dim() == 2:
                if total_energy is None:
                    total_energy = torch.tensor(0.0, device=mod.weight.device)
                total_energy = total_energy + dirichlet_energy_2d(mod.weight, normalization=self.normalization)
                count += 1
        if total_energy is None:
            return torch.tensor(0.0)
        if count == 0:
            return total_energy
        return self.weight_decay * (total_energy / count)
