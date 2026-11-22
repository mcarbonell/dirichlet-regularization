"""
Comparative Baselines Module
============================
Implements standard quantization and compression controls demanded in peer review:
  1. Random Orthogonal Rotation / Incoherence Transform (QuIP/QuaRot-style control)
  2. SVD Low-Rank Truncation (budget-matched low-rank baseline)
"""

import math
from typing import Optional, Tuple

import torch


def random_orthogonal_transform_2d(
    weight: torch.Tensor,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Applies random orthogonal rotation matrices Q_L and Q_R to a weight matrix:
        W_rot = Q_L @ W @ Q_R^T
    where Q_L in O(M) and Q_R in O(N).

    This serves as the critical control against Hadamard/QuIP-style incoherence:
    it rotates coordinates without imposing spatial continuity.
    """
    if seed is not None:
        gen = torch.Generator(device=weight.device).manual_seed(seed)
    else:
        gen = None

    m, n = weight.shape
    # Generate Haar-uniform random orthogonal matrices via QR + Mezzadri (2007) sign correction
    g_l = torch.randn((m, m), device=weight.device, dtype=weight.dtype, generator=gen)
    q_l, r_l = torch.linalg.qr(g_l)
    diag_l = torch.sign(torch.diag(r_l))
    diag_l = torch.where(diag_l == 0, torch.ones_like(diag_l), diag_l)
    q_l = q_l * diag_l.unsqueeze(0)

    g_r = torch.randn((n, n), device=weight.device, dtype=weight.dtype, generator=gen)
    q_r, r_r = torch.linalg.qr(g_r)
    diag_r = torch.sign(torch.diag(r_r))
    diag_r = torch.where(diag_r == 0, torch.ones_like(diag_r), diag_r)
    q_r = q_r * diag_r.unsqueeze(0)

    w_rot = q_l @ weight @ q_r.t()
    return w_rot, q_l, q_r


def svd_low_rank_approximation(
    weight: torch.Tensor,
    rank: Optional[int] = None,
    compression_ratio: Optional[float] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Computes optimal truncated SVD rank-k approximation of a weight matrix:
        W_approx = U[:, :k] @ diag(S[:k]) @ Vh[:k, :]

    Args:
        weight: [M, N] tensor
        rank: Explicit rank cutoff k.
        compression_ratio: If rank is None, determines k such that k*(M+N) achieves
                           the target compression relative to M*N FP32 elements.

    Returns:
        (W_approx, rank_k)
    """
    m, n = weight.shape
    max_rank = min(m, n)

    if rank is None:
        if compression_ratio is not None and compression_ratio > 1.0:
            # target_elements = (M * N) / compression_ratio
            # k * (M + N) <= target_elements => k = target_elements / (M + N)
            target_params = (m * n) / compression_ratio
            rank = max(1, min(max_rank, int(math.floor(target_params / (m + n)))))
        else:
            rank = max(1, max_rank // 4)
    else:
        rank = max(1, min(max_rank, rank))

    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    w_approx = u[:, :rank] @ torch.diag(s[:rank]) @ vh[:rank, :]
    return w_approx, rank


def spectral_hadamard_benchmark_control(weight: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
    """
    Applies a Haar-uniform random orthogonal 2D rotation control (incoherence
    baseline without imposing spatial manifold smoothing).
    """
    w_rot, _, _ = random_orthogonal_transform_2d(weight, seed=seed)
    return w_rot
