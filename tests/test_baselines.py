"""
Tests for comparative baselines: Random Orthogonal Rotations and SVD truncation.
"""

import pytest
import torch
from dreg.baselines import random_orthogonal_transform_2d, svd_low_rank_approximation


class TestBaselines:
    def test_orthogonal_transform_properties(self):
        torch.manual_seed(42)
        w = torch.randn(32, 64)
        w_rot, q_l, q_r = random_orthogonal_transform_2d(w, seed=123)

        # Shape preserved
        assert w_rot.shape == (32, 64)
        assert q_l.shape == (32, 32)
        assert q_r.shape == (64, 64)

        # Frobenius norm preserved (orthogonal invariance)
        f_orig = torch.norm(w, p="fro")
        f_rot = torch.norm(w_rot, p="fro")
        assert f_rot.item() == pytest.approx(f_orig.item(), rel=1e-4)

        # Q_L and Q_R are strictly orthogonal
        i_l = torch.eye(32)
        i_r = torch.eye(64)
        assert torch.allclose(q_l @ q_l.t(), i_l, atol=1e-4)
        assert torch.allclose(q_r @ q_r.t(), i_r, atol=1e-4)

    def test_svd_low_rank_approximation(self):
        torch.manual_seed(42)
        w = torch.randn(32, 32)
        w_approx, rank = svd_low_rank_approximation(w, rank=8)

        assert w_approx.shape == (32, 32)
        assert rank == 8
        # SVD rank constraint
        s = torch.linalg.svdvals(w_approx)
        assert (s > 1e-4).sum().item() <= 8

    def test_svd_compression_ratio(self):
        w = torch.randn(64, 128)
        w_approx, rank = svd_low_rank_approximation(w, compression_ratio=4.0)
        assert w_approx.shape == (64, 128)
        assert rank > 0
        assert rank < 64
