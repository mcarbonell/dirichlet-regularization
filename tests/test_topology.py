"""
Tests for topology module: Dirichlet energy and grid dimension utilities.
"""

import math
import pytest
import torch
import torch.nn as nn

from topospec.topology import dirichlet_energy_2d, get_grid_dimensions, DirichletLoss


class TestGetGridDimensions:
    """Tests for the get_grid_dimensions factor-pair finder."""

    def test_perfect_square(self):
        h, w = get_grid_dimensions(64)
        assert h * w == 64
        assert h == 8 and w == 8

    def test_prime_number(self):
        h, w = get_grid_dimensions(17)
        assert h * w == 17
        assert h == 1 and w == 17

    def test_rectangular(self):
        h, w = get_grid_dimensions(256)
        assert h * w == 256
        assert h == 16 and w == 16

    def test_aspect_ratio_close_to_one(self):
        """Factor pair should be as close to square as possible."""
        h, w = get_grid_dimensions(12)
        assert h * w == 12
        # 3x4 is closer to square than 2x6 or 1x12
        assert (h, w) == (3, 4)

    def test_one(self):
        h, w = get_grid_dimensions(1)
        assert h == 1 and w == 1


class TestDirichletEnergy2D:
    """Tests for the Dirichlet harmonic energy computation."""

    def test_constant_matrix_has_zero_energy(self):
        """A constant matrix has zero spatial gradient → zero Dirichlet energy."""
        w = torch.ones(8, 8) * 3.14
        energy = dirichlet_energy_2d(w)
        assert energy.item() == pytest.approx(0.0, abs=1e-7)

    def test_smooth_gradient_has_low_energy(self):
        """A smooth linear gradient should have low energy."""
        h, w_dim = 16, 16
        rows = torch.linspace(0, 1, h).unsqueeze(1).expand(h, w_dim)
        energy_smooth = dirichlet_energy_2d(rows)

        # Compare with random (high-frequency) weights
        torch.manual_seed(42)
        energy_random = dirichlet_energy_2d(torch.randn(h, w_dim))

        assert energy_smooth.item() < energy_random.item()

    def test_checkerboard_has_high_energy(self):
        """A checkerboard pattern is maximally non-smooth."""
        n = 16
        checker = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                checker[i, j] = (-1.0) ** (i + j)
        energy = dirichlet_energy_2d(checker)
        assert energy.item() > 1.0  # Very high energy

    def test_energy_is_nonnegative(self):
        torch.manual_seed(123)
        w = torch.randn(32, 64)
        energy = dirichlet_energy_2d(w)
        assert energy.item() >= 0.0

    def test_with_grid_shape_3d(self):
        """When grid_shape is provided and matches out_features, weight is reshaped to 3D sheet."""
        w = torch.randn(16, 32)  # out=16, in=32
        grid_shape = (4, 4)  # 4*4 = 16 = out_features
        energy = dirichlet_energy_2d(w, grid_shape=grid_shape)
        assert energy.item() >= 0.0
        assert torch.isfinite(energy)

    def test_gradient_flows(self):
        """Dirichlet energy should be differentiable."""
        w = torch.randn(8, 8, requires_grad=True)
        energy = dirichlet_energy_2d(w)
        energy.backward()
        assert w.grad is not None
        assert w.grad.shape == w.shape


class TestDirichletLoss:
    """Tests for the DirichletLoss module wrapper."""

    def test_on_simple_model(self):
        model = nn.Sequential(
            nn.Linear(32, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 10, bias=False),
        )
        loss_fn = DirichletLoss(weight_decay=0.01)
        loss = loss_fn(model.modules())
        assert loss.item() > 0.0
        assert torch.isfinite(loss)

    def test_with_no_linear_modules(self):
        """Should return 0.0 when no 2D weight modules exist."""
        model = nn.Sequential(nn.ReLU(), nn.Dropout(0.5))
        loss_fn = DirichletLoss(weight_decay=0.01)
        loss = loss_fn(model.modules())
        assert loss.item() == pytest.approx(0.0)

    def test_weight_decay_scales_loss(self):
        model = nn.Sequential(nn.Linear(16, 16, bias=False))
        loss_low = DirichletLoss(weight_decay=0.001)(model.modules())
        loss_high = DirichletLoss(weight_decay=1.0)(model.modules())
        assert loss_high.item() > loss_low.item()
