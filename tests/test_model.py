"""
Tests for model module: TopographicTransformer, TopographicLinear, config, generation.
"""

import pytest
import torch
import torch.nn as nn

from dreg.model import (
    TopographicTransformer,
    TopographicConfig,
    TopographicLinear,
    TopographicMLP,
    TopographicAttention,
    TopographicBlock,
)


class TestTopographicLinear:
    """Tests for the TopographicLinear layer."""

    def test_is_nn_linear_subclass(self):
        layer = TopographicLinear(32, 64)
        assert isinstance(layer, nn.Linear)

    def test_forward_shape(self):
        layer = TopographicLinear(32, 64)
        x = torch.randn(4, 10, 32)
        out = layer(x)
        assert out.shape == (4, 10, 64)

    def test_no_bias_by_default(self):
        layer = TopographicLinear(32, 64)
        assert layer.bias is None

    def test_dirichlet_energy_is_positive(self):
        layer = TopographicLinear(32, 64)
        energy = layer.dirichlet_energy()
        assert energy.item() > 0.0
        assert torch.isfinite(energy)

    def test_grid_shape(self):
        layer = TopographicLinear(32, 64)
        assert layer.grid_shape == (64, 32)


class TestTopographicConfig:
    """Tests for the dataclass config."""

    def test_defaults(self):
        cfg = TopographicConfig()
        assert cfg.vocab_size == 4096
        assert cfg.d_model == 256
        assert cfg.n_heads == 4
        assert cfg.n_layers == 6
        assert cfg.topo_lambda == 0.01

    def test_custom_values(self):
        cfg = TopographicConfig(vocab_size=128, d_model=64, n_heads=2, n_layers=2)
        assert cfg.vocab_size == 128
        assert cfg.d_model == 64


class TestTopographicTransformer:
    """Tests for the full Transformer model."""

    @pytest.fixture
    def small_model(self):
        cfg = TopographicConfig(
            vocab_size=128,
            d_model=64,
            n_heads=2,
            n_layers=2,
            ffn_dim=128,
            max_seq_len=32,
            topo_lambda=0.01,
        )
        return TopographicTransformer(cfg)

    def test_forward_output_shape(self, small_model):
        x = torch.randint(0, 128, (2, 16))
        logits, loss = small_model(x)
        assert logits.shape == (2, 16, 128)
        assert loss is None

    def test_forward_with_targets(self, small_model):
        x = torch.randint(0, 128, (2, 16))
        targets = torch.randint(0, 128, (2, 16))
        logits, loss = small_model(x, targets)
        assert logits.shape == (2, 16, 128)
        assert loss is not None
        assert loss.item() > 0.0

    def test_topographic_loss_positive(self, small_model):
        topo_loss = small_model.topographic_loss()
        assert topo_loss.item() > 0.0

    def test_topographic_loss_zero_lambda(self):
        cfg = TopographicConfig(
            vocab_size=128, d_model=64, n_heads=2, n_layers=2,
            ffn_dim=128, max_seq_len=32, topo_lambda=0.0,
        )
        model = TopographicTransformer(cfg)
        topo_loss = model.topographic_loss()
        assert topo_loss.item() == pytest.approx(0.0)

    def test_backward_pass(self, small_model):
        """Gradients should flow through both CE and topographic loss."""
        x = torch.randint(0, 128, (2, 16))
        targets = torch.randint(0, 128, (2, 16))
        logits, ce_loss = small_model(x, targets)
        topo_loss = small_model.topographic_loss()
        total_loss = ce_loss + topo_loss
        total_loss.backward()

        for name, p in small_model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_get_linear_projections(self, small_model):
        projs = small_model.get_linear_projections()
        assert len(projs) > 0
        for p in projs:
            assert isinstance(p, TopographicLinear)

    def test_generate_output_length(self, small_model):
        prompt = torch.randint(0, 128, (1, 4))
        output = small_model.generate(prompt, max_new_tokens=10)
        assert output.shape == (1, 14)  # 4 prompt + 10 generated

    def test_param_count(self, small_model):
        count = sum(p.numel() for p in small_model.parameters())
        assert count > 0
        # A 2-layer model with d=64, ffn=128, vocab=128 should be reasonable
        assert count < 1_000_000  # Sanity upper bound


class TestTopographicAttention:
    """Tests for the attention module."""

    def test_output_shape(self):
        cfg = TopographicConfig(d_model=64, n_heads=2)
        attn = TopographicAttention(cfg)
        x = torch.randn(2, 8, 64)
        out = attn(x)
        assert out.shape == (2, 8, 64)

    def test_causal_masking(self):
        """Output at position i should not depend on positions j > i."""
        cfg = TopographicConfig(d_model=64, n_heads=2)
        attn = TopographicAttention(cfg)
        attn.eval()
        x = torch.randn(1, 8, 64)

        with torch.no_grad():
            out_full = attn(x)
            # Truncate to first 4 tokens
            out_prefix = attn(x[:, :4, :])

        # First 4 positions should produce same output regardless of future tokens
        assert torch.allclose(out_full[:, :4, :], out_prefix, atol=1e-5)
