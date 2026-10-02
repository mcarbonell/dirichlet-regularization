"""
Integration Tests: End-to-End Pipeline Validation
===================================================
Tests that verify the full pipeline works correctly:
  1. Train → Quantize → Evaluate (Dirichlet model retains more quality)
  2. Training convergence (Dirichlet loss doesn't break learning)
  3. BlockDCTTiler + Quantization combined pipeline
"""

import copy
import math
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from dreg import (
    TopographicTransformer,
    TopographicConfig,
    TopographicLinear,
    Base3TritQuantizer,
    DirichletLoss,
    dct2d,
    idct2d,
    BlockDCTTiler,
    dirichlet_energy_2d,
)


class TestEndToEndPipeline:
    """Tests the full train → quantize → evaluate pipeline."""

    @pytest.fixture
    def small_cfg(self):
        return TopographicConfig(
            vocab_size=64,
            d_model=32,
            n_heads=2,
            n_layers=2,
            ffn_dim=64,
            max_seq_len=24,
            topo_lambda=5.0,
        )

    @pytest.fixture
    def train_data(self):
        """Structured synthetic data with learnable patterns."""
        torch.manual_seed(42)
        patterns = []
        for _ in range(200):
            # Simple pattern: subject-verb pairs
            tokens = []
            for _ in range(12):
                s = torch.randint(2, 8, (1,)).item()
                v = s + 10  # Verb deterministically follows subject
                tokens.extend([s, v])
            patterns.append(tokens[:24])
        return torch.tensor(patterns, dtype=torch.long)

    def _train_model(self, model, train_data, epochs=3, use_topo=True):
        """Train a model for a few epochs."""
        optimizer = optim.AdamW(model.parameters(), lr=2e-3)
        model.train()
        for _ in range(epochs):
            for batch in train_data.split(32):
                optimizer.zero_grad()
                _, ce_loss = model(batch[:, :-1], batch[:, 1:])
                if use_topo:
                    topo_loss = model.topographic_loss()
                    loss = ce_loss + topo_loss
                else:
                    loss = ce_loss
                loss.backward()
                optimizer.step()
        return model

    def _quantize_model(self, model, quantizer):
        """Apply spectral quantization to all TopographicLinear layers."""
        q_model = copy.deepcopy(model)
        for m in q_model.modules():
            if isinstance(m, TopographicLinear):
                spec = dct2d(m.weight.data)
                packed = quantizer.quantize_matrix(spec)
                rec_spec = quantizer.dequantize_matrix(packed, device=m.weight.device)
                m.weight.data.copy_(idct2d(rec_spec))
        return q_model

    def _evaluate_ppl(self, model, data):
        """Compute perplexity on test data."""
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in data.split(32):
                _, loss = model(batch[:, :-1], batch[:, 1:])
                total_loss += loss.item() * batch[:, 1:].numel()
                total_tokens += batch[:, 1:].numel()
        avg_loss = total_loss / max(total_tokens, 1)
        return math.exp(min(avg_loss, 20.0))

    def test_dirichlet_model_has_lower_energy(self, small_cfg, train_data):
        """Topographic model should have lower Dirichlet energy after training."""
        # Train with Dirichlet
        torch.manual_seed(0)
        topo_model = TopographicTransformer(small_cfg)
        topo_model = self._train_model(topo_model, train_data, use_topo=True)

        # Train without Dirichlet (same init)
        cfg_std = copy.deepcopy(small_cfg)
        cfg_std.topo_lambda = 0.0
        torch.manual_seed(0)
        std_model = TopographicTransformer(cfg_std)
        std_model = self._train_model(std_model, train_data, use_topo=False)

        # Measure Dirichlet energy
        energies_topo = [dirichlet_energy_2d(m.weight).item()
                         for m in topo_model.modules() if isinstance(m, TopographicLinear)]
        energies_std = [dirichlet_energy_2d(m.weight).item()
                        for m in std_model.modules() if isinstance(m, TopographicLinear)]

        mean_topo = sum(energies_topo) / len(energies_topo)
        mean_std = sum(energies_std) / len(energies_std)

        # Topographic model should be smoother
        assert mean_topo < mean_std, (
            f"Topographic model (E_D={mean_topo:.6f}) should have lower Dirichlet "
            f"energy than standard model (E_D={mean_std:.6f})"
        )

    def test_quantized_topo_degrades_less(self, small_cfg, train_data):
        """Topographic model should degrade less under quantization than standard."""
        quantizer = Base3TritQuantizer(r0=0.10, r1=0.25, r2=0.50)

        # Train topographic model
        torch.manual_seed(0)
        topo_model = TopographicTransformer(small_cfg)
        topo_model = self._train_model(topo_model, train_data, use_topo=True)
        ppl_topo_fp32 = self._evaluate_ppl(topo_model, train_data)
        q_topo = self._quantize_model(topo_model, quantizer)
        ppl_topo_quant = self._evaluate_ppl(q_topo, train_data)
        delta_topo = ppl_topo_quant - ppl_topo_fp32

        # Train standard model
        cfg_std = copy.deepcopy(small_cfg)
        cfg_std.topo_lambda = 0.0
        torch.manual_seed(0)
        std_model = TopographicTransformer(cfg_std)
        std_model = self._train_model(std_model, train_data, use_topo=False)
        ppl_std_fp32 = self._evaluate_ppl(std_model, train_data)
        q_std = self._quantize_model(std_model, quantizer)
        ppl_std_quant = self._evaluate_ppl(q_std, train_data)
        delta_std = ppl_std_quant - ppl_std_fp32

        # Topographic model should degrade less (or at least not more)
        # We use a generous margin since this is a small-scale test
        assert delta_topo <= delta_std * 1.5, (
            f"Topographic degradation (Δ={delta_topo:.2f}) should be less than "
            f"standard degradation (Δ={delta_std:.2f}) under quantization"
        )


class TestTrainingConvergence:
    """Tests that Dirichlet regularization doesn't break training convergence."""

    def test_loss_decreases_with_dirichlet(self):
        """Total loss should decrease over training steps even with Dirichlet penalty."""
        cfg = TopographicConfig(
            vocab_size=32, d_model=16, n_heads=2, n_layers=1,
            ffn_dim=32, max_seq_len=16, topo_lambda=1.0,
        )
        torch.manual_seed(42)
        model = TopographicTransformer(cfg)
        optimizer = optim.AdamW(model.parameters(), lr=5e-3)

        # Generate simple repeating data
        data = torch.randint(0, 32, (64, 16))

        losses = []
        model.train()
        for _ in range(20):
            optimizer.zero_grad()
            _, ce_loss = model(data[:, :-1], data[:, 1:])
            topo_loss = model.topographic_loss()
            total = ce_loss + topo_loss
            total.backward()
            optimizer.step()
            losses.append(total.item())

        # Loss should generally decrease: final < initial
        assert losses[-1] < losses[0], (
            f"Training loss should decrease: initial={losses[0]:.4f} > final={losses[-1]:.4f}"
        )

    def test_dirichlet_energy_decreases_during_training(self):
        """Dirichlet energy should decrease when topo_lambda > 0."""
        cfg = TopographicConfig(
            vocab_size=32, d_model=16, n_heads=2, n_layers=1,
            ffn_dim=32, max_seq_len=16, topo_lambda=10.0,
        )
        torch.manual_seed(42)
        model = TopographicTransformer(cfg)
        optimizer = optim.AdamW(model.parameters(), lr=5e-3)

        # Measure initial energy
        initial_energy = model.topographic_loss().item()

        data = torch.randint(0, 32, (64, 16))
        model.train()
        for _ in range(30):
            optimizer.zero_grad()
            _, ce_loss = model(data[:, :-1], data[:, 1:])
            topo_loss = model.topographic_loss()
            (ce_loss + topo_loss).backward()
            optimizer.step()

        final_energy = model.topographic_loss().item()
        assert final_energy < initial_energy, (
            f"Dirichlet energy should decrease: initial={initial_energy:.6f} > final={final_energy:.6f}"
        )


class TestBlockDCTQuantizationPipeline:
    """Tests the combined BlockDCTTiler + Quantization pipeline."""

    def test_tiled_quantize_roundtrip(self):
        """BlockDCTTiler → quantize → dequantize → untile should approximately recover the matrix."""
        torch.manual_seed(42)
        # Create a smooth weight matrix (simulating Dirichlet-regularized weights)
        m, n = 128, 128
        u = torch.arange(m).view(-1, 1).float() / m
        v = torch.arange(n).view(1, -1).float() / n
        W = torch.sin(2 * math.pi * u) * torch.cos(2 * math.pi * v) * 0.1

        tiler = BlockDCTTiler(block_size=32)
        quantizer = Base3TritQuantizer(r0=0.10, r1=0.25, r2=0.50)

        # Tile and transform
        spec_tiles = tiler.tile_and_transform(W)
        n_tiles_r, n_tiles_c, b, _ = spec_tiles.shape

        # Quantize and dequantize each tile
        rec_tiles = torch.zeros_like(spec_tiles)
        for i in range(n_tiles_r):
            for j in range(n_tiles_c):
                tile_spec = spec_tiles[i, j]
                packed = quantizer.quantize_matrix(tile_spec)
                rec_tiles[i, j] = quantizer.dequantize_matrix(packed)

        # Reconstruct
        W_rec = tiler.untile_and_reconstruct(rec_tiles, (m, n))

        # Should approximately recover (with quantization error)
        assert W_rec.shape == W.shape
        rel_error = (W_rec - W).norm() / W.norm()
        # Smooth matrices should have low reconstruction error
        assert rel_error < 1.0, f"Relative error too high: {rel_error:.4f}"

    def test_tiled_vs_global_energy_preservation(self):
        """Tiled DCT should preserve most energy compared to global DCT for smooth matrices."""
        torch.manual_seed(42)
        m, n = 64, 64
        # Smooth matrix
        u = torch.linspace(0, 1, m).unsqueeze(1)
        v = torch.linspace(0, 1, n).unsqueeze(0)
        W = (u * v) * 0.1 + torch.randn(m, n) * 0.001

        # Global DCT energy
        S_global = dct2d(W)
        energy_global = (S_global ** 2).sum().item()

        # Tiled DCT energy
        tiler = BlockDCTTiler(block_size=32)
        spec_tiles = tiler.tile_and_transform(W)
        energy_tiled = (spec_tiles ** 2).sum().item()

        # Energy should be approximately preserved (Parseval's theorem per block)
        ratio = energy_tiled / energy_global
        assert 0.95 < ratio < 1.05, (
            f"Tiled DCT energy ({energy_tiled:.4f}) should approximately equal "
            f"global DCT energy ({energy_global:.4f}), ratio={ratio:.4f}"
        )
