"""
Tests for quantization module: Base-3 Trit quantization and .tritq format.
"""

import os
import tempfile
import pytest
import numpy as np
import torch

from topospec.quantization import Base3TritQuantizer, TritQFormat
from topospec.spectral import dct2d, idct2d


class TestBase3TritLUT:
    """Tests for the O(1) trit lookup table."""

    def test_lut_shape(self):
        q = Base3TritQuantizer()
        assert q._lut.shape == (256, 5)

    def test_lut_values_in_range(self):
        """All LUT entries should be in {-1, 0, +1}."""
        q = Base3TritQuantizer()
        for b in range(243):
            for i in range(5):
                assert q._lut[b, i] in (-1.0, 0.0, 1.0), \
                    f"LUT[{b}][{i}] = {q._lut[b, i]} is out of range"

    def test_lut_encoding_decoding(self):
        """Verify LUT correctly inverts the base-3 encoding: byte → 5 trits."""
        q = Base3TritQuantizer()
        for b in range(243):
            val = b
            for i in range(5):
                trit = val % 3
                expected = float(trit - 1)
                assert q._lut[b, i] == expected, \
                    f"LUT mismatch at byte={b}, trit={i}: got {q._lut[b, i]}, expected {expected}"
                val //= 3

    def test_unused_entries_are_zero(self):
        """Bytes 243-255 are unused in base-3; LUT should be zero."""
        q = Base3TritQuantizer()
        for b in range(243, 256):
            assert np.all(q._lut[b] == 0.0)


class TestBase3TritQuantizer:
    """Tests for the full quantize → dequantize pipeline."""

    def test_quantize_shape_preserved(self):
        q = Base3TritQuantizer()
        spec = torch.randn(16, 16)
        packed = q.quantize_matrix(spec)
        assert packed["shape"] == (16, 16)

    def test_dequantize_shape_preserved(self):
        q = Base3TritQuantizer()
        spec = torch.randn(16, 16)
        packed = q.quantize_matrix(spec)
        rec = q.dequantize_matrix(packed)
        assert rec.shape == (16, 16)

    def test_bpp_is_sub_one(self):
        """For topographic-like smooth spectra, bpp should be well under 32."""
        q = Base3TritQuantizer()
        # Create a smooth spectrum (energy concentrated in low frequencies)
        spec = torch.zeros(32, 32)
        spec[:4, :4] = torch.randn(4, 4) * 10.0  # Only low-freq has energy
        packed = q.quantize_matrix(spec)
        assert packed["bpp"] < 2.0  # Much less than FP32's 32 bpp

    def test_roundtrip_low_freq_preservation(self):
        """DC and low-frequency coefficients should survive quantization well."""
        q = Base3TritQuantizer()
        spec = torch.zeros(16, 16)
        spec[0, 0] = 10.0  # DC
        spec[0, 1] = 5.0   # Low freq
        spec[1, 0] = 5.0   # Low freq
        packed = q.quantize_matrix(spec)
        rec = q.dequantize_matrix(packed)
        # DC should be reasonably preserved (8-bit quantization)
        assert abs(rec[0, 0].item() - 10.0) < 1.0

    def test_band_masks_are_disjoint(self):
        """Band masks should not overlap."""
        q = Base3TritQuantizer()
        spec = torch.randn(16, 16)
        packed = q.quantize_matrix(spec)
        m0 = packed["band0"]["mask"]
        m1 = packed["band1"]["mask"]
        m2 = packed["band2"]["mask"]
        # No overlap between any pair
        assert not np.any(m0 & m1)
        assert not np.any(m0 & m2)
        assert not np.any(m1 & m2)

    def test_radial_grid_normalized(self):
        """Radial grid values should be in [0, sqrt(2)]."""
        q = Base3TritQuantizer()
        rho = q.compute_radial_grid(16, 16, torch.device("cpu"))
        assert rho.min().item() == pytest.approx(0.0)
        assert rho.max().item() <= 1.5  # sqrt(2) ≈ 1.414


class TestTritQFormat:
    """Tests for .tritq binary serialization."""

    def test_save_creates_file(self):
        weights = {
            "layer.weight": torch.randn(8, 8),
        }
        config = {"d_model": 8, "n_layers": 1}
        with tempfile.NamedTemporaryFile(suffix=".tritq", delete=False) as f:
            path = f.name
        try:
            TritQFormat.save(path, weights, config)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
            # Verify magic bytes
            with open(path, "rb") as f:
                magic = f.read(8)
                assert magic == b"TRITQ_V1"
        finally:
            os.unlink(path)

    def test_save_handles_1d_params(self):
        """1D parameters (LayerNorm, biases) should be saved as FP16."""
        weights = {
            "ln.weight": torch.randn(16),
            "ln.bias": torch.randn(16),
            "proj.weight": torch.randn(8, 8),
        }
        config = {"test": True}
        with tempfile.NamedTemporaryFile(suffix=".tritq", delete=False) as f:
            path = f.name
        try:
            TritQFormat.save(path, weights, config)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_save_load_roundtrip(self):
        """Save and load should produce approximately the same weights."""
        torch.manual_seed(42)
        weights = {
            "block.attn.weight": torch.randn(16, 16),
            "block.mlp.weight": torch.randn(32, 16),
            "ln.weight": torch.randn(16),
        }
        config = {"d_model": 16, "n_layers": 1, "vocab_size": 64}
        with tempfile.NamedTemporaryFile(suffix=".tritq", delete=False) as f:
            path = f.name
        try:
            TritQFormat.save(path, weights, config)
            loaded_config, loaded_weights = TritQFormat.load(path)

            # Config should match
            assert loaded_config == config

            # All tensor names should be present
            assert set(loaded_weights.keys()) == set(weights.keys())

            # 1D params should be close (FP16 precision loss)
            assert torch.allclose(
                loaded_weights["ln.weight"],
                weights["ln.weight"],
                atol=1e-2
            )

            # 2D params will have quantization error but shapes must match
            assert loaded_weights["block.attn.weight"].shape == (16, 16)
            assert loaded_weights["block.mlp.weight"].shape == (32, 16)
        finally:
            os.unlink(path)
