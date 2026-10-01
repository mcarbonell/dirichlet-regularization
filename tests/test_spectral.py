"""
Tests for spectral module: DCT/IDCT transforms and BlockDCTTiler.
"""

import pytest
import torch

from topospec.spectral import dct_matrix_1d, dct2d, idct2d, BlockDCTTiler


class TestDCTMatrix:
    """Tests for the 1D orthonormal DCT-II basis matrix."""

    def test_orthogonality(self):
        """DCT matrix should be orthonormal: D @ D^T = I."""
        for n in [4, 8, 16, 32, 64]:
            d = dct_matrix_1d(n)
            identity = d @ d.t()
            expected = torch.eye(n)
            assert torch.allclose(identity, expected, atol=1e-5), \
                f"DCT matrix of size {n} is not orthonormal"

    def test_inverse_is_transpose(self):
        """For orthonormal DCT: D^{-1} = D^T."""
        n = 16
        d = dct_matrix_1d(n)
        d_inv = torch.linalg.inv(d)
        assert torch.allclose(d_inv, d.t(), atol=1e-5)

    def test_dc_component(self):
        """Row 0 should be constant = 1/sqrt(n) (DC component)."""
        n = 8
        d = dct_matrix_1d(n)
        dc_row = d[0, :]
        expected_val = 1.0 / (n ** 0.5)
        assert torch.allclose(dc_row, torch.full((n,), expected_val), atol=1e-6)

    def test_caching(self):
        """Same (n, device, dtype) should return cached tensor."""
        d1 = dct_matrix_1d(16)
        d2 = dct_matrix_1d(16)
        assert d1.data_ptr() == d2.data_ptr()  # Same memory


class TestDCT2D:
    """Tests for the separable 2D-DCT and IDCT."""

    def test_roundtrip_identity(self):
        """dct2d → idct2d should be the identity transform."""
        torch.manual_seed(42)
        for shape in [(8, 8), (16, 32), (7, 13), (64, 64)]:
            x = torch.randn(*shape)
            reconstructed = idct2d(dct2d(x))
            assert torch.allclose(x, reconstructed, atol=1e-4), \
                f"Roundtrip failed for shape {shape}, max_diff={( x - reconstructed).abs().max().item()}"

    def test_inverse_roundtrip(self):
        """idct2d → dct2d should also be the identity."""
        torch.manual_seed(99)
        x = torch.randn(16, 16)
        reconstructed = dct2d(idct2d(x))
        assert torch.allclose(x, reconstructed, atol=1e-5)

    def test_energy_preservation(self):
        """Parseval's theorem: ||X||_F == ||S||_F for orthonormal transforms."""
        torch.manual_seed(7)
        x = torch.randn(16, 16)
        s = dct2d(x)
        assert torch.allclose(
            torch.norm(x), torch.norm(s), atol=1e-4
        ), "Energy not preserved (Parseval violation)"

    def test_dc_coefficient(self):
        """The DC coefficient (0,0) of a constant matrix should encode the mean."""
        val = 5.0
        n = 8
        x = torch.full((n, n), val)
        s = dct2d(x)
        # DC coefficient = mean * sqrt(M) * sqrt(N) for orthonormal DCT
        expected_dc = val * n  # sqrt(1/n) * n * val = val * sqrt(n), twice for 2D
        # Actually: DC = sum * (1/sqrt(M)) * (1/sqrt(N)) = val * M * N / (sqrt(M) * sqrt(N)) = val * sqrt(M*N)
        expected_dc = val * (n * n) ** 0.5
        assert abs(s[0, 0].item() - expected_dc) < 1e-4

    def test_zero_input(self):
        x = torch.zeros(8, 8)
        s = dct2d(x)
        assert torch.allclose(s, torch.zeros_like(s))


class TestBlockDCTTiler:
    """Tests for the BlockDCTTiler that partitions matrices into B×B tiles."""

    def test_roundtrip_exact_size(self):
        """Matrix size that's an exact multiple of block_size."""
        torch.manual_seed(42)
        tiler = BlockDCTTiler(block_size=8)
        w = torch.randn(16, 24)
        spec_tiles = tiler.tile_and_transform(w)
        reconstructed = tiler.untile_and_reconstruct(spec_tiles, w.shape)
        assert torch.allclose(w, reconstructed, atol=1e-5)

    def test_roundtrip_needs_padding(self):
        """Matrix size that requires padding to reach block_size multiple."""
        torch.manual_seed(42)
        tiler = BlockDCTTiler(block_size=8)
        w = torch.randn(13, 19)  # Not multiples of 8
        spec_tiles = tiler.tile_and_transform(w)
        reconstructed = tiler.untile_and_reconstruct(spec_tiles, w.shape)
        assert reconstructed.shape == w.shape
        assert torch.allclose(w, reconstructed, atol=1e-5)

    def test_tile_shape(self):
        """Output tiles should have shape (M/B, N/B, B, B)."""
        tiler = BlockDCTTiler(block_size=8)
        w = torch.randn(16, 24)
        spec_tiles = tiler.tile_and_transform(w)
        assert spec_tiles.shape == (2, 3, 8, 8)  # 16/8=2, 24/8=3

    def test_different_block_sizes(self):
        torch.manual_seed(42)
        w = torch.randn(64, 64)
        for bs in [8, 16, 32, 64]:
            tiler = BlockDCTTiler(block_size=bs)
            spec = tiler.tile_and_transform(w)
            rec = tiler.untile_and_reconstruct(spec, w.shape)
            assert torch.allclose(w, rec, atol=1e-4), f"Failed for block_size={bs}"
