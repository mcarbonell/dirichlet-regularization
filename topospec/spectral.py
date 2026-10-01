"""
Spectral Transform & Block-DCT Tiling Module
============================================
Provides fast orthonormal 2D-DCT transforms, basis cache, and fixed-size block tiling
to decouple inverse transform arithmetic O(B^3) from model width D.
"""

import math
import torch
from typing import Tuple, Dict


_DCT_CACHE: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def dct_matrix_1d(n: int, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Constructs the orthonormal DCT-II basis matrix of size (n, n):
        D_{k, i} = sqrt(2/n) * cos(pi * (2i + 1) * k / (2n)),  k > 0
        D_{0, i} = sqrt(1/n)
    """
    key = (n, device, dtype)
    if key in _DCT_CACHE:
        return _DCT_CACHE[key]

    d = torch.zeros((n, n), device=device, dtype=dtype)
    i = torch.arange(n, device=device, dtype=dtype)
    for k in range(n):
        scale = math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)
        d[k, :] = scale * torch.cos(math.pi * (2.0 * i + 1.0) * k / (2.0 * n))

    _DCT_CACHE[key] = d
    return d


def dct2d(x: torch.Tensor) -> torch.Tensor:
    """
    Applies the separable orthonormal 2D-DCT-II to a matrix X:
        S = D_out @ X @ D_in^T
    """
    m, n = x.shape
    d_m = dct_matrix_1d(m, device=x.device, dtype=x.dtype)
    d_n = dct_matrix_1d(n, device=x.device, dtype=x.dtype)
    return d_m @ x @ d_n.t()


def idct2d(s: torch.Tensor) -> torch.Tensor:
    """
    Applies the separable orthonormal 2D-IDCT-II (inverse DCT) to a spectral matrix S:
        X = D_out^T @ S @ D_in
    """
    m, n = s.shape
    d_m = dct_matrix_1d(m, device=s.device, dtype=s.dtype)
    d_n = dct_matrix_1d(n, device=s.device, dtype=s.dtype)
    return d_m.t() @ s @ d_n


class BlockDCTTiler:
    """
    Splits arbitrary (M, N) weight matrices into uniform B x B blocks
    and applies local 2D-DCT transforms. Guarantees that decoding cost
    is strictly O(B^3) per block, solving the Amdahl scaling limit.
    """
    def __init__(self, block_size: int = 64):
        self.b = block_size

    def tile_and_transform(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Pads weight to multiples of B, splits into (M/B, N/B, B, B) tiles,
        and applies dct2d to each tile.
        """
        m, n = weight.shape
        pad_m = (self.b - (m % self.b)) % self.b
        pad_n = (self.b - (n % self.b)) % self.b

        if pad_m > 0 or pad_n > 0:
            padded = torch.nn.functional.pad(weight, (0, pad_n, 0, pad_m))
        else:
            padded = weight

        pm, pn = padded.shape
        tiles = padded.view(pm // self.b, self.b, pn // self.b, self.b).permute(0, 2, 1, 3)
        # Apply 2D DCT to each tile
        d_b = dct_matrix_1d(self.b, device=weight.device, dtype=weight.dtype)
        # Vectorized tile DCT: D @ Tile @ D^T
        spec_tiles = torch.einsum("ij, rcij, kj -> rcik", d_b, tiles, d_b)
        return spec_tiles

    def untile_and_reconstruct(self, spec_tiles: torch.Tensor, orig_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Applies inverse 2D-DCT to each tile and reassembles the original (M, N) matrix.
        """
        d_b = dct_matrix_1d(self.b, device=spec_tiles.device, dtype=spec_tiles.dtype)
        # Vectorized tile IDCT: D^T @ SpecTile @ D
        tiles = torch.einsum("ji, rcij, jk -> rcik", d_b, spec_tiles, d_b)
        pm_tiles, pn_tiles, b, _ = tiles.shape
        reassembled = tiles.permute(0, 2, 1, 3).contiguous().view(pm_tiles * b, pn_tiles * b)
        m, n = orig_shape
        return reassembled[:m, :n]
