"""
Topographic Spectral Transformers (topospec)
============================================
A mathematically principled framework for biological-inspired topographic regularization,
sub-1.0 bpp Base-3 quantum trit quantization, and zero-copy streaming DMA inference in silicon.
"""

from .topology import DirichletLoss, dirichlet_energy_2d, get_grid_dimensions
from .spectral import dct_matrix_1d, dct2d, idct2d, BlockDCTTiler
from .quantization import Base3TritQuantizer, TritQFormat
from .model import TopographicTransformer, TopographicConfig, TopographicLinear

__version__ = "1.0.0"
__all__ = [
    "DirichletLoss",
    "dirichlet_energy_2d",
    "get_grid_dimensions",
    "dct_matrix_1d",
    "dct2d",
    "idct2d",
    "BlockDCTTiler",
    "Base3TritQuantizer",
    "TritQFormat",
    "TopographicTransformer",
    "TopographicConfig",
    "TopographicLinear",
]
