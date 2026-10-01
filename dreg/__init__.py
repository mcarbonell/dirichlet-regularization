"""
Dirichlet Regularization (dreg)
===============================
A mathematically principled framework for Dirichlet spatial regularization of neural
network weights: cortical-inspired topographic smoothness that enables extreme spectral
compressibility. Includes sub-1.0 bpp Base-3 trit quantization and zero-copy DMA inference.
"""

from .topology import DirichletLoss, dirichlet_energy_2d, get_grid_dimensions
from .spectral import dct_matrix_1d, dct2d, idct2d, BlockDCTTiler
from .quantization import Base3TritQuantizer, TritQFormat
from .model import TopographicTransformer, TopographicConfig, TopographicLinear
from .baselines import random_orthogonal_transform_2d, svd_low_rank_approximation

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
    "random_orthogonal_transform_2d",
    "svd_low_rank_approximation",
]
