"""
Dirichlet Regularization (dreg)
===============================
A mathematically principled framework for Dirichlet spatial regularization of neural
network weights: cortical-inspired topographic smoothness that enables extreme spectral
compressibility. Includes sub-1.0 bpp Base-3 trit quantization and zero-copy DMA inference.
"""

from .baselines import random_orthogonal_transform_2d, svd_low_rank_approximation
from .model import TopographicConfig, TopographicLinear, TopographicTransformer
from .quantization import Base3TritQuantizer, TritQFormat
from .spectral import BlockDCTTiler, dct2d, dct_matrix_1d, idct2d
from .topology import (
    CosineDirichletScheduler,
    DirichletLoss,
    DirichletScheduler,
    StepCooldownScheduler,
    dirichlet_energy_2d,
    get_grid_dimensions,
)

__version__ = "1.0.0"
__all__ = [
    "DirichletLoss",
    "dirichlet_energy_2d",
    "get_grid_dimensions",
    "DirichletScheduler",
    "StepCooldownScheduler",
    "CosineDirichletScheduler",
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

