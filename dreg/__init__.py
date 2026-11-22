"""
Dirichlet Regularization (dreg)
===============================
A mathematically principled framework for 2D Dirichlet spatial regularization of neural
network weights: cortical-inspired topographic smoothness that concentrates 2D-DCT
spectral energy and improves low-bit Base-3 trit spectral quantization.
"""

from .baselines import random_orthogonal_transform_2d, svd_low_rank_approximation
from .model import TopographicConfig, TopographicLinear, TopographicTransformer
from .quantization import Base3TritQuantizer, TritQFormat
from .spectral import BlockDCTTiler, dct1d, dct2d, dct_matrix_1d, idct1d, idct2d, spectral_matvec
from .topology import (
    CosineDirichletScheduler,
    DirichletLoss,
    DirichletScheduler,
    StepCooldownScheduler,
    dirichlet_energy_2d,
    get_grid_dimensions,
)

__version__ = "1.1.0"
__all__ = [
    "DirichletLoss",
    "dirichlet_energy_2d",
    "get_grid_dimensions",
    "DirichletScheduler",
    "StepCooldownScheduler",
    "CosineDirichletScheduler",
    "dct_matrix_1d",
    "dct1d",
    "idct1d",
    "dct2d",
    "idct2d",
    "spectral_matvec",
    "BlockDCTTiler",
    "Base3TritQuantizer",
    "TritQFormat",
    "TopographicTransformer",
    "TopographicConfig",
    "TopographicLinear",
    "random_orthogonal_transform_2d",
    "svd_low_rank_approximation",
]

