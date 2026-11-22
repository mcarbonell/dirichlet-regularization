"""
Topological Dirichlet Regularization Module
===========================================
Implements 2D cortical manifold embedding and Dirichlet harmonic pinning
to eliminate permutation invariance and concentrate energy into low spatial frequencies.
"""

import math
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn


def get_grid_dimensions(n: int) -> Tuple[int, int]:
    """
    Finds the integer factor pair (H, W) such that H * W = n
    with the aspect ratio closest to 1.0 (square-like 2D cortical lattice).
    """
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    sqrt_n = int(math.isqrt(n))
    for h in range(sqrt_n, 0, -1):
        if n % h == 0:
            return h, n // h
    return 1, n


def dirichlet_energy_2d(
    weight: torch.Tensor,
    grid_shape: Optional[Tuple[int, int]] = None,
    normalization: str = "numel",
) -> torch.Tensor:
    """
    Computes the 2D Dirichlet Harmonic Energy of a weight matrix.

    Variational form (discrete unit lattice normalized by 2*M*N):
        E_var(W) = 1/(2*M*N) * sum_{(u,v) in E} ||w_u - w_v||^2

    Implementation note — factor 2:
        For backward compatibility the ``'numel'`` mode returns
        ``sum_sq / (M*N)`` which equals ``2 * E_var``.  The ``'edges'``
        mode (``sum_sq / num_edges``) is asymptotically equal to ``E_var``
        for large matrices since ``num_edges ≈ 2*M*N``.  Both modes are
        monotonic in roughness; only the absolute scale differs by ~2×.

    Args:
        weight: Tensor of shape (out_features, in_features)
        grid_shape: Optional (H, W) tuple. If it equals ``weight.shape``, the
                    natural 2D grid is used. If ``H * W`` matches ``weight.shape[0]``
                    or ``weight.shape[1]``, that axis is reshaped onto a 2D sheet.
        normalization: Normalization mode for the energy scalar:
            - 'numel' (default): Normalizes by total elements (M * N). Backward-compatible.
              Equals 2× the variational form.
            - 'edges': Normalizes by the exact number of grid edges ((M-1)*N + M*(N-1)),
              providing scale-invariant roughness estimation across varying matrix shapes.
              Asymptotically equal to the variational form.
            - 'rayleigh': Variance-normalized Rayleigh quotient of the 2D Neumann Laplacian:
              ``sum_sq / (2 * ||W||_F^2 + 1e-8)`` in ``[0, 4]``. Strictly scale-invariant
              (immune to L2 magnitude shrinkage) and dimensionless across matrix scales.
            - 'rayleigh_sg': Stop-gradient Rayleigh normalization:
              ``sum_sq / (2 * sg(||W||_F^2) + 1e-8)`` in ``[0, 4]``. Same forward value as
              'rayleigh', but detaches the Frobenius norm denominator so backward yields pure
              scale-invariant Neumann heat diffusion (-Delta W / ||W||_F^2) with zero
              anti-decay / outlier inflation.
            - 'none': Raw sum of squared differences without division.

    Returns:
        Scalar torch.Tensor representing the normalized Dirichlet energy.
    """
    if grid_shape is not None:
        h, w = grid_shape
        if weight.dim() == 2 and (h, w) == (weight.shape[0], weight.shape[1]):
            sheet = weight
        elif weight.shape[0] == h * w:
            sheet = weight.reshape(h, w, -1)
        elif weight.shape[1] == h * w:
            sheet = weight.reshape(-1, h, w).permute(1, 2, 0)
        else:
            raise ValueError(
                f"grid_shape {grid_shape} is incompatible with weight shape {tuple(weight.shape)}"
            )
    else:
        sheet = weight

    if sheet.dim() == 2:
        diff_h = sheet[1:, :] - sheet[:-1, :]
        diff_w = sheet[:, 1:] - sheet[:, :-1]
        sum_sq = diff_h.pow(2).sum() + diff_w.pow(2).sum()
        h, w = sheet.shape
        num_edges = (h - 1) * w + h * (w - 1)
    elif sheet.dim() == 3:
        diff_h = sheet[1:, :, :] - sheet[:-1, :, :]
        diff_w = sheet[:, 1:, :] - sheet[:, :-1, :]
        sum_sq = diff_h.pow(2).sum() + diff_w.pow(2).sum()
        h, w, c = sheet.shape
        num_edges = ((h - 1) * w + h * (w - 1)) * c
    else:
        raise ValueError(f"Unsupported weight tensor shape for Dirichlet energy: {sheet.shape}")

    if normalization == "numel":
        return sum_sq / max(sheet.numel(), 1)
    elif normalization == "edges":
        return sum_sq / max(num_edges, 1)
    elif normalization == "rayleigh":
        frob_sq = sheet.pow(2).sum()
        return sum_sq / (2.0 * frob_sq + 1e-8)
    elif normalization == "rayleigh_sg":
        frob_sq = sheet.pow(2).sum().detach()
        return sum_sq / (2.0 * frob_sq + 1e-8)
    elif normalization == "none":
        return sum_sq
    else:
        raise ValueError(
            f"Unknown normalization mode: {normalization}. Expected 'numel', 'edges', 'rayleigh', 'rayleigh_sg', or 'none'."
        )


class DirichletLoss(nn.Module):
    """
    PyTorch loss wrapper for Dirichlet harmonic pinning.
    Traverses 2D linear projections and accumulates topological gradient penalty.

    By default, ``nn.Embedding`` layers and modules matching ``exclude_names``
    (e.g. ``lm_head``) are skipped because token indices have no spatial topology.
    Frozen weights (``requires_grad=False``) are skipped and tied weights are
    deduplicated by storage pointer.

    Args:
        weight_decay: Coupling strength lambda_topo.
        normalization: Per-matrix normalization ('numel', 'edges', 'rayleigh', 'none').
        layer_weighting: Aggregation across layers:
            - 'uniform' (default): Equal average across layers (1/L).
            - 'param_weighted': Weighted by parameter share (M_l*N_l / sum M_j*N_j),
              eliminating per-weight gradient attenuation on wider layers (e.g. FFNs).
        exclude_embeddings: Skip nn.Embedding layers (default True).
        exclude_names: Module name substrings to skip (default ("lm_head",)).
    """
    def __init__(
        self,
        weight_decay: float = 0.01,
        normalization: str = "numel",
        layer_weighting: str = "uniform",
        exclude_embeddings: bool = True,
        exclude_names: Tuple[str, ...] = ("lm_head",),
    ):
        super().__init__()
        if layer_weighting not in {"uniform", "param_weighted"}:
            raise ValueError(
                f"Unknown layer_weighting: {layer_weighting}. Expected 'uniform' or 'param_weighted'."
            )
        self.weight_decay = weight_decay
        self.normalization = normalization
        self.layer_weighting = layer_weighting
        self.exclude_embeddings = exclude_embeddings
        self.exclude_names = tuple(n.lower() for n in exclude_names)

    def _is_excluded(self, name: str, mod: nn.Module) -> bool:
        if self.exclude_embeddings and isinstance(mod, nn.Embedding):
            return True
        if name:
            leaf = name.rsplit(".", 1)[-1].lower()
            if any(leaf == ex or ex in name.lower() for ex in self.exclude_names):
                return True
        return False

    def forward(self, target: Union[nn.Module, Iterable[nn.Module]]) -> torch.Tensor:
        if isinstance(target, nn.Module):
            named_iter = target.named_modules()
        else:
            named_iter = (("", mod) for mod in target)

        energies = []
        numels = []
        seen_ptrs = set()

        for name, mod in named_iter:
            if self._is_excluded(name, mod):
                continue
            weight = getattr(mod, "weight", None)
            if weight is None or not isinstance(weight, torch.Tensor) or weight.dim() != 2:
                continue
            if not weight.requires_grad:
                continue
            ptr = weight.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)

            energies.append(dirichlet_energy_2d(weight, normalization=self.normalization))
            numels.append(weight.numel())

        if not energies:
            return torch.tensor(0.0)

        if self.layer_weighting == "param_weighted":
            total_numel = float(sum(numels))
            agg = sum(e * (n / total_numel) for e, n in zip(energies, numels))
        else:
            agg = sum(energies) / len(energies)

        return self.weight_decay * agg


class DirichletScheduler:
    """
    Base dynamic schedule manager for Dirichlet loss weight_decay (surface tension).
    Allows scheduling lambda_topo over training steps (e.g. cosine annealing, step cooldown).
    """

    def __init__(
        self,
        loss_fn: DirichletLoss,
        total_steps: int,
        schedule: str = "cosine",
        initial_weight_decay: Optional[float] = None,
        min_weight_decay: float = 0.0,
        cooldown_ratio: float = 0.20,
    ):
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if not (0.0 <= cooldown_ratio <= 1.0):
            raise ValueError(f"cooldown_ratio must be in [0.0, 1.0], got {cooldown_ratio}")

        self.loss_fn = loss_fn
        self.total_steps = total_steps
        self.schedule = schedule.lower()
        self.initial_weight_decay = (
            float(initial_weight_decay) if initial_weight_decay is not None else float(loss_fn.weight_decay)
        )
        self.min_weight_decay = float(min_weight_decay)
        self.cooldown_ratio = float(cooldown_ratio)
        self.current_step = 0

        valid_schedules = {"constant", "step_cooldown", "cosine", "linear"}
        if self.schedule not in valid_schedules:
            raise ValueError(f"Unknown schedule: {self.schedule}. Supported: {sorted(valid_schedules)}")

    def get_weight_decay(self, step: Optional[int] = None) -> float:
        """Computes the scheduled weight_decay for the specified step."""
        t = self.current_step if step is None else step
        if t < 0:
            t = 0

        l0 = self.initial_weight_decay
        l_min = self.min_weight_decay
        t_total = self.total_steps

        if self.schedule == "constant":
            return l0
        elif self.schedule == "step_cooldown":
            cooldown_start = int((1.0 - self.cooldown_ratio) * t_total)
            return l0 if t < cooldown_start else l_min
        elif self.schedule == "cosine":
            if t >= t_total:
                return l_min
            progress = min(max(t / t_total, 0.0), 1.0)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return l_min + (l0 - l_min) * cosine_factor
        elif self.schedule == "linear":
            if t >= t_total:
                return l_min
            progress = min(max(t / t_total, 0.0), 1.0)
            return l0 + (l_min - l0) * progress
        return l0

    def step(self, step: Optional[int] = None) -> float:
        """
        Advances the scheduler by one step (or jumps to specified step),
        updates loss_fn.weight_decay, and returns the new value.
        """
        if step is not None:
            self.current_step = step
        wd = self.get_weight_decay(self.current_step)
        self.loss_fn.weight_decay = wd
        if step is None:
            self.current_step += 1
        else:
            self.current_step = step + 1
        return wd

    def state_dict(self) -> dict:
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "initial_weight_decay": self.initial_weight_decay,
            "min_weight_decay": self.min_weight_decay,
            "cooldown_ratio": self.cooldown_ratio,
            "schedule": self.schedule,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.current_step = state_dict["current_step"]
        self.total_steps = state_dict["total_steps"]
        self.initial_weight_decay = state_dict["initial_weight_decay"]
        self.min_weight_decay = state_dict["min_weight_decay"]
        self.cooldown_ratio = state_dict.get("cooldown_ratio", 0.20)
        self.schedule = state_dict["schedule"]
        self.loss_fn.weight_decay = self.get_weight_decay(self.current_step)


class StepCooldownScheduler(DirichletScheduler):
    """
    Convenience wrapper for two-stage step cooldown annealing:
    lambda(t) = lambda_0 for t < (1 - cooldown_ratio) * total_steps, else min_weight_decay.
    """

    def __init__(
        self,
        loss_fn: DirichletLoss,
        total_steps: int,
        cooldown_ratio: float = 0.20,
        initial_weight_decay: Optional[float] = None,
        min_weight_decay: float = 0.0,
    ):
        super().__init__(
            loss_fn=loss_fn,
            total_steps=total_steps,
            schedule="step_cooldown",
            initial_weight_decay=initial_weight_decay,
            min_weight_decay=min_weight_decay,
            cooldown_ratio=cooldown_ratio,
        )


class CosineDirichletScheduler(DirichletScheduler):
    """
    Convenience wrapper for smooth half-cosine decay annealing:
    lambda(t) = lambda_min + 0.5 * (lambda_0 - lambda_min) * (1 + cos(pi * t / T))
    """

    def __init__(
        self,
        loss_fn: DirichletLoss,
        total_steps: int,
        initial_weight_decay: Optional[float] = None,
        min_weight_decay: float = 0.0,
    ):
        super().__init__(
            loss_fn=loss_fn,
            total_steps=total_steps,
            schedule="cosine",
            initial_weight_decay=initial_weight_decay,
            min_weight_decay=min_weight_decay,
        )

