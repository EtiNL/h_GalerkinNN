# src/pde_dataset/time_projection.py
"""
Time-sampled projected datasets.

Goal: sample u(t, x) on a fixed spatial grid x_grid, then project each snapshot
onto a basis {phi_k(x)} using a 1D quadrature rule:
  c_k(t) = ∫ u(t,x) phi_k(x) dx  ≈  Σ_j w_j u(t, x_j) phi_k(x_j)

Returns a PyTorch Dataset with items:
  {'t': (1,), 'c': (K,), optionally 'k': (K,)}
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import Device
from .io import save_dataset, DatasetMetadata


# -------------------------
# Quadrature utilities
# -------------------------
def trapz_weights_1d(x: np.ndarray) -> np.ndarray:
    """Trapezoidal-rule weights for 1D grid x (possibly non-uniform)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be 1D with at least 2 points.")
    w = np.empty_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def projection_matrix(
    x_grid: np.ndarray,
    basis_eval: Callable[[np.ndarray], np.ndarray],
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build P of shape (nx, K) such that coeffs C = U @ P for U shape (nT, nx).

    basis_eval(x_grid) must return Phi of shape (K, nx): Phi[k,j]=phi_k(x_j).
    """
    x_grid = np.asarray(x_grid, dtype=float)
    Phi = np.asarray(basis_eval(x_grid), dtype=float)  # (K, nx)
    if Phi.ndim != 2 or Phi.shape[1] != x_grid.size:
        raise ValueError(f"basis_eval must return (K, nx), got {Phi.shape}.")

    if weights is None:
        weights = trapz_weights_1d(x_grid)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (x_grid.size,):
            raise ValueError("weights must have shape (nx,)")

    # c_k ≈ Σ_j w_j u_j Phi_{k,j}.
    # For matrix multiply: U (nT,nx) @ P (nx,K)
    # with P[j,k] = w_j * Phi[k,j]
    P = (weights[None, :] * Phi).T  # (nx,K)
    return P


# -------------------------
# Dataset
# -------------------------
@dataclass
class TimeProjectedDatasetConfig:
    n_time_samples: int = 200
    t_sampling: str = "grid"  # "grid" or "random"
    seed: Optional[int] = None

    # normalization (simple: affine on t and c)
    normalize_t: bool = False
    normalize_c: bool = False

    return_k_coords: bool = False  # include basis index coordinate
    pde_name: str = "unknown"


class TimeProjectedDataset(Dataset):
    """
    Stores tensors:
      t: (nT,1)
      c: (nT,K)
      optionally k_coords: (K,)
    """
    def __init__(
        self,
        config: TimeProjectedDatasetConfig,
        t: np.ndarray,
        c: np.ndarray,
        device: Device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.device = device
        self.dtype = dtype

        t = np.asarray(t, dtype=float).reshape(-1, 1)     # (nT,1)
        c = np.asarray(c, dtype=float)                    # (nT,K)
        if c.ndim != 2 or c.shape[0] != t.shape[0]:
            raise ValueError("c must be (nT,K) and match t.")

        # store normalizer stats (optional)
        self.t_mean = float(t.mean())
        self.t_std = float(t.std() + 1e-8)
        self.c_mean = c.mean(axis=0, keepdims=True)       # (1,K)
        self.c_std = c.std(axis=0, keepdims=True) + 1e-8  # (1,K)

        if config.normalize_t:
            t = (t - self.t_mean) / self.t_std
        if config.normalize_c:
            c = (c - self.c_mean) / self.c_std

        self.t = torch.tensor(t, device=device, dtype=dtype)
        self.c = torch.tensor(c, device=device, dtype=dtype)

        self.k_coords = None
        if config.return_k_coords:
            self.k_coords = torch.arange(self.c.shape[1], device=device, dtype=dtype)

    def __len__(self) -> int:
        return self.t.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        out = {"t": self.t[idx], "c": self.c[idx]}
        if self.k_coords is not None:
            out["k"] = self.k_coords
        return out

    def get_trajectory(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (t: (nT,), c: (nT,K)) in the dataset’s stored (possibly normalized) space."""
        return self.t.squeeze(1), self.c

    def save(self, filepath: str, format: str = "npz") -> None:
        data = {
            "t": self.t.detach().cpu().numpy(),
            "c": self.c.detach().cpu().numpy(),
        }
        if self.k_coords is not None:
            data["k"] = self.k_coords.detach().cpu().numpy()

        metadata = DatasetMetadata(
            dataset_type="time_projected",
            pde_name=self.config.pde_name,
            spatial_dim=1,
            n_samples=len(self),
            bounds=None,
            normalizer={
                "normalize_t": self.config.normalize_t,
                "normalize_c": self.config.normalize_c,
                "t_mean": self.t_mean,
                "t_std": self.t_std,
                "c_mean": self.c_mean.squeeze(0).tolist(),
                "c_std": self.c_std.squeeze(0).tolist(),
            },
            extra_info={
                "n_time_samples": self.config.n_time_samples,
                "t_sampling": self.config.t_sampling,
                "return_k_coords": self.config.return_k_coords,
                "K": int(self.c.shape[1]),
            }
        )
        save_dataset(filepath, data, metadata, format=format)


# -------------------------
# Factory
# -------------------------
def create_time_projected_dataset(
    solution_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_grid: np.ndarray,
    t_min: float,
    t_max: float,
    basis_eval: Callable[[np.ndarray], np.ndarray],  # returns (K, nx)
    n_time_samples: int = 200,
    t_sampling: str = "grid",
    seed: Optional[int] = None,
    weights: Optional[np.ndarray] = None,
    device: Device = "cpu",
    dtype: torch.dtype = torch.float32,
    normalize_t: bool = False,
    normalize_c: bool = False,
    return_k_coords: bool = False,
    pde_name: str = "unknown",
) -> TimeProjectedDataset:
    x_grid = np.asarray(x_grid, dtype=float)

    rng = np.random.default_rng(seed)
    if t_sampling == "grid":
        t = np.linspace(t_min, t_max, n_time_samples, dtype=float)
    elif t_sampling == "random":
        t = rng.uniform(t_min, t_max, size=n_time_samples).astype(float)
        t.sort()
    else:
        raise ValueError("t_sampling must be 'grid' or 'random'")

    P = projection_matrix(x_grid, basis_eval=basis_eval, weights=weights)  # (nx,K)

    # Evaluate U on mesh (nT,nx) in one call if possible
    Tm = np.repeat(t[:, None], x_grid.size, axis=1)
    Xm = np.repeat(x_grid[None, :], t.size, axis=0)
    U = solution_function(Tm, Xm)  # (nT,nx)

    C = np.asarray(U, dtype=float) @ P  # (nT,K)

    cfg = TimeProjectedDatasetConfig(
        n_time_samples=n_time_samples,
        t_sampling=t_sampling,
        seed=seed,
        normalize_t=normalize_t,
        normalize_c=normalize_c,
        return_k_coords=return_k_coords,
        pde_name=pde_name,
    )
    return TimeProjectedDataset(cfg, t=t, c=C, device=device, dtype=dtype)
