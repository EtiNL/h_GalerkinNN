"""
Time-sampled projected datasets.

Goal: sample u(t, x) on a fixed spatial grid x_grid, then project each snapshot
onto a basis {phi_k(x)} using a 1D quadrature rule:
  c_k(t) = ∫ u(t,x) phi_k(x) dx  ≈  Σ_j w_j u(t, x_j) phi_k(x_j)

Returns a PyTorch Dataset with items:
  {'t': (1,), 'c': (K,), optionally 'k': (K,)}
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any, Tuple, Union, List
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import Device
from .io import save_dataset, load_dataset, DatasetMetadata


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
class NeuralGalerkinDatasetConfig:
    n_time_samples: int
    t_sampling: str
    seed: Optional[int]
    normalize_t: bool
    normalize_c: bool
    return_k_coords: bool
    pde_name: str


class NeuralGalerkinDataset(Dataset):
    """
    Stores tensors:
      t: (M,nT)   (or normalized)
      c: (M,nT,K) (or normalized)
      optionally k_coords: (K,)
    Item = one trajectory (one initial condition):
      {'t': (nT,), 'c': (nT,K), 'id': int}
    """
    def __init__(
        self,
        config: NeuralGalerkinDatasetConfig,
        t: np.ndarray,          # (M,nT)
        c: np.ndarray,          # (M,nT,K)
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        x_grid: Optional[np.ndarray] = None,
        basis_matrix: Optional[np.ndarray] = None,
    ):
        self.config = config
        self.device = device
        self.dtype = dtype

        t = np.asarray(t, dtype=float)
        c = np.asarray(c, dtype=float)

        if t.ndim != 2:
            raise ValueError("t must be (M,nT)")
        if c.ndim != 3 or c.shape[0] != t.shape[0] or c.shape[1] != t.shape[1]:
            raise ValueError("c must be (M,nT,K) and match t on first 2 dims")

        self.M, self.nT, self.K = c.shape

        # global normalizer stats (across all ICs and times)
        self.t_mean = float(t.mean())
        self.t_std = float(t.std() + 1e-8)
        self.c_mean = c.reshape(-1, self.K).mean(axis=0, keepdims=True)       # (1,K)
        self.c_std  = c.reshape(-1, self.K).std(axis=0, keepdims=True) + 1e-8 # (1,K)

        if config.normalize_t:
            t = (t - self.t_mean) / self.t_std
        if config.normalize_c:
            c = (c - self.c_mean[None, None, :]) / self.c_std[None, None, :]

        self.t = torch.tensor(t, device=device, dtype=torch.float64)
        self.c = torch.tensor(c, device=device, dtype=dtype)


        self.k_coords = None
        if config.return_k_coords:
            self.k_coords = torch.arange(self.K, device=device, dtype=dtype)

        self.x_grid = None
        self.Phi = None
        if x_grid is not None:
            self.x_grid = np.asarray(x_grid, dtype=float)

        if basis_matrix is not None:
            Phi_np = np.asarray(basis_matrix, dtype=float)
            if Phi_np.ndim != 2:
                raise ValueError("basis_matrix must be (K,nx)")
            if Phi_np.shape[0] != self.K:
                raise ValueError(f"basis_matrix first dim must be K={self.K}, got {Phi_np.shape[0]}")
            if self.x_grid is not None and Phi_np.shape[1] != self.x_grid.size:
                raise ValueError("basis_matrix second dim must match x_grid length")
            self.Phi = torch.tensor(Phi_np, device=device, dtype=dtype)


    def __len__(self) -> int:
        return self.M

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        out = {"t": self.t[idx], "c": self.c[idx], "id": torch.tensor(idx, device=self.device)}
        if self.k_coords is not None:
            out["k"] = self.k_coords
        return out

    def get_trajectory(self, idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """(t: (nT,), c: (nT,K)) for a given IC index."""
        return self.t[idx], self.c[idx]

    def get_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(t: (M,nT), c: (M,nT,K))"""
        return self.t, self.c

    def denormalize_c(self, c: torch.Tensor) -> torch.Tensor:
        """c is (...,K) in stored space -> physical coeffs."""
        if not self.config.normalize_c:
            return c
        mean = torch.as_tensor(self.c_mean, device=c.device, dtype=c.dtype)  # (1,K)
        std  = torch.as_tensor(self.c_std,  device=c.device, dtype=c.dtype)  # (1,K)
        return c * std + mean

    def denormalize_t(self, t: torch.Tensor) -> torch.Tensor:
        if not self.config.normalize_t:
            return t
        return t * self.t_std + self.t_mean
    
    def reconstruct_u(self, c_traj: torch.Tensor, denormalize: bool = True) -> torch.Tensor:
        """
        c_traj: (nT,K) or (K,)
        returns: (nT,nx) or (nx,)
        """
        if self.Phi is None:
            raise ValueError("No basis_matrix stored in dataset (Phi is None).")
        if denormalize:
            c_traj = self.denormalize_c(c_traj)
        return c_traj.to(self.Phi.device, self.Phi.dtype) @ self.Phi  # matmul works for 1D/2D

    def get_reconstruction_grid(self) -> np.ndarray:
        if self.x_grid is None:
            raise ValueError("No x_grid stored in dataset.")
        return self.x_grid

    def save(self, filepath: str, format: str = "npz") -> None:
        """
        Save in *physical* space (i.e. unnormalized t,c), and store normalizer stats in metadata.
        This makes reload + (re)normalization consistent and avoids double-normalizing.
        """
        # --- physical payload ---
        t_phys = self.denormalize_t(self.t).detach().cpu().numpy()    # (M,nT)
        c_phys = self.denormalize_c(self.c).detach().cpu().numpy()    # (M,nT,K)

        data = {"t": t_phys, "c": c_phys}

        if self.k_coords is not None:
            data["k"] = self.k_coords.detach().cpu().numpy()

        if self.x_grid is not None:
            data["x_grid"] = np.asarray(self.x_grid, dtype=float)

        if self.Phi is not None:
            data["Phi"] = self.Phi.detach().cpu().numpy()            # (K,nx)

        # Save basis parameters
        if hasattr(self, 'transformation_matrix') and self.transformation_matrix is not None:
            data["transformation_matrix"] = np.asarray(self.transformation_matrix, dtype=float)

        # --- bounds (optional, but useful) ---
        bounds = {
            "t_min": float(t_phys.min()),
            "t_max": float(t_phys.max()),
        }
        if "x_grid" in data:
            bounds["x_min"] = float(data["x_grid"].min())
            bounds["x_max"] = float(data["x_grid"].max())

        # --- metadata ---
        metadata = DatasetMetadata(
            dataset_type="neural_galerkin",
            pde_name=self.config.pde_name,
            spatial_dim=1,
            n_samples=int(self.M),  # number of trajectories / ICs
            bounds=bounds,
            normalizer={
                "normalize_t": bool(self.config.normalize_t),
                "normalize_c": bool(self.config.normalize_c),
                "t_mean": float(self.t_mean),
                "t_std": float(self.t_std),
                "c_mean": self.c_mean.squeeze(0).tolist(),
                "c_std": self.c_std.squeeze(0).tolist(),
            },
            extra_info={
                "M": int(self.M),
                "nT": int(self.nT),
                "K": int(self.K),
                "n_time_samples": int(self.config.n_time_samples),
                "t_sampling": str(self.config.t_sampling),
                "return_k_coords": bool(self.config.return_k_coords),
                "seed": self.config.seed,
                "has_reconstruction": bool(self.x_grid is not None and self.Phi is not None),
                "nx": int(data["x_grid"].shape[0]) if "x_grid" in data else None,
                # Save basis construction parameters
                "hermite_scale": float(self.hermite_scale) if hasattr(self, 'hermite_scale') else None,
                "hermite_shift": float(self.hermite_shift) if hasattr(self, 'hermite_shift') else None,
                "orthonormalize": bool(self.orthonormalize) if hasattr(self, 'orthonormalize') else False,
            },
        )

        save_dataset(filepath, data=data, metadata=metadata, format=format)

    @classmethod
    def load(
        cls,
        filepath: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "NeuralGalerkinDataset":
        """
        Load dataset saved with .save(). Data is stored in *physical* space in the file;
        we rebuild the dataset and apply normalization according to saved metadata.
        """
        data_torch, metadata = load_dataset(filepath, device="cpu", dtype=torch.float32)

        # tensors -> numpy (constructor expects numpy)
        def to_np(key: str) -> Optional[np.ndarray]:
            if key not in data_torch:
                return None
            v = data_torch[key]
            if isinstance(v, torch.Tensor):
                return v.detach().cpu().numpy()
            return np.asarray(v)

        t = to_np("t")              # (M,nT) physical
        c = to_np("c")              # (M,nT,K) physical
        x_grid = to_np("x_grid")    # (nx,) or None
        Phi = to_np("Phi")          # (K,nx) or None
        T_matrix = to_np("transformation_matrix")  # ✅ Load transformation matrix

        extra = metadata.extra_info or {}
        norm = metadata.normalizer or {}

        cfg = NeuralGalerkinDatasetConfig(
            n_time_samples=int(extra.get("n_time_samples", t.shape[1])),
            t_sampling=str(extra.get("t_sampling", "unknown")),
            seed=extra.get("seed", None),
            normalize_t=bool(norm.get("normalize_t", False)),
            normalize_c=bool(norm.get("normalize_c", False)),
            return_k_coords=bool(extra.get("return_k_coords", False)),
            pde_name=str(metadata.pde_name),
        )

        ds = cls(
            config=cfg,
            t=t,
            c=c,
            device=device,
            dtype=dtype,
            x_grid=x_grid,
            basis_matrix=Phi,
        )

        # restore exact stats (important for exact denormalize)
        if "t_mean" in norm: ds.t_mean = float(norm["t_mean"])
        if "t_std"  in norm: ds.t_std  = float(norm["t_std"])
        if "c_mean" in norm: ds.c_mean = np.asarray(norm["c_mean"], dtype=float)[None, :]
        if "c_std"  in norm: ds.c_std  = np.asarray(norm["c_std"], dtype=float)[None, :]

        # Restore basis parameters
        if "hermite_scale" in extra and extra["hermite_scale"] is not None:
            ds.hermite_scale = float(extra["hermite_scale"])
        if "hermite_shift" in extra and extra["hermite_shift"] is not None:
            ds.hermite_shift = float(extra["hermite_shift"])
        if "orthonormalize" in extra:
            ds.orthonormalize = bool(extra["orthonormalize"])
        
        if T_matrix is not None:
            ds.transformation_matrix = T_matrix
        else:
            ds.transformation_matrix = None

        return ds


# -------------------------
# Factory
# -------------------------
def create_NeuralGalerkin_dataset(
    solution_functions: List[Callable[[np.ndarray, np.ndarray], np.ndarray]],
    x_grid: np.ndarray,
    t_min: float,
    t_max: float,
    basis_eval: Callable[[np.ndarray], np.ndarray],  # returns (K, nx)
    n_time_samples: int = 200,
    t_sampling: str = "grid",
    seed: Optional[int] = None,
    weights: Optional[np.ndarray] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    normalize_t: bool = False,
    normalize_c: bool = False,
    return_k_coords: bool = False,
    pde_name: str = "unknown",
) -> NeuralGalerkinDataset:
    x_grid = np.asarray(x_grid, dtype=float)

    rng = np.random.default_rng(seed)
    M = len(solution_functions)
    if M == 0:
        raise ValueError("solution_functions must be a non-empty list")

    # time axis: shared across ICs or independent per IC
    if t_sampling == "grid":
        t = np.linspace(t_min, t_max, n_time_samples, dtype=float)          # (nT,)
        T_all = np.repeat(t[None, :], M, axis=0)                            # (M,nT)
    elif t_sampling == "random":
        T_all = rng.uniform(t_min, t_max, size=(M, n_time_samples)).astype(float)  # (M,nT)
        T_all.sort(axis=1)
    else:
        raise ValueError("t_sampling must be 'grid' or 'random'")

    # basis matrix for reconstruction
    Phi = np.asarray(basis_eval(x_grid), dtype=float)  # (K,nx)
    if Phi.ndim != 2 or Phi.shape[1] != x_grid.size:
        raise ValueError(f"basis_eval(x_grid) must return (K,nx); got {Phi.shape}")

    # projection operator
    P = projection_matrix(x_grid, basis_eval=basis_eval, weights=weights)  # (nx,K)

    # Evaluate and project each trajectory
    C_all = np.empty((M, n_time_samples, Phi.shape[0]), dtype=float)       # (M,nT,K)
    for m, sol_fn in enumerate(solution_functions):
        t_m = T_all[m]  # (nT,)
        Tm = np.repeat(t_m[:, None], x_grid.size, axis=1)                  # (nT,nx)
        Xm = np.repeat(x_grid[None, :], t_m.size, axis=0)                  # (nT,nx)
        U = np.asarray(sol_fn(Tm, Xm), dtype=float)                        # (nT,nx)
        if U.shape != (t_m.size, x_grid.size):
            raise ValueError(f"solution_functions[{m}] returned {U.shape}, expected {(t_m.size, x_grid.size)}")
        C_all[m] = U @ P                                                   # (nT,K)

    cfg = NeuralGalerkinDatasetConfig(
        n_time_samples=n_time_samples,
        t_sampling=t_sampling,
        seed=seed,
        normalize_t=normalize_t,
        normalize_c=normalize_c,
        return_k_coords=return_k_coords,
        pde_name=pde_name,
    )
    return NeuralGalerkinDataset(
        cfg,
        t=T_all,
        c=C_all,
        device=device,
        dtype=dtype,
        x_grid=x_grid,
        basis_matrix=Phi,
    )
