# src/data_burgers/sanity_check.py

import os
import sys
from typing import Dict, Any, Optional

# ---------------------------------------------------------------------------
# Make this file work both as:
#   python -m data_burgers.sanity_check
# and as:
#   python src/data_burgers/sanity_check.py
# ---------------------------------------------------------------------------

if __name__ == "__main__" and __package__ is None:
    # When executed as a script, __package__ is None and the parent package
    # 'data_burgers' is not known. We add the src/ directory (one level above
    # data_burgers/) to sys.path so that `import data_burgers` works.
    this_file = os.path.abspath(__file__)
    src_root = os.path.dirname(os.path.dirname(this_file))  # .../code/src
    if src_root not in sys.path:
        sys.path.insert(0, src_root)

# Now we can always use absolute imports
from data_burgers.burgers_dataset import (
    BurgersInitialCondition,
    BurgersInitialConditions,
    BurgersSolution,
)

from pde_dataset import load_dataset  # auto NPZ/HDF5/etc loader
import numpy as np


def _error_stats(u_ds: np.ndarray, u_ref: np.ndarray) -> Dict[str, float]:
    """Compute basic absolute/relative error statistics."""
    u_ds = np.asarray(u_ds)
    u_ref = np.asarray(u_ref)

    abs_err = np.abs(u_ds - u_ref)
    denom = np.maximum(np.abs(u_ref), 1e-12)
    rel_err = abs_err / denom

    return {
        "max_abs": float(abs_err.max()),
        "mean_abs": float(abs_err.mean()),
        "max_rel": float(rel_err.max()),
        "mean_rel": float(rel_err.mean()),
    }


def _build_reference_solution(
    initial_condition: Optional[BurgersInitialCondition] = None,
    hz: float = 0.1,
    ht: float = 0.05,
    Tmax: float = 2.0,
    L: float = 6.0,
) -> BurgersSolution:
    """Construct BurgersSolution with given parameters."""
    if initial_condition is None:
        initial_condition = BurgersInitialConditions.sine()

    return BurgersSolution(
        initial_condition=initial_condition,
        hz=hz,
        ht=ht,
        Tmax=Tmax,
        L=L,
    )


# ---------------------------------------------------------------------------
# PINN dataset check
# ---------------------------------------------------------------------------

def check_pinn_dataset_file(
    pinn_path: str,
    solution: BurgersSolution,
    n_samples: int = 1000,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """
    Check a saved PINN dataset file against a reference BurgersSolution.

    Args:
        pinn_path: Path to '<base>_pinn.npz'.
        solution: BurgersSolution instance (same params as used for dataset).
        n_samples: Number of random points for the comparison.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with keys 'collocation', 'boundary', 'initial', each mapping
        to an error stats dict (possibly empty if that set is empty).
    """
    data, meta = load_dataset(pinn_path, device="cpu")
    rng = np.random.default_rng(seed)

    def _sample_block(name: str) -> Dict[str, float]:
        t_key = f"t_{name}"
        x_key = f"x_{name}"
        u_key = f"u_{name}"
        if t_key not in data or u_key not in data:
            return {}

        t = data[t_key].cpu().numpy()
        x = data[x_key].cpu().numpy()
        u_ds = data[u_key].cpu().numpy()

        if t.size == 0:
            return {}

        m = min(n_samples, t.size)
        idx = rng.choice(t.size, size=m, replace=False)

        t_s = t[idx]
        x_s = x[idx]
        u_s = u_ds[idx]

        u_ref = solution(t_s, x_s)
        return _error_stats(u_s, u_ref)

    return {
        "collocation": _sample_block("collocation"),
        "boundary": _sample_block("boundary"),
        "initial": _sample_block("initial"),
    }


# ---------------------------------------------------------------------------
# Sequential dataset check
# ---------------------------------------------------------------------------

def check_sequential_dataset_file(
    sequential_path: str,
    solution: BurgersSolution,
) -> Dict[str, float]:
    """
    Check a saved sequential dataset file against a reference BurgersSolution.

    Compares the full T/X/U grid.

    Args:
        sequential_path: Path to '<base>_sequential.npz'.
        solution: BurgersSolution instance.

    Returns:
        Error stats dict for the full grid.
    """
    data, meta = load_dataset(sequential_path, device="cpu")

    T = data["T"].cpu().numpy()
    X = data["X"].cpu().numpy()
    U = data["U"].cpu().numpy()

    u_ref = solution(T, X)
    return _error_stats(U, u_ref)


# ---------------------------------------------------------------------------
# Galerkin dataset check
# ---------------------------------------------------------------------------

def check_galerkin_dataset_file(
    galerkin_path: str,
    solution: BurgersSolution,
    t_snapshot: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Check a saved Galerkin dataset file against a reference BurgersSolution.

    Args:
        galerkin_path: Path to '<base>_galerkin.npz'.
        solution: BurgersSolution instance.
        t_snapshot: Time at which the Galerkin dataset was built.
                    If None, defaults to solution.Tmax / 2.0.

    Returns:
        Dict with 'interior' and 'boundary' error stats.
    """
    if t_snapshot is None:
        t_snapshot = solution.Tmax / 2.0

    data, meta = load_dataset(galerkin_path, device="cpu")

    x_int = data["x_interior"].cpu().numpy()
    u_int_ds = data.get("u_interior", None)
    x_bnd = data["x_boundary"].cpu().numpy()
    u_bnd_ds = data.get("u_boundary", None)

    results: Dict[str, Dict[str, float]] = {}

    if u_int_ds is not None and u_int_ds.numel() > 0:
        u_int_ds_np = u_int_ds.cpu().numpy()
        t_int = np.full_like(x_int, t_snapshot)
        u_int_ref = solution(t_int, x_int)
        results["interior"] = _error_stats(u_int_ds_np, u_int_ref)

    if u_bnd_ds is not None and u_bnd_ds.numel() > 0:
        u_bnd_ds_np = u_bnd_ds.cpu().numpy()
        t_bnd = np.full_like(x_bnd, t_snapshot)
        u_bnd_ref = solution(t_bnd, x_bnd)
        results["boundary"] = _error_stats(u_bnd_ds_np, u_bnd_ref)

    return results


# ---------------------------------------------------------------------------
# High-level convenience entry point
# ---------------------------------------------------------------------------

def run_full_sanity_check(
    base_path: str = "./burgers_data",
    initial_condition: Optional[BurgersInitialCondition] = None,
    hz: float = 0.1,
    ht: float = 0.05,
    Tmax: float = 2.0,
    L: float = 6.0,
    n_pinn_samples: int = 1000,
    seed: int = 0,
    t_snapshot: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run sanity checks for all three Burgers datasets (PINN, Sequential, Galerkin).

    Assumes you previously called:
        generate_all_burgers_datasets(..., save_path=base_path)

    which produces:
        base_path + "_pinn.npz"
        base_path + "_sequential.npz"
        base_path + "_galerkin.npz"

    Returns:
        Nested dict with error statistics for each dataset type.
    """
    solution = _build_reference_solution(
        initial_condition=initial_condition,
        hz=hz,
        ht=ht,
        Tmax=Tmax,
        L=L,
    )

    pinn_path = f"{base_path}_pinn.npz"
    seq_path = f"{base_path}_sequential.npz"
    gal_path = f"{base_path}_galerkin.npz"

    results: Dict[str, Any] = {}

    results["pinn"] = check_pinn_dataset_file(
        pinn_path=pinn_path,
        solution=solution,
        n_samples=n_pinn_samples,
        seed=seed,
    )

    results["sequential"] = {
        "grid": check_sequential_dataset_file(
            sequential_path=seq_path,
            solution=solution,
        )
    }

    results["galerkin"] = check_galerkin_dataset_file(
        galerkin_path=gal_path,
        solution=solution,
        t_snapshot=t_snapshot,
    )

    return results


if __name__ == "__main__":
    # Example CLI usage
    stats = run_full_sanity_check(
        base_path="./burgers_data",
        Tmax=2.0,
    )

    import pprint
    pprint.pprint(stats)
