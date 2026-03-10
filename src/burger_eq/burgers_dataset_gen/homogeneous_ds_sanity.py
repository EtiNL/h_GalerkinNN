import math
import numpy as np
import torch

from homogeneity_utils_np import compute_Gdn


# ============================================================
# Basic linear-algebra helpers
# ============================================================

@torch.no_grad()
def to_torch(x, device=None, dtype=None):
    if isinstance(x, torch.Tensor):
        y = x
        if device is not None:
            y = y.to(device=device)
        if dtype is not None:
            y = y.to(dtype=dtype)
        return y
    return torch.tensor(x, device=device, dtype=dtype)


@torch.no_grad()
def batch_apply_dilation_dn(c: torch.Tensor, s: torch.Tensor, Gdn: torch.Tensor, chunk_rows: int = 2048) -> torch.Tensor:
    """
    Apply d_n(s) row-wise:
        out_i = exp(s_i Gdn) c_i

    Args:
        c:   (B, K)
        s:   (B,)
        Gdn: (K, K)

    Returns:
        out: (B, K)
    """
    if c.ndim != 2:
        raise ValueError("c must have shape (B, K)")
    if s.ndim != 1 or s.shape[0] != c.shape[0]:
        raise ValueError("s must have shape (B,)")

    B, K = c.shape
    out = torch.empty_like(c)

    for start in range(0, B, chunk_rows):
        end = min(start + chunk_rows, B)
        M = torch.matrix_exp(s[start:end, None, None] * Gdn[None, :, :])   # (b, K, K)
        out[start:end] = torch.bmm(M, c[start:end].unsqueeze(-1)).squeeze(-1)

    return out


@torch.no_grad()
def batch_apply_inverse_dilation_dn(c: torch.Tensor, r: torch.Tensor, Gdn: torch.Tensor, chunk_rows: int = 2048) -> torch.Tensor:
    """
    Compute a = d_n(-log r) c row-wise.
    """
    s = -torch.log(r)
    return batch_apply_dilation_dn(c, s, Gdn, chunk_rows=chunk_rows)


@torch.no_grad()
def rebuild_c_from_a_r(a: torch.Tensor, r: torch.Tensor, Gdn: torch.Tensor, chunk_rows: int = 2048) -> torch.Tensor:
    """
    Compute c = d_n(log r) a row-wise.
    """
    s = torch.log(r)
    return batch_apply_dilation_dn(a, s, Gdn, chunk_rows=chunk_rows)


@torch.no_grad()
def relative_l2_error(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12, dim=-1) -> torch.Tensor:
    """
    relerr = ||x-y|| / (||y|| + eps)
    """
    num = torch.linalg.vector_norm(x - y, dim=dim)
    den = torch.linalg.vector_norm(y, dim=dim).clamp_min(eps)
    return num / den


@torch.no_grad()
def relative_l2_error_scalar(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    num = torch.linalg.vector_norm((x - y).reshape(-1))
    den = torch.linalg.vector_norm(y.reshape(-1)).clamp_min(eps)
    return float((num / den).item())


# ============================================================
# Basis / reduced-space helpers
# ============================================================

@torch.no_grad()
def basis_gram_from_dataset(ds, device=None, dtype=torch.float64):
    """
    Approximate Gram matrix of the basis stored in ds on ds.x_grid.
    """
    x_grid = to_torch(ds.x_grid, device=device, dtype=dtype)              # (nz,)
    Phi = to_torch(ds.basis_matrix, device=device, dtype=dtype)           # shape expected: (K, nz) or (nz, K)

    if Phi.ndim != 2:
        raise ValueError("ds.basis_matrix must be 2D")

    # Try to infer orientation
    if Phi.shape[1] == x_grid.numel():
        Phi_kn = Phi
    elif Phi.shape[0] == x_grid.numel():
        Phi_kn = Phi.t()
    else:
        raise ValueError("basis_matrix shape incompatible with x_grid")

    w = torch.empty_like(x_grid)
    w[0] = 0.5 * (x_grid[1] - x_grid[0])
    w[-1] = 0.5 * (x_grid[-1] - x_grid[-2])
    w[1:-1] = 0.5 * (x_grid[2:] - x_grid[:-2])

    G = (Phi_kn * w.unsqueeze(0)) @ Phi_kn.t()
    return G


@torch.no_grad()
def euclidean_norm_stats(x: torch.Tensor):
    """
    Returns min, mean, max of row-wise Euclidean norms.
    """
    nrm = torch.linalg.vector_norm(x, dim=-1)
    return {
        "min": float(nrm.min().item()),
        "mean": float(nrm.mean().item()),
        "max": float(nrm.max().item()),
    }


# ============================================================
# Main internal dataset sanity check
# ============================================================

@torch.no_grad()
def sanity_check_homogeneous_dataset(
    ds,
    K: int | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    chunk_rows: int = 2048,
    safe_margin: float = 0.7,
    verbose: bool = True,
):
    r"""
    Internal sanity checks for a homogeneous Burgers dataset.

    Assumes the dataset produced by burgers_homogeneous_ds stores:
        ds.c                : (N, T, K) homogeneous reduced coefficients
        ds.d_norm           : (N, T)    r(t) = ||u(t)||_d
        ds.angular_coeffs   : (N, T, K) a(t) = Pi_n phi(t)
        ds.solve_z_range
        ds.proj_z_range
        ds.outside_query_fraction
        ds.t

    Checks:
      1) shapes / finiteness
      2) monotone increasing times
      3) admissibility threshold r_cut = Z_proj / Z_solve
      4) safer threshold r_cut_safe = Z_proj / (safe_margin * Z_solve)
      5) reconstruction consistency:
             c ?= d_n(log r) a
      6) inverse consistency:
             a ?= d_n(-log r) c
      7) Euclidean norm behavior of a(t)
      8) approximate orthonormality of basis on ds.x_grid

    Returns:
        report: dict
    """
    dev = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    c = to_torch(ds.c, device=dev, dtype=dtype)                     # (N,T,K)
    t = to_torch(ds.t, device=dev, dtype=dtype)                     # (N,T)
    r = to_torch(ds.d_norm, device=dev, dtype=dtype)                # (N,T)
    a = to_torch(ds.angular_coeffs, device=dev, dtype=dtype)        # (N,T,K)

    if K is None:
        K = c.shape[-1]

    Gdn = to_torch(compute_Gdn(K), device=dev, dtype=dtype)

    report = {}

    # --------------------------------------------------------
    # 1) basic shapes / finiteness
    # --------------------------------------------------------
    report["shape_c"] = tuple(c.shape)
    report["shape_t"] = tuple(t.shape)
    report["shape_r"] = tuple(r.shape)
    report["shape_a"] = tuple(a.shape)

    report["all_finite_c"] = bool(torch.isfinite(c).all().item())
    report["all_finite_t"] = bool(torch.isfinite(t).all().item())
    report["all_finite_r"] = bool(torch.isfinite(r).all().item())
    report["all_finite_a"] = bool(torch.isfinite(a).all().item())

    report["r_min"] = float(r.min().item())
    report["r_mean"] = float(r.mean().item())
    report["r_max"] = float(r.max().item())

    # --------------------------------------------------------
    # 2) time monotonicity
    # --------------------------------------------------------
    dt = t[:, 1:] - t[:, :-1]
    report["time_strictly_increasing"] = bool((dt > 0).all().item())
    report["t_min"] = float(t.min().item())
    report["t_max"] = float(t.max().item())

    # --------------------------------------------------------
    # 3) admissibility thresholds
    # --------------------------------------------------------
    Z_proj = max(abs(float(ds.proj_z_range[0])), abs(float(ds.proj_z_range[1])))
    Z_solve = max(abs(float(ds.solve_z_range[0])), abs(float(ds.solve_z_range[1])))

    r_cut = Z_proj / Z_solve
    r_cut_safe = Z_proj / (safe_margin * Z_solve)

    report["Z_proj"] = float(Z_proj)
    report["Z_solve"] = float(Z_solve)
    report["r_cut"] = float(r_cut)
    report["r_cut_safe"] = float(r_cut_safe)

    report["fraction_r_ge_r_cut"] = float((r >= r_cut).float().mean().item())
    report["fraction_r_ge_r_cut_safe"] = float((r >= r_cut_safe).float().mean().item())

    per_time_valid = (r >= r_cut).float().mean(dim=0)
    per_time_valid_safe = (r >= r_cut_safe).float().mean(dim=0)

    report["per_time_valid_fraction_min"] = float(per_time_valid.min().item())
    report["per_time_valid_fraction_mean"] = float(per_time_valid.mean().item())
    report["per_time_valid_fraction_safe_min"] = float(per_time_valid_safe.min().item())
    report["per_time_valid_fraction_safe_mean"] = float(per_time_valid_safe.mean().item())

    if hasattr(ds, "outside_query_fraction"):
        report["outside_query_fraction"] = float(ds.outside_query_fraction)

    # --------------------------------------------------------
    # 4) reconstruction consistency: c ?= d_n(log r) a
    # --------------------------------------------------------
    c_flat = c.reshape(-1, K)
    a_flat = a.reshape(-1, K)
    r_flat = r.reshape(-1)

    c_rebuilt = rebuild_c_from_a_r(a_flat, r_flat, Gdn, chunk_rows=chunk_rows)
    rel_rebuild = relative_l2_error(c_rebuilt, c_flat, dim=-1)

    report["rebuild_relerr_mean"] = float(rel_rebuild.mean().item())
    report["rebuild_relerr_median"] = float(rel_rebuild.median().item())
    report["rebuild_relerr_max"] = float(rel_rebuild.max().item())

    # --------------------------------------------------------
    # 5) inverse consistency: a ?= d_n(-log r) c
    # --------------------------------------------------------
    a_recovered = batch_apply_inverse_dilation_dn(c_flat, r_flat, Gdn, chunk_rows=chunk_rows)
    rel_inverse = relative_l2_error(a_recovered, a_flat, dim=-1)

    report["inverse_relerr_mean"] = float(rel_inverse.mean().item())
    report["inverse_relerr_median"] = float(rel_inverse.median().item())
    report["inverse_relerr_max"] = float(rel_inverse.max().item())

    # --------------------------------------------------------
    # 6) angular coefficient Euclidean norm behavior
    # --------------------------------------------------------
    report["angular_coeff_norm_stats"] = euclidean_norm_stats(a_flat)

    # --------------------------------------------------------
    # 7) basis Gram check on projection grid
    # --------------------------------------------------------
    try:
        Gram = basis_gram_from_dataset(ds, device=dev, dtype=dtype)
        I = torch.eye(Gram.shape[0], device=dev, dtype=dtype)
        gram_err = relative_l2_error_scalar(Gram, I)
        report["basis_gram_relerr_to_identity"] = gram_err
        report["basis_gram_max_abs_offdiag"] = float((Gram - torch.diag(torch.diag(Gram))).abs().max().item())
        report["basis_gram_diag_min"] = float(torch.diag(Gram).min().item())
        report["basis_gram_diag_max"] = float(torch.diag(Gram).max().item())
    except Exception as e:
        report["basis_gram_check_error"] = str(e)

    if verbose:
        print("=" * 70)
        print("INTERNAL HOMOGENEOUS DATASET SANITY CHECK")
        print("=" * 70)
        for k, v in report.items():
            print(f"{k}: {v}")

    return report


# ============================================================
# Optional stronger check:
# compare two datasets under scaling symmetry
# ============================================================

@torch.no_grad()
def sanity_check_scaling_pair(
    ds_ref,
    ds_scaled,
    s: float,
    K: int | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    chunk_rows: int = 2048,
    verbose: bool = True,
):
    r"""
    Check the homogeneous scaling law between two datasets built from:
        u0_scaled = d(s) u0_ref

    For Burgers degree ν = 2, the reduced homogeneous target should satisfy
        c_h(t ; d(s)u0) = d_n(s) c_h(e^{2s} t ; u0)

    This function assumes:
      - same number of trajectories N
      - same K
      - ds_ref.t and ds_scaled.t are common per-dataset time grids
      - ds_scaled time grid lies inside the scaled ref window after mapping
        t_ref = e^{2s} t_scaled

    It interpolates ds_ref in time and compares against ds_scaled.

    Returns:
        report: dict
    """
    dev = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    c_ref = to_torch(ds_ref.c, device=dev, dtype=dtype)          # (N, T1, K)
    t_ref = to_torch(ds_ref.t, device=dev, dtype=dtype)          # (N, T1)
    c_scaled = to_torch(ds_scaled.c, device=dev, dtype=dtype)    # (N, T2, K)
    t_scaled = to_torch(ds_scaled.t, device=dev, dtype=dtype)    # (N, T2)

    if K is None:
        K = c_ref.shape[-1]

    Gdn = to_torch(compute_Gdn(K), device=dev, dtype=dtype)

    N = c_ref.shape[0]
    if c_scaled.shape[0] != N:
        raise ValueError("ds_ref and ds_scaled must have the same number of trajectories")
    if c_scaled.shape[-1] != K:
        raise ValueError("Mismatch in reduced dimension K")

    # Use first row because time grid is repeated across trajectories
    tref = t_ref[0]
    tsc = t_scaled[0]

    # target times in ref dataset after scaling
    tref_query = torch.exp(torch.tensor(2.0 * s, device=dev, dtype=dtype)) * tsc

    # Restrict to overlapping time interval
    keep = (tref_query >= tref[0]) & (tref_query <= tref[-1])
    if keep.sum() == 0:
        raise RuntimeError("No overlapping times after t -> e^{2s} t scaling")

    tref_query = tref_query[keep]
    c_scaled_kept = c_scaled[:, keep, :]   # (N, Tkeep, K)

    # linear interpolation in time for c_ref
    idx = torch.searchsorted(tref, tref_query, right=False).clamp(1, tref.numel() - 1)
    t_lo = tref[idx - 1]
    t_hi = tref[idx]
    w = (tref_query - t_lo) / (t_hi - t_lo)

    c_lo = c_ref[:, idx - 1, :]
    c_hi = c_ref[:, idx, :]
    c_ref_interp = c_lo * (1.0 - w)[None, :, None] + c_hi * w[None, :, None]

    # apply d_n(s)
    B = N * c_ref_interp.shape[1]
    svec = torch.full((B,), float(s), device=dev, dtype=dtype)
    pred = batch_apply_dilation_dn(c_ref_interp.reshape(B, K), svec, Gdn, chunk_rows=chunk_rows)
    pred = pred.reshape(N, c_ref_interp.shape[1], K)

    rel = relative_l2_error(pred.reshape(-1, K), c_scaled_kept.reshape(-1, K), dim=-1)

    report = {
        "s": float(s),
        "n_trajectories": int(N),
        "n_times_compared": int(c_ref_interp.shape[1]),
        "pair_scaling_relerr_mean": float(rel.mean().item()),
        "pair_scaling_relerr_median": float(rel.median().item()),
        "pair_scaling_relerr_max": float(rel.max().item()),
        "t_scaled_min": float(tsc[keep].min().item()),
        "t_scaled_max": float(tsc[keep].max().item()),
        "mapped_t_ref_min": float(tref_query.min().item()),
        "mapped_t_ref_max": float(tref_query.max().item()),
    }

    if verbose:
        print("=" * 70)
        print("SCALING-PAIR SANITY CHECK")
        print("=" * 70)
        for k, v in report.items():
            print(f"{k}: {v}")

    return report


# ============================================================
# Optional batch wrapper for several scales
# ============================================================

@torch.no_grad()
def sanity_check_scaling_pairs(
    ref_ds,
    scaled_ds_dict: dict,
    K: int | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    chunk_rows: int = 2048,
    verbose: bool = True,
):
    """
    scaled_ds_dict example:
        {
            -0.5: ds_m05,
             0.5: ds_p05,
             1.0: ds_p10,
        }
    """
    reports = {}
    for s, ds_scaled in scaled_ds_dict.items():
        reports[float(s)] = sanity_check_scaling_pair(
            ds_ref=ref_ds,
            ds_scaled=ds_scaled,
            s=float(s),
            K=K,
            device=device,
            dtype=dtype,
            chunk_rows=chunk_rows,
            verbose=verbose,
        )
    return reports


# ============================================================
# High-level convenience wrapper
# ============================================================

@torch.no_grad()
def full_homogeneous_ds_report(
    ds,
    scaled_ds_dict: dict | None = None,
    K: int | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    chunk_rows: int = 2048,
    safe_margin: float = 0.7,
    verbose: bool = True,
):
    """
    Runs:
      - internal dataset checks
      - optional pairwise scaling checks if scaled datasets are provided
    """
    report = {
        "internal": sanity_check_homogeneous_dataset(
            ds=ds,
            K=K,
            device=device,
            dtype=dtype,
            chunk_rows=chunk_rows,
            safe_margin=safe_margin,
            verbose=verbose,
        )
    }

    if scaled_ds_dict is not None and len(scaled_ds_dict) > 0:
        report["pair_scaling"] = sanity_check_scaling_pairs(
            ref_ds=ds,
            scaled_ds_dict=scaled_ds_dict,
            K=K,
            device=device,
            dtype=dtype,
            chunk_rows=chunk_rows,
            verbose=verbose,
        )

    return report