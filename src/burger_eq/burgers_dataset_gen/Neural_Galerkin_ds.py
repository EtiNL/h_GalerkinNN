import math
import numpy as np
import torch
import sys
import os

# -------------------------
# ensure src/ is importable
# -------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from pde_dataset import (
    NeuralGalerkinDatasetConfig,
    NeuralGalerkinDataset,
)
from homogeneity_utils_np import compute_Gdn
import warnings


@torch.no_grad()
def sobol_sphere(N: int, K: int, device, dtype, seed: int = 0) -> torch.Tensor:
    """Generate N unit-norm vectors in K dimensions using Sobol sequence."""
    eng = torch.quasirandom.SobolEngine(dimension=K, scramble=True, seed=seed)
    U = eng.draw(N + 1)[1:].to(device=device, dtype=torch.float64)
    
    u_eps = 1e-6 if dtype == torch.float32 else 1e-12
    U = U.clamp(u_eps, 1.0 - u_eps)
    
    G = math.sqrt(2.0) * torch.erfinv(2.0 * U - 1.0)
    
    nrm = torch.linalg.vector_norm(G, dim=1, keepdim=True)
    bad = (~torch.isfinite(nrm)) | (nrm < 1e-12) | (~torch.isfinite(G).all(dim=1, keepdim=True))
    if bad.any():
        idx = bad.squeeze(1).nonzero(as_tuple=False).squeeze(1)
        G[idx].zero_()
        G[idx, 0] = 1.0
        nrm = torch.linalg.vector_norm(G, dim=1, keepdim=True)
    
    G = (G / nrm).to(dtype=dtype)
    return G


@torch.no_grad()
def trapz_weights_1d_torch(x: torch.Tensor) -> torch.Tensor:
    """Trapezoidal quadrature weights."""
    w = torch.empty_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


@torch.no_grad()
def hermite_basis_x_torch(x: torch.Tensor, K: int, scale: float, shift: float) -> torch.Tensor:
    """
    Stable Hermite functions basis (orthonormal in L²(ℝ)) with scaling.
    Returns: (K, nx) tensor where Phi[k, :] is the k-th basis function.
    """
    y = (x - shift) / scale
    y_flat = y.reshape(-1)
    M = y_flat.numel()
    
    yd = y_flat.to(torch.float64)
    Phi = torch.empty((K, M), device=x.device, dtype=torch.float64)
    
    phi0 = (math.pi ** (-0.25)) * torch.exp(-0.5 * yd * yd)
    Phi[0] = phi0
    
    if K >= 2:
        Phi[1] = math.sqrt(2.0) * yd * phi0
    
    for k in range(1, K - 1):
        a = math.sqrt(2.0 / (k + 1))
        b = math.sqrt(k / (k + 1))
        Phi[k + 1] = a * yd * Phi[k] - b * Phi[k - 1]
    
    Phi = Phi.reshape(K, *y.shape).to(dtype=x.dtype)
    return Phi / math.sqrt(scale)


@torch.no_grad()
def interp_time_batch(t_grid: torch.Tensor, C_grid: torch.Tensor, t_query: torch.Tensor) -> torch.Tensor:
    """Linear interpolation in time for coefficient trajectories."""
    tq = t_query.clamp(t_grid[0], t_grid[-1])
    idx = torch.searchsorted(t_grid, tq, right=False).clamp(1, t_grid.numel() - 1)
    t_lo = t_grid[idx - 1]
    t_hi = t_grid[idx]
    w = (tq - t_lo) / (t_hi - t_lo)
    
    C_lo = C_grid[:, idx - 1, :]
    C_hi = C_grid[:, idx, :]
    return C_lo * (1.0 - w)[None, :, None] + C_hi * w[None, :, None]


def burgers_neural_ds(
    solver,
    N: int,
    K: int,
    hz: float = 0.1,
    Tmax: float = 2.0,
    z_range=(-7.0, 7.0),
    L: float = 10.0,
    n_quad_points: int = 200,
    q_n: int = 8192,
    z_batch_size: int = 300,
    sphere_mode: str = "coeff",
    n_time_samples: int = 200,
    t_sampling: str = "grid",
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    hermite_scale: float | None = None,
    hermite_shift: float | None = None,
):
    """
    Generate Neural Galerkin dataset for Burgers equation.

    Args:
        hermite_scale: Override basis scale. Default (None) computes from z_range.
            Use 1.0 to match the standard Hermite basis (no rescaling).
        hermite_shift: Override basis shift. Default (None) computes from z_range.
            Use 0.0 to match the standard Hermite basis (no shifting).
    """
    device = solver.device

    # Spatial grid
    z = torch.arange(z_range[0], z_range[1] + 0.5 * hz, hz, device=device, dtype=dtype)

    # Hermite basis parameters
    shift = hermite_shift if hermite_shift is not None else 0.5 * (z_range[0] + z_range[1])
    scale = hermite_scale if hermite_scale is not None else (z_range[1] - z_range[0]) / 6.0
    
    # Build basis on z-grid
    Phi_z = hermite_basis_x_torch(z, K, scale=scale, shift=shift)  # (K, nz)
    w_z = trapz_weights_1d_torch(z)                                 # (nz,)

    # Projection matrix
    P = (w_z.unsqueeze(0) * Phi_z).t().contiguous()  # (nz, K)
    
    # Generate ICs on unit sphere
    A = sobol_sphere(N, K, device=device, dtype=dtype, seed=seed)  # (N, K)
    
    if sphere_mode == "l2":
        M = (Phi_z * w_z.unsqueeze(0)) @ Phi_z.t()
        M = 0.5 * (M + M.t()) + 1e-10 * torch.eye(K, device=device, dtype=dtype)
        Lc = torch.linalg.cholesky(M)
        Ccoeff = torch.linalg.solve_triangular(Lc.t(), A.t(), upper=True).t()
    else:
        Ccoeff = A
    
    def make_x0_fn(c_coeff: torch.Tensor):
        """Returns a function that evaluates x0 at any point."""
        def f(x):
            is_torch = isinstance(x, torch.Tensor)
            if not is_torch:
                x = torch.tensor(x, device=device, dtype=dtype)
            Phi_x = hermite_basis_x_torch(x, K, scale=scale, shift=shift)  # (K, nx)
            u_x = c_coeff @ Phi_x
            return u_x if is_torch else u_x.detach().cpu().numpy()
        return f
    
    x0_list = [make_x0_fn(Ccoeff[i]) for i in range(N)]
    
    # Solve PDE
    ht = hz ** 2
    z_vals, t_vals, C_grid = solver.solve_parallel_projected(
        x0_list=x0_list,
        G_list=None,
        hz=hz,
        ht=ht,
        Tmax=Tmax,
        z_range=z_range,
        L=L,
        n_quad_points=n_quad_points,
        q_n=q_n,
        P=P,
        z_batch_size=z_batch_size,
        compute_G_if_missing=True,
        enforce_exact_ic=True,
        interp="cubic",
    )
    
    # Sample time points
    t_grid = torch.tensor(t_vals, device=device, dtype=dtype)
    
    if t_sampling == "grid":
        t_target = torch.linspace(0.0, Tmax, n_time_samples, device=device, dtype=dtype)
    elif t_sampling == "random":
        rng = np.random.default_rng(seed)
        tt = rng.uniform(0.0, Tmax, size=(n_time_samples,)).astype(np.float64)
        tt.sort()
        tt[0] = 0.0
        for i in range(1, n_time_samples):
            if tt[i] <= tt[i - 1]:
                tt[i] = np.nextafter(tt[i - 1], np.float64(Tmax))
        t_target = torch.tensor(tt, device=device, dtype=dtype)
    else:
        raise ValueError("t_sampling must be 'grid' or 'random'")
    
    C_target = interp_time_batch(t_grid, C_grid, t_target)
    
    if not torch.isfinite(C_target).all():
        raise RuntimeError("Non-finite values in C_target!")
    
    # Create numpy arrays ONCE
    T_all = t_target.detach().cpu().numpy()[None, :].repeat(N, axis=0)  # (N, nT)
    C_all = C_target.detach().cpu().numpy()  # (N, nT, K)
    Phi_np = Phi_z.detach().cpu().numpy()
    
    cfg = NeuralGalerkinDatasetConfig(
        n_time_samples=int(T_all.shape[1]),
        t_sampling=t_sampling,
        seed=seed,
        return_k_coords=False,
        pde_name="burgers",
    )
    
    ds = NeuralGalerkinDataset(
        config=cfg,
        t=T_all,
        c=C_all,
        device=str(device),
        dtype=dtype,
        x_grid=z_vals,
        basis_matrix=Phi_np,
    )
    
    # Store basis parameters for projection of arbitrary ICs
    ds.hermite_scale = float(scale)
    ds.hermite_shift = float(shift)

    return ds
    
#========================================h Neural ODE ds =========================================
@torch.no_grad()
def burgers_d_norm_from_snapshots(U: torch.Tensor, z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Canonical d-norm for Burgers dilation d(s)u(x)=e^s u(e^s x).

    For this dilation:
        ||u||_d = ||u||_{L2}^2.

    Args:
        U: snapshots (..., nz)
        z: 1D spatial grid (nz,)
        eps: lower clamp for numerical safety

    Returns:
        r = ||u||_d with shape U.shape[:-1]
    """
    if U.ndim < 2:
        raise ValueError("U must have shape (..., nz)")
    if z.ndim != 1 or z.numel() != U.shape[-1]:
        raise ValueError("z must be 1D and match the last dimension of U")

    w = trapz_weights_1d_torch(z).to(device=U.device, dtype=U.dtype)  # (nz,)
    r = torch.sum(U * U * w.view(*([1] * (U.ndim - 1)), -1), dim=-1)
    return r.clamp_min(eps)


@torch.no_grad()
def interp_rows_uniform_1d(
    table: torch.Tensor,
    query: torch.Tensor,
    x_grid: torch.Tensor,
    mode: str = "cubic",
    chunk_rows: int = 4096,
) -> torch.Tensor:
    """
    Row-wise interpolation on a common uniform grid.

    Args:
        table: (B, nx)
        query: (B, nq)
        x_grid: (nx,) uniform grid
        mode: 'linear' or 'cubic'
        chunk_rows: number of rows processed per chunk

    Returns:
        out: (B, nq)
    """
    if table.ndim != 2 or query.ndim != 2:
        raise ValueError("table and query must both be 2D")
    if table.shape[0] != query.shape[0]:
        raise ValueError("table and query must have the same number of rows")
    if x_grid.ndim != 1 or x_grid.numel() < 2:
        raise ValueError("x_grid must be 1D with at least 2 points")

    x0 = x_grid[0]
    dx = x_grid[1] - x_grid[0]
    if not torch.allclose(x_grid[1:] - x_grid[:-1], dx.expand_as(x_grid[1:] - x_grid[:-1]), atol=1e-6, rtol=1e-6):
        raise ValueError("interp_rows_uniform_1d assumes a uniform spatial grid")

    nx = x_grid.numel()
    out = torch.empty((table.shape[0], query.shape[1]), device=table.device, dtype=table.dtype)

    for start in range(0, table.shape[0], chunk_rows):
        end = min(start + chunk_rows, table.shape[0])

        tab = table[start:end]   # (b, nx)
        q = query[start:end]     # (b, nq)
        pos = (q - x0) / dx

        if mode == "linear":
            i0 = torch.floor(pos).to(torch.long).clamp(0, nx - 2)
            frac = (pos - i0.to(pos.dtype)).clamp(0.0, 1.0)

            v0 = torch.gather(tab, 1, i0)
            v1 = torch.gather(tab, 1, i0 + 1)
            out[start:end] = v0 * (1.0 - frac) + v1 * frac

        elif mode == "cubic":
            # Catmull-Rom cubic interpolation
            i1 = torch.floor(pos).to(torch.long).clamp(0, nx - 2)
            t = (pos - i1.to(pos.dtype)).clamp(0.0, 1.0)

            im1 = (i1 - 1).clamp(0, nx - 1)
            i0 = i1.clamp(0, nx - 1)
            ip1 = (i1 + 1).clamp(0, nx - 1)
            ip2 = (i1 + 2).clamp(0, nx - 1)

            p0 = torch.gather(tab, 1, im1)
            p1 = torch.gather(tab, 1, i0)
            p2 = torch.gather(tab, 1, ip1)
            p3 = torch.gather(tab, 1, ip2)

            t2 = t * t
            t3 = t2 * t

            out[start:end] = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
        else:
            raise ValueError("mode must be 'linear' or 'cubic'")

    return out


@torch.no_grad()
def apply_projected_dilation_chunked(
    a_flat: torch.Tensor,
    r_flat: torch.Tensor,
    Gdn: torch.Tensor,
    chunk_rows: int = 2048,
) -> torch.Tensor:
    """
    Apply c = d_n(log r) a = exp((log r) Gdn) a row-wise.

    Uses the Burgers/Hermite structure:
        Gdn = 0.5 I + Xi,   Xi^T = -Xi
    so
        exp((log r) Gdn) = sqrt(r) * exp((log r) Xi).

    Args:
        a_flat: (B, K)
        r_flat: (B,)
        Gdn: (K, K)
        chunk_rows: batch size for matrix exponentials

    Returns:
        c_flat: (B, K)
    """
    if a_flat.ndim != 2:
        raise ValueError("a_flat must have shape (B, K)")
    if r_flat.ndim != 1 or r_flat.shape[0] != a_flat.shape[0]:
        raise ValueError("r_flat must have shape (B,) matching a_flat")

    B, K = a_flat.shape
    out = torch.empty_like(a_flat)

    I = torch.eye(K, device=a_flat.device, dtype=a_flat.dtype)
    Xi = Gdn - 0.5 * I

    for start in range(0, B, chunk_rows):
        end = min(start + chunk_rows, B)
        s = torch.log(r_flat[start:end]).to(dtype=a_flat.dtype)  # (b,)

        # (b, K, K)
        rot = torch.matrix_exp(s[:, None, None] * Xi[None, :, :])

        # (b, K)
        rotated = torch.bmm(rot, a_flat[start:end].unsqueeze(-1)).squeeze(-1)
        out[start:end] = torch.sqrt(r_flat[start:end]).unsqueeze(-1) * rotated

    return out


def burgers_homogeneous_ds(
    solver,
    N: int,
    K: int,
    hz: float = 0.1,
    Tmax: float = 2.0,
    z_range=(-7.0, 7.0),
    solve_z_range=None,
    L: float = 10.0,
    n_quad_points: int = 200,
    q_n: int = 8192,
    z_batch_size: int = 300,
    sphere_mode: str = "coeff",
    n_time_samples: int = 200,
    t_sampling: str = "grid",
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    hermite_scale: float | None = None,
    hermite_shift: float | None = None,
    snapshot_interp: str = "cubic",
    row_interp_chunk: int = 4096,
    dilation_chunk: int = 2048,
    r_eps: float = 1e-8,
):
    """
    Build the homogeneous Neural ODE dataset for Burgers.

    The target is
        c_h(t) = d_n(log r(t)) Pi_n(phi(t)),
    where
        r(t)   = ||u(t)||_d = ||u(t)||_{L2}^2,
        phi(t) = d(-log r(t))u(t).

    Important:
        phi(t, z) = (1 / r(t)) * u(t, z / r(t)).

    Args:
        solve_z_range:
            Spatial range on which the PDE is solved and stored.
            This should usually be wider than z_range because the rescaled query
            z / r(t) may fall outside the projection grid.
    """
    device = solver.device
    proj_z_range = z_range
    if solve_z_range is None:
        solve_z_range = proj_z_range

    # ------------------------------------------------------------------
    # 1) projection grid / basis
    # ------------------------------------------------------------------
    z_proj = torch.arange(
        proj_z_range[0],
        proj_z_range[1] + 0.5 * hz,
        hz,
        device=device,
        dtype=dtype,
    )

    shift = hermite_shift if hermite_shift is not None else 0.5 * (proj_z_range[0] + proj_z_range[1])
    scale = hermite_scale if hermite_scale is not None else (proj_z_range[1] - proj_z_range[0]) / 6.0

    Phi_proj = hermite_basis_x_torch(z_proj, K, scale=scale, shift=shift)  # (K, nz_proj)
    w_proj = trapz_weights_1d_torch(z_proj)                                  # (nz_proj,)
    P_proj = (w_proj.unsqueeze(0) * Phi_proj).t().contiguous()               # (nz_proj, K)

    # ------------------------------------------------------------------
    # 2) initial conditions from Hermite coefficients
    # ------------------------------------------------------------------
    A = sobol_sphere(N, K, device=device, dtype=dtype, seed=seed)  # (N, K)

    if sphere_mode == "l2":
        M = (Phi_proj * w_proj.unsqueeze(0)) @ Phi_proj.t()
        M = 0.5 * (M + M.t()) + 1e-10 * torch.eye(K, device=device, dtype=dtype)
        Lc = torch.linalg.cholesky(M)
        Ccoeff = torch.linalg.solve_triangular(Lc.t(), A.t(), upper=True).t()
    elif sphere_mode == "coeff":
        Ccoeff = A
    else:
        raise ValueError("sphere_mode must be 'coeff' or 'l2'")

    def make_x0_fn(c_coeff: torch.Tensor):
        def f(x):
            is_torch = isinstance(x, torch.Tensor)
            if not is_torch:
                x = torch.tensor(x, device=device, dtype=dtype)
            Phi_x = hermite_basis_x_torch(x, K, scale=scale, shift=shift)
            u_x = c_coeff @ Phi_x
            return u_x if is_torch else u_x.detach().cpu().numpy()
        return f

    x0_list = [make_x0_fn(Ccoeff[i]) for i in range(N)]

    # ------------------------------------------------------------------
    # 3) solve PDE and keep full snapshots (not only projections)
    # ------------------------------------------------------------------
    ht = hz ** 2
    z_vals_solve, t_vals, U_grid = solver.solve_parallel_projected(
        x0_list=x0_list,
        G_list=None,
        hz=hz,
        ht=ht,
        Tmax=Tmax,
        z_range=solve_z_range,
        L=L,
        n_quad_points=n_quad_points,
        q_n=q_n,
        P=None,  # full field snapshots are required here
        z_batch_size=z_batch_size,
        compute_G_if_missing=True,
        enforce_exact_ic=True,
        interp=snapshot_interp,
    )
    # U_grid: (N, nT_full, nz_solve)

    t_grid = torch.tensor(t_vals, device=device, dtype=dtype)
    if t_grid.numel() < 2 and n_time_samples > 1:
        raise ValueError(
            "The solver returned only one time point. "
            "Increase Tmax or decrease hz so that ht = hz**2 < Tmax."
        )

    if t_sampling == "grid":
        t_target = torch.linspace(0.0, Tmax, n_time_samples, device=device, dtype=dtype)
    elif t_sampling == "random":
        rng = np.random.default_rng(seed)
        tt = rng.uniform(0.0, Tmax, size=(n_time_samples,)).astype(np.float64)
        tt.sort()
        tt[0] = 0.0
        for i in range(1, n_time_samples):
            if tt[i] <= tt[i - 1]:
                tt[i] = np.nextafter(tt[i - 1], np.float64(Tmax))
        t_target = torch.tensor(tt, device=device, dtype=dtype)
    else:
        raise ValueError("t_sampling must be 'grid' or 'random'")

    U_target = interp_time_batch(t_grid, U_grid, t_target)  # (N, nT, nz_solve)
    if not torch.isfinite(U_target).all():
        raise RuntimeError("Non-finite values in U_target")

    # ------------------------------------------------------------------
    # 4) exact Burgers d-norm: r(t) = ||u(t)||_d = ||u(t)||_{L2}^2
    # ------------------------------------------------------------------
    z_solve = torch.tensor(z_vals_solve, device=device, dtype=dtype)
    r_target = burgers_d_norm_from_snapshots(U_target, z_solve, eps=r_eps)  # (N, nT)

    # ------------------------------------------------------------------
    # 5) angular snapshots phi(t, z) = (1/r) u(t, z/r)
    # ------------------------------------------------------------------
    query = z_proj.view(1, 1, -1) / r_target.unsqueeze(-1)  # (N, nT, nz_proj)

    outside_frac = ((query < z_solve[0]) | (query > z_solve[-1])).float().mean().item()
    if outside_frac > 0.0:
        warnings.warn(
            f"{100.0 * outside_frac:.2f}% of angular resampling points fall outside solve_z_range. "
            f"Increase solve_z_range to reduce truncation in phi(t,z) = u(t, z/r)/r."
        )

    B = N * n_time_samples
    phi_flat = interp_rows_uniform_1d(
        table=U_target.reshape(B, -1),
        query=query.reshape(B, -1),
        x_grid=z_solve,
        mode=snapshot_interp,
        chunk_rows=row_interp_chunk,
    )
    phi_flat = phi_flat / r_target.reshape(-1, 1)  # divide by r
    phi_target = phi_flat.reshape(N, n_time_samples, -1)  # (N, nT, nz_proj)

    # ------------------------------------------------------------------
    # 6) project angular part: a(t) = Pi_n phi(t)
    # ------------------------------------------------------------------
    a_target = phi_target @ P_proj  # (N, nT, K)
    if not torch.isfinite(a_target).all():
        raise RuntimeError("Non-finite values in angular coefficients a_target")

    # ------------------------------------------------------------------
    # 7) reconstruct homogeneous reduced target:
    #       c_h(t) = d_n(log r(t)) a(t)
    # ------------------------------------------------------------------
    Gdn = torch.tensor(compute_Gdn(K), device=device, dtype=dtype)
    c_h = apply_projected_dilation_chunked(
        a_flat=a_target.reshape(B, K),
        r_flat=r_target.reshape(B),
        Gdn=Gdn,
        chunk_rows=dilation_chunk,
    ).reshape(N, n_time_samples, K)

    if not torch.isfinite(c_h).all():
        raise RuntimeError("Non-finite values in homogeneous coefficients c_h")

    # ------------------------------------------------------------------
    # 8) dataset object
    # ------------------------------------------------------------------
    T_all = t_target.detach().cpu().numpy()[None, :].repeat(N, axis=0)  # (N, nT)
    C_all = c_h.detach().cpu().numpy()
    Phi_np = Phi_proj.detach().cpu().numpy()

    cfg = NeuralGalerkinDatasetConfig(
        n_time_samples=int(T_all.shape[1]),
        t_sampling=t_sampling,
        seed=seed,
        return_k_coords=False,
        pde_name="burgers_homogeneous",
    )

    ds = NeuralGalerkinDataset(
        config=cfg,
        t=T_all,
        c=C_all,
        device=str(device),
        dtype=dtype,
        x_grid=z_proj.detach().cpu().numpy(),
        basis_matrix=Phi_np,
    )

    # extra metadata / diagnostics
    ds.hermite_scale = float(scale)
    ds.hermite_shift = float(shift)
    ds.solve_z_range = tuple(float(v) for v in solve_z_range)
    ds.proj_z_range = tuple(float(v) for v in proj_z_range)
    ds.d_norm = r_target.detach()              # (N, nT)
    ds.angular_coeffs = a_target.detach()      # (N, nT, K)
    ds.outside_query_fraction = float(outside_frac)

    return ds

if __name__ == "__main__":
    from burger_eq.burgers_analytic import BurgersParallelSolver
    import torch

    solver = BurgersParallelSolver(device="cuda")

    neural_galerkin_dataset = burgers_neural_ds(
        solver=solver,
        N=1024,
        K=5,
        hz=0.1,
        Tmax=2.0,
        L=10.0,
        n_quad_points=200,
        q_n=8192,
        z_batch_size=300,
        sphere_mode="coeff",
        n_time_samples=200,
        t_sampling="grid",
        seed=42,
        dtype=torch.float32,
    )

    # Verify the fix worked
    ic_norms = torch.norm(neural_galerkin_dataset.c[:, 0, :], dim=1)
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")
    print(f"Dataset shape: {neural_galerkin_dataset.c.shape}")
    print(f"IC norms: min={ic_norms.min():.3f}, max={ic_norms.max():.3f}, mean={ic_norms.mean():.3f}")
    print(f"Coefficient range: [{neural_galerkin_dataset.c.min():.3f}, {neural_galerkin_dataset.c.max():.3f}]")

    if ic_norms.mean() > 0.5:
        print("\n✅ Dataset is GOOD! Saving...")
        neural_galerkin_dataset.save("src/burger_eq/neural_galerkin_ds.npz", format="npz")
        print("✅ Saved to: src/burger_eq/neural_galerkin_ds.npz")
    else:
        print("\n❌ ERROR: ICs are still near zero!")
        print(f"First IC: {neural_galerkin_dataset.c[0, 0, :]}")
        print("NOT saving broken dataset.")

    