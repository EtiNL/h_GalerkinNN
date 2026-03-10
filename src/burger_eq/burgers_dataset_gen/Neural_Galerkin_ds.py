import math
import warnings
import numpy as np
import torch
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from pde_dataset import NeuralGalerkinDatasetConfig, NeuralGalerkinDataset
from homogeneity_utils_np import compute_Gdn


@torch.no_grad()
def sobol_sphere(N: int, K: int, device, dtype, seed: int = 0) -> torch.Tensor:
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
    return (G / nrm).to(dtype=dtype)


@torch.no_grad()
def trapz_weights_1d_torch(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 1 or x.numel() < 2:
        raise ValueError("x must be 1D with at least 2 points")
    w = torch.empty_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


@torch.no_grad()
def hermite_basis_x_torch(x: torch.Tensor, K: int, scale: float, shift: float) -> torch.Tensor:
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
    tq = t_query.clamp(t_grid[0], t_grid[-1])
    idx = torch.searchsorted(t_grid, tq, right=False).clamp(1, t_grid.numel() - 1)
    t_lo = t_grid[idx - 1]
    t_hi = t_grid[idx]
    w = (tq - t_lo) / (t_hi - t_lo)
    C_lo = C_grid[:, idx - 1, :]
    C_hi = C_grid[:, idx, :]
    return C_lo * (1.0 - w)[None, :, None] + C_hi * w[None, :, None]


@torch.no_grad()
def burgers_d_norm_from_snapshots(U: torch.Tensor, z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if U.ndim < 2:
        raise ValueError("U must have shape (..., nz)")
    if z.ndim != 1 or z.numel() != U.shape[-1]:
        raise ValueError("z must be 1D and match the last dimension of U")

    w = trapz_weights_1d_torch(z).to(device=U.device, dtype=U.dtype)
    view_shape = [1] * U.ndim
    view_shape[-1] = -1
    r = torch.sum(U * U * w.view(*view_shape), dim=-1)
    return r.clamp_min(eps)


@torch.no_grad()
def interp_rows_uniform_1d(
    table: torch.Tensor,
    query: torch.Tensor,
    x_grid: torch.Tensor,
    mode: str = "cubic",
    chunk_rows: int = 4096,
    uniform_tol: float = 5e-4,
) -> torch.Tensor:
    if table.ndim != 2 or query.ndim != 2:
        raise ValueError("table and query must both be 2D")
    if table.shape[0] != query.shape[0]:
        raise ValueError("table and query must have the same number of rows")
    if x_grid.ndim != 1 or x_grid.numel() < 2:
        raise ValueError("x_grid must be 1D with at least 2 points")

    nx = x_grid.numel()
    x0 = x_grid[0]
    x1 = x_grid[-1]
    dx = (x1 - x0) / (nx - 1)

    x_affine = x0 + dx * torch.arange(nx, device=x_grid.device, dtype=x_grid.dtype)
    max_dev = torch.max(torch.abs(x_grid - x_affine)).item()
    scale = max(1.0, torch.max(torch.abs(x_grid)).item(), abs(dx.item()) * nx)
    if max_dev > uniform_tol * scale:
        raise ValueError(
            f"interp_rows_uniform_1d expected an affine grid, but max deviation "
            f"is {max_dev:.3e} (tol={uniform_tol * scale:.3e})."
        )

    out = torch.empty((table.shape[0], query.shape[1]), device=table.device, dtype=table.dtype)

    for start in range(0, table.shape[0], chunk_rows):
        end = min(start + chunk_rows, table.shape[0])
        tab = table[start:end]
        q = query[start:end]
        pos = (q - x0) / dx

        if mode == "linear":
            i0 = torch.floor(pos).to(torch.long).clamp(0, nx - 2)
            frac = (pos - i0.to(pos.dtype)).clamp(0.0, 1.0)
            v0 = torch.gather(tab, 1, i0)
            v1 = torch.gather(tab, 1, i0 + 1)
            out[start:end] = v0 * (1.0 - frac) + v1 * frac

        elif mode == "cubic":
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
        s = torch.log(r_flat[start:end]).to(dtype=a_flat.dtype)
        rot = torch.matrix_exp(s[:, None, None] * Xi[None, :, :])
        rotated = torch.bmm(rot, a_flat[start:end].unsqueeze(-1)).squeeze(-1)
        out[start:end] = torch.sqrt(r_flat[start:end]).unsqueeze(-1) * rotated

    return out


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
    device = solver.device

    z = torch.arange(z_range[0], z_range[1] + 0.5 * hz, hz, device=device, dtype=dtype)

    shift = hermite_shift if hermite_shift is not None else 0.5 * (z_range[0] + z_range[1])
    scale = hermite_scale if hermite_scale is not None else (z_range[1] - z_range[0]) / 6.0

    Phi_z = hermite_basis_x_torch(z, K, scale=scale, shift=shift)
    w_z = trapz_weights_1d_torch(z)
    P = (w_z.unsqueeze(0) * Phi_z).t().contiguous()

    A = sobol_sphere(N, K, device=device, dtype=dtype, seed=seed)

    if sphere_mode == "l2":
        M = (Phi_z * w_z.unsqueeze(0)) @ Phi_z.t()
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

    t_grid = torch.tensor(t_vals, device=device, dtype=dtype)

    if t_sampling == "grid":
        t_target = torch.linspace(0.0, Tmax, n_time_samples, device=device, dtype=dtype)
    elif t_sampling == "random":
        rng = np.random.default_rng(seed)
        tt = rng.uniform(0.0, Tmax, size=(n_time_samples,)).astype(np.float64)
        tt.sort()
        tt[0] = 0.0
        t_target = torch.tensor(tt, device=device, dtype=dtype)
    else:
        raise ValueError("t_sampling must be 'grid' or 'random'")

    C_target = interp_time_batch(t_grid, C_grid, t_target)

    cfg = NeuralGalerkinDatasetConfig(
        n_time_samples=int(t_target.numel()),
        t_sampling=t_sampling,
        seed=seed,
        return_k_coords=False,
        pde_name="burgers",
    )

    ds = NeuralGalerkinDataset(
        config=cfg,
        t=t_target.detach().cpu().numpy()[None, :].repeat(N, axis=0),
        c=C_target.detach().cpu().numpy(),
        device=str(device),
        dtype=dtype,
        x_grid=z_vals,
        basis_matrix=Phi_z.detach().cpu().numpy(),
    )

    ds.hermite_scale = float(scale)
    ds.hermite_shift = float(shift)
    return ds


@torch.no_grad()
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
    enforce_common_valid_window: bool = True,
    min_valid_fraction: float = 0.95,
):
    device = solver.device
    proj_z_range = z_range
    if solve_z_range is None:
        solve_z_range = proj_z_range

    z_proj = torch.arange(
        proj_z_range[0],
        proj_z_range[1] + 0.5 * hz,
        hz,
        device=device,
        dtype=dtype,
    )

    shift = hermite_shift if hermite_shift is not None else 0.5 * (proj_z_range[0] + proj_z_range[1])
    scale = hermite_scale if hermite_scale is not None else (proj_z_range[1] - proj_z_range[0]) / 6.0

    Phi_proj = hermite_basis_x_torch(z_proj, K, scale=scale, shift=shift)
    w_proj = trapz_weights_1d_torch(z_proj)
    P_proj = (w_proj.unsqueeze(0) * Phi_proj).t().contiguous()

    A = sobol_sphere(N, K, device=device, dtype=dtype, seed=seed)

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
        P=None,
        z_batch_size=z_batch_size,
        compute_G_if_missing=True,
        enforce_exact_ic=True,
        interp=snapshot_interp,
    )

    t_grid = torch.tensor(t_vals, device=device, dtype=dtype)
    if t_sampling == "grid":
        t_target = torch.linspace(0.0, Tmax, n_time_samples, device=device, dtype=dtype)
    elif t_sampling == "random":
        rng = np.random.default_rng(seed)
        tt = rng.uniform(0.0, Tmax, size=(n_time_samples,)).astype(np.float64)
        tt.sort()
        tt[0] = 0.0
        t_target = torch.tensor(tt, device=device, dtype=dtype)
    else:
        raise ValueError("t_sampling must be 'grid' or 'random'")

    U_target = interp_time_batch(t_grid, U_grid, t_target)

    z_solve = torch.linspace(
        float(z_vals_solve[0]),
        float(z_vals_solve[-1]),
        len(z_vals_solve),
        device=device,
        dtype=dtype,
    )

    r_target = burgers_d_norm_from_snapshots(U_target, z_solve, eps=r_eps)

    Z_proj = max(abs(float(proj_z_range[0])), abs(float(proj_z_range[1])))
    Z_solve = max(abs(float(solve_z_range[0])), abs(float(solve_z_range[1])))
    r_cut = Z_proj / Z_solve

    r_min = float(r_target.min().item())
    r_max = float(r_target.max().item())
    print(f"[homogeneous ds] r(t) range: [{r_min:.4e}, {r_max:.4e}]")
    print(f"[homogeneous ds] admissibility threshold r_cut = {r_cut:.4e}")

    valid_mask = r_target >= r_cut
    valid_frac_per_time = valid_mask.float().mean(dim=0)

    if enforce_common_valid_window:
        keep_t = valid_frac_per_time >= min_valid_fraction
    else:
        keep_t = torch.ones_like(valid_frac_per_time, dtype=torch.bool)

    if keep_t.sum() == 0:
        raise RuntimeError(
            "No homogeneous time samples remain after validity filtering. "
            "Reduce Tmax or enlarge solve_z_range."
        )

    if keep_t.sum() < t_target.numel():
        dropped = int((~keep_t).sum().item())
        print(f"[homogeneous ds] dropping {dropped} / {t_target.numel()} time samples due to small r(t)")

    t_target = t_target[keep_t]
    U_target = U_target[:, keep_t, :]
    r_target = r_target[:, keep_t]

    query = z_proj.view(1, 1, -1) / r_target.unsqueeze(-1)

    outside_frac = ((query < z_solve[0]) | (query > z_solve[-1])).float().mean().item()
    if outside_frac > 0.0:
        warnings.warn(
            f"{100.0 * outside_frac:.2f}% of angular resampling points still fall outside solve_z_range.",
            UserWarning,
        )

    B = U_target.shape[0] * U_target.shape[1]
    phi_flat = interp_rows_uniform_1d(
        table=U_target.reshape(B, -1),
        query=query.reshape(B, -1),
        x_grid=z_solve,
        mode=snapshot_interp,
        chunk_rows=row_interp_chunk,
    )
    phi_flat = phi_flat / r_target.reshape(-1, 1)
    phi_target = phi_flat.reshape(U_target.shape[0], U_target.shape[1], -1)

    a_target = phi_target @ P_proj
    Gdn = torch.tensor(compute_Gdn(K), device=device, dtype=dtype)

    c_h = apply_projected_dilation_chunked(
        a_flat=a_target.reshape(B, K),
        r_flat=r_target.reshape(B),
        Gdn=Gdn,
        chunk_rows=dilation_chunk,
    ).reshape(U_target.shape[0], U_target.shape[1], K)

    cfg = NeuralGalerkinDatasetConfig(
        n_time_samples=int(t_target.numel()),
        t_sampling=t_sampling,
        seed=seed,
        return_k_coords=False,
        pde_name="burgers_homogeneous",
    )

    ds = NeuralGalerkinDataset(
        config=cfg,
        t=t_target.detach().cpu().numpy()[None, :].repeat(N, axis=0),
        c=c_h.detach().cpu().numpy(),
        device=str(device),
        dtype=dtype,
        x_grid=z_proj.detach().cpu().numpy(),
        basis_matrix=Phi_proj.detach().cpu().numpy(),
    )

    ds.hermite_scale = float(scale)
    ds.hermite_shift = float(shift)
    ds.solve_z_range = tuple(float(v) for v in solve_z_range)
    ds.proj_z_range = tuple(float(v) for v in proj_z_range)
    ds.d_norm = r_target.detach()
    ds.angular_coeffs = a_target.detach()
    ds.outside_query_fraction = float(outside_frac)
    ds.r_cut = float(r_cut)
    ds.valid_frac_per_time = valid_frac_per_time.detach().cpu()

    return ds

if __name__ == "__main__":
    from burger_eq.burgers_analytic import BurgersParallelSolver

    solver = BurgersParallelSolver(device="cuda")

    hds = burgers_homogeneous_ds(
        solver=solver,
        N=128,
        K=5,
        hz=0.1,
        Tmax=2.0,
        z_range=(-7.0, 7.0),
        solve_z_range=(-200.0, 200.0),
        L=10.0,
        n_quad_points=200,
        q_n=8192,
        z_batch_size=300,
        sphere_mode="coeff",
        n_time_samples=100,
        t_sampling="grid",
        seed=42,
        dtype=torch.float32,
        hermite_scale=1.0,
        hermite_shift=0.0,
        enforce_common_valid_window=True,
        min_valid_fraction=0.95,
    )

    print("=" * 60)
    print("HOMOGENEOUS DATASET VERIFICATION")
    print("=" * 60)
    print(f"Dataset shape: {tuple(hds.c.shape)}")
    print(f"outside_query_fraction: {hds.outside_query_fraction:.4f}")
    print(f"d_norm range: [{hds.d_norm.min().item():.4e}, {hds.d_norm.max().item():.4e}]")