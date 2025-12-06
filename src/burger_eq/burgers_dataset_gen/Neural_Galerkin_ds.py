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
    save_dataset,
    load_dataset,
    DatasetMetadata,
)

# -------------------------
# helpers
# -------------------------
@torch.no_grad()
def sobol_sphere(N: int, K: int, device, dtype, seed: int = 0) -> torch.Tensor:
    # scramble=True avoids pathological dyadic endpoints; seed makes it reproducible
    eng = torch.quasirandom.SobolEngine(dimension=K, scramble=True, seed=seed)

    # avoid the very first Sobol point (often all zeros)
    U = eng.draw(N + 1)[1:].to(device=device, dtype=torch.float64)

    # float32-safe clamp (eps is too small for erfinv stability on some GPUs)
    u_eps = 1e-6 if dtype == torch.float32 else 1e-12
    U = U.clamp(u_eps, 1.0 - u_eps)

    # compute in float64 for stability, then cast
    G = math.sqrt(2.0) * torch.erfinv(2.0 * U - 1.0)  # (N,K) float64

    # robust normalization: fix any bad rows deterministically
    nrm = torch.linalg.vector_norm(G, dim=1, keepdim=True)  # float64
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
    w = torch.empty_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w

@torch.no_grad()
def hermite_basis_x_torch(x: torch.Tensor, K: int, scale: float, shift: float) -> torch.Tensor:
    """
    Stable Hermite *functions* basis (orthonormal in L2(R)) with scaling:
      y = (x - shift)/scale
      Phi[k](x) = (1/sqrt(scale)) * ϕ_k(y)

    Recurrence (stable, avoids inf*0):
      ϕ0 = π^{-1/4} exp(-y^2/2)
      ϕ1 = sqrt(2) y ϕ0
      ϕ_{k+1} = sqrt(2/(k+1)) y ϕ_k - sqrt(k/(k+1)) ϕ_{k-1}
    """
    y = (x - shift) / scale
    y_flat = y.reshape(-1)
    M = y_flat.numel()

    # compute in float64 for stability, cast back
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

def burgers_neural_ds(
    solver,
    N: int,
    K: int,
    hz: float = 0.1,
    Tmax: float = 5.0,
    z_range=(-7.0, 7.0),
    L: float = 6.0,
    n_quad_points: int = 200,
    q_n: int = 8192,
    z_batch_size: int = 4096,
    sphere_mode: str = "coeff",   # "coeff" or "l2"
    n_time_samples: int = 200,
    t_sampling: str = "grid",     # "grid" or "random"
    seed: int = 0,
    normalize_t: bool = False,
    normalize_c: bool = False,
    return_k_coords: bool = False,
    dtype: torch.dtype = torch.float32,
):
    device = solver.device

    # spatial grid used by solver
    z = torch.arange(z_range[0], z_range[1] + 0.5 * hz, hz, device=device, dtype=dtype)

    # Hermite params
    shift = 0.5 * (z_range[0] + z_range[1])
    scale = (z_range[1] - z_range[0]) / 6.0

    # basis + projection matrix
    Phi_z = hermite_basis_x_torch(z, K, scale=scale, shift=shift)          # (K,nz)
    w_z = trapz_weights_1d_torch(z)                                        # (nz,)
    P = (w_z.unsqueeze(0) * Phi_z).t().contiguous()                        # (nz,K)

    # coefficients on sphere
    A = A = sobol_sphere(N, K, device=device, dtype=dtype, seed=seed)      # (N,K)
    if sphere_mode == "l2":
        M = (Phi_z * w_z.unsqueeze(0)) @ Phi_z.t()
        M = 0.5 * (M + M.t()) + 1e-10 * torch.eye(K, device=device, dtype=dtype)
        Lc = torch.linalg.cholesky(M)
        Ccoeff = torch.linalg.solve_triangular(Lc.t(), A.t(), upper=True).t()
    else:
        Ccoeff = A

    # q_grid consistent with solver
    q_min = z_range[0] - 2.0 * math.sqrt(Tmax) * L
    q_max = z_range[1] + 2.0 * math.sqrt(Tmax) * L
    q_grid = torch.linspace(q_min, q_max, q_n, device=device, dtype=dtype)

    # x0 table on q_grid (solver will sample these once then interpolate internally)
    Phi_q = hermite_basis_x_torch(q_grid, K, scale=scale, shift=shift)     # (K,q_n)
    x0_table = Ccoeff @ Phi_q                                              # (N,q_n)

    if not torch.isfinite(x0_table).all():
        bad = (~torch.isfinite(x0_table)).nonzero(as_tuple=False)[0]
        raise RuntimeError(
            f"Non-finite x0_table at idx={tuple(int(i) for i in bad)} value={x0_table[tuple(bad.tolist())]}"
        )


    def row_fn(table_row: torch.Tensor):
        def f(q):
            if isinstance(q, torch.Tensor):
                return table_row
            return table_row.detach().cpu().numpy()
        return f

    x0_list = [row_fn(x0_table[i]) for i in range(N)]

    # solve (projected)
    ht = hz ** 2
    z_vals, t_vals, C_grid = solver.solve_parallel_projected(
        x0_list=x0_list,
        G_list=None,                    # solver computes G from x0
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
        interp="cubic",
    enforce_exact_ic=True,
)

    # time grid -> target times (make "random" strictly increasing to avoid dt=0)
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

    C_target = interp_time_batch(t_grid, C_grid, t_target)                 # (N,nT,K)

    # final sanity (fail fast if something is still wrong)
    if not torch.isfinite(C_target).all():
        bad = (~torch.isfinite(C_target)).nonzero(as_tuple=False)[0]
        raise RuntimeError(f"Non-finite C_target at idx={tuple(int(i) for i in bad)} value={C_target[tuple(bad.tolist())]}")

    # build NeuralGalerkinDataset
    T_all = t_target.detach().cpu().numpy()[None, :].repeat(N, axis=0)     # (N,nT)
    C_all = C_target.detach().cpu().numpy()                                # (N,nT,K)
    Phi_np = Phi_z.detach().cpu().numpy()                                  # (K,nz)

    cfg = NeuralGalerkinDatasetConfig(
        n_time_samples=int(T_all.shape[1]),
        t_sampling=t_sampling,
        seed=seed,
        normalize_t=normalize_t,
        normalize_c=normalize_c,
        return_k_coords=return_k_coords,
        pde_name="burgers",
    )

    return NeuralGalerkinDataset(
        config=cfg,
        t=T_all,
        c=C_all,
        device=str(device),
        dtype=dtype,
        x_grid=z_vals,
        basis_matrix=Phi_np,
    )

if __name__ == "__main__":
    from burger_eq.burgers_analytic import BurgersParallelSolver

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
        seed=0,
        normalize_t=False,
        normalize_c=False,
        return_k_coords=False,
        dtype=torch.float32,
    )

    neural_galerkin_dataset.save("src/burger_eq/neural_galerkin_ds.npz", format="npz")
