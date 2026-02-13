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

    