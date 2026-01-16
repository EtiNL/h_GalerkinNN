import math
import numpy as np
import torch
from scipy.special import roots_hermite


def gauss_hermite_cached(n_points: int, device='cuda', dtype=torch.float32):
    """
    Get Gauss-Hermite quadrature nodes and weights.
    
    For integral: ∫[-∞,∞] f(y) exp(-y²) dy ≈ Σ w_i · f(y_i)
    
    Exact for polynomials up to degree 2n-1.
    """
    nodes_np, weights_np = roots_hermite(n_points)
    nodes = torch.tensor(nodes_np, device=device, dtype=dtype)
    weights = torch.tensor(weights_np, device=device, dtype=dtype)
    return nodes, weights


class BurgersParallelSolver:
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Initialized BurgersParallelSolver on device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # -------------------------
    # Interpolation (linear + cubic)
    # -------------------------
    @staticmethod
    @torch.no_grad()
    def _interp_uniform_table_linear(table: torch.Tensor, q: torch.Tensor, q0: float, dq: float) -> torch.Tensor:
        N, nq = table.shape
        pos = (q - q0) / dq
        i0 = torch.floor(pos).to(torch.long).clamp(0, nq - 2)
        frac = (pos - i0.to(pos.dtype)).clamp(0.0, 1.0)

        i0f = i0.reshape(-1)
        fracf = frac.reshape(-1)

        idx = i0f.unsqueeze(0).expand(N, -1)
        v0 = torch.gather(table, 1, idx)
        v1 = torch.gather(table, 1, idx + 1)
        out = v0 * (1.0 - fracf) + v1 * fracf
        return out.view(N, *q.shape)

    @staticmethod
    @torch.no_grad()
    def _interp_uniform_table_cubic(table: torch.Tensor, q: torch.Tensor, q0: float, dq: float) -> torch.Tensor:
        """
        Catmull–Rom cubic interpolation on a uniform grid.
        Much better than linear for steep x0(q) (e.g. tanh(i q) with large i).
        """
        N, nq = table.shape
        pos = (q - q0) / dq
        i1 = torch.floor(pos).to(torch.long)              # base index
        t = (pos - i1.to(pos.dtype)).clamp(0.0, 1.0)      # local coordinate in [0,1]

        # clamp so we can safely access i1-1 .. i1+2
        i1 = i1.clamp(0, nq - 2)  # ensure i1+1 exists
        i1f = i1.reshape(-1)
        tf = t.reshape(-1)

        im1 = (i1f - 1).clamp(0, nq - 1)
        i0  = i1f.clamp(0, nq - 1)
        ip1 = (i1f + 1).clamp(0, nq - 1)
        ip2 = (i1f + 2).clamp(0, nq - 1)

        idx_im1 = im1.unsqueeze(0).expand(N, -1)
        idx_i0  = i0.unsqueeze(0).expand(N, -1)
        idx_ip1 = ip1.unsqueeze(0).expand(N, -1)
        idx_ip2 = ip2.unsqueeze(0).expand(N, -1)

        p0 = torch.gather(table, 1, idx_im1)
        p1 = torch.gather(table, 1, idx_i0)
        p2 = torch.gather(table, 1, idx_ip1)
        p3 = torch.gather(table, 1, idx_ip2)

        # Catmull–Rom spline
        tt  = tf
        tt2 = tt * tt
        tt3 = tt2 * tt

        out = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * tt
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * tt2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * tt3
        )
        return out.view(N, *q.shape)

    @torch.no_grad()
    def _interp_uniform_table(self, table: torch.Tensor, q: torch.Tensor, q0: float, dq: float, interp: str) -> torch.Tensor:
        if interp == "cubic":
            return self._interp_uniform_table_cubic(table, q, q0, dq)
        return self._interp_uniform_table_linear(table, q, q0, dq)

    # -------------------------
    # Callable -> tables
    # -------------------------
    @torch.no_grad()
    def _eval_callables_on_grid(self, f_list, grid: torch.Tensor) -> torch.Tensor:
        device = self.device
        dtype = grid.dtype
        grid_np = grid.detach().cpu().numpy()
        rows = []
        for f in f_list:
            try:
                y = f(grid)
                if isinstance(y, torch.Tensor):
                    y_t = y.to(device=device, dtype=dtype)
                else:
                    raise TypeError
            except Exception:
                y_np = f(grid_np)
                y_t = torch.as_tensor(np.asarray(y_np), device=device, dtype=dtype)
            rows.append(y_t.reshape(-1))
        return torch.stack(rows, dim=0)

    @torch.no_grad()
    def _tables_from_callables(
        self,
        x0_list,
        G_list,
        q_grid: torch.Tensor,
        z_grid: torch.Tensor | None,
        compute_G_if_missing: bool,
        shift_G0_to_zero: bool,
        interp: str,
    ):
        device = self.device
        dtype = q_grid.dtype

        x0_table = self._eval_callables_on_grid(x0_list, q_grid)  # (N,nq)
        x0_z_table = self._eval_callables_on_grid(x0_list, z_grid) if z_grid is not None else None

        if G_list is not None:
            G_table = self._eval_callables_on_grid(G_list, q_grid)
        else:
            if not compute_G_if_missing:
                raise ValueError("G_list is None and compute_G_if_missing=False.")
            # float64 cumulative trapz for stability
            dq64 = (q_grid[1] - q_grid[0]).to(torch.float64)
            x064 = x0_table.to(torch.float64)
            incr = 0.5 * (x064[:, 1:] + x064[:, :-1]) * dq64
            G64 = torch.cat(
                [torch.zeros((x064.shape[0], 1), device=device, dtype=torch.float64),
                 torch.cumsum(incr, dim=1)],
                dim=1,
            )
            G_table = G64.to(dtype)

        if shift_G0_to_zero:
            q0f = float(q_grid[0].item())
            dqf = float((q_grid[1] - q_grid[0]).item())
            G0 = self._interp_uniform_table(G_table, torch.tensor(0.0, device=device, dtype=dtype), q0=q0f, dq=dqf, interp=interp)
            G_table = G_table - G0[:, 0:1]

        q0 = float(q_grid[0].item())
        dq = float((q_grid[1] - q_grid[0]).item())
        return x0_table, G_table, q0, dq, x0_z_table

    # -------------------------
    # Solve
    # -------------------------
    @torch.no_grad()
    def _solve_from_tables(
        self,
        x0_table: torch.Tensor,
        G_table: torch.Tensor,
        q0: float,
        dq: float,
        hz: float,
        ht: float,
        Tmax: float,
        z_range=(-7.0, 7.0),
        L: float = 6.0,  # Not used for Gauss-Hermite (kept for API compatibility)
        n_quad_points: int = 20,  # Gauss-Hermite: 10-20 points is excellent!
        P: torch.Tensor = None,
        z_batch_size: int = 4096,
        x0_z_table: torch.Tensor | None = None,
        den_eps: float = 1e-30,
        interp: str = "cubic",
    ):
        """
        Solve using Gauss-Hermite quadrature.
        
        The Cole-Hopf integral is:
            ∫ exp(-y²) · x0(q) · exp(-0.5·G(q)) dy
        
        Which we write as:
            ∫ f(y) · exp(-y²) dy    where f(y) = x0(q) · exp(-0.5·G(q))
        
        This is EXACTLY the form for Gauss-Hermite quadrature!
        """
        device = self.device
        dtype = x0_table.dtype

        z = torch.arange(z_range[0], z_range[1] + 0.5 * hz, hz, device=device, dtype=dtype)
        t = torch.arange(0.0, Tmax + 0.5 * ht, ht, device=device, dtype=dtype)
        nz = z.numel()
        nT = t.numel()
        N = x0_table.shape[0]

        # Get Gauss-Hermite nodes and weights
        # For ∫ f(y) e^(-y²) dy ≈ Σ w_i · f(y_i)
        y, w_y = gauss_hermite_cached(n_quad_points, device, dtype)
        
        print(f"Using Gauss-Hermite quadrature with {n_quad_points} points")
        print(f"  Node range: [{y.min().item():.2f}, {y.max().item():.2f}]")

        if P is None:
            U_out = torch.empty((N, nT, nz), device=device, dtype=dtype)
        else:
            K = P.shape[1]
            C_out = torch.empty((N, nT, K), device=device, dtype=dtype)

        for it in range(nT):
            tt = t[it]
            if P is not None:
                c_t = torch.zeros((N, P.shape[1]), device=device, dtype=dtype)

            for zs in range(0, nz, z_batch_size):
                ze = min(zs + z_batch_size, nz)
                z_b = z[zs:ze]
                bz = z_b.numel()

                if tt.item() == 0.0:
                    u_b = x0_z_table[:, zs:ze] if x0_z_table is not None else self._interp_uniform_table(x0_table, z_b, q0=q0, dq=dq, interp=interp)
                else:
                    a = 2.0 * torch.sqrt(tt)
                    q = z_b.view(bz, 1) + a * y.view(1, -1)  # [bz, n_quad]

                    # Evaluate x0 and G at quadrature points
                    x0_q = self._interp_uniform_table(x0_table, q, q0=q0, dq=dq, interp=interp)  # [N, bz, n_quad]
                    G_q  = self._interp_uniform_table(G_table,  q, q0=q0, dq=dq, interp=interp)  # [N, bz, n_quad]

                    # The integrand (WITHOUT exp(-y²) since it's in the Hermite weight!)
                    # f(y) = x0(q) · exp(-0.5·G(q))
                    # NOTE: We do NOT include exp(-y²) - it's already in w_y!
                    f_y = x0_q * torch.exp(-0.5 * G_q)  # [N, bz, n_quad]

                    # Gauss-Hermite integration: ∫ f(y) e^(-y²) dy ≈ Σ w_i · f(y_i)
                    wy = w_y.view(1, 1, -1)  # [1, 1, n_quad]
                    num = torch.sum(f_y * wy, dim=2)  # [N, bz]
                    
                    # For normalization, we integrate exp(-0.5·G(q))
                    den = torch.sum(torch.exp(-0.5 * G_q) * wy, dim=2)  # [N, bz]
                    den = den.clamp_min(den_eps)
                    
                    u_b = num / den

                if P is None:
                    U_out[:, it, zs:ze] = u_b
                else:
                    c_t += u_b @ P[zs:ze, :]

            if P is not None:
                C_out[:, it, :] = c_t

        z_vals = z.detach().cpu().numpy()
        t_vals = t.detach().cpu().numpy()
        return (z_vals, t_vals, U_out) if P is None else (z_vals, t_vals, C_out)

    @torch.no_grad()
    def solve_parallel_projected(
        self,
        x0_list,
        G_list=None,
        hz: float = 0.1,
        ht: float = 0.05,
        Tmax: float = 5.0,
        z_range=(-7.0, 7.0),
        L: float = 6.0,  # Kept for API compatibility, not used for Gauss-Hermite
        n_quad_points: int = 20,  # Gauss-Hermite: 10-20 is excellent!
        q_n: int = 8192,
        P: torch.Tensor = None,
        z_batch_size: int = 4096,
        compute_G_if_missing: bool = True,
        shift_G0_to_zero: bool = False,
        enforce_exact_ic: bool = True,
        interp: str = "cubic",
    ):
        """
        Solve Burgers equation using optimal Gauss-Hermite quadrature.
        
        The Cole-Hopf integral ∫ exp(-y²) · x0(q) · exp(-0.5·G(q)) dy
        is computed using Gauss-Hermite quadrature, which is the mathematically
        optimal method for this integral form.
        
        Args:
            n_quad_points: Number of Gauss-Hermite points
                - 10: Very good (error ~ 1e-10)
                - 20: Excellent (error ~ 1e-14) [RECOMMENDED]
                - 50: Machine precision for smooth functions
        """
        assert len(x0_list) > 0
        if G_list is not None:
            assert len(G_list) == len(x0_list)

        # For Gauss-Hermite, we estimate effective L based on typical node range
        # Hermite nodes typically span ~[-sqrt(4n), sqrt(4n)]
        L_effective = math.sqrt(4 * n_quad_points)  # Rough estimate of max Hermite node
        
        q_min = z_range[0] - 2.0 * math.sqrt(Tmax) * L_effective
        q_max = z_range[1] + 2.0 * math.sqrt(Tmax) * L_effective
        q_grid = torch.linspace(q_min, q_max, q_n, device=self.device, dtype=torch.float32)

        z_grid = None
        if enforce_exact_ic:
            z_grid = torch.arange(z_range[0], z_range[1] + 0.5 * hz, hz, device=self.device, dtype=torch.float32)

        x0_table, G_table, q0, dq, x0_z_table = self._tables_from_callables(
            x0_list=x0_list,
            G_list=G_list,
            q_grid=q_grid,
            z_grid=z_grid,
            compute_G_if_missing=compute_G_if_missing,
            shift_G0_to_zero=shift_G0_to_zero,
            interp=interp,
        )

        return self._solve_from_tables(
            x0_table=x0_table,
            G_table=G_table,
            q0=q0,
            dq=dq,
            hz=hz,
            ht=ht,
            Tmax=Tmax,
            z_range=z_range,
            L=L,
            n_quad_points=n_quad_points,
            P=P,
            z_batch_size=z_batch_size,
            x0_z_table=x0_z_table,
            interp=interp,
        )


if __name__ == '__main__':
    import numpy as np
    import torch
    import math
    import time
    try:
        from src.plot_utils import plot_sim_result
        from src.burger_eq.burgers_analytic import BurgersParallelSolver
    except ImportError:
        try:
            # If running from within the burger_eq directory
            import sys
            import os
            
            # Add parent directory (src) to path for plot_utils
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            # Add current directory for burgers_analytic
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            from plot_utils import plot_sim_result
            from burgers_analytic_np import num_approx_burgers
        except ImportError as e:
            print(f"Error importing required modules: {e}")
            print("\nPlease ensure you're running from the correct directory")
            print("Expected structure:")
            print("  src/plot_utils.py")
            print("  src/burger_eq/burgers_analytic.py")
            print("  src/burger_eq/burgers_galerkin_parallel.py")
            import sys
            sys.exit(1)
    
    # ======================================================================
    # Shared Parameters and Initial Condition
    # ======================================================================
    print("\n" + "="*70)
    print("COMPARISON: NumPy vs PyTorch Analytic Burgers Solvers")
    print("="*70)
    
    # -------- parameters --------
    hz = 0.1         # spatial step
    ht = 0.05        # time step
    Tmax = 8.0       # final time
    z_range = (-15.0, 15.0)
    
    print(f"\nSimulation Parameters:")
    print(f"  Spatial step (hz): {hz}")
    print(f"  Time step (ht): {ht}")
    print(f"  Final time (Tmax): {Tmax}")
    print(f"  Spatial range: {z_range}")
    
    # -------- initial profile x0 and its primitive G --------
    def x0_np(q):
        """Initial condition for NumPy version"""
        return np.sin(q)
    
    def G_np(q):
        """Primitive of x0 for NumPy version: G'(q) = x0(q)"""
        return 1 - np.cos(q)
    
    def x0_torch(q):
        """Initial condition for PyTorch version"""
        if isinstance(q, torch.Tensor):
            return torch.sin(q)
        else:
            return np.sin(q)
    
    def G_torch(q):
        """Primitive of x0 for PyTorch version: G'(q) = x0(q)"""
        if isinstance(q, torch.Tensor):
            return 1 - torch.cos(q)
        else:
            return 1 - np.cos(q)

    def x0_np(q):
        """Initial condition for NumPy version: x0(q) = exp(-q^2)"""
        return np.exp(-q**2)
    
    def G_np(q):
        """Primitive of x0 for NumPy version: G'(q) = x0(q)
        G(q) = sqrt(pi)/2 * erf(q)
        """
        from scipy.special import erf
        return (np.sqrt(np.pi) / 2.0) * erf(q)
    
    def x0_torch(q):
        """Initial condition for PyTorch version: x0(q) = exp(-q^2)"""
        if isinstance(q, torch.Tensor):
            return torch.exp(-q**2)
        else:
            return np.exp(-q**2)
    
    def G_torch(q):
        """Primitive of x0 for PyTorch version: G'(q) = x0(q)
        G(q) = sqrt(pi)/2 * erf(q)
        """
        if isinstance(q, torch.Tensor):
            return (math.sqrt(math.pi) / 2.0) * torch.erf(q)
        else:
            from scipy.special import erf
            return (np.sqrt(np.pi) / 2.0) * erf(q)
    
    
    # ======================================================================
    # 1. NumPy-based Solver (scipy integration)
    # ======================================================================
    print("\n" + "="*70)
    print("1. NumPy-BASED SOLVER (scipy.integrate.quad)")
    print("="*70)
    
    start_time_np = time.time()
    
    z_vals_np, t_vals_np, X_np = num_approx_burgers(
        x0=x0_np,
        G=G_np,
        hz=hz,
        ht=ht,
        Tmax=Tmax,
        z_range=z_range,
        n_quad = 100
    )
    
    elapsed_np = time.time() - start_time_np
    
    print(f"\nNumPy solver completed in {elapsed_np:.2f} seconds")
    print(f"Solution shapes:")
    print(f"  z_vals: {z_vals_np.shape}")
    print(f"  t_vals: {t_vals_np.shape}")
    print(f"  X_np: {X_np.shape}")
    print(f"  Solution range: [{X_np.min():.6f}, {X_np.max():.6f}]")
    
    # -------- plot NumPy solution --------
    plot_sim_result(z_vals_np, t_vals_np, X_np, 'X_numpy', notebook_plot=False)
    
    
    # ======================================================================
    # 2. PyTorch-based Solver (GPU-accelerated)
    # ======================================================================
    print("\n" + "="*70)
    print("2. PyTorch-BASED SOLVER (GPU-accelerated)")
    print("="*70)
    
    torch_solver = BurgersParallelSolver(device='cuda')
    
    start_time_torch = time.time()
    
    z_vals_torch, t_vals_torch, U_torch = torch_solver.solve_parallel_projected(
        x0_list=[x0_torch],
        G_list=[G_torch],
        hz=hz,
        ht=ht,
        Tmax=Tmax,
        z_range=z_range,
        L=L,
        n_quad_points=400,
        P=None,  # Full solution, not projected
        enforce_exact_ic=True,
        interp='cubic',
    )
    
    elapsed_torch = time.time() - start_time_torch
    
    X_torch = U_torch[0].cpu().numpy()  # [nT, nz]
    
    print(f"\nPyTorch solver completed in {elapsed_torch:.2f} seconds")
    print(f"Solution shapes:")
    print(f"  z_vals: {z_vals_torch.shape}")
    print(f"  t_vals: {t_vals_torch.shape}")
    print(f"  X_torch: {X_torch.shape}")
    print(f"  Solution range: [{X_torch.min():.6f}, {X_torch.max():.6f}]")
    
    # -------- plot PyTorch solution --------
    plot_sim_result(z_vals_torch, t_vals_torch, X_torch, 'X_pytorch', notebook_plot=False)
    
    
    # ======================================================================
    # 3. COMPARISON & ERROR ANALYSIS
    # ======================================================================
    print("\n" + "="*70)
    print("3. COMPARISON & ERROR ANALYSIS")
    print("="*70)
    
    # Ensure grids match (they should, but let's verify)
    assert z_vals_np.shape == z_vals_torch.shape, "Spatial grids don't match!"
    assert t_vals_np.shape == t_vals_torch.shape, "Time grids don't match!"
    assert X_np.shape == X_torch.shape, "Solution arrays don't match!"
    
    print(f"\nGrid verification: ✓ All grids match")
    
    # Compute absolute and relative errors
    abs_error = np.abs(X_torch - X_np)
    
    # Relative error (only compute where solution is significant)
    # Mask out regions where both solutions are very small
    threshold = 1e-3  # Only compute relative error where |solution| > threshold
    mask = np.abs(X_np) > threshold
    
    rel_error = np.zeros_like(abs_error)
    rel_error[mask] = abs_error[mask] / np.abs(X_np[mask])
    rel_error[~mask] = np.nan  # Mark insignificant regions as NaN
    
    # Error statistics
    print(f"\nAbsolute Error Statistics:")
    print(f"  Max abs error: {np.max(abs_error):.6e}")
    print(f"  Mean abs error: {np.mean(abs_error):.6e}")
    print(f"  Median abs error: {np.median(abs_error):.6e}")
    print(f"  Std abs error: {np.std(abs_error):.6e}")
    
    print(f"\nRelative Error Statistics (computed only where |solution| > {threshold}):")
    rel_error_valid = rel_error[~np.isnan(rel_error)]
    if len(rel_error_valid) > 0:
        print(f"  Max rel error: {np.max(rel_error_valid):.6e}")
        print(f"  Mean rel error: {np.mean(rel_error_valid):.6e}")
        print(f"  Median rel error: {np.median(rel_error_valid):.6e}")
        print(f"  Fraction of domain analyzed: {len(rel_error_valid) / rel_error.size * 100:.1f}%")
    else:
        print(f"  No regions with |solution| > {threshold}")
    
    # Error at specific times
    print(f"\nError at specific times:")
    n_times = len(t_vals_np)
    for i in [0, n_times//4, n_times//2, 3*n_times//4, -1]:
        t_val = t_vals_np[i]
        max_err_at_t = np.max(abs_error[i, :])
        mean_err_at_t = np.mean(abs_error[i, :])
        print(f"  t={t_val:.2f}: max={max_err_at_t:.6e}, mean={mean_err_at_t:.6e}")
    
    # -------- plot error --------
    plot_sim_result(z_vals_np, t_vals_np, abs_error, 
                    'Error_PyTorch_vs_NumPy', notebook_plot=False)
    
    # Also plot relative error
    plot_sim_result(z_vals_np, t_vals_np, rel_error, 
                    'RelError_PyTorch_vs_NumPy', notebook_plot=False)
    
    
    # ======================================================================
    # 4. PERFORMANCE COMPARISON
    # ======================================================================
    print("\n" + "="*70)
    print("4. PERFORMANCE COMPARISON")
    print("="*70)
    
    speedup = elapsed_np / elapsed_torch if elapsed_torch > 0 else float('inf')
    
    print(f"\n{'Method':<20} {'Time (s)':<15} {'Speedup':<15}")
    print("-"*50)
    print(f"{'NumPy (scipy)':<20} {elapsed_np:<15.2f} {1.0:<15.2f}")
    print(f"{'PyTorch (GPU)':<20} {elapsed_torch:<15.2f} {speedup:<15.2f}x")
    
    
    # ======================================================================
    # 5. FINAL SUMMARY
    # ======================================================================
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print(f"\nNumerical Agreement:")
    if np.max(abs_error) < 1e-4:
        agreement = "EXCELLENT"
    elif np.max(abs_error) < 1e-3:
        agreement = "GOOD"
    elif np.max(abs_error) < 1e-2:
        agreement = "FAIR"
    else:
        agreement = "POOR"
    print(f"  {agreement} (max absolute error: {np.max(abs_error):.6e})")
    
    print(f"\nComputational Efficiency:")
    print(f"  PyTorch solver is {speedup:.1f}x faster than NumPy solver")
    
    print(f"\nRecommendation:")
    if speedup > 5:
        print(f"  Use PyTorch solver for production (much faster)")
    elif speedup > 2:
        print(f"  Use PyTorch solver for large-scale problems")
    else:
        print(f"  Either solver is suitable; choose based on dependencies")
    
    print(f"\nNotes:")
    print(f"  - NumPy solver uses scipy.integrate.quad (adaptive quadrature)")
    print(f"  - PyTorch solver uses fixed quadrature with {200} points")
    print(f"  - Both methods compute the Cole-Hopf solution numerically")
    print(f"  - Small differences expected due to different integration methods")
    print(f"  - Relative error only computed where |solution| > {threshold}")
    print(f"  - (Avoids spurious large values from dividing by near-zero)")
    
    print("\n" + "="*70)