import math
import numpy as np
import torch
from scipy.special import roots_hermite


# =====================================================================
# Gauss-Hermite Quadrature
# =====================================================================

def gauss_hermite_cached(n_points: int, device='cuda', dtype=torch.float64):
    """Get Gauss-Hermite quadrature nodes and weights (using scipy)."""
    nodes_np, weights_np = roots_hermite(n_points)
    nodes = torch.tensor(nodes_np, device=device, dtype=dtype)
    weights = torch.tensor(weights_np, device=device, dtype=dtype)
    return nodes, weights


# =====================================================================
# Hermite Functions for ROM
# =====================================================================

def hermite_polynomial_torch(n: int, x: torch.Tensor) -> torch.Tensor:
    """Compute Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return 2.0 * x
    
    H_prev_prev, H_prev = torch.ones_like(x), 2.0 * x
    for k in range(1, n):
        H_current = 2.0 * x * H_prev - 2.0 * k * H_prev_prev
        H_prev_prev, H_prev = H_prev, H_current
    
    return H_prev


def hermite_function_torch(n: int, y: torch.Tensor) -> torch.Tensor:
    """Compute normalized Hermite function."""
    norm = 1.0 / math.sqrt(2**n * math.factorial(n) * math.sqrt(math.pi))
    return norm * torch.exp(-0.5 * y * y) * hermite_polynomial_torch(n, y)


# =====================================================================
# ROM Matrix Computation
# =====================================================================

def compute_A(n: int, device='cuda', dtype=torch.float32):
    """Compute linear operator A for Burgers equation."""
    A = torch.zeros(n, n, device=device, dtype=dtype)
    for i in range(n):
        A[i, i] = -(i + 0.5)
        if i >= 2:
            A[i, i-2] = math.sqrt(i / 2.0)
        if i + 2 < n:
            A[i, i+2] = math.sqrt((i + 1) / 2.0)
    return A


def compute_Bfull(n: int, n_quad: int = 100, device='cuda', dtype=torch.float64):
    """Compute nonlinear tensor Bfull for Burgers equation (vectorized)."""
    print(f"\nComputing Bfull (n={n}, n_quad={n_quad})...")
    
    nodes, weights = gauss_hermite_cached(n_quad, device, dtype)
    
    psi = torch.zeros(n, n_quad, device=device, dtype=dtype)
    for i in range(n):
        psi[i, :] = hermite_function_torch(i, nodes)
    
    dpsi = torch.zeros(n, n_quad, device=device, dtype=dtype)
    for i in range(n):
        if i == 0:
            dpsi[i, :] = -nodes * psi[i, :]
        else:
            dpsi[i, :] = math.sqrt(i / 2.0) * psi[i-1, :]
            if i + 1 < n:
                dpsi[i, :] -= math.sqrt((i + 1) / 2.0) * psi[i+1, :]
    
    exp_factor = torch.exp(nodes**2)
    dpsi_weighted = dpsi * weights * exp_factor
    Bfull = torch.einsum('iq,jq,kq->ijk', dpsi_weighted, psi, psi)
    Bfull = 0.5 * (Bfull + Bfull.transpose(1, 2))
    
    print(f"✓ Bfull computed: {Bfull.shape}")
    return Bfull


def compute_Bn(Bfull: torch.Tensor, c: torch.Tensor):
    """
    Compute state-dependent Bn matrix: Bn[i,j] = 0.5 * sum_k c[k] * Bfull[i,j,k]
    
    Args:
        Bfull: [K, K, K] tensor
        c: [B, K] or [K] tensor
    
    Returns:
        Bn: [B, K, K] or [K, K] tensor
    """
    if c.ndim == 1:
        Bn = 0.5 * torch.einsum('ijk,k->ij', Bfull, c)
    else:
        Bn = 0.5 * torch.einsum('ijk,bk->bij', Bfull, c)
    return Bn


def compute_Gdn(n: int, device='cuda', dtype=torch.float32):
    """
    Compute the generator matrix Gdn for the dilation group.
    
    For Burgers: d(s)x(z) = e^s x(e^s z) gives α=1, β=1
    - Diagonal: 0.5
    - Upper off-diagonal (i, i+2): √((i+1)/2)
    - Lower off-diagonal (i, i-2): -√(i/2)
    """
    Gdn = torch.zeros(n, n, device=device, dtype=dtype)
    for i in range(n):
        Gdn[i, i] = 0.5
        if i >= 2:
            Gdn[i, i-2] = -math.sqrt(i / 2.0)
        if i + 2 < n:
            Gdn[i, i+2] = math.sqrt((i + 1) / 2.0)
    return Gdn


# =====================================================================
# Galerkin Solver
# =====================================================================

class BurgersGalerkinSolver:
    """Parallel Galerkin solver for multiple initial conditions."""
    
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Initialized BurgersGalerkinSolver on device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    @torch.no_grad()
    def _compute_initial_coefficients(
        self,
        x0_list,
        n: int,
        n_quad: int = 200,
        dtype=torch.float64,
    ):
        """Compute initial Hermite coefficients for all x0 functions."""
        device = self.device
        N = len(x0_list)
        
        nodes, weights = gauss_hermite_cached(n_quad, device, dtype)
        
        # Evaluate all x0 functions on quadrature nodes
        x0_vals = torch.zeros(N, n_quad, device=device, dtype=dtype)
        nodes_np = nodes.cpu().numpy()
        
        for i, x0 in enumerate(x0_list):
            try:
                vals = x0(nodes)
                if isinstance(vals, torch.Tensor):
                    x0_vals[i] = vals.to(device=device, dtype=dtype)
                else:
                    raise TypeError
            except Exception:
                vals_np = x0(nodes_np)
                x0_vals[i] = torch.tensor(vals_np, device=device, dtype=dtype)
        
        # Compute Hermite functions at quadrature nodes
        psi = torch.zeros(n, n_quad, device=device, dtype=dtype)
        for k in range(n):
            psi[k] = hermite_function_torch(k, nodes)
        
        # Compute coefficients via quadrature: c[k] = ∫ x0(y) ψ_k(y) dy
        exp_factor = torch.exp(nodes**2)
        weighted_psi = psi * weights * exp_factor  # [n, n_quad]
        
        C0 = torch.einsum('nq,bq->bn', weighted_psi, x0_vals)  # [N, n]
        
        return C0
    
    @torch.no_grad()
    def solve_parallel(
        self,
        x0_list,
        n: int,
        ht: float,
        Tmax: float,
        n_quad: int = 200,
        dtype_compute=torch.float64,
        dtype_output=torch.float32,
    ):
        """
        Solve Galerkin system for multiple initial conditions in parallel.
        
        Args:
            x0_list: List of initial condition functions
            n: Dimension of Galerkin projection
            ht: Time step
            Tmax: Final time
            n_quad: Number of quadrature points for computing coefficients
            dtype_compute: Computation dtype (float64 for stability)
            dtype_output: Output dtype
        
        Returns:
            t_vals: Time values (numpy array)
            C_all: Coefficient trajectories [N, nT, n] (torch tensor)
        """
        device = self.device
        N = len(x0_list)
        
        # Time grid
        t_vals = np.arange(0.0, Tmax + 0.5 * ht, ht)
        nT = len(t_vals)
        
        print(f"\n{'='*60}")
        print(f"GALERKIN SOLVER")
        print(f"{'='*60}")
        print(f"Number of ICs: {N}")
        print(f"Galerkin dimension: {n}")
        print(f"Time steps: {nT} (ht={ht}, Tmax={Tmax})")
        
        # Pre-compute matrices
        A = compute_A(n, device, dtype_compute)
        Bfull = compute_Bfull(n, n_quad=n_quad, device=device, dtype=dtype_compute)
        I = torch.eye(n, device=device, dtype=dtype_compute)
        
        # Compute initial coefficients
        print(f"\nComputing initial coefficients...")
        C0 = self._compute_initial_coefficients(x0_list, n, n_quad, dtype_compute)
        print(f"✓ Initial coefficients computed: {C0.shape}")
        
        # Allocate output
        C_all = torch.zeros(N, nT, n, device=device, dtype=dtype_compute)
        C_all[:, 0, :] = C0
        
        # Time integration
        print(f"\nTime integration...")
        C_prev = C0.clone()
        
        for it in range(1, nT):
            # Compute Bn for all ICs: [N, n, n]
            Bn = compute_Bn(Bfull, C_prev)
            
            # Compute M = (I - ht*A - ht*Bn)^{-1} for each IC
            # M_inv[b] = I - ht*A - ht*Bn[b]
            M_inv = I.unsqueeze(0) - ht * A.unsqueeze(0) - ht * Bn  # [N, n, n]
            
            # Solve M * C_next = C_prev for each IC
            C_next = torch.linalg.solve(M_inv, C_prev.unsqueeze(-1)).squeeze(-1)  # [N, n]
            
            C_all[:, it, :] = C_next
            C_prev = C_next
            
            if (it + 1) % max(1, nT // 10) == 0:
                print(f"  Step {it+1}/{nT} ({100*(it+1)/nT:.1f}%)")
        
        print(f"✓ Integration complete!")
        
        # Convert to output dtype
        C_all = C_all.to(dtype_output)
        
        return t_vals, C_all
    
    @torch.no_grad()
    def reconstruct_solution(
        self,
        C_all: torch.Tensor,
        z_vals: np.ndarray | torch.Tensor,
        batch_size: int = 512,
    ):
        """
        Reconstruct full solution u(t,z) from coefficients.
        
        Args:
            C_all: Coefficient trajectories [N, nT, n]
            z_vals: Spatial grid points
            batch_size: Batch size for z evaluation
        
        Returns:
            U_all: Solution [N, nT, nz]
        """
        device = self.device
        dtype = C_all.dtype
        
        if isinstance(z_vals, np.ndarray):
            z_vals = torch.tensor(z_vals, device=device, dtype=dtype)
        else:
            z_vals = z_vals.to(device=device, dtype=dtype)
        
        N, nT, n = C_all.shape
        nz = len(z_vals)
        
        U_all = torch.zeros(N, nT, nz, device=device, dtype=dtype)
        
        # Pre-compute Hermite functions
        for zs in range(0, nz, batch_size):
            ze = min(zs + batch_size, nz)
            z_batch = z_vals[zs:ze]
            bz = ze - zs
            
            # Compute ψ_k(z) for all k and z in batch
            psi = torch.zeros(n, bz, device=device, dtype=dtype)
            for k in range(n):
                psi[k] = hermite_function_torch(k, z_batch)
            
            # Reconstruct: u(t,z) = Σ_k c_k(t) * ψ_k(z)
            # C_all: [N, nT, n], psi: [n, bz] -> U_batch: [N, nT, bz]
            U_batch = torch.einsum('btk,kz->btz', C_all, psi)
            U_all[:, :, zs:ze] = U_batch
        
        return U_all


# =====================================================================
# Homogeneous Galerkin Solver
# =====================================================================

class BurgersHomogeneousGalerkinSolver:
    """Parallel homogeneous Galerkin solver with radial-angular decomposition."""
    
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Initialized BurgersHomogeneousGalerkinSolver on device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    @torch.no_grad()
    def _compute_norm_d(
        self,
        x0,
        n: int,
        n_quad: int = 200,
        max_steps: int = 200,
        eps: float = 0.001,
        dtype=torch.float64,
    ):
        """Compute dilation norm using analytical scaling property.
        
        For d(s)x(z) = e^s x(e^s z), we have: ||d(s)x||_H = e^{s/2} ||x||_H
        So we can compute ||x0||_H once and use bisection on the formula.
        """
        device = self.device
        
        nodes, weights = gauss_hermite_cached(n_quad, device, dtype)
        exp_factor = torch.exp(nodes**2)
        
        # Compute ||x0||_H once
        try:
            x0_vals = x0(nodes)
            if not isinstance(x0_vals, torch.Tensor):
                x0_vals = torch.tensor(x0_vals, device=device, dtype=dtype)
        except:
            x0_vals_np = x0(nodes.cpu().numpy())
            x0_vals = torch.tensor(x0_vals_np, device=device, dtype=dtype)
        
        # ∫ x0(z)² dz using Gauss-Hermite
        integrand = x0_vals**2 * weights * exp_factor
        norm_x0_squared = torch.sum(integrand)
        norm_x0 = torch.sqrt(norm_x0_squared.clamp(min=0))
        
        # Use analytical formula: ||d(-s)x||_H = e^{-s/2} ||x||_H
        # Bisect to find s where e^{-s/2} ||x||_H = 1
        # i.e., s = -2 log(||x||_H)
        # But use bisection to match numpy logic exactly
        
        s_inf = -3.0
        s_sup = 3.0
        s_mid = 0.5 * (s_sup + s_inf)
        
        # Compute norms using scaling formula
        n_inf = torch.exp(torch.tensor(-s_inf / 2.0, device=device, dtype=dtype)) * norm_x0
        n_sup = torch.exp(torch.tensor(-s_sup / 2.0, device=device, dtype=dtype)) * norm_x0
        n_mid = torch.exp(torch.tensor(-s_mid / 2.0, device=device, dtype=dtype)) * norm_x0
        
        if n_sup >= 1.0 or n_inf <= 1.0:
            # Adjust range
            s_inf = -5.0 if n_inf <= 1.0 else s_inf
            s_sup = 5.0 if n_sup >= 1.0 else s_sup
            s_mid = 0.5 * (s_sup + s_inf)
            n_inf = torch.exp(torch.tensor(-s_inf / 2.0, device=device, dtype=dtype)) * norm_x0
            n_sup = torch.exp(torch.tensor(-s_sup / 2.0, device=device, dtype=dtype)) * norm_x0
            n_mid = torch.exp(torch.tensor(-s_mid / 2.0, device=device, dtype=dtype)) * norm_x0
        
        assert n_sup.item() < 1.0, f"n_sup = {n_sup.item():.6f} >= 1.0, ||x0||_H = {norm_x0.item():.6f}"
        assert n_inf.item() > 1.0, f"n_inf = {n_inf.item():.6f} <= 1.0, ||x0||_H = {norm_x0.item():.6f}"
        
        steps = 0
        while (abs(1.0 - n_mid.item()) > eps) and steps < max_steps:
            s_mid = 0.5 * (s_sup + s_inf)
            n_mid = torch.exp(torch.tensor(-s_mid / 2.0, device=device, dtype=dtype)) * norm_x0
            
            if n_mid.item() < 1.0:
                n_sup = n_mid.item()
                s_sup = s_mid
            else:
                n_inf = n_mid.item()
                s_inf = s_mid
            
            steps += 1
        
        assert steps < max_steps, "max norm_d steps reached"
        
        return torch.exp(torch.tensor(s_mid, device=device, dtype=dtype))
    
    @torch.no_grad()
    def _compute_initial_decomposition(
        self,
        x0_list,
        n: int,
        n_quad: int = 200,
        dtype=torch.float64,
    ):
        """Compute initial radial-angular decomposition - mimics numpy version."""
        device = self.device
        N = len(x0_list)
        
        r0 = torch.zeros(N, device=device, dtype=dtype)
        Phi0 = torch.zeros(N, n, device=device, dtype=dtype)
        
        nodes, weights = gauss_hermite_cached(n_quad, device, dtype)
        exp_factor = torch.exp(nodes**2)
        
        # Pre-compute Hermite functions at quadrature nodes
        psi = torch.zeros(n, n_quad, device=device, dtype=dtype)
        for k in range(n):
            psi[k] = hermite_function_torch(k, nodes)
        
        for i, x0 in enumerate(x0_list):
            # Step 1: Compute r = ||x0||_d
            r0[i] = self._compute_norm_d(x0, n, n_quad, dtype=dtype)
            
            # Step 2: Compute Φ = Π_n(d(-log r) x0)
            # This mimics: Pi_n(dilation(-np.log(norm_d_x0), x0), n)
            # where dilation(s, x) returns lambda z: exp(s) * x(exp(s) * z)
            
            log_r = torch.log(r0[i])
            s = -log_r  # s = -log(r), so d(s) = d(-log r)
            scale = torch.exp(s)  # scale = exp(-log r) = 1/r
            
            # Evaluate dilated function at quadrature nodes
            # d(-log r)x(z) = e^{-log r} x(e^{-log r} z) = (1/r) x(z/r)
            dilated_nodes = scale * nodes  # nodes / r
            
            try:
                vals = x0(dilated_nodes)
                if not isinstance(vals, torch.Tensor):
                    vals = torch.tensor(vals, device=device, dtype=dtype)
            except:
                vals_np = x0(dilated_nodes.cpu().numpy())
                vals = torch.tensor(vals_np, device=device, dtype=dtype)
            
            # Apply amplitude scaling
            vals_dilated = scale * vals  # (1/r) * x0(z/r)
            
            # Project: c_k = ∫ [d(-log r)x(y)] * ψ_k(y) dy
            weighted_vals = vals_dilated * weights * exp_factor
            Phi0[i] = torch.einsum('kq,q->k', psi, weighted_vals)
            
            # Normalize
            Phi0[i] = Phi0[i] / torch.norm(Phi0[i])
        
        return r0, Phi0
    
    @torch.no_grad()
    def solve_parallel(
        self,
        x0_list,
        n: int,
        ht: float,
        Tmax: float,
        nu: float = 2.0,
        n_quad: int = 200,
        dtype_compute=torch.float64,
        dtype_output=torch.float32,
    ):
        """
        Solve homogeneous Galerkin system in parallel.
        
        Returns:
            t_vals: Time values
            r_all: Radial components [N, nT]
            Phi_all: Angular components [N, nT, n]
        """
        device = self.device
        N = len(x0_list)
        
        t_vals = np.arange(0.0, Tmax + 0.5 * ht, ht)
        nT = len(t_vals)
        
        print(f"\n{'='*60}")
        print(f"HOMOGENEOUS GALERKIN SOLVER")
        print(f"{'='*60}")
        print(f"Number of ICs: {N}")
        print(f"Dimension: {n}, ν={nu}")
        print(f"Time steps: {nT}")
        
        # Pre-compute matrices
        A = compute_A(n, device, dtype_compute)
        Bfull = compute_Bfull(n, n_quad, device, dtype_compute)
        Gdn = compute_Gdn(n, device, dtype_compute)
        I = torch.eye(n, device=device, dtype=dtype_compute)
        
        # Initial decomposition
        print(f"\nComputing initial decomposition...")
        r0, Phi0 = self._compute_initial_decomposition(x0_list, n, n_quad, dtype_compute)
        print(f"✓ r0: {r0.shape}, Phi0: {Phi0.shape}")
        
        # Allocate outputs
        r_all = torch.zeros(N, nT, device=device, dtype=dtype_compute)
        Phi_all = torch.zeros(N, nT, n, device=device, dtype=dtype_compute)
        
        r_all[:, 0] = r0
        Phi_all[:, 0, :] = Phi0
        
        # Time integration
        print(f"\nTime integration...")
        r_prev = r0.clone()
        Phi_prev = Phi0.clone()
        
        for it in range(1, nT):
            # Compute Bn[b] for each IC
            Bn = compute_Bn(Bfull, Phi_prev)  # [N, n, n]
            
            # Compute α[b] = Φ^T (A + B_n) Φ / (Φ^T G_d Φ)
            num = torch.einsum('bn,bnm,bm->b', Phi_prev, A.unsqueeze(0) + Bn, Phi_prev)
            den = torch.einsum('bn,nm,bm->b', Phi_prev, Gdn, Phi_prev)
            alpha = num / den  # [N]
            
            # Update Φ: (I - ht r^ν (A + B_n - α G_d)) Φ_new = Φ_prev
            r_nu = r_prev ** nu
            M_inv = I.unsqueeze(0) - ht * r_nu.view(N, 1, 1) * (
                A.unsqueeze(0) + Bn - alpha.view(N, 1, 1) * Gdn.unsqueeze(0)
            )
            
            Phi_next = torch.linalg.solve(M_inv, Phi_prev.unsqueeze(-1)).squeeze(-1)
            Phi_next = Phi_next / torch.norm(Phi_next, dim=1, keepdim=True)
            
            # Update r: r_new = r_prev / (1 - ht r^ν α)
            r_next = r_prev / (1 - ht * r_nu * alpha)
            
            r_all[:, it] = r_next
            Phi_all[:, it, :] = Phi_next
            
            r_prev = r_next
            Phi_prev = Phi_next
            
            if (it + 1) % max(1, nT // 10) == 0:
                print(f"  Step {it+1}/{nT} ({100*(it+1)/nT:.1f}%)")
        
        print(f"✓ Integration complete!")
        
        return t_vals, r_all.to(dtype_output), Phi_all.to(dtype_output)
    
    @torch.no_grad()
    def reconstruct_solution(
        self,
        r_all: torch.Tensor,
        Phi_all: torch.Tensor,
        z_vals: np.ndarray | torch.Tensor,
        batch_size: int = 512,
    ):
        """
        Reconstruct u(t,z) = r(t) Σ Φ_k(t) ψ_k(r(t) z)
        
        Args:
            r_all: [N, nT]
            Phi_all: [N, nT, n]
            z_vals: Spatial grid
        
        Returns:
            U_all: [N, nT, nz]
        """
        device = self.device
        dtype = r_all.dtype
        
        if isinstance(z_vals, np.ndarray):
            z_vals = torch.tensor(z_vals, device=device, dtype=dtype)
        
        N, nT = r_all.shape
        n = Phi_all.shape[2]
        nz = len(z_vals)
        
        U_all = torch.zeros(N, nT, nz, device=device, dtype=dtype)
        
        for zs in range(0, nz, batch_size):
            ze = min(zs + batch_size, nz)
            z_batch = z_vals[zs:ze]
            bz = ze - zs
            
            # For each (ic, time), compute Σ Φ_k ψ_k(r*z)
            for b in range(N):
                for it in range(nT):
                    r_t = r_all[b, it]
                    Phi_t = Phi_all[b, it, :]  # [n]
                    
                    # Compute ψ_k(r*z)
                    z_scaled = r_t * z_batch
                    psi = torch.zeros(n, bz, device=device, dtype=dtype)
                    for k in range(n):
                        psi[k] = hermite_function_torch(k, z_scaled)
                    
                    # u = r * Σ Φ_k ψ_k
                    u = r_t * torch.einsum('k,kz->z', Phi_t, psi)
                    U_all[b, it, zs:ze] = u
        
        return U_all


if __name__ == '__main__':
    # Try to import with proper package structure
    # plot_utils is in src/, burgers files are in src/burger_eq/
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
            from burgers_analytic import BurgersParallelSolver
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
    
    # -------- parameters --------
    hz = 0.1         # spatial step
    ht = 0.05        # time step
    Tmax = 2.0       # final time (reduced for faster comparison)
    z_range = (-7.0, 7.0)
    
    # -------- initial profile x0 --------
    def x0(q):
        """Initial condition x0(q) = exp(-q^2)"""
        return torch.exp(-q**2) if isinstance(q, torch.Tensor) else np.exp(-q**2)
    
    def G(q):
        """Primitive of x0: G'(q) = x0(q)
        G(q) = sqrt(pi)/2 * erf(q)
        """
        if isinstance(q, torch.Tensor):
            return (math.sqrt(math.pi) / 2.0) * torch.erf(q)
        else:
            from scipy.special import erf
            return (np.sqrt(np.pi) / 2.0) * erf(q)
    
    z_vals = np.arange(z_range[0], z_range[1] + hz, hz)
    
    # ======================================================================
    # 1. EXACT SOLUTION (BurgersParallelSolver)
    # ======================================================================
    print("\n" + "="*60)
    print("1. EXACT BURGERS SOLUTION")
    print("="*60)
    
    exact_solver = BurgersParallelSolver(device='cuda')
    
    z_vals_exact, t_vals_exact, U_exact = exact_solver.solve_parallel_projected(
        x0_list=[x0],
        G_list=[G],
        hz=hz,
        ht=ht,
        Tmax=Tmax,
        z_range=z_range,
        L=6.0,
        n_quad_points=200,
        P=None,  # Full solution, not projected
        enforce_exact_ic=True,
        interp='cubic',
    )
    
    X_exact = U_exact[0].cpu().numpy()  # [nT, nz]
    
    print(f"\nExact solution shapes:")
    print(f"  z_vals: {z_vals_exact.shape}")
    print(f"  t_vals: {t_vals_exact.shape}")
    print(f"  X_exact: {X_exact.shape}")
    
    # -------- plot exact solution --------
    plot_sim_result(z_vals_exact, t_vals_exact, X_exact, 'X_exact', notebook_plot=False)
    
    
    # ======================================================================
    # 2. STANDARD GALERKIN
    # ======================================================================
    print("\n" + "="*60)
    print("2. STANDARD GALERKIN")
    print("="*60)
    
    n_gal = 15       # dim sub vector space of projection
    
    gal_solver = BurgersGalerkinSolver(device='cuda')
    
    t_vals_gal, C_all = gal_solver.solve_parallel(
        x0_list=[x0],
        n=n_gal,
        ht=ht,
        Tmax=Tmax,
    )
    
    U_gal = gal_solver.reconstruct_solution(C_all, z_vals)
    X_gal = U_gal[0].cpu().numpy()  # [nT, nz]
    
    print(f"\nGalerkin solution shapes:")
    print(f"  X_gal: {X_gal.shape}")
    
    # -------- plot Galerkin solution --------
    plot_sim_result(z_vals_exact, t_vals_gal, X_gal, 'X_galerkin', notebook_plot=False)
    
    # -------- compute and plot error --------
    error_gal = np.abs(X_gal - X_exact)
    print(f"\nGalerkin Error Statistics:")
    print(f"  Max abs error: {np.max(error_gal):.6e}")
    print(f"  Mean abs error: {np.mean(error_gal):.6e}")
    print(f"  Final time error: {np.max(error_gal[-1, :]):.6e}")
    
    plot_sim_result(z_vals_exact, t_vals_gal, error_gal, 
                    f'Error_Galerkin_n{n_gal}', notebook_plot=False)
    
    
    # ======================================================================
    # 3. HOMOGENEOUS GALERKIN
    # ======================================================================
    print("\n" + "="*60)
    print("3. HOMOGENEOUS GALERKIN")
    print("="*60)
    
    n_hgal = 15      # dim sub vector space of projection
    
    hgal_solver = BurgersHomogeneousGalerkinSolver(device='cuda')
    
    t_vals_hgal, r_all, Phi_all = hgal_solver.solve_parallel(
        x0_list=[x0],
        n=n_hgal,
        ht=ht,
        Tmax=Tmax,
        nu=2.0,
    )
    
    U_hgal = hgal_solver.reconstruct_solution(r_all, Phi_all, z_vals)
    X_hgal = U_hgal[0].cpu().numpy()  # [nT, nz]
    
    print(f"\nHomogeneous Galerkin solution shapes:")
    print(f"  X_hgal: {X_hgal.shape}")
    print(f"  r range: [{r_all.min().item():.4f}, {r_all.max().item():.4f}]")
    
    # -------- plot H-Galerkin solution --------
    plot_sim_result(z_vals_exact, t_vals_hgal, X_hgal, 'X_homogeneous_galerkin', notebook_plot=False)
    
    # -------- compute and plot error --------
    error_hgal = np.abs(X_hgal - X_exact)
    print(f"\nHomogeneous Galerkin Error Statistics:")
    print(f"  Max abs error: {np.max(error_hgal):.6e}")
    print(f"  Mean abs error: {np.mean(error_hgal):.6e}")
    print(f"  Final time error: {np.max(error_hgal[-1, :]):.6e}")
    
    plot_sim_result(z_vals_exact, t_vals_hgal, error_hgal, 
                    f'Error_HGalerkin_n{n_hgal}', notebook_plot=False)
    
    
    # ======================================================================
    # 4. COMPARISON SUMMARY
    # ======================================================================
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"\nParameters: hz={hz}, ht={ht}, Tmax={Tmax}")
    print(f"Initial condition: x0(q) = exp(-q^2)")
    print(f"\n{'Method':<25} {'n':<5} {'Max Error':<15} {'Mean Error':<15}")
    print("-"*60)
    print(f"{'Standard Galerkin':<25} {n_gal:<5} {np.max(error_gal):<15.6e} {np.mean(error_gal):<15.6e}")
    print(f"{'Homogeneous Galerkin':<25} {n_hgal:<5} {np.max(error_hgal):<15.6e} {np.mean(error_hgal):<15.6e}")
    print("="*60)