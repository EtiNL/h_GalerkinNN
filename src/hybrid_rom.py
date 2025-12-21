"""
Hybrid ROM + Neural ODE Components
===================================
ROM dynamics and hybrid model for residual learning.
"""

import torch
import torch.nn as nn
from scipy.special import roots_hermite
from torchdiffeq import odeint as odeint_fwd
import math
from tqdm import tqdm


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


# =====================================================================
# Burgers Galerkin ROM
# =====================================================================

class BurgersGalerkinROM(nn.Module):
    """
    Burgers equation ROM using Galerkin projection.
    
    Dynamics: dc/dt = A @ c + Bn(c) @ c
    where Bn(c)[i,j] = 0.5 * sum_k c[k] * Bfull[i,j,k]
    """
    
    def __init__(self, K: int, device='cuda', dtype=torch.float32, n_quad: int = 100):
        """
        Args:
            K: Number of Galerkin modes
            device: 'cuda' or 'cpu'
            dtype: Data type for A (Bfull uses float64 internally)
            n_quad: Number of quadrature points
        """
        super().__init__()
        self.K = K
        
        print(f"\n{'='*60}")
        print(f"Initializing Burgers Galerkin ROM")
        print(f"{'='*60}")
        print(f"Modes: {K}")
        print(f"Device: {device}")
        
        # Compute ROM matrices
        A = compute_A(K, device=device, dtype=dtype)
        Bfull = compute_Bfull(K, n_quad=n_quad, device=device, dtype=torch.float64)
        
        # Register as buffers
        self.register_buffer('A', A)
        self.register_buffer('Bfull', Bfull.to(dtype))
        
        print(f"\n✓ ROM initialized")
        print(f"  A: {self.A.shape}, dtype: {self.A.dtype}")
        print(f"  Bfull: {self.Bfull.shape}, dtype: {self.Bfull.dtype}")
    
    def forward(self, t, c):
        """
        ROM dynamics: dc/dt = A @ c + Bn(c) @ c
        
        Args:
            t: Time (not used, required by odeint)
            c: State vector [B, K] or [K]
        
        Returns:
            dc_dt: Time derivative [B, K] or [K]
        """
        squeeze_back = False
        if c.ndim == 1:
            c = c.unsqueeze(0)
            squeeze_back = True
        
        # Linear term
        linear_term = c @ self.A.T
        
        # Nonlinear term
        Bn = compute_Bn(self.Bfull, c)
        
        if Bn.ndim == 2:
            nonlinear_term = (Bn @ c.unsqueeze(-1)).squeeze(-1)
        else:
            nonlinear_term = torch.bmm(Bn, c.unsqueeze(-1)).squeeze(-1)
        
        dc_dt = linear_term + nonlinear_term
        
        if squeeze_back:
            dc_dt = dc_dt.squeeze(0)
        
        return dc_dt
    
    @torch.no_grad()
    def integrate(self, c0, t, method='dopri5', rtol=1e-6, atol=1e-6, options=None):
        """
        Integrate ROM from initial condition.
        
        Args:
            c0: Initial condition [B, K] or [K]
            t: Time points [nT]
        
        Returns:
            c_rom: ROM trajectory [nT, B, K] or [nT, K]
        """
        squeeze_output = False
        if c0.ndim == 1:
            c0 = c0.unsqueeze(0)
            squeeze_output = True
        
        c_rom = odeint_fwd(self.forward, c0, t, method=method, rtol=rtol, atol=atol, options=options)
        
        if squeeze_output:
            c_rom = c_rom.squeeze(1)
        
        return c_rom


# =====================================================================
# Hybrid Model
# =====================================================================

import torch
import torch.nn as nn
from torchdiffeq import odeint as odeint_fwd

class HybridROMNeuralODE(nn.Module):
    """
    Option B (correct residual-state hybrid):

    State is y = [c_rom, r] with shape (..., 2K)

      dc_rom/dt = ROM(t, c_rom)
      dr/dt     = NN(t, c_rom)              (variant B1, simplest)
      c_hybrid  = c_rom + r
    """

    def __init__(self, neural_ode_func, rom_dynamics, learn_rom: bool = False):
        super().__init__()
        self.neural_ode = neural_ode_func
        self.rom = rom_dynamics

        if not learn_rom:
            for p in self.rom.parameters():
                p.requires_grad = False

    def forward(self, t, y):
        """
        y: [B, 2K] or [2K]
        returns dy/dt with same shape
        """
        squeeze_back = False
        if y.ndim == 1:
            y = y.unsqueeze(0)
            squeeze_back = True

        K = y.shape[-1] // 2
        assert 2 * K == y.shape[-1], "y must have last dim 2K"

        c_rom = y[:, :K]
        r     = y[:, K:]

        dc_rom = self.rom(t, c_rom)

        # Residual dynamics conditioned on the full hybrid state
        c_full = c_rom + r
        dr = self.neural_ode(t, c_full)

        dy = torch.cat([dc_rom, dr], dim=-1)
        return dy.squeeze(0) if squeeze_back else dy

    @torch.no_grad()
    def predict(self, c0, t, method="dopri5", rtol=1e-6, atol=1e-6, options=None, return_components=False):
        """
        Returns:
          c_pred: [nT, B, K] (or [nT, K] if c0 was [K])

        If return_components=True:
          (c_rom, r, c_pred)
        """
        squeeze_B = False
        if c0.ndim == 1:
            c0B = c0.unsqueeze(0)
            squeeze_B = True
        else:
            c0B = c0

        r0 = torch.zeros_like(c0B)
        y0 = torch.cat([c0B, r0], dim=-1)          # [B, 2K]

        Y = odeint_fwd(self.forward, y0, t, method=method, rtol=rtol, atol=atol, options=options)
        # Y: [nT, B, 2K]

        K = c0B.shape[-1]
        c_rom = Y[..., :K]
        r     = Y[..., K:]
        c_pred = c_rom + r

        if squeeze_B:
            c_rom = c_rom[:, 0, :]
            r     = r[:, 0, :]
            c_pred = c_pred[:, 0, :]

        if return_components:
            return c_rom, r, c_pred
        return c_pred

    @torch.no_grad()
    def get_rom_prediction(self, c0, t, method='dopri5', rtol=1e-6, atol=1e-6, options=None):
        """
        Pure ROM rollout (for diagnostics/baseline).
        """
        return self.rom.integrate(c0, t, method=method, rtol=rtol, atol=atol, options=options)

