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
    def integrate(self, c0, t, method='dopri5', rtol=1e-6, atol=1e-6):
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
        
        c_rom = odeint_fwd(self.forward, c0, t, method=method, rtol=rtol, atol=atol)
        
        if squeeze_output:
            c_rom = c_rom.squeeze(1)
        
        return c_rom


# =====================================================================
# Hybrid Model
# =====================================================================

class HybridROMNeuralODE(nn.Module):
    """
    Hybrid model: ROM + Neural ODE for residual learning.
    
    Training:
    - ROM provides baseline: c_rom = ROM(c0, t)
    - Neural ODE learns residual: c_residual = c_true - c_rom
    - Final prediction: c_hybrid = c_rom + c_residual
    """
    
    def __init__(self, neural_ode_func, rom_dynamics, learn_rom=False):
        """
        Args:
            neural_ode_func: CoeffODEFunc for learning residuals
            rom_dynamics: BurgersGalerkinROM with known dynamics
            learn_rom: If True, allow ROM parameters to be trained
        """
        super().__init__()
        
        self.neural_ode = neural_ode_func
        self.rom = rom_dynamics
        
        # Freeze ROM parameters by default
        if not learn_rom:
            for param in self.rom.parameters():
                param.requires_grad = False
    
    def forward(self, t, c):
        """Combined dynamics: d(c_rom + c_residual)/dt"""
        dc_rom = self.rom(t, c)
        dc_residual = self.neural_ode(t, c)
        return dc_rom + dc_residual
    
    @torch.no_grad()
    def get_rom_prediction(self, c0, t, method='dopri5', rtol=1e-6, atol=1e-6):
        """Get ROM-only prediction."""
        return self.rom.integrate(c0, t, method=method, rtol=rtol, atol=atol)
    
    def get_residual(self, c0, t, c_true, method='dopri5', rtol=1e-6, atol=1e-6):
        """
        Compute residual: c_true - ROM(c0, t)
        
        Returns:
            residual: [nT, B, K] - what Neural ODE should learn
            c_rom: [nT, B, K] - ROM prediction
        """
        c_rom = self.get_rom_prediction(c0, t, method=method, rtol=rtol, atol=atol)
        residual = c_true - c_rom
        return residual, c_rom
    
    @torch.no_grad()
    def predict(self, c0, t, method='dopri5', rtol=1e-6, atol=1e-6, return_components=False):
        """
        Hybrid prediction: ROM + Neural ODE correction
        
        Args:
            c0: Initial condition [B, K] or [K]
            t: Time points [nT]
            return_components: If True, return (c_rom, c_residual, c_hybrid)
        
        Returns:
            c_pred: Hybrid prediction
            OR (c_rom, c_residual, c_pred) if return_components=True
        """
        # ROM prediction
        c_rom = self.get_rom_prediction(c0, t, method=method, rtol=rtol, atol=atol)
        
        # Neural ODE starts from zero residual
        if c0.ndim == 1:
            residual_c0 = torch.zeros_like(c0)
        else:
            residual_c0 = torch.zeros_like(c0)
        
        # Integrate Neural ODE for residual
        c_residual = odeint_fwd(
            self.neural_ode.forward,
            residual_c0,
            t,
            method=method,
            rtol=rtol,
            atol=atol
        )
        
        # Combine
        c_pred = c_rom + c_residual
        
        if return_components:
            return c_rom, c_residual, c_pred
        
        return c_pred
