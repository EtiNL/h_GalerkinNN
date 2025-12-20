"""
Neural ODE Model and Utilities
================================
Core Neural ODE components and utility functions.
"""

import math
import numpy as np
import torch
import torch.nn as nn


# =====================================================================
# Neural ODE Model
# =====================================================================

class CoeffODEFunc(nn.Module):
    """
    Neural ODE function for learning dynamics in coefficient space.
    
    Args:
        K: Number of Galerkin modes
        hidden: Hidden layer size
        time_dependent: If True, concatenate time to input
    """
    
    def __init__(self, K: int, hidden: int = 256, time_dependent: bool = True, num_layers: int = 1):
        super().__init__()
        self.time_dependent = time_dependent
        inp = K + (1 if time_dependent else 0)
        
        layers = []
        layers.append(nn.Linear(inp, hidden))
        layers.append(nn.Tanh())
        for i in range(num_layers):
            layers.append(nn.Linear(hidden, hidden))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden, K))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, t, c):
        """
        Forward pass.
        
        Args:
            t: time scalar () or (B,)
            c: state (K,) or (B, K)
        
        Returns:
            dc/dt: same shape as c
        """
        squeeze_back = False
        if c.ndim == 1:
            c = c.unsqueeze(0)
            squeeze_back = True

        B = c.shape[0]

        if self.time_dependent:
            t = t.to(device=c.device)

            if t.ndim == 0:
                tt = t.to(dtype=c.dtype).expand(B, 1)
            elif t.ndim == 1:
                if t.shape[0] == 1:
                    tt = t.to(dtype=c.dtype).expand(B, 1)
                else:
                    assert t.shape[0] == B, f"Time batch size {t.shape[0]} != state batch size {B}"
                    tt = t.to(dtype=c.dtype).view(B, 1)
            else:
                raise ValueError(f"Unsupported time tensor shape {t.shape}")

            x = torch.cat([c, tt], dim=1)
        else:
            x = c

        out = self.net(x)
        if squeeze_back:
            out = out.squeeze(0)
        return out


# =====================================================================
# Utility Functions
# =====================================================================

def trapz_weights_1d(x: np.ndarray) -> np.ndarray:
    """Compute trapezoidal weights for 1D integration."""
    x = np.asarray(x, dtype=float)
    w = np.empty_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def _to_stored_time(ds, t_phys: torch.Tensor) -> torch.Tensor:
    """Convert physical time to stored time (with normalization if applicable)."""
    if ds.config.normalize_t:
        return (t_phys - ds.t_mean) / ds.t_std
    return t_phys


def _u_to_numpy_on_zgrid(U_tnx: np.ndarray, x_grid: np.ndarray, z_vals: np.ndarray) -> np.ndarray:
    """Interpolate spatial solution onto z_vals grid."""
    if np.allclose(x_grid, z_vals):
        return U_tnx
    out = np.empty((U_tnx.shape[0], z_vals.size), dtype=float)
    for i in range(U_tnx.shape[0]):
        out[i] = np.interp(z_vals, x_grid, U_tnx[i])
    return out


class AffineCoeffTransform:
    """Affine transformation for coefficient whitening (normalization)."""
    
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def encode(self, c: torch.Tensor) -> torch.Tensor:
        """Transform to normalized space."""
        return (c - self.mean) / self.std

    def decode(self, c_hat: torch.Tensor) -> torch.Tensor:
        """Transform back to physical space."""
        return c_hat * self.std + self.mean


# =====================================================================
# Hermite Basis Functions
# =====================================================================

@torch.no_grad()
def hermite_basis_x_torch(x: torch.Tensor, K: int, scale: float, shift: float) -> torch.Tensor:
    """
    Compute Hermite basis functions at points x.
    
    Args:
        x: Spatial points
        K: Number of basis functions
        scale: Hermite scale parameter
        shift: Hermite shift parameter
    
    Returns:
        Phi: [K, ...] basis functions evaluated at x
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


# =====================================================================
# Rollout and Projection
# =====================================================================

@torch.no_grad()
def rollout(func, t_stored, c0, method="dopri5", rtol=1e-6, atol=1e-6, options=None):
    """
    Integrate ODE function from initial condition.
    
    Args:
        func: ODE function (Neural ODE or ROM)
        t_stored: Time points
        c0: Initial condition
        method: ODE solver method
        rtol, atol: Tolerances
        options: Additional solver options
    
    Returns:
        c_pred: Predicted trajectory [nT, K]
    """
    from torchdiffeq import odeint as odeint_fwd
    
    if c0.ndim == 1:
        c0 = c0.unsqueeze(0)
    c_pred = odeint_fwd(func, c0, t_stored, method=method, rtol=rtol, atol=atol, options=options)
    return c_pred.squeeze(1)


@torch.no_grad()
def project_u0_to_c0_stored(ds, u0_callable) -> torch.Tensor:
    """
    Project initial condition u0(x) onto Galerkin coefficient space.
    
    Args:
        ds: Dataset with basis information
        u0_callable: Function u0(x) returning spatial initial condition
    
    Returns:
        c0: Initial coefficients [K]
    """
    if ds.Phi is None:
        raise ValueError("Dataset has no stored basis_matrix (ds.Phi is None).")
    
    if not hasattr(ds, 'hermite_scale') or not hasattr(ds, 'hermite_shift'):
        raise ValueError("Dataset missing hermite_scale/hermite_shift.")
    
    device = ds.c.device
    dtype = ds.c.dtype
    K = ds.K
    
    x_grid = ds.get_reconstruction_grid()
    u0_np = np.asarray(u0_callable(x_grid), dtype=float).reshape(-1)
    
    w_np = trapz_weights_1d(x_grid)
    
    u0 = torch.tensor(u0_np, device=device, dtype=dtype)
    w = torch.tensor(w_np, device=device, dtype=dtype)
    
    x_torch = torch.tensor(x_grid, device=device, dtype=dtype)
    Phi_z_original = hermite_basis_x_torch(x_torch, K, scale=ds.hermite_scale, shift=ds.hermite_shift)
    
    if hasattr(ds, 'orthonormalize') and ds.orthonormalize:
        if ds.transformation_matrix is None:
            raise ValueError("Dataset was orthonormalized but transformation_matrix is missing!")
        T = torch.as_tensor(ds.transformation_matrix, device=device, dtype=dtype)
        Phi_z = T @ Phi_z_original
    else:
        Phi_z = Phi_z_original
    
    P = (w.unsqueeze(0) * Phi_z).t().contiguous()
    c0_phys = P.t() @ u0
    
    if ds.config.normalize_c:
        mean = torch.as_tensor(ds.c_mean, device=device, dtype=dtype).squeeze(0)
        std = torch.as_tensor(ds.c_std, device=device, dtype=dtype).squeeze(0)
        return (c0_phys - mean) / std
    
    return c0_phys
