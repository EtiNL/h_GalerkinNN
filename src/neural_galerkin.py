"""
Fast Neural Galerkin NeuralODE training + prediction utilities (GPU-friendly).

Key changes (speed + fixes + does not remove the final NeuralODE objective):
- Removes DataLoader for CUDA-resident datasets (no pin_memory issues, far less overhead).
- One-time preprocessing of ALL trajectories into a single packed tensor:
    * sort by time, optional subsample, drop non-increasing times, finiteness checks.
- Fully batched odeint on shared time grid (typical for your Burgers dataset).
- Optional *derivative pretraining* (teacher forcing) to avoid “doesn’t learn / flat loss”
  then fine-tunes with the exact same NeuralODE loss as before.
- Optional whitening of coefficients for training stability when ds.config.normalize_c=False
  (invertible affine transform; prediction/reconstruction returns physical coeffs).

If your dataset uses a common time grid (grid sampling), this is much faster and usually
converges better than the DataLoader-based loop.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from tqdm import tqdm
from torchdiffeq import odeint as odeint_fwd

from plot_utils import plot_sim_result


# -----------------------------
# Model
# -----------------------------
class CoeffODEFunc(nn.Module):
    def __init__(self, K: int, hidden: int = 256, time_dependent: bool = True):
        super().__init__()
        self.time_dependent = time_dependent
        inp = K + (1 if time_dependent else 0)
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, K),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, t, c):
        """
        t: scalar () or (B,)
        c: (K,) or (B,K)

        - torchdiffeq case: t is scalar, c is (B,K) or (K,)
        - derivative pretrain: t can be (B,), c is (B,K)
        """
        # Ensure c is at least 2D: (B,K)
        squeeze_back = False
        if c.ndim == 1:
            c = c.unsqueeze(0)      # (1,K)
            squeeze_back = True     # remember to squeeze at the end

        B = c.shape[0]

        if self.time_dependent:
            t = t.to(device=c.device)  # keep dtype float64 if needed for solver

            if t.ndim == 0:
                # scalar time: broadcast to batch
                tt = t.to(dtype=c.dtype).expand(B, 1)  # (B,1)
            elif t.ndim == 1:
                # batched time: t.shape[0] should be 1 or B
                if t.shape[0] == 1:
                    tt = t.to(dtype=c.dtype).expand(B, 1)
                else:
                    assert t.shape[0] == B, \
                        f"Time batch size {t.shape[0]} != state batch size {B}"
                    tt = t.to(dtype=c.dtype).view(B, 1)
            else:
                raise ValueError(f"Unsupported time tensor shape {t.shape} (expected () or (B,)).")

            x = torch.cat([c, tt], dim=1)  # (B, K+1)
        else:
            x = c

        out = self.net(x)  # (B,K)
        if squeeze_back:
            out = out.squeeze(0)  # return (K,) if input was (K,)
        return out



# -----------------------------
# Utilities
# -----------------------------
def trapz_weights_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    w = np.empty_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def _to_stored_time(ds, t_phys: torch.Tensor) -> torch.Tensor:
    if ds.config.normalize_t:
        return (t_phys - ds.t_mean) / ds.t_std
    return t_phys


def _u_to_numpy_on_zgrid(U_tnx: np.ndarray, x_grid: np.ndarray, z_vals: np.ndarray) -> np.ndarray:
    if np.allclose(x_grid, z_vals):
        return U_tnx
    out = np.empty((U_tnx.shape[0], z_vals.size), dtype=float)
    for i in range(U_tnx.shape[0]):
        out[i] = np.interp(z_vals, x_grid, U_tnx[i])
    return out


class AffineCoeffTransform:
    """
    Optional coefficient whitening:
      c_hat = (c - mean) / std
      c = c_hat * std + mean

    Used to stabilize training when ds.config.normalize_c=False.
    This is invertible and prediction/reconstruction output is still in physical space.
    """
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        # mean/std expected shape (K,) on the target device/dtype
        self.mean = mean
        self.std = std

    def encode(self, c: torch.Tensor) -> torch.Tensor:
        return (c - self.mean) / self.std

    def decode(self, c_hat: torch.Tensor) -> torch.Tensor:
        return c_hat * self.std + self.mean


# -----------------------------
# One-time trajectory packing (fast + consistent)
# -----------------------------
@torch.no_grad()
def pack_dataset_trajectories(
    ds,
    time_subsample: int | None,
    require_shared_time: bool = True,
):
    """
    Returns:
      t_shared: (nT,) float64 (on ds device)
      C_all:    (M,nT,K) float32
    """
    device = ds.c.device
    M, _, K = ds.c.shape

    # Build one reference processed time grid from trajectory 0
    t0, c0 = ds.get_trajectory(0)
    t_ref, c_ref = _prep_one_traj(t0, c0, time_subsample=time_subsample)
    nT = t_ref.numel()

    C_all = torch.empty((M, nT, K), device=device, dtype=ds.c.dtype)
    C_all[0] = c_ref

    # Check + pack all
    for i in range(1, M):
        ti, ci = ds.get_trajectory(i)
        ti2, ci2 = _prep_one_traj(ti, ci, time_subsample=time_subsample)

        if require_shared_time and not torch.equal(ti2, t_ref):
            # If you truly have random per-IC times, disable require_shared_time and handle lists.
            raise RuntimeError(
                f"Trajectory {i} has a different time grid after preprocessing. "
                f"Set require_shared_time=False or regenerate dataset with grid sampling."
            )

        if not torch.isfinite(ci2).all():
            bad = torch.nonzero(~torch.isfinite(ci2), as_tuple=False)[0]
            raise RuntimeError(f"Trajectory {i}: non-finite c at {tuple(int(x) for x in bad)} value={ci2[tuple(bad.tolist())]}")

        C_all[i] = ci2

    return t_ref, C_all


@torch.no_grad()
def _prep_one_traj(t: torch.Tensor, c: torch.Tensor, time_subsample: int | None):
    """
    Same logic as your original _traj_mse preprocessing:
      - sort by time
      - optional subsample (unique indices)
      - drop non-increasing times
    Output:
      t: (nT,) float64
      c: (nT,K) (original dtype)
    """
    order = torch.argsort(t)
    t = t[order].to(torch.float64)
    c = c[order]

    if time_subsample is not None and t.numel() > time_subsample:
        nT = t.numel()
        lin = torch.linspace(0, nT - 1, time_subsample, device=t.device)
        ii = torch.unique_consecutive(lin.round().long())
        t = t[ii]
        c = c[ii]

    dt = t[1:] - t[:-1]
    keep = torch.cat([torch.ones(1, dtype=torch.bool, device=t.device), dt > 0])
    t = t[keep]
    c = c[keep]

    if t.numel() < 2:
        raise RuntimeError(f"Need >=2 increasing times, got {t.numel()}")
    if not torch.isfinite(t).all():
        raise RuntimeError("Non-finite times in t")
    if not torch.isfinite(c).all():
        bad = torch.nonzero(~torch.isfinite(c), as_tuple=False)[0]
        raise RuntimeError(f"Non-finite c at {tuple(int(x) for x in bad)} value={c[tuple(bad.tolist())]}")

    return t.contiguous(), c.contiguous()


# -----------------------------
# Fast losses
# -----------------------------
def _mse_ode_batch(
    func,
    t: torch.Tensor,      # (nT,) float64
    cB: torch.Tensor,     # (B,nT,K) float32
    method: str,
    rtol: float,
    atol: float,
    ode_options: dict | None,
):
    y0 = cB[:, 0, :]  # (B,K)
    pred_tBK = odeint_fwd(func, y0, t, method=method, rtol=rtol, atol=atol, options=ode_options)  # (nT,B,K)
    pred_BtK = pred_tBK.permute(1, 0, 2).contiguous()
    return torch.mean((pred_BtK - cB) ** 2)


@torch.no_grad()
def eval_mse_ode(
    func,
    t: torch.Tensor,
    C: torch.Tensor,
    ids: list[int],
    batch_ics: int,
    method: str,
    rtol: float,
    atol: float,
    ode_options: dict | None,
):
    func.eval()
    tot = 0.0
    n = 0
    for s in range(0, len(ids), batch_ics):
        b_ids = ids[s : s + batch_ics]
        cB = C[b_ids]
        loss = _mse_ode_batch(func, t, cB, method, rtol, atol, ode_options)
        tot += float(loss.item()) * len(b_ids)
        n += len(b_ids)
    return tot / max(1, n)


# -----------------------------
# Optional derivative pretraining (teacher forcing)
# -----------------------------
def _finite_difference_dc_dt(t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    t: (nT,) float64
    c: (nT,K) float32
    returns dc/dt: (nT,K) float32 (same device)
    """
    # Use central differences inside, forward/backward at ends
    t0 = t[:-1]
    t1 = t[1:]
    dt = (t1 - t0).to(c.dtype)  # (nT-1,)
    dc = c[1:] - c[:-1]         # (nT-1,K)
    slope = dc / dt[:, None]    # (nT-1,K)

    # Build (nT,K)
    K = c.shape[1]
    out = torch.empty((c.shape[0], K), device=c.device, dtype=c.dtype)
    out[0] = slope[0]
    out[-1] = slope[-1]
    if c.shape[0] > 2:
        out[1:-1] = 0.5 * (slope[:-1] + slope[1:])
    return out


def pretrain_rhs_derivative_matching(
    func,
    t: torch.Tensor,
    C: torch.Tensor,
    train_ids: list[int],
    epochs: int = 200,
    lr: float = 1e-3,
    batch_ics: int = 128,
    time_batch: int = 64,
):
    """
    Fast pretraining: minimize ||f(t_i, c_i) - dc/dt||^2 on random (IC, time) samples.
    This is a warm-start ONLY; you still fine-tune with the NeuralODE objective after.

    Does not constrain final accuracy; it just makes training actually move early.
    """
    device = C.device
    opt = torch.optim.Adam(func.parameters(), lr=lr)

    # Precompute dc/dt for all trajectories once
    dC = torch.empty_like(C)
    for i in range(C.shape[0]):
        dC[i] = _finite_difference_dc_dt(t, C[i])

    nT = t.numel()
    for ep in tqdm(range(1, epochs + 1), desc="pretrain RHS (derivative match)"):
        func.train()
        opt.zero_grad(set_to_none=True)

        # sample ICs
        idx_ic = torch.randint(0, len(train_ids), (min(batch_ics, len(train_ids)),), device=device)
        ic_ids = [train_ids[int(j)] for j in idx_ic.tolist()]

        # sample times (avoid always hitting t=0)
        tidx = torch.randint(0, nT, (time_batch,), device=device)
        t_s = t[tidx]  # (Tb,) float64

        # build batch (B*Tb, K)
        c_s = C[ic_ids][:, tidx, :].reshape(-1, C.shape[2])
        dc_s = dC[ic_ids][:, tidx, :].reshape(-1, C.shape[2])

        # evaluate f at matching scalar times (vectorize by repeating t)
        # torchdiffeq calls func with scalar t; here we call directly:
        tt = t_s.to(dtype=c_s.dtype).repeat(len(ic_ids), 1).reshape(-1)  # (B*Tb,)
        pred = func(tt, c_s)  # supports t vector since we cast/expand inside forward
        loss = torch.mean((pred - dc_s) ** 2)

        loss.backward()
        opt.step()

    return func


# -----------------------------
# Training
# -----------------------------
def _split_indices(M: int, val_frac: float, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(M, generator=g).tolist()
    n_val = max(1, int(math.floor(val_frac * M))) if M > 1 else 0
    val_ids = perm[:n_val]
    train_ids = perm[n_val:]
    return train_ids, val_ids


def train_neural_ode_on_neural_galerkin_dataset(
    ds,
    val_frac: float = 0.2,
    split_seed: int = 0,
    epochs: int = 2000,
    lr: float = 1e-3,
    hidden: int = 512,
    time_dependent: bool = True,
    method: str = "dopri5",
    rtol: float = 1e-6,
    atol: float = 1e-6,
    batch_ics: int = 64,                  # avoid near-full-batch; helps learning
    time_subsample: int | None = 150,     # train-time speedup
    grad_clip: float = 1.0,
    print_every: int = 5,
    ode_options: dict | None = None,
    # Stabilizers:
    whiten_if_needed: bool = True,        # affine whitening if ds.normalize_c=False
    # Warm-start:
    pretrain_derivative: bool = True,
    pretrain_epochs: int = 10,
    pretrain_lr: float = 1e-3,
    # lr-scheduler
    lr_schedule: str = "cosine",  # or "plateau" or "step"
    lr_min: float = 1e-5,
):
    device = ds.c.device
    M, _, K = ds.c.shape

    # Pack dataset once (fast)
    t_shared, C_all = pack_dataset_trajectories(ds, time_subsample=time_subsample, require_shared_time=True)

    # Optional whitening if dataset is not already normalized
    transform = None
    if whiten_if_needed and not ds.config.normalize_c:
        mean = torch.as_tensor(ds.c_mean, device=device, dtype=ds.c.dtype).squeeze(0)  # (K,)
        std = torch.as_tensor(ds.c_std, device=device, dtype=ds.c.dtype).squeeze(0)    # (K,)
        transform = AffineCoeffTransform(mean, std)
        C_train_space = transform.encode(C_all)
    else:
        C_train_space = C_all

    train_ids, val_ids = _split_indices(M, val_frac=val_frac, seed=split_seed)

    func = CoeffODEFunc(K, hidden=hidden, time_dependent=time_dependent).to(device)

    # Warm-start to avoid “flat loss / doesn’t learn”
    if pretrain_derivative and pretrain_epochs > 0:
        func = pretrain_rhs_derivative_matching(
            func,
            t=t_shared,
            C=C_train_space,
            train_ids=train_ids,
            epochs=pretrain_epochs,
            lr=pretrain_lr,
            batch_ics=min(256, max(16, batch_ics)),
            time_batch=min(128, t_shared.numel()),
        )

    opt = torch.optim.Adam(func.parameters(), lr=lr)
    
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=lr_min
        )
    elif lr_schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=50, verbose=True
        )
    elif lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=epochs//3, gamma=0.5
        )
    

    train_curve = []
    val_curve = []

    for ep in tqdm(range(1, epochs + 1), desc="train NeuralODE"):
        func.train()

        # Shuffle ICs each epoch
        perm = torch.randperm(len(train_ids), device=device)
        train_ids_shuf = [train_ids[int(i)] for i in perm.tolist()]

        tot = 0.0
        n = 0

        for s in range(0, len(train_ids_shuf), batch_ics):
            b_ids = train_ids_shuf[s : s + batch_ics]
            cB = C_train_space[b_ids]  # (B,nT,K)

            opt.zero_grad(set_to_none=True)
            loss = _mse_ode_batch(func, t_shared, cB, method, rtol, atol, ode_options)
            loss.backward()

            if grad_clip is not None:
                nn.utils.clip_grad_norm_(func.parameters(), grad_clip)
            
            if lr_schedule == "plateau":
                scheduler.step(train_mse)
            else:
                scheduler.step()

            tot += float(loss.detach().item()) * len(b_ids)
            n += len(b_ids)

        train_mse = tot / max(1, n)
        train_curve.append(train_mse)

        if (ep % print_every == 0 or ep == epochs) and len(val_ids) > 0:
            val_mse = eval_mse_ode(func, t_shared, C_train_space, val_ids, batch_ics, method, rtol, atol, ode_options)
            val_curve.append(val_mse)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_curve, mode="lines", name="train"))
    if len(val_curve) > 0:
        # val is sampled every print_every; plot against those epochs
        x_val = list(range(print_every, print_every * len(val_curve) + 1, print_every))
        if x_val[-1] != epochs:
            x_val[-1] = epochs
        fig.add_trace(go.Scatter(x=x_val, y=val_curve, mode="lines", name="val"))
    fig.update_layout(title="Neural ODE training curves", xaxis_title="epoch", yaxis_title="MSE")
    fig.show()

    info = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "train_curve": train_curve,
        "val_curve": val_curve,
        "t_shared": t_shared,
        "transform": transform,  # used by predict utilities below
        "time_subsample": time_subsample,
    }
    return func, info


# -----------------------------
# Rollout / projection helpers (transform-aware)
# -----------------------------
@torch.no_grad()
def rollout(
    func,
    t_stored: torch.Tensor,
    c0: torch.Tensor,
    method="dopri5",
    rtol=1e-6,
    atol=1e-6,
    options=None,
) -> torch.Tensor:
    if c0.ndim == 1:
        c0 = c0.unsqueeze(0)  # (1,K)
    c_pred = odeint_fwd(func, c0, t_stored, method=method, rtol=rtol, atol=atol, options=options)
    return c_pred.squeeze(1)  # (nT,K)

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
def project_u0_to_c0_stored(ds, u0_callable) -> torch.Tensor:
    """
    Project arbitrary initial condition onto the dataset's basis.
    
    This correctly handles:
    - Scaled and shifted Hermite basis
    - Orthonormalized basis (if used during dataset generation)
    - Proper quadrature weights
    
    Args:
        ds: NeuralGalerkinDataset with stored basis information
        u0_callable: Function that takes x (array or tensor) and returns u0(x)
    
    Returns:
        c0_stored: (K,) coefficients in the dataset's storage space
    """
    if ds.Phi is None:
        raise ValueError("Dataset has no stored basis_matrix (ds.Phi is None).")
    
    # Check if basis parameters are available
    if not hasattr(ds, 'hermite_scale') or not hasattr(ds, 'hermite_shift'):
        raise ValueError(
            "Dataset missing hermite_scale/hermite_shift. "
            "Re-generate dataset with updated burgers_neural_ds function."
        )
    
    device = ds.c.device
    dtype = ds.c.dtype
    K = ds.K
    
    # Get reconstruction grid
    x_grid = ds.get_reconstruction_grid()
    
    # Evaluate u0 on the grid
    u0_np = np.asarray(u0_callable(x_grid), dtype=float).reshape(-1)
    if u0_np.shape[0] != x_grid.size:
        raise ValueError(
            f"u0(x_grid) must return shape (nx,), got {u0_np.shape} for nx={x_grid.size}"
        )
    
    # Quadrature weights
    w_np = trapz_weights_1d(x_grid)
    
    # Convert to torch
    u0 = torch.tensor(u0_np, device=device, dtype=dtype)  # (nx,)
    w = torch.tensor(w_np, device=device, dtype=dtype)    # (nx,)
    
    # Reconstruct the basis at the grid points using stored parameters
    x_torch = torch.tensor(x_grid, device=device, dtype=dtype)
    Phi_z_original = hermite_basis_x_torch(
        x_torch, K, 
        scale=ds.hermite_scale, 
        shift=ds.hermite_shift
    )  # (K, nx)
    
    # Apply orthonormalization transformation if it was used
    if hasattr(ds, 'orthonormalize') and ds.orthonormalize:
        if ds.transformation_matrix is None:
            raise ValueError("Dataset was orthonormalized but transformation_matrix is missing!")
        T = torch.as_tensor(ds.transformation_matrix, device=device, dtype=dtype)
        Phi_z = T @ Phi_z_original  # (K, nx)
    else:
        Phi_z = Phi_z_original
    
    # Build projection matrix (same as in burgers_neural_ds)
    P = (w.unsqueeze(0) * Phi_z).t().contiguous()  # (nx, K)
    
    # Project: c = P.t() @ u0
    c0_phys = P.t() @ u0  # (K,)
    
    # Handle normalization (stored space)
    if ds.config.normalize_c:
        mean = torch.as_tensor(ds.c_mean, device=device, dtype=dtype).squeeze(0)
        std = torch.as_tensor(ds.c_std, device=device, dtype=dtype).squeeze(0)
        return (c0_phys - mean) / std
    
    return c0_phys


@torch.no_grad()
def predict_and_plot_vs_reference_surface(
    func,
    ds,
    u0_callable,
    t_vals: np.ndarray,
    z_vals: np.ndarray,
    X_ref: np.ndarray,
    method: str = "dopri5",
    rtol: float = 1e-6,
    atol: float = 1e-6,
    ode_options: dict | None = None,
    notebook_plot: bool = True,
    # if you trained with whitening (info["transform"]), pass it here:
    transform: AffineCoeffTransform | None = None,
):
    device = ds.c.device

    t_phys = torch.tensor(t_vals, device=device, dtype=ds.t.dtype)
    t_stored = _to_stored_time(ds, t_phys)
    t_stored, _ = torch.sort(t_stored)

    c0_stored = project_u0_to_c0_stored(ds, u0_callable)  # (K,)

    # If training used whitening in coefficient space, encode IC, rollout, decode back.
    if transform is not None:
        c0_train = transform.encode(c0_stored)
        c_pred_train = rollout(func, t_stored, c0_train, method=method, rtol=rtol, atol=atol, options=ode_options)
        c_pred_stored = transform.decode(c_pred_train)
    else:
        c_pred_stored = rollout(func, t_stored, c0_stored, method=method, rtol=rtol, atol=atol, options=ode_options)

    # reconstruct (dataset handles ds.config.normalize_c inside reconstruct_u)
    U_pred_tx = ds.reconstruct_u(c_pred_stored, denormalize=True)
    U_pred_tx_np = U_pred_tx.detach().cpu().numpy()

    x_grid = ds.get_reconstruction_grid()
    U_pred_tz = _u_to_numpy_on_zgrid(U_pred_tx_np, x_grid, np.asarray(z_vals, float))

    X_ref = np.asarray(X_ref, dtype=float)
    if X_ref.shape != U_pred_tz.shape:
        raise ValueError(f"Shape mismatch: X_ref {X_ref.shape} vs U_pred {U_pred_tz.shape}")

    plot_sim_result(z_vals, t_vals, U_pred_tz, "u_pred (Neural ODE)", notebook_plot=notebook_plot)
    plot_sim_result(z_vals, t_vals, X_ref, "u_ref", notebook_plot=notebook_plot)
    plot_sim_result(z_vals, t_vals, np.abs(X_ref - U_pred_tz), "abs error", notebook_plot=notebook_plot)

    return {
        "t_phys": t_vals,
        "z_vals": z_vals,
        "U_pred": U_pred_tz,
        "U_ref": X_ref,
        "abs_err": np.abs(X_ref - U_pred_tz),
        "c_pred_stored": c_pred_stored,
    }


@torch.no_grad()
def predict(
    func: nn.Module,
    dataset,
    ic_idx: int = 0,
    t_eval: torch.Tensor | None = None,
    method: str = "dopri5",
    rtol: float = 1e-6,
    atol: float = 1e-6,
    ode_options: dict | None = None,
    reconstruct: bool = True,
    compare_ground_truth: bool = True,
    plot: bool = True,
    transform: AffineCoeffTransform | None = None,
):
    func.eval()
    device = dataset.c.device

    if t_eval is None:
        t_true, c_true = dataset.get_trajectory(ic_idx)
        t_eval = t_true.clone()
    else:
        t_eval = t_eval.to(device)
        _, c_true = dataset.get_trajectory(ic_idx)

    t_eval, order = torch.sort(t_eval)
    c_true = c_true[order]

    c0 = c_true[0]  # (K,)

    if transform is not None:
        c0_train = transform.encode(c0)
        c_pred_train = rollout(func, t_eval, c0_train, method=method, rtol=rtol, atol=atol, options=ode_options)
        c_pred = transform.decode(c_pred_train)
    else:
        c_pred = rollout(func, t_eval, c0, method=method, rtol=rtol, atol=atol, options=ode_options)

    results = {"t": t_eval.detach().cpu(), "c_pred": c_pred.detach().cpu()}

    c_pred_phys = dataset.denormalize_c(c_pred)
    c_true_phys = dataset.denormalize_c(c_true)
    t_phys = dataset.denormalize_t(t_eval)

    results["c_pred_phys"] = c_pred_phys.detach().cpu()
    results["t_phys"] = t_phys.detach().cpu()

    if compare_ground_truth:
        results["c_true"] = c_true.detach().cpu()
        results["c_true_phys"] = c_true_phys.detach().cpu()
        mse_coeff = torch.mean((c_pred - c_true) ** 2).item()
        results["mse_coeff"] = mse_coeff
        print(f"MSE (coefficient space): {mse_coeff:.6e}")

    if reconstruct:
        try:
            x_grid = dataset.get_reconstruction_grid()
            results["x_grid"] = x_grid

            u_pred = dataset.reconstruct_u(c_pred, denormalize=True)
            results["u_pred"] = u_pred.detach().cpu()

            if compare_ground_truth:
                u_true = dataset.reconstruct_u(c_true, denormalize=True)
                results["u_true"] = u_true.detach().cpu()
                mse_spatial = torch.mean((u_pred - u_true) ** 2).item()
                results["mse_spatial"] = mse_spatial
                print(f"MSE (spatial domain): {mse_spatial:.6e}")
        except ValueError as e:
            print(f"Could not reconstruct spatial solution: {e}")
            reconstruct = False

    if plot:
        _plot_predictions(results, reconstruct, compare_ground_truth)

    return results


def _plot_predictions(results: dict, reconstruct: bool, compare_ground_truth: bool):
    from plotly.subplots import make_subplots

    t_phys = results["t_phys"].numpy()
    c_pred = results["c_pred_phys"].numpy()
    K = c_pred.shape[1]
    n_plot = min(5, K)

    fig1 = go.Figure()
    for k in range(n_plot):
        fig1.add_trace(go.Scatter(x=t_phys, y=c_pred[:, k], mode="lines", name=f"c_{k} (pred)"))
        if compare_ground_truth:
            c_true = results["c_true_phys"].numpy()
            fig1.add_trace(go.Scatter(x=t_phys, y=c_true[:, k], mode="lines", name=f"c_{k} (true)", opacity=0.7))
    fig1.update_layout(
        title=f"Predicted Galerkin Coefficients (first {n_plot} modes)",
        xaxis_title="Time",
        yaxis_title="Coefficient value",
        height=500,
    )
    fig1.show()

    if reconstruct and "u_pred" in results:
        u_pred = results["u_pred"].numpy()
        x_grid = results["x_grid"]
        nT = u_pred.shape[0]
        n_snapshots = min(6, nT)
        snapshot_indices = np.linspace(0, nT - 1, n_snapshots, dtype=int)

        fig2 = go.Figure()
        for i in snapshot_indices:
            fig2.add_trace(go.Scatter(x=x_grid, y=u_pred[i], mode="lines", name=f"t={t_phys[i]:.2f} (pred)"))
            if compare_ground_truth and "u_true" in results:
                u_true = results["u_true"].numpy()
                fig2.add_trace(
                    go.Scatter(x=x_grid, y=u_true[i], mode="lines", name=f"t={t_phys[i]:.2f} (true)", opacity=0.5)
                )
        fig2.update_layout(title="Spatial Solution Snapshots", xaxis_title="x", yaxis_title="u(t, x)", height=500)
        fig2.show()

        if compare_ground_truth and "u_true" in results:
            u_true = results["u_true"].numpy()
            error = np.abs(u_pred - u_true)

            fig3 = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=("Prediction", "Ground Truth", "Absolute Error"),
                horizontal_spacing=0.1,
            )
            fig3.add_trace(go.Heatmap(z=u_pred.T, x=t_phys, y=x_grid, colorscale="RdBu"), row=1, col=1)
            fig3.add_trace(go.Heatmap(z=u_true.T, x=t_phys, y=x_grid, colorscale="RdBu"), row=1, col=2)
            fig3.add_trace(go.Heatmap(z=error.T, x=t_phys, y=x_grid, colorscale="Reds"), row=1, col=3)

            fig3.update_layout(title="Spatiotemporal Solution", height=400, showlegend=False)
            fig3.show()
        else:
            fig3 = go.Figure(data=go.Heatmap(z=u_pred.T, x=t_phys, y=x_grid, colorscale="RdBu"))
            fig3.update_layout(title="Spatiotemporal Solution (Prediction)", xaxis_title="Time", yaxis_title="x", height=400)
            fig3.show()


# -----------------------------
# Example main
# -----------------------------
if __name__ == "__main__":
    from pde_dataset.neural_galerkin_dataset import NeuralGalerkinDataset

    ds = NeuralGalerkinDataset.load(
        filepath="burger_eq/neural_galerkin_ds.npz",
        device="cuda",
        dtype=torch.float32,
    )

    func, info = train_neural_ode_on_neural_galerkin_dataset(
        ds=ds,
        epochs=2000,
        lr=1e-3,
        hidden=256,
        time_dependent=True,
        method="dopri5",
        batch_ics=64,
        time_subsample=150,
        print_every=100,
        ode_options=None,
        whiten_if_needed=True,
        pretrain_derivative=True,
        pretrain_epochs=200,
        pretrain_lr=1e-3,
    )

    results = predict(
        func=func,
        dataset=ds,
        ic_idx=0,
        method="dopri5",
        reconstruct=True,
        compare_ground_truth=True,
        plot=True,
        transform=info["transform"],
    )

    print("\nPrediction completed!")
    print(f"Time points: {results['t'].shape}")
    print(f"Predicted coefficients: {results['c_pred'].shape}")
    if "u_pred" in results:
        print(f"Reconstructed spatial solution: {results['u_pred'].shape}")
