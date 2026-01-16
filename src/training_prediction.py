"""
Training and Prediction Functions
"""

import math
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from torchdiffeq import odeint as odeint_fwd, odeint_adjoint

from neural_ode import (
    AffineCoeffTransform,
    rollout,
    project_u0_to_c0_stored,
    _to_stored_time,
    _u_to_numpy_on_zgrid,
)


# =====================================================================
# Dataset Preparation
# =====================================================================

@torch.no_grad()
def pack_dataset_trajectories(ds, time_subsample: int | None, require_shared_time: bool = True):
    """Pack dataset trajectories into a single tensor."""
    device = ds.c.device
    M, _, K = ds.c.shape

    t0, c0 = ds.get_trajectory(0)
    t_ref, c_ref = _prep_one_traj(t0, c0, time_subsample=time_subsample)
    nT = t_ref.numel()

    C_all = torch.empty((M, nT, K), device=device, dtype=ds.c.dtype)
    C_all[0] = c_ref

    for i in range(1, M):
        ti, ci = ds.get_trajectory(i)
        ti2, ci2 = _prep_one_traj(ti, ci, time_subsample=time_subsample)

        if require_shared_time and not torch.equal(ti2, t_ref):
            raise RuntimeError(f"Trajectory {i} has different time grid.")

        if not torch.isfinite(ci2).all():
            bad = torch.nonzero(~torch.isfinite(ci2), as_tuple=False)[0]
            raise RuntimeError(f"Trajectory {i}: non-finite c at {tuple(int(x) for x in bad)}")

        C_all[i] = ci2

    return t_ref, C_all


@torch.no_grad()
def _prep_one_traj(t: torch.Tensor, c: torch.Tensor, time_subsample: int | None):
    """Prepare single trajectory (sort, subsample, validate)."""
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
        raise RuntimeError(f"Non-finite c at {tuple(int(x) for x in bad)}")

    return t.contiguous(), c.contiguous()


def _split_indices(M: int, val_frac: float, seed: int):
    """Split indices into train/val."""
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(M, generator=g).tolist()
    n_val = max(1, int(math.floor(val_frac * M))) if M > 1 else 0
    val_ids = perm[:n_val]
    train_ids = perm[n_val:]
    return train_ids, val_ids

# =====================================================================
# LR scheduler
# =====================================================================

def _make_lr_scheduler(optimizer, lr_scheduler_cfg, epochs):
    if lr_scheduler_cfg is None:
        return None, None

    sched_type = lr_scheduler_cfg.get("type", None)

    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=lr_scheduler_cfg.get("min_lr", 0.0),
        )
        step_on = "epoch"

    elif sched_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_scheduler_cfg.get("factor", 0.5),
            patience=lr_scheduler_cfg.get("patience", 2),
            threshold=lr_scheduler_cfg.get("threshold", 1e-6),
            threshold_mode=lr_scheduler_cfg.get("threshold_mode", "rel"),
            min_lr=lr_scheduler_cfg.get("min_lr", 0.0),
        )
        step_on = "val"

    else:
        raise ValueError(f"Unknown lr scheduler type: {sched_type}")

    return scheduler, step_on


# =====================================================================
# Loss Functions
# =====================================================================

def _mse_ode_batch(func, t, cB, method, rtol, atol, ode_options, use_adjoint=True):
    """Compute MSE loss for a batch.

    Args:
        use_adjoint: If True, use adjoint method for O(1) memory backprop.
                     This is essential for training to avoid OOM errors.
    """
    y0 = cB[:, 0, :]
    if use_adjoint:
        # Adjoint method: O(1) memory w.r.t. number of solver steps
        pred_tBK = odeint_adjoint(func, y0, t, method=method, rtol=rtol, atol=atol, options=ode_options)
    else:
        # Standard method: stores all intermediate states (memory hungry)
        pred_tBK = odeint_fwd(func, y0, t, method=method, rtol=rtol, atol=atol, options=ode_options)
    pred_BtK = pred_tBK.permute(1, 0, 2).contiguous()
    return torch.mean((pred_BtK - cB) ** 2)


@torch.no_grad()
def eval_mse_ode(func, t, C, ids, batch_ics, method, rtol, atol, ode_options):
    """Evaluate MSE on a set of trajectories."""
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


# =====================================================================
# Derivative Pretraining
# =====================================================================

def _finite_difference_dc_dt(t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Compute dc/dt using finite differences."""
    t0 = t[:-1]
    t1 = t[1:]
    dt = (t1 - t0).to(c.dtype)
    dc = c[1:] - c[:-1]
    slope = dc / dt[:, None]

    K = c.shape[1]
    out = torch.empty((c.shape[0], K), device=c.device, dtype=c.dtype)
    out[0] = slope[0]
    out[-1] = slope[-1]
    if c.shape[0] > 2:
        out[1:-1] = 0.5 * (slope[:-1] + slope[1:])
    return out


def pretrain_rhs_derivative_matching(
    func, t, C, train_ids,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_ics: int = 128,
    time_batch: int = 64,
):
    """Pretrain by matching derivatives."""
    device = C.device
    opt = torch.optim.Adam(func.parameters(), lr=lr)

    dC = torch.empty_like(C)
    for i in range(C.shape[0]):
        dC[i] = _finite_difference_dc_dt(t, C[i])

    nT = t.numel()
    for ep in tqdm(range(1, epochs + 1), desc="Pretrain derivative matching"):
        func.train()
        opt.zero_grad(set_to_none=True)

        idx_ic = torch.randint(0, len(train_ids), (min(batch_ics, len(train_ids)),), device=device)
        ic_ids = [train_ids[int(j)] for j in idx_ic.tolist()]

        tidx = torch.randint(0, nT, (time_batch,), device=device)
        t_s = t[tidx]

        c_s = C[ic_ids][:, tidx, :].reshape(-1, C.shape[2])
        dc_s = dC[ic_ids][:, tidx, :].reshape(-1, C.shape[2])

        tt = t_s.to(dtype=c_s.dtype).repeat(len(ic_ids), 1).reshape(-1)
        pred = func(tt, c_s)
        loss = torch.mean((pred - dc_s) ** 2)

        loss.backward()
        opt.step()

    return func


# =====================================================================
# Training: Pure Neural ODE
# =====================================================================

def train_neural_ode_on_neural_galerkin_dataset(
    neural_ode_func,
    ds,
    val_frac: float = 0.25,
    split_seed: int = 0,
    epochs: int = 2000,
    weight_decay: float = 1e-5,
    hidden: int = 256,
    num_layers: int = 1,
    time_dependent: bool = True,
    method: str = "dopri5",
    rtol: float = 1e-6,
    atol: float = 1e-6,
    batch_ics: int = 64,
    time_subsample: int | None = 150,
    grad_clip: float = 1.0,
    print_every: int = 10,
    ode_options: dict | None = None,
    whiten_if_needed: bool = True,
    pretrain_derivative: bool = True,
    pretrain_epochs: int = 200,
    pretrain_lr: float = 1e-3,
    lr_scheduler: dict | None = None,
    early_stopping_patience: int = 20,
    early_stopping_min_delta: float = 1e-7,
    use_adjoint=True,
):
    """
    Train pure Neural ODE on dataset.

    Returns:
        func: Trained Neural ODE
        info: Training information dictionary
    """
    from neural_ode import CoeffODEFunc

    device = ds.c.device
    M, _, K = ds.c.shape

    # Pack dataset
    t_shared, C_all = pack_dataset_trajectories(ds, time_subsample=time_subsample, require_shared_time=True)

    # Optional whitening
    transform = None
    if whiten_if_needed and not ds.config.normalize_c:
        mean = torch.as_tensor(ds.c_mean, device=device, dtype=ds.c.dtype).squeeze(0)
        std = torch.as_tensor(ds.c_std, device=device, dtype=ds.c.dtype).squeeze(0)
        transform = AffineCoeffTransform(mean, std)
        C_train_space = transform.encode(C_all)
    else:
        C_train_space = C_all

    train_ids, val_ids = _split_indices(M, val_frac=val_frac, seed=split_seed)

    print(f"\n{'='*60}")
    print("NEURAL ODE TRAINING")
    print(f"{'='*60}")
    print(f"Parameters: {sum(p.numel() for p in neural_ode_func.parameters()):,}")
    print(f"Train ICs: {len(train_ids)}, Val ICs: {len(val_ids)}")

    # Derivative pretraining
    if pretrain_derivative and pretrain_epochs > 0:
        neural_ode_func = pretrain_rhs_derivative_matching(
            neural_ode_func,
            t=t_shared,
            C=C_train_space,
            train_ids=train_ids,
            epochs=pretrain_epochs,
            lr=pretrain_lr,
            batch_ics=min(256, max(16, batch_ics)),
            time_batch=min(128, t_shared.numel()),
        )

    # Optimizer
    opt = torch.optim.Adam(neural_ode_func.parameters(), lr=1e-3, weight_decay=weight_decay)

    # LR scheduler
    scheduler, step_on = _make_lr_scheduler(opt, lr_scheduler, epochs)

    # Training loop
    train_curve = []
    val_curve = []
    val_epochs = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for ep in tqdm(range(1, epochs + 1), desc="Training Neural ODE"):
        neural_ode_func.train()
        perm = torch.randperm(len(train_ids), device=device)
        train_ids_shuf = [train_ids[int(i)] for i in perm.tolist()]

        tot = 0.0
        n = 0
        for s in range(0, len(train_ids_shuf), batch_ics):
            b_ids = train_ids_shuf[s : s + batch_ics]
            cB = C_train_space[b_ids]

            opt.zero_grad(set_to_none=True)
            loss = _mse_ode_batch(neural_ode_func, t_shared, cB, method, rtol, atol, ode_options, use_adjoint=use_adjoint)
            loss.backward()

            if grad_clip is not None:
                nn.utils.clip_grad_norm_(neural_ode_func.parameters(), grad_clip)

            opt.step()

            tot += float(loss.detach().item()) * len(b_ids)
            n += len(b_ids)

        train_mse = tot / max(1, n)
        train_curve.append(train_mse)

        if scheduler is not None and step_on == "epoch":
            scheduler.step()

        # Validation
        if len(val_ids) > 0 and (ep % print_every == 0 or ep == epochs):
            val_batch = min(64, batch_ics)
            val_mse = eval_mse_ode(neural_ode_func, t_shared, C_train_space, val_ids, val_batch, method, rtol, atol, ode_options)
            val_curve.append(val_mse)
            val_epochs.append(ep)

            if scheduler is not None and step_on == "val":
                scheduler.step(val_mse)

            current_lr = opt.param_groups[0]['lr']
            print(f"Epoch {ep:4d} | Train: {train_mse:.6e}, Val: {val_mse:.6e} | "
                  f"LR: {current_lr:.6e}, Patience: {patience_counter}/{early_stopping_patience}")

            if val_mse < best_val_loss - early_stopping_min_delta:
                best_val_loss = val_mse
                best_epoch = ep
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in neural_ode_func.state_dict().items()}
                print(f"  ✓ New best model!")
            else:
                patience_counter += print_every

            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {ep}")
                break

    # Restore best model
    if best_state is not None:
        neural_ode_func.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"\n✓ Restored best model from epoch {best_epoch}")

    # Plot
    _plot_training_curves(train_curve, val_curve, val_epochs, best_epoch, title="Neural ODE Training")

    info = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "train_curve": train_curve,
        "val_curve": val_curve,
        "val_epochs": val_epochs,
        "t_shared": t_shared,
        "transform": transform,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_train_loss": train_curve[best_epoch-1] if best_epoch > 0 else train_curve[-1],
    }

    return neural_ode_func, info

# =====================================================================
# Training: Hybrid ROM + Neural ODE
# =====================================================================

def _diagnose_rom_consistency(
    hybrid_model,
    t_shared,
    C_all,
    C_rom_all,
    rom_consistency_tol,
    method="dopri5",
    rtol=1e-6,
    atol=1e-6,
    ode_options=None,
    n_check: int = 8,
):
    """
    Compare:
      (A) precomputed ROM rollout: C_rom_all[i]
      (B) c_rom extracted from augmented integration with hybrid_model.forward
          starting at y0=[c0, 0]

    Returns a dict of error stats.
    """
    device = C_all.device
    M, nT, K = C_all.shape

    n_check = min(n_check, M)
    ids = torch.randperm(M, device=device)[:n_check]

    c0 = C_all[ids, 0, :]                                   # [B, K]
    y0 = torch.cat([c0, torch.zeros_like(c0)], dim=-1)       # [B, 2K]

    with torch.no_grad():
        Y = odeint_fwd(
            hybrid_model.forward, y0, t_shared,
            method=method, rtol=rtol, atol=atol, options=ode_options
        ).permute(1, 0, 2)                                   # [B, nT, 2K]

        c_rom_from_aug = Y[..., :K]                          # [B, nT, K]
        c_rom_precomp = C_rom_all[ids]                       # [B, nT, K]

        diff = c_rom_from_aug - c_rom_precomp
        abs_diff = diff.abs()

        stats = {
            "ids": ids.detach().cpu(),
            "max_abs": float(abs_diff.max().item()),
            "mean_abs": float(abs_diff.mean().item()),
            "rmse": float(torch.sqrt(torch.mean(diff ** 2)).item()),
            "per_ic_max_abs": abs_diff.amax(dim=(1, 2)).detach().cpu(),  # [B]
        }

    # Heuristic warning threshold (tune if needed)
    if stats["max_abs"] > rom_consistency_tol:
        print("\nROM consistency diagnostic (augmented ROM vs precomputed ROM)")
        print(f"  checked ICs: {n_check}")
        print(f"  max|diff|   : {stats['max_abs']:.3e}")
        print(f"  mean|diff|  : {stats['mean_abs']:.3e}")
        print(f"  RMSE        : {stats['rmse']:.3e}")
        print(f"  per-IC max|diff|: {stats['per_ic_max_abs'].numpy()}")
        print("  ⚠ WARNING: ROM rollouts differ noticeably. "
              "Residual targets may be inconsistent with model ROM.")

    return stats



def _pretrain_residual_derivative_matching(
    func,                      # predicts dr/dt
    t: torch.Tensor,           # [nT]
    C_rom: torch.Tensor,       # [M, nT, K] inputs to func
    R: torch.Tensor,           # [M, nT, K] residual trajectories
    train_ids,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_ics: int = 128,
    time_batch: int = 64,
):
    device = C_rom.device
    opt = torch.optim.Adam(func.parameters(), lr=lr)

    # finite-diff dR/dt
    dR = torch.empty_like(R)
    for i in range(R.shape[0]):
        dR[i] = _finite_difference_dc_dt(t, R[i])

    nT = t.numel()
    for _ in tqdm(range(1, epochs + 1), desc="Pretrain residual derivative matching"):
        func.train()
        opt.zero_grad(set_to_none=True)

        idx_ic = torch.randint(0, len(train_ids), (min(batch_ics, len(train_ids)),), device=device)
        ic_ids = [train_ids[int(j)] for j in idx_ic.tolist()]

        tidx = torch.randint(0, nT, (time_batch,), device=device)
        t_s = t[tidx]

        c_rom_s = C_rom[ic_ids][:, tidx, :].reshape(-1, C_rom.shape[2])   # [B*time_batch, K]
        r_s     = R[ic_ids][:, tidx, :].reshape(-1, R.shape[2])           # [B*time_batch, K]
        dR_s    = dR[ic_ids][:, tidx, :].reshape(-1, dR.shape[2])         # [B*time_batch, K]

        c_full_s = c_rom_s + r_s

        tt = t_s.to(dtype=c_rom_s.dtype).repeat(len(ic_ids), 1).reshape(-1)
        pred = func(tt, c_full_s)  # dr/dt conditioned on (c_rom + r)
        loss = torch.mean((pred - dR_s) ** 2)

        loss.backward()
        opt.step()

    return func


def train_hybrid_rom_neural_ode(
    neural_ode_func,
    rom_dynamics,
    t_shared,
    C_all,
    train_ids,
    val_ids,
    epochs: int = 1000,
    batch_ics: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
    method: str = "dopri5",
    rtol: float = 1e-6,
    atol: float = 1e-6,
    ode_options: dict = None,
    print_every: int = 10,
    early_stopping_patience: int = 20,
    early_stopping_min_delta: float = 1e-7,
    lr_scheduler: dict | None = None,
    train_on_residuals: bool = True,
    pretrain_derivative: bool = True,
    pretrain_epochs: int = 10,
    pretrain_lr: float = 1e-3,
    assert_rom_consistency: bool = True,
    rom_consistency_tol: float = 1e-4,
    rom_consistency_ncheck: int = 8,
):
    """
    Matches your HybridROMNeuralODE implementation:

      y = [c_rom, r]
      dc_rom/dt = ROM(t, c_rom)
      dr/dt     = NN(t, c_rom)
      c_pred    = c_rom + r

    Dataset layout:
      C_all: [M, nT, K]
      t_shared: [nT]
    """
    from hybrid_rom import HybridROMNeuralODE

    device = C_all.device
    M, nT, K = C_all.shape

    hybrid_model = HybridROMNeuralODE(
        neural_ode_func=neural_ode_func,
        rom_dynamics=rom_dynamics,
        learn_rom=False
    ).to(device)

    print(f"\n{'='*70}")
    print("HYBRID ROM + NEURAL ODE TRAINING")
    print(f"{'='*70}")
    print(f"Neural ODE parameters: {sum(p.numel() for p in hybrid_model.neural_ode.parameters()):,}")
    print(f"Train ICs: {len(train_ids)}, Val ICs: {len(val_ids)}")

    # ROM trajectories for residual targets / pretraining
    need_rom = train_on_residuals or (pretrain_derivative and pretrain_epochs > 0)

    C_rom_all = None
    if need_rom:
        print("\nComputing ROM predictions...")
        with torch.no_grad():
            C_rom_all = torch.empty_like(C_all)

            chunk = 256  # tune
            for s in tqdm(range(0, M, chunk), desc="ROM baseline (batched)"):
                e = min(s + chunk, M)
                c0 = C_all[s:e, 0, :]                      # [B, K]
                C_rom_tBK = hybrid_model.get_rom_prediction(
                    c0, t_shared,
                    method=method, rtol=rtol, atol=atol, options=ode_options
                )                                           # [nT, B, K]
                C_rom_all[s:e] = C_rom_tBK.permute(1, 0, 2)


        rom_mse_train = torch.mean((C_rom_all[train_ids] - C_all[train_ids]) ** 2).item()
        rom_mse_val = torch.mean((C_rom_all[val_ids] - C_all[val_ids]) ** 2).item() if len(val_ids) > 0 else None

        print(f"ROM Baseline - Train MSE: {rom_mse_train:.6e}", end="")
        if rom_mse_val is not None:
            print(f", Val MSE: {rom_mse_val:.6e}")
        else:
            print()

        # ROM vs augmented-trajectory consistency assertion (only needed for residual training)
        if train_on_residuals and assert_rom_consistency:
            stats = _diagnose_rom_consistency(
                hybrid_model=hybrid_model,
                t_shared=t_shared,
                C_all=C_all,
                C_rom_all=C_rom_all,
                rom_consistency_tol = rom_consistency_tol,
                method=method,
                rtol=rtol,
                atol=atol,
                ode_options=ode_options,
                n_check=rom_consistency_ncheck,
            )
            if stats["max_abs"] > rom_consistency_tol:
                raise RuntimeError(
                    f"ROM consistency check failed: max|diff|={stats['max_abs']:.3e} "
                    f"> tol={rom_consistency_tol:.3e}. "
                    "Your residual targets C_all - C_rom_all may be inconsistent with the ROM "
                    "trajectory used inside the augmented integration. "
                    "Common causes: ode_options mismatch, different solver tolerances/options, "
                    "or ROM.integrate not using options."
                )

    else:
        rom_mse_train, rom_mse_val = None, None

    # Targets
    if train_on_residuals:
        R_all = C_all - C_rom_all            # [M, nT, K]
    else:
        R_all = None

    # Derivative pretraining (matches dr/dt = NN(t, c_rom))
    if pretrain_derivative and pretrain_epochs > 0:
        assert C_rom_all is not None
        # residual trajectories used to build finite-diff targets
        R_for_pretrain = (C_all - C_rom_all)

        hybrid_model.neural_ode = _pretrain_residual_derivative_matching(
            func=hybrid_model.neural_ode,
            t=t_shared,
            C_rom=C_rom_all,
            R=R_for_pretrain,
            train_ids=train_ids,
            epochs=pretrain_epochs,
            lr=pretrain_lr,
            batch_ics=min(256, max(16, batch_ics)),
            time_batch=min(128, t_shared.numel()),
        )

    # Optimizer (NN only)
    opt = torch.optim.Adam(hybrid_model.neural_ode.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler, scheduler_step_on = _make_lr_scheduler( opt, lr_scheduler_cfg=lr_scheduler, epochs=epochs)

    train_curve, val_curve, val_epochs = [], [], []
    best_val_loss, best_epoch, patience_counter, best_state = float("inf"), 0, 0, None

    def _roll_augmented(c0_batch):
        """
        c0_batch: [B, K]
        returns Y: [B, nT, 2K]
        """
        y0 = torch.cat([c0_batch, torch.zeros_like(c0_batch)], dim=-1)  # [B, 2K]
        Y = odeint_fwd(
            hybrid_model.forward, y0, t_shared,
            method=method, rtol=rtol, atol=atol, options=ode_options
        ).permute(1, 0, 2)  # [B, nT, 2K]
        return Y

    for ep in tqdm(range(1, epochs + 1), desc="Training Hybrid"):
        hybrid_model.neural_ode.train()

        perm = torch.randperm(len(train_ids), device=device)
        train_ids_shuf = [train_ids[int(i)] for i in perm.tolist()]

        tot_loss, n_samples = 0.0, 0

        for s in range(0, len(train_ids_shuf), batch_ics):
            b_ids = train_ids_shuf[s : s + batch_ics]
            B = len(b_ids)

            opt.zero_grad(set_to_none=True)

            c0 = C_all[b_ids, 0, :]                  # [B, K]
            Y = _roll_augmented(c0)                  # [B, nT, 2K]
            c_rom = Y[..., :K]
            r_pred = Y[..., K:]
            c_pred = c_rom + r_pred

            if train_on_residuals:
                r_target = R_all[b_ids]              # [B, nT, K]
                loss = torch.mean((r_pred - r_target) ** 2)
            else:
                c_target = C_all[b_ids]              # [B, nT, K]
                loss = torch.mean((c_pred - c_target) ** 2)

            loss.backward()

            if grad_clip is not None:
                nn.utils.clip_grad_norm_(hybrid_model.neural_ode.parameters(), grad_clip)

            opt.step()

            tot_loss += float(loss.detach().item()) * B
            n_samples += B

        train_loss = tot_loss / max(1, n_samples)
        train_curve.append(train_loss)

        if scheduler is not None and scheduler_step_on == "epoch":
            scheduler.step()

        # Validation
        if len(val_ids) > 0 and (ep % print_every == 0 or ep == epochs):
            hybrid_model.neural_ode.eval()

            with torch.no_grad():
                val_sum, val_n = 0.0, 0
                chunk = min(64, batch_ics)

                for s in range(0, len(val_ids), chunk):
                    b_ids = val_ids[s : s + chunk]
                    B = len(b_ids)

                    c0 = C_all[b_ids, 0, :]
                    Y = _roll_augmented(c0)
                    c_rom = Y[..., :K]
                    r_pred = Y[..., K:]
                    c_pred = c_rom + r_pred

                    if train_on_residuals:
                        r_target = R_all[b_ids]
                        vloss = torch.mean((r_pred - r_target) ** 2)
                    else:
                        c_target = C_all[b_ids]
                        vloss = torch.mean((c_pred - c_target) ** 2)

                    val_sum += float(vloss.item()) * B
                    val_n += B

                val_loss = val_sum / max(1, val_n)
                val_curve.append(val_loss)
                val_epochs.append(ep)

            if scheduler is not None and scheduler_step_on == "val":
                scheduler.step(val_loss)

            current_lr = opt.param_groups[0]["lr"]

            hybrid_train = train_loss
            hybrid_val   = val_loss

            if rom_mse_val is not None:
                improvement = (rom_mse_val - hybrid_val) / rom_mse_val * 100
                print(
                    f"\n Epoch {ep:4d} | "
                    f"Train: {hybrid_train:.6e}, Val: {hybrid_val:.6e} | "
                    f"ROM Val: {rom_mse_val:.6e} | "
                    f"Improvement: {improvement:+.2f}% | "
                    f"LR: {current_lr:.6e}"
                )
            else:
                print(
                    f"\n Epoch {ep:4d} | "
                    f"Train: {hybrid_train:.6e}, Val: {hybrid_val:.6e} | "
                    f"LR: {current_lr:.6e}"
                )

            if val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = val_loss
                best_epoch = ep
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in hybrid_model.neural_ode.state_dict().items()}
                print("  ✓ New best model!")
            else:
                patience_counter += print_every

            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {ep}")
                break

    # Restore best NN
    if best_state is not None:
        hybrid_model.neural_ode.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"\n✓ Restored best model from epoch {best_epoch}")

    _plot_training_curves(
        train_curve, val_curve, val_epochs, best_epoch,
        title="Hybrid ROM + Neural ODE Training",
        rom_baseline=rom_mse_train if (train_on_residuals and rom_mse_train is not None) else None
    )

    info = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "train_curve": train_curve,
        "val_curve": val_curve,
        "val_epochs": val_epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "rom_mse_train": rom_mse_train,
        "rom_mse_val": rom_mse_val,
        "train_on_residuals": train_on_residuals,
        "pretrain_derivative": pretrain_derivative,
        "pretrain_epochs": pretrain_epochs,
        "pretrain_lr": pretrain_lr,
    }
    return hybrid_model, info





# =====================================================================
# Prediction Functions
# =====================================================================

@torch.no_grad()
def predict_test(
    func,
    dataset,
    ic_idx=0,
    t_eval=None,
    method="dopri5",
    rtol=1e-6,
    atol=1e-6,
    ode_options=None,
    reconstruct=True,
    compare_ground_truth=True,
    plot=True,
    transform=None,
    is_hybrid: bool = False,
):
    func.eval()
    device = dataset.c.device

    # --- time + ground truth ---
    if t_eval is None:
        t_true, c_true = dataset.get_trajectory(ic_idx)
        t_eval = t_true.clone()
    else:
        t_eval = t_eval.to(device)
        _, c_true = dataset.get_trajectory(ic_idx)

    t_eval, order = torch.sort(t_eval)
    c_true = c_true[order]
    c0 = c_true[0]

    # --- training-space IC ---
    c0_train = transform.encode(c0) if transform is not None else c0

    # --- rollout ---
    if is_hybrid:
        c_rom_tr, r_tr, c_pred_tr = func.predict(
            c0_train,
            t_eval,
            method=method,
            rtol=rtol,
            atol=atol,
            options=ode_options,
            return_components=True,
        )
    else:
        c_pred_tr = rollout(
            func, t_eval, c0_train,
            method=method, rtol=rtol, atol=atol, options=ode_options
        )

    # --- decode ---
    if transform is not None:
        c_pred = transform.decode(c_pred_tr)
        if is_hybrid:
            c_rom = transform.decode(c_rom_tr)
            r = transform.decode(r_tr)
    else:
        c_pred = c_pred_tr
        if is_hybrid:
            c_rom, r = c_rom_tr, r_tr

    # --- physical space ---
    c_pred_phys = dataset.denormalize_c(c_pred)
    c_true_phys = dataset.denormalize_c(c_true)
    t_phys = dataset.denormalize_t(t_eval)

    results = {
        "t": t_eval.detach().cpu(),
        "t_phys": t_phys.detach().cpu(),
        "c_pred": c_pred.detach().cpu(),
        "c_pred_phys": c_pred_phys.detach().cpu(),
    }

    # --- metrics ---
    if compare_ground_truth:
        mse_coeff = torch.mean((c_pred - c_true) ** 2).item()
        results.update({
            "c_true": c_true.detach().cpu(),
            "c_true_phys": c_true_phys.detach().cpu(),
            "mse_coeff": mse_coeff,
        })

        print(f"MSE (coefficient space): {mse_coeff:.6e}")

    # --- hybrid extras ---
    if is_hybrid:
        c_rom_phys = dataset.denormalize_c(c_rom)
        results.update({
            "c_rom": c_rom_phys.detach().cpu(),
            "c_residual": dataset.denormalize_c(r).detach().cpu(),
            "c_hybrid": c_pred_phys.detach().cpu(),
        })

        if compare_ground_truth:
            mse_rom = torch.mean((c_rom - c_true) ** 2).item()
            mse_hybrid = mse_coeff
            improvement = (mse_rom - mse_hybrid) / mse_rom * 100 if mse_rom > 0 else float("nan")

            results.update({
                "mse_rom": mse_rom,
                "mse_hybrid": mse_hybrid,
                "improvement_pct": improvement,
            })

            print(f"ROM MSE:     {mse_rom:.6e}")
            print(f"Hybrid MSE:  {mse_hybrid:.6e}")
            print(f"Improvement: {improvement:+.2f}%")

    # --- reconstruction ---
    if reconstruct:
        try:
            x_grid = dataset.get_reconstruction_grid()
            u_pred = dataset.reconstruct_u(c_pred, denormalize=True)
            results["u_pred"] = u_pred.detach().cpu()

            if compare_ground_truth:
                u_true = dataset.reconstruct_u(c_true, denormalize=True)
                results["u_true"] = u_true.detach().cpu()
                mse_spatial = torch.mean((u_pred - u_true) ** 2).item()
                results["mse_spatial"] = mse_spatial
                print(f"MSE (spatial domain): {mse_spatial:.6e}")
        except Exception as e:
            print(f"Could not reconstruct: {e}")

    # --- plotting ---
    if plot:
        _plot_predictions(results, reconstruct, compare_ground_truth)
    return results



@torch.no_grad()
def predict_and_comparison_plot(
    func, ds, u0_callable, t_vals, z_vals, X_ref,
    method="dopri5", rtol=1e-6, atol=1e-6, ode_options=None,
    notebook_plot=True, transform=None, is_hybrid=False,
):
    """
    Predict and compare with reference solution.

    If is_hybrid=True, func must be HybridROMNeuralODE and we show ROM + hybrid.
    """
    from plot_utils import plot_sim_result

    device = ds.c.device

    t_phys = torch.tensor(t_vals, device=device, dtype=ds.t.dtype)
    t_stored = _to_stored_time(ds, t_phys)
    t_stored, _ = torch.sort(t_stored)

    c0_stored = project_u0_to_c0_stored(ds, u0_callable)

    c0_train = transform.encode(c0_stored) if transform is not None else c0_stored

    if is_hybrid:
        c_rom_tr, c_residual_tr, c_pred_tr = func.predict(
            c0_train, t_stored,
            method=method, rtol=rtol, atol=atol, options=ode_options,
            return_components=True,
        )

        if transform is not None:
            c_rom = transform.decode(c_rom_tr)
            c_pred_stored = transform.decode(c_pred_tr)
        else:
            c_rom = c_rom_tr
            c_pred_stored = c_pred_tr

    else:
        c_pred_stored = rollout(func, t_stored, c0_train, method, rtol, atol, ode_options)
        if transform is not None:
            c_pred_stored = transform.decode(c_pred_stored)

    # Reconstruct in physical space
    U_pred_tx = ds.reconstruct_u(c_pred_stored, denormalize=True)
    U_pred_tx_np = U_pred_tx.detach().cpu().numpy()

    x_grid = ds.get_reconstruction_grid()
    U_pred_tz = _u_to_numpy_on_zgrid(U_pred_tx_np, x_grid, np.asarray(z_vals, float))

    X_ref = np.asarray(X_ref, dtype=float)

    title_pred = "u_pred (Hybrid)" if is_hybrid else "u_pred (Neural ODE)"

    if is_hybrid:
        U_rom_tx = ds.reconstruct_u(c_rom, denormalize=True)
        U_rom_tz = _u_to_numpy_on_zgrid(U_rom_tx.detach().cpu().numpy(), x_grid, np.asarray(z_vals, float))
        plot_sim_result(z_vals, t_vals, U_rom_tz, "u_ROM", notebook_plot=notebook_plot)
    plot_sim_result(z_vals, t_vals, U_pred_tz, title_pred, notebook_plot=notebook_plot)
    plot_sim_result(z_vals, t_vals, X_ref, "u_ref", notebook_plot=notebook_plot)
    plot_sim_result(z_vals, t_vals, np.abs(X_ref - U_pred_tz), "abs error", notebook_plot=notebook_plot)

    return {
        "t_phys": t_vals,
        "z_vals": z_vals,
        "U_pred": U_pred_tz,
        "U_ref": X_ref,
        "abs_err": np.abs(X_ref - U_pred_tz),
        "c_pred_stored": c_pred_stored,
        **({"c_rom_stored": c_rom} if is_hybrid else {}),
    }

@torch.no_grad()
def eval_hybrid_val_mse(
    hybrid_model,
    t_shared,
    C_all,
    val_ids,
    method="dopri5",
    rtol=1e-6,
    atol=1e-6,
    ode_options=None,
    batch_ics: int = 64,
):
    """
    Returns mean MSE over val_ids in coefficient space.
    Batched over initial conditions.
    """
    device = C_all.device
    hybrid_model.eval()

    M, nT, K = C_all.shape
    ids = list(val_ids)
    if len(ids) == 0:
        return float("nan")

    mse_sum = 0.0
    n = 0

    for s in range(0, len(ids), batch_ics):
        b_ids = ids[s : s + batch_ics]
        c_true = C_all[b_ids]               # [B, nT, K]
        c0 = c_true[:, 0, :]                # [B, K]

        # HybridROMNeuralODE.predict supports batched c0 -> [nT, B, K]
        c_pred_tBK = hybrid_model.predict(
            c0, t_shared, method=method, rtol=rtol, atol=atol, options=ode_options, return_components=False
        )
        c_pred = c_pred_tBK.permute(1, 0, 2).contiguous()  # [B, nT, K]

        mse = torch.mean((c_pred - c_true) ** 2)           # scalar mean over B,nT,K
        mse_sum += float(mse.item()) * len(b_ids)
        n += len(b_ids)

    return mse_sum / max(1, n)

# =====================================================================
# Visualization Functions
# =====================================================================

def _plot_training_curves(train_curve, val_curve, val_epochs, best_epoch, title="Training", rom_baseline=None):
    """Plot training curves."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_curve, mode="lines", name="Train", line=dict(width=2)))
    
    if len(val_curve) > 0:
        fig.add_trace(go.Scatter(x=val_epochs, y=val_curve, mode="lines+markers",
                                name="Val", line=dict(width=2)))
        
        if best_epoch > 0:
            fig.add_vline(x=best_epoch, line_dash="dash", line_color="green",
                         annotation_text=f"Best (epoch {best_epoch})")
    
    if rom_baseline is not None:
        fig.add_hline(y=rom_baseline, line_dash="dot", line_color="red",
                     annotation_text="ROM baseline")
    
    fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title="MSE Loss",
                     yaxis_type="log", height=500)
    fig.show()


def _plot_predictions(results, reconstruct, compare_ground_truth):
    """
    Unified plotter:
      - Non-hybrid: plots predicted vs true.
      - Hybrid: additionally plots ROM baseline.
    Expects:
      results["t_phys"], results["c_pred_phys"]
      optionally results["c_true_phys"]
      optionally results["c_rom"] (hybrid)
    """
    t_phys = results["t_phys"].numpy()

    c_pred = results["c_pred_phys"].numpy()
    K = c_pred.shape[1]
    n_plot = min(5, K)

    has_true = compare_ground_truth and ("c_true_phys" in results)
    has_rom = "c_rom" in results  # hybrid marker

    if has_true:
        c_true = results["c_true_phys"].numpy()
    if has_rom:
        c_rom = results["c_rom"].numpy()

    fig = go.Figure()

    for k in range(n_plot):
        # Hybrid / predicted (same style you already like)
        fig.add_trace(go.Scatter(
            x=t_phys, y=c_pred[:, k],
            mode="lines",
            name=f"c_{k} (pred)",
        ))

        # ROM baseline (hybrid only)
        if has_rom:
            fig.add_trace(go.Scatter(
                x=t_phys, y=c_rom[:, k],
                mode="lines",
                name=f"c_{k} (ROM)",
                opacity=0.8,
                line=dict(dash="dash"),
            ))

        # Ground truth
        if has_true:
            fig.add_trace(go.Scatter(
                x=t_phys, y=c_true[:, k],
                mode="lines",
                name=f"c_{k} (true)",
                opacity=0.7,
            ))

    title = "Predicted Coefficients (Hybrid + ROM)" if has_rom else "Predicted Coefficients"

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Coefficient value",
        height=500,
        legend=dict(orientation="h"),
    )
    fig.show()
