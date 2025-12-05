"""
Fast Neural Galerkin NeuralODE training + prediction utilities.

Optimizations included (no change in objective / accuracy):
- Model casts time to c.dtype internally (avoid float64 through NN).
- Uses forward odeint for eval/rollout (adjoint not needed under no_grad).
- Caches per-trajectory preprocessing (sort/subsample/drop duplicates) once.
- Vectorized batched odeint when time grids are identical across batch.

Important: This version assumes ds tensors already live on GPU.
Therefore DataLoader uses pin_memory=False.
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset

import plotly.graph_objects as go
from tqdm import tqdm
import numpy as np

from torchdiffeq import odeint as odeint_fwd
from torchdiffeq import odeint_adjoint as odeint_adj

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
        # Keep solver time possibly float64, but do NOT push float64 through the NN.
        if self.time_dependent:
            tt = t.to(dtype=c.dtype, device=c.device).expand(c.shape[0], 1)
            x = torch.cat([c, tt], dim=1)
        else:
            x = c
        return self.net(x)


# -----------------------------
# Small utilities
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


# -----------------------------
# Fast cached trajectory prep
# -----------------------------
class PreparedTrajDataset(Dataset):
    """
    Wraps a base dataset whose __getitem__ returns {"t":(nT,), "c":(nT,K), "id":...}
    and caches the expensive preprocessing:
      - sort by time
      - optional subsample to fixed length
      - drop non-increasing times

    This causes NO change vs your original per-step _traj_mse logic,
    except that it is done once per trajectory.

    NOTE: keeps tensors on same device as base dataset (often CUDA).
    """
    def __init__(self, base_ds: Dataset, time_subsample: int | None):
        self.base = base_ds
        self.time_subsample = time_subsample
        self._cache: dict[int, dict[str, torch.Tensor]] = {}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        if idx in self._cache:
            return self._cache[idx]

        item = self.base[idx]
        t = item["t"]
        c = item["c"]

        # sort by time
        order = torch.argsort(t)
        t = t[order].to(torch.float64)
        c = c[order]

        # optional subsample
        if self.time_subsample is not None and t.numel() > self.time_subsample:
            nT = t.numel()
            lin = torch.linspace(0, nT - 1, self.time_subsample, device=t.device)
            ii = torch.unique_consecutive(lin.round().long())
            t = t[ii]
            c = c[ii]

        # drop non-increasing times
        dt = t[1:] - t[:-1]
        keep = torch.cat([torch.ones(1, dtype=torch.bool, device=t.device), dt > 0])
        t = t[keep]
        c = c[keep]

        # safety (fail early if dataset is still dirty)
        if t.numel() < 2:
            raise RuntimeError(f"Trajectory {idx}: need >=2 increasing times, got {t.numel()}")
        if not torch.isfinite(c).all():
            bad = torch.nonzero(~torch.isfinite(c), as_tuple=False)[0]
            raise RuntimeError(f"Trajectory {idx}: non-finite c at {tuple(int(x) for x in bad)} value={c[tuple(bad.tolist())]}")

        out = {"t": t.contiguous(), "c": c.contiguous(), "id": item["id"]}
        if "k" in item:
            out["k"] = item["k"]

        self._cache[idx] = out
        return out


# -----------------------------
# Rollout / projection helpers
# -----------------------------
@torch.no_grad()
def rollout(func, t_stored: torch.Tensor, c0_stored: torch.Tensor,
            method="dopri5", rtol=1e-6, atol=1e-6, options=None) -> torch.Tensor:
    if c0_stored.ndim == 1:
        c0_stored = c0_stored.unsqueeze(0)  # (1,K)
    c_pred = odeint_fwd(func, c0_stored, t_stored, method=method, rtol=rtol, atol=atol, options=options)
    return c_pred.squeeze(1)  # (nT,K)


@torch.no_grad()
def project_u0_to_c0_stored(ds, u0_callable) -> torch.Tensor:
    if ds.Phi is None:
        raise ValueError("Dataset has no stored basis_matrix (ds.Phi is None).")

    x_grid = ds.get_reconstruction_grid()  # numpy (nx,)
    u0_np = np.asarray(u0_callable(x_grid), dtype=float).reshape(-1)  # (nx,)
    if u0_np.shape[0] != x_grid.size:
        raise ValueError(f"u0(x_grid) must return shape (nx,), got {u0_np.shape} for nx={x_grid.size}")

    w_np = trapz_weights_1d(x_grid)  # (nx,)

    device = ds.c.device
    dtype = ds.c.dtype

    u0 = torch.tensor(u0_np, device=device, dtype=dtype)     # (nx,)
    w  = torch.tensor(w_np,  device=device, dtype=dtype)     # (nx,)

    c0_phys = ds.Phi @ (w * u0)  # (K,)

    if ds.config.normalize_c:
        mean = torch.as_tensor(ds.c_mean, device=device, dtype=dtype).squeeze(0)
        std  = torch.as_tensor(ds.c_std,  device=device, dtype=dtype).squeeze(0)
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
):
    device = ds.c.device

    t_phys = torch.tensor(t_vals, device=device, dtype=ds.t.dtype)
    t_stored = _to_stored_time(ds, t_phys)
    t_stored, _ = torch.sort(t_stored)

    c0_st = project_u0_to_c0_stored(ds, u0_callable)
    c_pred_st = rollout(func, t_stored, c0_st, method=method, rtol=rtol, atol=atol, options=ode_options)

    U_pred_tx = ds.reconstruct_u(c_pred_st, denormalize=True)
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
        "c0_stored": c0_st,
        "c_pred_stored": c_pred_st,
    }


# -----------------------------
# Prediction (single IC from dataset)
# -----------------------------
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
):
    func.eval()
    device = dataset.device

    if t_eval is None:
        t_true, c_true = dataset.get_trajectory(ic_idx)
        t_eval = t_true.clone()
    else:
        t_eval = t_eval.to(device)
        _, c_true = dataset.get_trajectory(ic_idx)

    t_eval, order = torch.sort(t_eval)
    c_true = c_true[order]
    c0 = c_true[0].unsqueeze(0)

    c_pred = odeint_fwd(func, c0, t_eval, method=method, rtol=rtol, atol=atol, options=ode_options).squeeze(1)

    results = {"t": t_eval.cpu(), "c_pred": c_pred.cpu()}

    c_pred_phys = dataset.denormalize_c(c_pred)
    c_true_phys = dataset.denormalize_c(c_true)
    t_phys = dataset.denormalize_t(t_eval)

    results["c_pred_phys"] = c_pred_phys.cpu()
    results["t_phys"] = t_phys.cpu()

    if compare_ground_truth:
        results["c_true"] = c_true.cpu()
        results["c_true_phys"] = c_true_phys.cpu()
        mse_coeff = torch.mean((c_pred - c_true) ** 2).item()
        results["mse_coeff"] = mse_coeff
        print(f"MSE (coefficient space): {mse_coeff:.6e}")

    if reconstruct:
        try:
            x_grid = dataset.get_reconstruction_grid()
            results["x_grid"] = x_grid

            u_pred = dataset.reconstruct_u(c_pred, denormalize=True)
            results["u_pred"] = u_pred.cpu()

            if compare_ground_truth:
                u_true = dataset.reconstruct_u(c_true, denormalize=True)
                results["u_true"] = u_true.cpu()

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
    fig1.update_layout(title=f"Predicted Galerkin Coefficients (first {n_plot} modes)",
                       xaxis_title="Time", yaxis_title="Coefficient value", height=500)
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
                fig2.add_trace(go.Scatter(x=x_grid, y=u_true[i], mode="lines", name=f"t={t_phys[i]:.2f} (true)",
                                          opacity=0.5))
        fig2.update_layout(title="Spatial Solution Snapshots", xaxis_title="x", yaxis_title="u(t, x)", height=500)
        fig2.show()

        if compare_ground_truth and "u_true" in results:
            u_true = results["u_true"].numpy()
            error = np.abs(u_pred - u_true)

            fig3 = make_subplots(rows=1, cols=3,
                                 subplot_titles=("Prediction", "Ground Truth", "Absolute Error"),
                                 horizontal_spacing=0.1)

            fig3.add_trace(go.Heatmap(z=u_pred.T, x=t_phys, y=x_grid, colorscale="RdBu", name="pred"), row=1, col=1)
            fig3.add_trace(go.Heatmap(z=u_true.T, x=t_phys, y=x_grid, colorscale="RdBu", name="true"), row=1, col=2)
            fig3.add_trace(go.Heatmap(z=error.T, x=t_phys, y=x_grid, colorscale="Reds", name="error"), row=1, col=3)

            fig3.update_xaxes(title_text="Time", row=1, col=1)
            fig3.update_xaxes(title_text="Time", row=1, col=2)
            fig3.update_xaxes(title_text="Time", row=1, col=3)
            fig3.update_yaxes(title_text="x", row=1, col=1)
            fig3.update_layout(title="Spatiotemporal Solution", height=400, showlegend=False)
            fig3.show()
        else:
            fig3 = go.Figure(data=go.Heatmap(z=u_pred.T, x=t_phys, y=x_grid, colorscale="RdBu"))
            fig3.update_layout(title="Spatiotemporal Solution (Prediction)", xaxis_title="Time", yaxis_title="x",
                               height=400)
            fig3.show()


# -----------------------------
# Train helpers
# -----------------------------
def _split_indices(M: int, val_frac: float, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(M, generator=g).tolist()
    n_val = max(1, int(math.floor(val_frac * M))) if M > 1 else 0
    val_ids = perm[:n_val]
    train_ids = perm[n_val:]
    return train_ids, val_ids


def _traj_mse_batch_shared_t(
    func,
    t: torch.Tensor,          # (nT,) float64
    cB: torch.Tensor,         # (B,nT,K) float32
    method: str,
    rtol: float,
    atol: float,
    use_adjoint: bool,
    options: dict | None,
):
    y0 = cB[:, 0, :]  # (B,K)
    ode = odeint_adj if use_adjoint else odeint_fwd
    pred_tBK = ode(func, y0, t, method=method, rtol=rtol, atol=atol, options=options)  # (nT,B,K)
    pred_BtK = pred_tBK.permute(1, 0, 2).contiguous()
    return torch.mean((pred_BtK - cB) ** 2)


@torch.no_grad()
def eval_dataset_mse(func, loader, method, rtol, atol, ode_options=None):
    func.eval()
    tot = 0.0
    n = 0
    for batch in loader:
        tB = batch["t"]
        cB = batch["c"]
        B = tB.shape[0]

        t0 = tB[0]
        shared_t = all(torch.equal(tB[b], t0) for b in range(1, B))
        if shared_t:
            loss = _traj_mse_batch_shared_t(func, t0, cB, method, rtol, atol, use_adjoint=False, options=ode_options)
            tot += float(loss.item()) * B
            n += B
        else:
            for b in range(B):
                t = tB[b]
                c = cB[b]
                y0 = c[0].unsqueeze(0)
                pred = odeint_fwd(func, y0, t, method=method, rtol=rtol, atol=atol, options=ode_options).squeeze(1)
                tot += float(torch.mean((pred - c) ** 2).item())
                n += 1
    return tot / max(1, n)


def train_neural_ode_on_neural_galerkin_dataset(
    ds,
    val_frac: float = 0.2,
    split_seed: int = 0,
    epochs: int = 2000,
    lr: float = 1e-3,
    hidden: int = 256,
    time_dependent: bool = True,
    method: str = "dopri5",
    rtol: float = 1e-6,
    atol: float = 1e-6,
    batch_ics: int = 8,
    time_subsample: int | None = 150,
    grad_clip: float = 1.0,
    print_every: int = 50,
    use_adjoint_train: bool = False,
    ode_options: dict | None = None,
):
    device = ds.c.device
    M, _, K = ds.c.shape

    prep_ds = PreparedTrajDataset(ds, time_subsample=time_subsample)

    train_ids, val_ids = _split_indices(M, val_frac=val_frac, seed=split_seed)
    train_set = Subset(prep_ds, train_ids)
    val_set   = Subset(prep_ds, val_ids) if len(val_ids) > 0 else None

    # IMPORTANT: pin_memory must be False because data are CUDA tensors.
    train_loader = DataLoader(
        train_set,
        batch_size=min(batch_ics, len(train_set)),
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=min(batch_ics, len(val_set)),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    ) if val_set is not None else None

    func = CoeffODEFunc(K, hidden=hidden, time_dependent=time_dependent).to(device)
    opt = torch.optim.Adam(func.parameters(), lr=lr)

    train_curve, val_curve = [], []

    for ep in tqdm(range(1, epochs + 1), "training Neural Galerkin ODE"):
        func.train()
        train_tot = 0.0
        train_n = 0

        for batch in train_loader:
            tB = batch["t"]  # (B,nT') float64 on cuda
            cB = batch["c"]  # (B,nT',K) float32 on cuda
            B = tB.shape[0]

            opt.zero_grad(set_to_none=True)

            t0 = tB[0]
            shared_t = all(torch.equal(tB[b], t0) for b in range(1, B))

            if shared_t:
                loss = _traj_mse_batch_shared_t(
                    func, t0, cB, method, rtol, atol,
                    use_adjoint=use_adjoint_train,
                    options=ode_options,
                )
            else:
                ode = odeint_adj if use_adjoint_train else odeint_fwd
                loss_acc = 0.0
                for b in range(B):
                    t = tB[b]
                    c = cB[b]
                    y0 = c[0].unsqueeze(0)
                    pred = ode(func, y0, t, method=method, rtol=rtol, atol=atol, options=ode_options).squeeze(1)
                    loss_acc = loss_acc + torch.mean((pred - c) ** 2)
                loss = loss_acc / B

            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(func.parameters(), grad_clip)
            opt.step()

            train_tot += float(loss.detach().item()) * B
            train_n += B

        train_mse = train_tot / max(1, train_n)
        train_curve.append(train_mse)

        if val_loader is not None and (ep % print_every == 0 or ep == epochs):
            val_mse = eval_dataset_mse(func, val_loader, method, rtol, atol, ode_options=ode_options)
            val_curve.append(val_mse)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_curve, mode="lines", name="train"))
    if len(val_curve) > 0:
        fig.add_trace(go.Scatter(y=val_curve, mode="lines", name="val"))
    fig.update_layout(title="Neural ODE training curves", xaxis_title="epoch", yaxis_title="MSE")
    fig.show()

    return func, {"train_ids": train_ids, "val_ids": val_ids, "train_curve": train_curve, "val_curve": val_curve}


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
        batch_ics=8,
        time_subsample=150,
        print_every=100,
        use_adjoint_train=False,
        ode_options=None,
    )

    results = predict(
        func=func,
        dataset=ds,
        ic_idx=0,
        method="dopri5",
        reconstruct=True,
        compare_ground_truth=True,
        plot=True,
    )

    print("\nPrediction completed!")
    print(f"Time points: {results['t'].shape}")
    print(f"Predicted coefficients: {results['c_pred'].shape}")
    if "u_pred" in results:
        print(f"Reconstructed spatial solution: {results['u_pred'].shape}")
