import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchdiffeq import odeint_adjoint as odeint
import plotly.graph_objects as go
from tqdm import tqdm
import numpy as np
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
        if self.time_dependent:
            tt = t.expand(c.shape[0], 1)
            x = torch.cat([c, tt], dim=1)
        else:
            x = c
        return self.net(x)

# If you don't want to import it, copy the same implementation you used in dataset code
def trapz_weights_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    w = np.empty_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w

def _to_stored_time(ds, t_phys: torch.Tensor) -> torch.Tensor:
    # physical -> stored (possibly normalized)
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

@torch.no_grad()
def rollout(func, t_stored: torch.Tensor, c0_stored: torch.Tensor,
            method="dopri5", rtol=1e-6, atol=1e-6) -> torch.Tensor:
    if c0_stored.ndim == 1:
        c0_stored = c0_stored.unsqueeze(0)  # (1,K)
    c_pred = odeint(func, c0_stored, t_stored, method=method, rtol=rtol, atol=atol)  # (nT,1,K)
    return c_pred.squeeze(1)  # (nT,K)

@torch.no_grad()
def project_u0_to_c0_stored(ds, u0_callable) -> torch.Tensor:
    """
    u0_callable: function z -> u0(z) (scalar or numpy array)
    returns c0 in *stored* coeff space (normalized if ds.normalize_c=True).
    """
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

    # c0_phys[k] = sum_j w_j * u0_j * Phi[k,j]
    c0_phys = ds.Phi @ (w * u0)  # (K,)

    # map to stored (normalized) space if needed
    if ds.config.normalize_c:
        mean = torch.as_tensor(ds.c_mean, device=device, dtype=dtype).squeeze(0)  # (K,)
        std  = torch.as_tensor(ds.c_std,  device=device, dtype=dtype).squeeze(0)  # (K,)
        c0_st = (c0_phys - mean) / std
        return c0_st

    return c0_phys

@torch.no_grad()
def predict_and_plot_vs_reference_surface(
    func,
    ds,
    u0_callable,                # <-- NEW: use this initial condition, not ds[ic_idx]
    t_vals: np.ndarray,
    z_vals: np.ndarray,
    X_ref: np.ndarray,
    method: str = "dopri5",
    rtol: float = 1e-6,
    atol: float = 1e-6,
    notebook_plot: bool = True,
):
    device = ds.c.device

    # eval times: physical -> stored
    t_phys = torch.tensor(t_vals, device=device, dtype=ds.t.dtype)
    t_stored = _to_stored_time(ds, t_phys)
    t_stored, _ = torch.sort(t_stored)

    # project u0 to c0 in stored space
    c0_st = project_u0_to_c0_stored(ds, u0_callable)  # (K,)

    # rollout in stored space
    c_pred_st = rollout(func, t_stored, c0_st, method=method, rtol=rtol, atol=atol)  # (nT,K)

    # reconstruct u on ds.x_grid and denormalize coeffs inside reconstruct_u
    U_pred_tx = ds.reconstruct_u(c_pred_st, denormalize=True)  # (nT,nx)
    U_pred_tx_np = U_pred_tx.detach().cpu().numpy()

    # map to z_vals if grids differ
    x_grid = ds.get_reconstruction_grid()
    U_pred_tz = _u_to_numpy_on_zgrid(U_pred_tx_np, x_grid, np.asarray(z_vals, float))

    # compare with reference
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



@torch.no_grad()
def predict(
    func: nn.Module,
    dataset,  # NeuralGalerkinDataset
    ic_idx: int = 0,
    t_eval: torch.Tensor | None = None,
    method: str = "dopri5",
    rtol: float = 1e-6,
    atol: float = 1e-6,
    reconstruct: bool = True,
    compare_ground_truth: bool = True,
    plot: bool = True,
):
    """
    Predict trajectory from initial condition and optionally reconstruct spatial solution.
    
    Args:
        func: Trained CoeffODEFunc model
        dataset: NeuralGalerkinDataset containing the data
        ic_idx: Index of initial condition to use (default: 0)
        t_eval: Time points for evaluation. If None, uses dataset times for this IC
        method: ODE solver method (default: "dopri5")
        rtol: Relative tolerance for ODE solver
        atol: Absolute tolerance for ODE solver
        reconstruct: If True, reconstruct spatial solution u(t,x) from coefficients
        compare_ground_truth: If True, compute error vs ground truth
        plot: If True, create visualization plots
    
    Returns:
        dict with keys:
            - 't': time points (nT,)
            - 'c_pred': predicted coefficients (nT, K)
            - 'c_true': ground truth coefficients (nT, K) if available
            - 'u_pred': reconstructed spatial solution (nT, nx) if reconstruct=True
            - 'u_true': ground truth spatial solution (nT, nx) if available
            - 'x_grid': spatial grid (nx,) if reconstruct=True
            - 'mse_coeff': MSE in coefficient space
            - 'mse_spatial': MSE in spatial domain if reconstruct=True
    """
    func.eval()
    device = dataset.device
    
    # Get time points and initial condition
    if t_eval is None:
        t_true, c_true = dataset.get_trajectory(ic_idx)
        t_eval = t_true.clone()
    else:
        t_eval = t_eval.to(device)
        _, c_true = dataset.get_trajectory(ic_idx)
    
    # Sort time points
    t_eval, order = torch.sort(t_eval)
    c0 = c_true[0].unsqueeze(0)  # (1, K) - use first time point as IC
    
    # Predict coefficients
    c_pred = odeint(func, c0, t_eval, method=method, rtol=rtol, atol=atol)
    c_pred = c_pred.squeeze(1)  # (nT, K)
    
    # Prepare output dictionary
    results = {
        't': t_eval.cpu(),
        'c_pred': c_pred.cpu(),
    }
    
    # Denormalize coefficients if needed
    c_pred_phys = dataset.denormalize_c(c_pred)
    c_true_phys = dataset.denormalize_c(c_true)
    t_phys = dataset.denormalize_t(t_eval)
    
    results['c_pred_phys'] = c_pred_phys.cpu()
    results['t_phys'] = t_phys.cpu()
    
    # Compare with ground truth if requested
    if compare_ground_truth:
        results['c_true'] = c_true.cpu()
        results['c_true_phys'] = c_true_phys.cpu()
        
        # MSE in coefficient space
        mse_coeff = torch.mean((c_pred - c_true) ** 2).item()
        results['mse_coeff'] = mse_coeff
        print(f"MSE (coefficient space): {mse_coeff:.6e}")
    
    # Reconstruct spatial solution if requested
    if reconstruct:
        try:
            x_grid = dataset.get_reconstruction_grid()
            results['x_grid'] = x_grid
            
            # Reconstruct predicted solution
            u_pred = dataset.reconstruct_u(c_pred, denormalize=True)  # (nT, nx)
            results['u_pred'] = u_pred.cpu()
            
            # Reconstruct ground truth if available
            if compare_ground_truth:
                u_true = dataset.reconstruct_u(c_true, denormalize=True)  # (nT, nx)
                results['u_true'] = u_true.cpu()
                
                # MSE in spatial domain
                mse_spatial = torch.mean((u_pred - u_true) ** 2).item()
                results['mse_spatial'] = mse_spatial
                print(f"MSE (spatial domain): {mse_spatial:.6e}")
        
        except ValueError as e:
            print(f"Could not reconstruct spatial solution: {e}")
            reconstruct = False
    
    # Create visualizations if requested
    if plot:
        _plot_predictions(results, reconstruct, compare_ground_truth)
    
    return results


def _plot_predictions(results: dict, reconstruct: bool, compare_ground_truth: bool):
    """Helper function to create visualization plots."""
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    t_phys = results['t_phys'].numpy()
    
    # Plot 1: Coefficient trajectories
    c_pred = results['c_pred_phys'].numpy()  # (nT, K)
    K = c_pred.shape[1]
    
    # Select a few coefficients to plot (avoid plotting too many)
    n_plot = min(5, K)
    fig1 = go.Figure()
    
    for k in range(n_plot):
        fig1.add_trace(go.Scatter(
            x=t_phys, y=c_pred[:, k],
            mode='lines',
            name=f'c_{k} (pred)',
            line=dict(dash='solid')
        ))
        
        if compare_ground_truth:
            c_true = results['c_true_phys'].numpy()
            fig1.add_trace(go.Scatter(
                x=t_phys, y=c_true[:, k],
                mode='lines',
                name=f'c_{k} (true)',
                line=dict(dash='dash'),
                opacity=0.7
            ))
    
    fig1.update_layout(
        title=f"Predicted Galerkin Coefficients (first {n_plot} modes)",
        xaxis_title="Time",
        yaxis_title="Coefficient value",
        height=500
    )
    fig1.show()
    
    # Plot 2: Spatial solution if available
    if reconstruct and 'u_pred' in results:
        u_pred = results['u_pred'].numpy()  # (nT, nx)
        x_grid = results['x_grid']
        
        # Select time snapshots to plot
        nT = u_pred.shape[0]
        n_snapshots = min(6, nT)
        snapshot_indices = np.linspace(0, nT - 1, n_snapshots, dtype=int)
        
        fig2 = go.Figure()
        
        for i in snapshot_indices:
            fig2.add_trace(go.Scatter(
                x=x_grid, y=u_pred[i],
                mode='lines',
                name=f't={t_phys[i]:.2f} (pred)',
                line=dict(dash='solid')
            ))
            
            if compare_ground_truth and 'u_true' in results:
                u_true = results['u_true'].numpy()
                fig2.add_trace(go.Scatter(
                    x=x_grid, y=u_true[i],
                    mode='lines',
                    name=f't={t_phys[i]:.2f} (true)',
                    line=dict(dash='dash'),
                    opacity=0.5
                ))
        
        fig2.update_layout(
            title="Spatial Solution Snapshots",
            xaxis_title="x",
            yaxis_title="u(t, x)",
            height=500
        )
        fig2.show()
        
        # Plot 3: Spatiotemporal heatmap
        if compare_ground_truth and 'u_true' in results:
            u_true = results['u_true'].numpy()
            error = np.abs(u_pred - u_true)
            
            fig3 = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Prediction', 'Ground Truth', 'Absolute Error'),
                horizontal_spacing=0.1
            )
            
            fig3.add_trace(
                go.Heatmap(z=u_pred.T, x=t_phys, y=x_grid, colorscale='RdBu', name='pred'),
                row=1, col=1
            )
            fig3.add_trace(
                go.Heatmap(z=u_true.T, x=t_phys, y=x_grid, colorscale='RdBu', name='true'),
                row=1, col=2
            )
            fig3.add_trace(
                go.Heatmap(z=error.T, x=t_phys, y=x_grid, colorscale='Reds', name='error'),
                row=1, col=3
            )
            
            fig3.update_xaxes(title_text="Time", row=1, col=1)
            fig3.update_xaxes(title_text="Time", row=1, col=2)
            fig3.update_xaxes(title_text="Time", row=1, col=3)
            fig3.update_yaxes(title_text="x", row=1, col=1)
            
            fig3.update_layout(
                title="Spatiotemporal Solution",
                height=400,
                showlegend=False
            )
            fig3.show()
        else:
            # Just plot prediction
            fig3 = go.Figure(data=go.Heatmap(
                z=u_pred.T,
                x=t_phys,
                y=x_grid,
                colorscale='RdBu'
            ))
            fig3.update_layout(
                title="Spatiotemporal Solution (Prediction)",
                xaxis_title="Time",
                yaxis_title="x",
                height=400
            )
            fig3.show()


def _split_indices(M: int, val_frac: float, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(M, generator=g).tolist()
    n_val = max(1, int(math.floor(val_frac * M))) if M > 1 else 0
    val_ids = perm[:n_val]
    train_ids = perm[n_val:]
    return train_ids, val_ids


def _traj_mse(func, t: torch.Tensor, c: torch.Tensor, method, rtol, atol, time_subsample: int | None):
    # t: (nT,), c: (nT,K)
    if time_subsample is not None and time_subsample < t.numel():
        pick = torch.randperm(t.numel(), device=t.device)[:time_subsample]
        pick, _ = torch.sort(pick)
        t = t[pick]
        c = c[pick]

    t, order = torch.sort(t)
    c = c[order]

    c0 = c[0].unsqueeze(0)  # (1,K)
    c_pred = odeint(func, c0, t, method=method, rtol=rtol, atol=atol).squeeze(1)  # (nT,K)
    return torch.mean((c_pred - c) ** 2)


@torch.no_grad()
def eval_dataset_mse(func, loader, method, rtol, atol):
    func.eval()
    tot = 0.0
    n = 0
    for batch in loader:
        tB = batch["t"]  # (B,nT)
        cB = batch["c"]  # (B,nT,K)
        B = tB.shape[0]
        for b in range(B):
            loss = _traj_mse(func, tB[b], cB[b], method, rtol, atol, time_subsample=None)
            tot += float(loss.item())
            n += 1
    return tot / max(1, n)


def train_neural_ode_on_neural_galerkin_dataset(
    ds,                          # NeuralGalerkinDataset
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
    time_subsample: int | None = 150,   # train-time speedup
    grad_clip: float = 1.0,
    print_every: int = 50,
):
    device = ds.c.device
    M, nT, K = ds.c.shape

    train_ids, val_ids = _split_indices(M, val_frac=val_frac, seed=split_seed)
    train_set = Subset(ds, train_ids)
    val_set   = Subset(ds, val_ids) if len(val_ids) > 0 else None

    train_loader = DataLoader(train_set, batch_size=min(batch_ics, len(train_set)),
                              shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_set, batch_size=min(batch_ics, len(val_set)),
                              shuffle=False, drop_last=False) if val_set is not None else None

    func = CoeffODEFunc(K, hidden=hidden, time_dependent=time_dependent).to(device)
    opt = torch.optim.Adam(func.parameters(), lr=lr)

    train_curve, val_curve = [], []

    for ep in tqdm(range(1, epochs + 1), "training Neural Galerkin ODE"):
        func.train()
        train_tot = 0.0
        train_n = 0

        for batch in train_loader:
            tB = batch["t"]  # (B,nT)
            cB = batch["c"]  # (B,nT,K)
            B = tB.shape[0]

            opt.zero_grad(set_to_none=True)

            loss_batch = 0.0
            for b in range(B):
                loss_b = _traj_mse(func, tB[b], cB[b], method, rtol, atol, time_subsample=time_subsample)
                loss_batch = loss_batch + loss_b

            loss_batch = loss_batch / B
            loss_batch.backward()

            if grad_clip is not None:
                nn.utils.clip_grad_norm_(func.parameters(), grad_clip)
            opt.step()

            train_tot += float(loss_batch.detach().item()) * B
            train_n += B

        train_mse = train_tot / max(1, train_n)
        train_curve.append(train_mse)

        if val_loader is not None:
            val_mse = eval_dataset_mse(func, val_loader, method=method, rtol=rtol, atol=atol)
            val_curve.append(val_mse)
        else:
            val_mse = None

    # Plot loss curves (Plotly)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_curve, mode="lines", name="train"))
    if len(val_curve) > 0:
        fig.add_trace(go.Scatter(y=val_curve, mode="lines", name="val"))
    fig.update_layout(
        title="Neural ODE training curves",
        xaxis_title="epoch",
        yaxis_title="MSE",
    )
    fig.show()

    return func, {"train_ids": train_ids, "val_ids": val_ids, "train_curve": train_curve, "val_curve": val_curve}



if __name__ == "__main__":
    from data_burgers import create_burgers_NeuralGalerkin_dataset, BurgersInitialConditions
    import torch
    from pde_dataset.neural_galerkin_dataset import NeuralGalerkinDataset

    # ds = create_burgers_NeuralGalerkin_dataset(
    #     initial_conditions=[
    #         BurgersInitialConditions.gaussian(amplitude=1.0, center=-1.0, width=0.8),
    #         BurgersInitialConditions.gaussian(amplitude=2.0, center=0.0,  width=0.5),
    #         BurgersInitialConditions.gaussian(amplitude=1.5, center=1.0,  width=1.0),
    #     ],
    #     Tmax=2.0,
    #     n_basis=48,
    #     n_time_samples=400,
    #     t_sampling="random",
    #     device="cuda",
    #     normalize_c=True,
    #     normalize_t=False,
    #     seed=42,
    # )

    ds = NeuralGalerkinDataset.load(
        filepath="datasets/burgers_neural_galerkin.npz",
        device="cuda",
        dtype=torch.float32,
    )

    func, info = train_neural_ode_on_neural_galerkin_dataset(
        ds=ds,
        epochs=2000,
        lr=1e-3,
        hidden=256,
        time_dependent=True,   # recommended with multi-IC data
        method="dopri5",
        batch_ics=3,
        time_subsample=150,
        print_every=100,
    )
    
    # Predict and visualize first trajectory
    results = predict(
        func=func,
        dataset=ds,
        ic_idx=0,
        method="dopri5",
        reconstruct=True,      # Reconstruct spatial solution u(t,x)
        compare_ground_truth=True,  # Compare with ground truth
        plot=True              # Create visualizations
    )

    print(f"\nPrediction completed!")
    print(f"Time points: {results['t'].shape}")
    print(f"Predicted coefficients: {results['c_pred'].shape}")
    if 'u_pred' in results:
        print(f"Reconstructed spatial solution: {results['u_pred'].shape}")
    
    # --- build reference surface with your h-Galerkin solver ---
    n = ds.K
    hz = 0.1
    ht = hz**2
    Tmax = 2.0

    def x0(q):
        return hermit(0, q)

    z_vals, t_vals, X_hgal = num_approx_hgalerkin(x0, n=n, hz=hz, ht=ht, Tmax=Tmax)

    # --- predict at those times and plot using plot_sim_result ---
    results = predict_and_plot_vs_reference_surface(
        func=func,
        ds=ds,
        ic_idx=0,
        t_vals=t_vals,
        z_vals=z_vals,
        X_ref=X_hgal,
        method="dopri5",
        notebook_plot=False,  # True if in notebook
    )