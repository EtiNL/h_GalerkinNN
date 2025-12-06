import math
import numpy as np
import torch


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
        L: float = 6.0,
        n_quad_points: int = 200,
        P: torch.Tensor = None,
        z_batch_size: int = 4096,
        x0_z_table: torch.Tensor | None = None,
        den_eps: float = 1e-30,
        interp: str = "cubic",
    ):
        device = self.device
        dtype = x0_table.dtype

        z = torch.arange(z_range[0], z_range[1] + 0.5 * hz, hz, device=device, dtype=dtype)
        t = torch.arange(0.0, Tmax + 0.5 * ht, ht, device=device, dtype=dtype)
        nz = z.numel()
        nT = t.numel()
        N = x0_table.shape[0]

        y = torch.linspace(-L, L, n_quad_points, device=device, dtype=dtype)
        dy = (y[1] - y[0])
        w_y = torch.ones_like(y)
        w_y[0] = 0.5
        w_y[-1] = 0.5
        y_sq = (y * y).view(1, 1, -1)

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
                    q = z_b.view(bz, 1) + a * y.view(1, -1)

                    x0_q = self._interp_uniform_table(x0_table, q, q0=q0, dq=dq, interp=interp)
                    G_q  = self._interp_uniform_table(G_table,  q, q0=q0, dq=dq, interp=interp)

                    logw = -(y_sq + 0.5 * G_q)                      # (N,bz,nq)
                    m = torch.max(logw, dim=2, keepdim=True).values # (N,bz,1)
                    w = torch.exp(logw - m)                         # scaled weights

                    wy = w_y.view(1, 1, -1)
                    num = torch.sum(x0_q * w * wy, dim=2) * dy
                    den = torch.sum(w * wy, dim=2) * dy
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
        L: float = 6.0,
        n_quad_points: int = 200,
        q_n: int = 8192,
        P: torch.Tensor = None,
        z_batch_size: int = 4096,
        compute_G_if_missing: bool = True,
        shift_G0_to_zero: bool = False,
        enforce_exact_ic: bool = True,
        interp: str = "cubic",   # NEW: "cubic" (recommended) or "linear"
    ):
        assert len(x0_list) > 0
        if G_list is not None:
            assert len(G_list) == len(x0_list)

        q_min = z_range[0] - 2.0 * math.sqrt(Tmax) * L
        q_max = z_range[1] + 2.0 * math.sqrt(Tmax) * L
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