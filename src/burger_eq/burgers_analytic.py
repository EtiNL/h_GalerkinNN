import math
import torch

class BurgersParallelSolver:
    """
    Optimized parallel Burgers equation solver using PyTorch.
    Supports batch processing of multiple initial conditions with GPU acceleration.
    """
    
    def __init__(self, device: str = None):
        """
        Initialize solver.
        
        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Initialized BurgersParallelSolver on device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    @staticmethod
    @torch.no_grad()
    def _interp_uniform_table(table: torch.Tensor, q: torch.Tensor, q0: float, dq: float) -> torch.Tensor:
        """
        table: (N, nq) sampled at q_grid[i]=q0+i*dq
        q: (...,) query points
        returns: (N, *q.shape)
        """
        N, nq = table.shape
        pos = (q - q0) / dq
        i0 = torch.floor(pos).to(torch.long).clamp(0, nq - 2)
        frac = (pos - i0.to(pos.dtype)).clamp(0.0, 1.0)

        i0f = i0.reshape(-1)                      # (M,)
        fracf = frac.reshape(-1)                  # (M,)

        idx = i0f.unsqueeze(0).expand(N, -1)      # (N,M)
        v0 = torch.gather(table, 1, idx)
        v1 = torch.gather(table, 1, idx + 1)
        out = v0 * (1.0 - fracf) + v1 * fracf     # (N,M)

        return out.view(N, *q.shape)

    @torch.no_grad()
    def _tables_from_callables(
        self,
        x0_list,
        G_list,
        q_grid: torch.Tensor,
        compute_G_if_missing: bool = True,
        shift_G0_to_zero: bool = False,
    ):
        """
        Builds x0_table, G_table on q_grid.

        - Evaluates each callable ONCE on the full grid (vectorized in q).
        - Loops over initial conditions only during this table construction stage.
          The solver itself is fully batched across ICs.

        Returns: (x0_table, G_table, q0, dq)
        """
        device = self.device
        dtype = q_grid.dtype
        q_np = q_grid.detach().cpu().numpy()

        # x0_table
        x0_rows = []
        for f in x0_list:
            # try torch path
            try:
                y = f(q_grid)
                if isinstance(y, torch.Tensor):
                    y_t = y.to(device=device, dtype=dtype)
                else:
                    raise TypeError
            except Exception:
                y_np = f(q_np)
                y_t = torch.as_tensor(np.asarray(y_np), device=device, dtype=dtype)
            x0_rows.append(y_t.reshape(-1))
        x0_table = torch.stack(x0_rows, dim=0)  # (N, nq)

        # G_table
        if G_list is not None:
            assert len(G_list) == len(x0_list)
            G_rows = []
            for f in G_list:
                try:
                    y = f(q_grid)
                    if isinstance(y, torch.Tensor):
                        y_t = y.to(device=device, dtype=dtype)
                    else:
                        raise TypeError
                except Exception:
                    y_np = f(q_np)
                    y_t = torch.as_tensor(np.asarray(y_np), device=device, dtype=dtype)
                G_rows.append(y_t.reshape(-1))
            G_table = torch.stack(G_rows, dim=0)
        else:
            if not compute_G_if_missing:
                raise ValueError("G_list is None and compute_G_if_missing=False.")
            dq = float((q_grid[1] - q_grid[0]).item())
            incr = 0.5 * (x0_table[:, 1:] + x0_table[:, :-1]) * dq
            G_table = torch.cat(
                [torch.zeros((x0_table.shape[0], 1), device=device, dtype=dtype),
                 torch.cumsum(incr, dim=1)],
                dim=1,
            )

        if shift_G0_to_zero:
            # subtract G(0) row-wise using interpolation on the uniform grid
            q0 = float(q_grid[0].item())
            dq = float((q_grid[1] - q_grid[0]).item())
            G0 = self._interp_uniform_table(G_table, torch.tensor(0.0, device=device, dtype=dtype), q0=q0, dq=dq)
            G_table = G_table - G0[:, 0:1]

        q0 = float(q_grid[0].item())
        dq = float((q_grid[1] - q_grid[0]).item())
        return x0_table, G_table, q0, dq

    @torch.no_grad()
    def _solve_from_tables(
        self,
        x0_table: torch.Tensor,   # (N, nq)
        G_table: torch.Tensor,    # (N, nq)
        q0: float,
        dq: float,
        hz: float,
        ht: float,
        Tmax: float,
        z_range=(-7.0, 7.0),
        L: float = 6.0,
        n_quad_points: int = 200,
        P: torch.Tensor = None,   # (nz, K) projection matrix, optional
        z_batch_size: int = 4096,
    ):
        device = self.device
        dtype = x0_table.dtype

        z = torch.arange(z_range[0], z_range[1] + 0.5*hz, hz, device=device, dtype=dtype)
        t = torch.arange(0.0, Tmax + 0.5*ht, ht, device=device, dtype=dtype)
        nz = z.numel()
        nT = t.numel()
        N = x0_table.shape[0]

        y = torch.linspace(-L, L, n_quad_points, device=device, dtype=dtype)
        dy = (y[1] - y[0]).item()
        w_y = torch.ones_like(y)
        w_y[0] = 0.5
        w_y[-1] = 0.5
        y_sq = y * y

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
                z_b = z[zs:ze]  # (bz,)
                bz = z_b.numel()

                if tt.item() == 0.0:
                    u_b = self._interp_uniform_table(x0_table, z_b, q0=q0, dq=dq)  # (N,bz)
                else:
                    a = 2.0 * torch.sqrt(tt)
                    q = z_b.view(bz, 1) + a * y.view(1, -1)  # (bz,nq)

                    x0_q = self._interp_uniform_table(x0_table, q, q0=q0, dq=dq)  # (N,bz,nq)
                    G_q  = self._interp_uniform_table(G_table,  q, q0=q0, dq=dq)  # (N,bz,nq)

                    exp_term = torch.exp(-y_sq.view(1,1,-1) - 0.5 * G_q)           # (N,bz,nq)
                    num = torch.sum(x0_q * exp_term * w_y.view(1,1,-1), dim=2) * dy
                    den = torch.sum(exp_term       * w_y.view(1,1,-1), dim=2) * dy
                    u_b = num / den                                                # (N,bz)

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
    ):
        """
        Public API: takes lists of callables like your old solve_parallel, but
        converts them to tables first, then solves fully batched across ICs.

        Note: If your lambdas capture loop variables, use i=i (see example below).
        """
        assert len(x0_list) > 0
        if G_list is not None:
            assert len(G_list) == len(x0_list)

        # q grid wide enough for q = z + 2*sqrt(t)*y, y in [-L,L]
        q_min = z_range[0] - 2.0 * math.sqrt(Tmax) * L
        q_max = z_range[1] + 2.0 * math.sqrt(Tmax) * L
        q_grid = torch.linspace(q_min, q_max, q_n, device=self.device, dtype=torch.float32)

        x0_table, G_table, q0, dq = self._tables_from_callables(
            x0_list=x0_list,
            G_list=G_list,
            q_grid=q_grid,
            compute_G_if_missing=compute_G_if_missing,
            shift_G0_to_zero=shift_G0_to_zero,
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
        )