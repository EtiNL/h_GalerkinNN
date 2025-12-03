import numpy as np
from scipy.integrate import quad
from tqdm import tqdm

def num_approx_burgers(x0, G, hz, ht, Tmax, z_range = (-7.0, 7.0), L=np.inf):

    assert z_range[0] < z_range[1], f"z_range not defined correctly: {z_range}"

    # -------- grid --------
    z_vals = np.arange(z_range[0], z_range[1] + hz, hz)
    t_vals = np.arange(0.0, Tmax + ht, ht)
    Z, T   = np.meshgrid(z_vals, t_vals)
    X_tr   = np.zeros_like(Z, dtype=float)

    def x_of_tz(z, t, L=6.0):
        """Compute x(t,z) via
        x(t,z) = ∫ x0(z+2√t y) e^{-y^2 - 0.5 G(z+2√t y)} dy
                    / ∫ e^{-y^2 - 0.5 G(z+2√t y)} dy,
        integrating y on [-L, L].
        """
        if t <= 0.0:
            return x0(z)   # limit t -> 0

        sqrt_t = np.sqrt(t)

        def num_integrand(y):
            q = z + 2.0 * sqrt_t * y
            return np.exp(-y**2 - 0.5 * G(q)) * x0(q)

        def den_integrand(y):
            q = z + 2.0 * sqrt_t * y
            return np.exp(-y**2 - 0.5 * G(q))

        num, _ = quad(num_integrand, -L, L, limit=200)
        den, _ = quad(den_integrand, -L, L, limit=200)

        return num / den

    # -------- compute X_tr on the (t,z) grid --------
    for i, t in tqdm(enumerate(t_vals), desc="Computing exact Burgers soluton..."):
        for j, z in enumerate(z_vals):
            X_tr[i, j] = x_of_tz(z, t)

    return z_vals, t_vals, X_tr

if __name__ == '__main__':

    from plot_utils import plot_sim_result


    # -------- parameters --------
    hz   = 0.1      # spatial step
    ht   = 0.05     # time step
    Tmax = 5      # final time

    # -------- initial profile x0 and its primitive G --------
    def x0(q):
        """Initial condition x0(q)"""
        return  np.sin(q)

    def G(q):
        """Primitive of x0: G'(q) = x0(q).
        """
        return 1 - np.cos(q)

    z_vals, t_vals, X_tr = num_approx_burgers(x0, G, hz, ht, Tmax)

    # -------- interactive surface plot (Plotly) --------
    plot_sim_result(z_vals, t_vals, X_tr, notebook_plot=False)
