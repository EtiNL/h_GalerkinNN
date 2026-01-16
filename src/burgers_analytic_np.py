import numpy as np
from scipy.special import roots_hermite
from scipy.integrate import quad
from tqdm import tqdm

def num_approx_burgers(x0, G, hz, ht, Tmax, z_range = (-7.0, 7.0), n_quad=20,
                       quadrature_method='adaptive'):
    """
    Compute Burgers solution using Cole-Hopf transformation with numerical quadrature.

    The Cole-Hopf integral is:
        ∫ exp(-y²) · x0(q) · exp(-0.5·G(q)) dy
        = ∫ f(y) · exp(-y²) dy    where f(y) = x0(q) · exp(-0.5·G(q))

    Args:
        x0: Initial condition function
        G: Primitive of initial condition (∫x0 dq)
        hz: Spatial step size
        ht: Time step size
        Tmax: Final time
        z_range: Spatial domain (z_min, z_max)
        n_quad: Number of quadrature points for Gauss-Hermite (default 20, only used if method='gauss-hermite')
        quadrature_method: Quadrature method to use (default 'adaptive')
            - 'adaptive': Scipy adaptive quadrature (more accurate for complex integrands)
            - 'gauss-hermite': Gauss-Hermite quadrature (fast, exact for polynomial integrands)
    """

    assert z_range[0] < z_range[1], f"z_range not defined correctly: {z_range}"
    assert quadrature_method in ['gauss-hermite', 'adaptive'], \
        f"quadrature_method must be 'gauss-hermite' or 'adaptive', got '{quadrature_method}'"

    # -------- Setup quadrature method --------
    if quadrature_method == 'gauss-hermite':
        # For ∫ f(y) e^(-y²) dy ≈ Σ w_i · f(y_i)
        y_nodes, w_nodes = roots_hermite(n_quad)
        print(f"Using Gauss-Hermite quadrature with {n_quad} points")
        print(f"  Node range: [{y_nodes.min():.2f}, {y_nodes.max():.2f}]")
    else:  # adaptive
        print(f"Using scipy adaptive quadrature")
        print(f"  Integration limits: automatic")

    # -------- grid --------
    z_vals = np.arange(z_range[0], z_range[1] + hz, hz)
    t_vals = np.arange(0.0, Tmax + ht, ht)
    Z, T   = np.meshgrid(z_vals, t_vals)
    X_tr   = np.zeros_like(Z, dtype=float)

    def x_of_tz(z, t):
        """Compute x(t,z) via numerical quadrature.

        x(t,z) = ∫ x0(q) e^{-y² - 0.5·G(q)} dy / ∫ e^{-y² - 0.5·G(q)} dy
        where q = z + 2√t·y
        """
        if t <= 0.0:
            return x0(z)   # limit t -> 0

        sqrt_t = np.sqrt(t)

        if quadrature_method == 'gauss-hermite':
            # Gauss-Hermite: ∫ f(y) e^(-y²) dy ≈ Σ w_i · f(y_i)
            # where f(y) = x0(q) · exp(-0.5·G(q)) for numerator
            # and f(y) = exp(-0.5·G(q)) for denominator
            # IMPORTANT: We do NOT include exp(-y²) in f(y) since it's
            # already in the Gauss-Hermite weights!

            # Compute q values at Gauss-Hermite nodes
            q_vals = z + 2.0 * sqrt_t * y_nodes  # [n_quad]

            # Evaluate x0 and G at quadrature points
            x0_vals = np.array([x0(q) for q in q_vals])  # [n_quad]
            G_vals = np.array([G(q) for q in q_vals])    # [n_quad]

            # The integrand WITHOUT exp(-y²) (it's in the weights!)
            # f(y) = x0(q) · exp(-0.5·G(q))
            f_num = x0_vals * np.exp(-0.5 * G_vals)  # [n_quad]
            f_den = np.exp(-0.5 * G_vals)             # [n_quad]

            # Gauss-Hermite integration: ∫ f(y) e^(-y²) dy ≈ Σ w_i · f(y_i)
            num = np.sum(w_nodes * f_num)
            den = np.sum(w_nodes * f_den)

        else:  # adaptive
            # Scipy adaptive quadrature: ∫ f(y) dy from -∞ to +∞
            # Here we INCLUDE exp(-y²) in the integrand

            def integrand_num(y):
                """Numerator: x0(q) · exp(-y² - 0.5·G(q))"""
                q = z + 2.0 * sqrt_t * y
                return x0(q) * np.exp(-y**2 - 0.5 * G(q))

            def integrand_den(y):
                """Denominator: exp(-y² - 0.5·G(q))"""
                q = z + 2.0 * sqrt_t * y
                return np.exp(-y**2 - 0.5 * G(q))

            # Integrate from -∞ to +∞
            num, _ = quad(integrand_num, -np.inf, np.inf)
            den, _ = quad(integrand_den, -np.inf, np.inf)

        return num / den

    # -------- compute X_tr on the (t,z) grid --------
    for i, t in tqdm(enumerate(t_vals), desc="Computing exact Burgers solution..."):
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