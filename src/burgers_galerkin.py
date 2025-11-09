import numpy as np
from scipy.integrate import quad
from scipy.special import eval_hermite, factorial
from scipy.linalg import inv

def hermit(k,y):
    return 1/np.sqrt(2**k*factorial(k)*np.sqrt(np.pi))*np.exp(-y**2/2)*eval_hermite(k,y)

def compute_A(n):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = -(i + 0.5)
        if i >= 2:
            A[i, i-2] = np.sqrt(i / 2.0)
        if i + 2 < n:
            A[i, i+2] = np.sqrt((i + 1) / 2.0)
    return A

def compute_Bfull(n):
    Bfull = np.zeros((n, n, n))

    def Bfull_integrand(i, j, k, y):
        if i == 0:
            dphi_i = -y * hermit(0, y)
        else:
            dphi_i = (np.sqrt(i/2)   * hermit(i-1, y)
                    - np.sqrt((i+1)/2) * hermit(i+1, y))
        return hermit(k, y) * hermit(j, y) * dphi_i

    for i in range(n):
        for j in range(n):
            for k in range(j+1):
                num, _ = quad(lambda y:Bfull_integrand(i,j,k,y), -np.inf, np.inf, limit=200)
                Bfull[i,j,k] = num
                Bfull[i,k,j] = num

    return Bfull

def compute_Bn(Bfull, x, n):
    Bn = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            Bn[i,j] = 0.5*np.dot(x, Bfull[i,j,:])
    
    return Bn

            
def num_approx_hermite_galerkin(x0, n, hz, ht, Tmax):

    # -------- grid --------
    z_vals = np.arange(-7.0, 7.0 + hz, hz)
    t_vals = np.arange(0.0, Tmax + ht, ht)
    Z, T = np.meshgrid(z_vals, t_vals)
    X_gal = np.zeros_like(Z, dtype=float)
    X_coefs = np.zeros((len(t_vals), n), dtype=float)

    E = np.eye(n)
    An = compute_A(n)
    Bfull = compute_Bfull(n)

    def compute_X_coefs(i):
        if i == 0:
            def x0_hk_integrand(k, y):
                return x0(y)*hermit(k,y)
            for k in range(n):
                num, _ = quad(lambda y:x0_hk_integrand(k,y), -np.inf, np.inf, limit=200)
                X_coefs[i,k] = num
        else:
            Bn = compute_Bn(Bfull, X_coefs[i-1,:], n)
            M = inv(E - ht * An - ht * Bn)
            X_coefs[i,:] = M @ X_coefs[i-1,:]


    for i, t in enumerate(t_vals):
        compute_X_coefs(i)
        for j, z in enumerate(z_vals):
            val = 0
            for k in range(n):
                val += X_coefs[i,k]*hermit(k,z)
            X_gal[i, j] = val

    return z_vals, t_vals, X_gal


if __name__ == '__main__':

    from plot_utils import plot_sim_result

    # -------- parameters --------
    n = 15           # dim sub vector space of projection
    hz   = 0.1      # spatial step
    ht   = 0.05     # time step
    Tmax = 5   # final time

    # -------- initial profile x0 --------
    def x0(q):
        """Initial condition x0(q)"""
        return  np.exp(-q**2)

    z_vals, t_vals, X_gal = num_approx_hermite_galerkin(x0, n, hz, ht, Tmax)

    # interactive surface plot (x, t, M)
    plot_sim_result(z_vals, t_vals, X_gal, notebook_plot=False)
