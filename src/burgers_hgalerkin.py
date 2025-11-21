import numpy as np
from scipy.integrate import quad
from scipy.special import eval_hermite, factorial
from scipy.linalg import inv
from homogeneity_utils import norm_d, dilation, dilation_n, compute_Gdn, radial_angular_decomp


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

def Pi_n(x,n):
    coefs = np.zeros(n)
    for k in range(n):
        num, _ = quad(lambda y:x(y)*hermit(k,y), -np.inf, np.inf, limit=200)
        coefs[k] = num
    return coefs

def compute_alpha(Phi, An, Bn, Gdn):
    num = Phi @ (An+Bn) @ Phi.T
    den = Phi @ (Gdn) @ Phi.T
    return num/den
            
def num_approx_hgalerkin(x0, n, hz, ht, Tmax):
    nu = 2

    # -------- grid --------
    z_vals = np.arange(-7.0, 7.0 + hz, hz)
    t_vals = np.arange(0.0, Tmax + ht, ht)
    Z, T = np.meshgrid(z_vals, t_vals)
    X_gal = np.zeros_like(Z, dtype=float)
    
    Phi_coefs = np.zeros((len(t_vals), n), dtype=float)
    r_coefs = np.zeros((len(t_vals)), dtype=float)

    E = np.eye(n)
    An = compute_A(n)
    Bfull = compute_Bfull(n)
    Gdn = compute_Gdn(n)

    def compute_coefs(i):
        if i == 0:
            norm_d_x0 = norm_d(x0)
            r_coefs[0] = norm_d_x0
            Phi_coefs[0,:] = Pi_n(dilation(-np.log(norm_d_x0), x0),n)
        else:
            Phi_p = Phi_coefs[i-1,:]
            Bn = compute_Bn(Bfull, Phi_p, n)
            alpha = compute_alpha(Phi_p, An, Bn, Gdn)
            r_p = r_coefs[i-1]
            

            M1 = inv(E - ht*(r_p**nu)*(An + Bn - alpha*Gdn))
            Phi = M1 @ Phi_p

            r =  r_p/(1-ht*(r_p**nu)*alpha)

            Phi_coefs[i,:] = Phi/np.linalg.norm(Phi)
            r_coefs[i] = r


    for i, t in enumerate(t_vals):
        compute_coefs(i)
        for j, z in enumerate(z_vals):
            val = 0
            assert r_coefs[i] > 0, f"at time step {i}, r_coef_1 = {r_coefs[i]} <= 0"

            X = dilation_n(Gdn, np.log(r_coefs[i]), Phi_coefs[i,:])

            for k in range(n):
                val += X[k]*hermit(k,z)

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

    z_vals, t_vals, X_hgal = num_approx_hgalerkin(x0, n, hz, ht, Tmax)

    # interactive surface plot (x, t, M)
    plot_sim_result(z_vals, t_vals, X_hgal, 'x_hgal', notebook_plot=False)
