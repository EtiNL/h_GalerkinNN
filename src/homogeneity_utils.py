from scipy.linalg import expm 
from scipy.integrate import quad
import numpy as np

def dilation(s,x):
    return lambda z: np.exp(s)*x(np.exp(s)*z)

def norm_h(x, tol=1e-4):
    num, _ = quad(lambda z: x(z)**2, -np.inf, np.inf, limit=200)

    # clamp small negative values due to numerical noise
    if num < 0:
        if num > -tol:
            num = 0.0
        else:
            raise ValueError(f"norm_h integral is significantly negative: {num}")

    return np.sqrt(num)

def norm_d(x, max_steps = 200, eps = 0.001):
    s_inf = -3.0
    s_sup = 3.0
    s_mid = 0.5*(s_sup + s_inf)

    n_inf = norm_h(dilation(-s_inf, x))
    n_sup = norm_h(dilation(-s_sup, x))
    n_mid = norm_h(dilation(-s_mid, x))

    assert n_sup < 1.0, f"n_sup = {n_sup} > 1.0"
    assert n_inf > 1.0, f"n_inf = {n_inf} < 1.0"
    

    steps = 0
    while (np.abs(1.0 - n_mid) > eps) and steps < max_steps:
        s_mid = 0.5*(s_sup + s_inf)
        n_mid = norm_h(dilation(-s_mid, x))

        if n_mid < 1.0:
            n_sup = n_mid
            s_sup = s_mid
        
        else:
            n_inf = n_mid
            s_inf = s_mid
        
        steps += 1

    assert steps < max_steps, "max norm_d steps reached"
    
    return np.exp(s_mid)

#--------------------------------Finite dimensional analogues-------------------------------------

def norm_s(x):
    P=np.eye(x.shape[0])
    n_squared = x.T @ P @ x
    assert n_squared >= 0, "norm_s negative"
    return np.sqrt(n_squared)

def compute_Gdn(n):
    """
    Compute the generator matrix Gdn for the dilation group
    in the finite-dimensional Hermite basis projection.
    
    For the Burgers equation with dilation d(s)x(z) = e^s x(e^s z),
    we have α=1, β=1, giving:
    - Diagonal: (2α-β)/2 = 0.5
    - Upper off-diagonal (i, i+2): β√((i+1)/2) = √((i+1)/2)
    - Lower off-diagonal (i, i-2): -β√(i/2) = -√(i/2)
    
    This matches equation (4.11) in the article.
    """
    Gdn = np.zeros((n, n))
    for i in range(n):
        Gdn[i, i] = 0.5
        
        if i >= 2:
            Gdn[i, i-2] = -np.sqrt(i / 2.0)
        
        if i + 2 < n:
            Gdn[i, i+2] = np.sqrt((i + 1) / 2.0)
    
    test_phi = np.random.randn(n)
    test_phi = test_phi / np.linalg.norm(test_phi)
    if not np.isclose(test_phi @ Gdn @ test_phi,0.5): 
        print(f"Phi @ Gdn @ Phi.T = {test_phi @ Gdn @ test_phi}!=0.5")
    return Gdn

def dilation_n(Gdn, s, x):
    return expm(s*Gdn)@x

def norm_dn(Gdn, x, max_steps = 200, eps = 0.01):
    s_inf = -3.0
    s_sup = 3.0
    s_mid = 0.5*(s_sup + s_inf)

    n_inf = norm_s(dilation_n(Gdn, -s_inf, x))
    n_sup = norm_s(dilation_n(Gdn, -s_sup, x))
    n_mid = norm_s(dilation_n(Gdn, -s_mid, x))

    assert n_sup < 1.0, f"n_sup = {n_sup} > 1.0"
    assert n_inf > 1.0, f"n_inf = {n_inf} < 1.0"

    steps = 0
    while (np.abs(1.0 - n_mid) > eps) and steps < max_steps:
        s_mid = 0.5*(s_sup + s_inf)
        n_mid = norm_s(dilation_n(Gdn, -s_mid, x))

        if n_mid < 1.0:
            n_sup = n_mid
            s_sup = s_mid
        
        else:
            n_inf = n_mid
            s_inf = s_mid
        
        steps += 1
    
    return np.exp(s_mid)

def radial_angular_decomp(Gdn, x):
    nd = norm_dn(Gdn, x.T)
    Pi_dn = (dilation_n(Gdn, -np.log(nd), x.T)).T

    return nd, Pi_dn
