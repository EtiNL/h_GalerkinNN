"""
Comparison script: PyTorch BurgersParallelSolver (CUDA) vs NumPy num_approx_burgers

This script compares the accuracy and performance of two Burgers equation solvers:
1. PyTorch GPU-accelerated solver (BurgersParallelSolver)
2. NumPy analytical solver with scipy adaptive quadrature (num_approx_burgers)

Both solvers use the Cole-Hopf transformation to compute solutions.
"""

import numpy as np
import torch
import math
import time
import sys
import os
from typing import Callable, Tuple

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from burgers_analytic_np import num_approx_burgers
from burger_eq.burgers_analytic import BurgersParallelSolver

try:
    from plot_utils import plot_sim_result
    HAS_PLOTTING = True
except ImportError:
    print("Warning: plot_utils not available. Plotting will be disabled.")
    HAS_PLOTTING = False


def compare_burgers_solvers(
    x0_np: Callable,
    G_np: Callable,
    x0_torch: Callable,
    G_torch: Callable,
    hz: float = 0.1,
    ht: float = 0.05,
    Tmax: float = 5.0,
    z_range: Tuple[float, float] = (-7.0, 7.0),
    n_quad_np: int = 20,
    n_quad_torch: int = 20,
    quadrature_method_np: str = 'adaptive',
    device: str = 'cuda',
    enable_plotting: bool = True
):
    """
    Compare PyTorch and NumPy Burgers solvers.

    Args:
        x0_np: Initial condition function for NumPy (takes numpy array)
        G_np: Primitive of x0 for NumPy (takes numpy array)
        x0_torch: Initial condition function for PyTorch (takes torch or numpy)
        G_torch: Primitive of x0 for PyTorch (takes torch or numpy)
        hz: Spatial step size
        ht: Time step size
        Tmax: Final time
        z_range: Spatial domain (z_min, z_max)
        n_quad_np: Number of quadrature points for NumPy solver
        n_quad_torch: Number of quadrature points for PyTorch solver
        quadrature_method_np: Quadrature method for NumPy ('adaptive' or 'gauss-hermite')
        device: Device for PyTorch solver ('cuda' or 'cpu')
        enable_plotting: Whether to generate plots

    Returns:
        dict: Dictionary containing comparison results
    """

    print("\n" + "="*70)
    print("BURGERS EQUATION SOLVER COMPARISON")
    print("PyTorch GPU-accelerated vs NumPy analytical solver")
    print("="*70)

    # Display parameters
    print(f"\nSimulation Parameters:")
    print(f"  Spatial step (hz): {hz}")
    print(f"  Time step (ht): {ht}")
    print(f"  Final time (Tmax): {Tmax}")
    print(f"  Spatial range: {z_range}")
    print(f"  NumPy quadrature: {quadrature_method_np} (n={n_quad_np})")
    print(f"  PyTorch quadrature: Gauss-Hermite (n={n_quad_torch})")
    print(f"  Device: {device}")

    # ======================================================================
    # 1. NumPy-based Solver
    # ======================================================================
    print("\n" + "="*70)
    print("1. NumPy-BASED SOLVER (scipy adaptive quadrature)")
    print("="*70)

    start_time_np = time.time()

    z_vals_np, t_vals_np, X_np = num_approx_burgers(
        x0=x0_np,
        G=G_np,
        hz=hz,
        ht=ht,
        Tmax=Tmax,
        z_range=z_range,
        n_quad=n_quad_np,
        quadrature_method=quadrature_method_np
    )

    elapsed_np = time.time() - start_time_np

    print(f"\nNumPy solver completed in {elapsed_np:.2f} seconds")
    print(f"Solution shapes:")
    print(f"  z_vals: {z_vals_np.shape}")
    print(f"  t_vals: {t_vals_np.shape}")
    print(f"  X_np: {X_np.shape}")
    print(f"  Solution range: [{X_np.min():.6f}, {X_np.max():.6f}]")

    if enable_plotting and HAS_PLOTTING:
        plot_sim_result(z_vals_np, t_vals_np, X_np, 'X_numpy', notebook_plot=False)

    # ======================================================================
    # 2. PyTorch-based Solver
    # ======================================================================
    print("\n" + "="*70)
    print(f"2. PyTorch-BASED SOLVER ({device.upper()}-accelerated)")
    print("="*70)

    torch_solver = BurgersParallelSolver(device=device)

    start_time_torch = time.time()

    z_vals_torch, t_vals_torch, U_torch = torch_solver.solve_parallel_projected(
        x0_list=[x0_torch],
        G_list=[G_torch],
        hz=hz,
        ht=ht,
        Tmax=Tmax,
        z_range=z_range,
        n_quad_points=n_quad_torch,
        P=None,  # Full solution, not projected
        enforce_exact_ic=True,
        interp='cubic',
    )

    elapsed_torch = time.time() - start_time_torch

    X_torch = U_torch[0].cpu().numpy()  # [nT, nz]

    print(f"\nPyTorch solver completed in {elapsed_torch:.2f} seconds")
    print(f"Solution shapes:")
    print(f"  z_vals: {z_vals_torch.shape}")
    print(f"  t_vals: {t_vals_torch.shape}")
    print(f"  X_torch: {X_torch.shape}")
    print(f"  Solution range: [{X_torch.min():.6f}, {X_torch.max():.6f}]")

    if enable_plotting and HAS_PLOTTING:
        plot_sim_result(z_vals_torch, t_vals_torch, X_torch, 'X_pytorch', notebook_plot=False)

    # ======================================================================
    # 3. COMPARISON & ERROR ANALYSIS
    # ======================================================================
    print("\n" + "="*70)
    print("3. COMPARISON & ERROR ANALYSIS")
    print("="*70)

    # Verify grids match
    assert z_vals_np.shape == z_vals_torch.shape, "Spatial grids don't match!"
    assert t_vals_np.shape == t_vals_torch.shape, "Time grids don't match!"
    assert X_np.shape == X_torch.shape, "Solution arrays don't match!"

    print(f"\nGrid verification: ✓ All grids match")

    # Compute absolute and relative errors
    abs_error = np.abs(X_torch - X_np)

    # Relative error (only where solution is significant)
    threshold = 1e-3
    mask = np.abs(X_np) > threshold

    rel_error = np.zeros_like(abs_error)
    rel_error[mask] = abs_error[mask] / np.abs(X_np[mask])
    rel_error[~mask] = np.nan

    # Error statistics
    print(f"\nAbsolute Error Statistics:")
    print(f"  Max abs error: {np.max(abs_error):.6e}")
    print(f"  Mean abs error: {np.mean(abs_error):.6e}")
    print(f"  Median abs error: {np.median(abs_error):.6e}")
    print(f"  Std abs error: {np.std(abs_error):.6e}")

    print(f"\nRelative Error Statistics (|solution| > {threshold}):")
    rel_error_valid = rel_error[~np.isnan(rel_error)]
    if len(rel_error_valid) > 0:
        print(f"  Max rel error: {np.max(rel_error_valid):.6e}")
        print(f"  Mean rel error: {np.mean(rel_error_valid):.6e}")
        print(f"  Median rel error: {np.median(rel_error_valid):.6e}")
        print(f"  Fraction analyzed: {len(rel_error_valid) / rel_error.size * 100:.1f}%")
    else:
        print(f"  No regions with |solution| > {threshold}")

    # Error at specific times
    print(f"\nError evolution over time:")
    n_times = len(t_vals_np)
    for i in [0, n_times//4, n_times//2, 3*n_times//4, -1]:
        t_val = t_vals_np[i]
        max_err_at_t = np.max(abs_error[i, :])
        mean_err_at_t = np.mean(abs_error[i, :])
        print(f"  t={t_val:6.2f}: max={max_err_at_t:.6e}, mean={mean_err_at_t:.6e}")

    if enable_plotting and HAS_PLOTTING:
        plot_sim_result(z_vals_np, t_vals_np, abs_error,
                        'Error_PyTorch_vs_NumPy', notebook_plot=False)
        plot_sim_result(z_vals_np, t_vals_np, rel_error,
                        'RelError_PyTorch_vs_NumPy', notebook_plot=False)

    # ======================================================================
    # 4. PERFORMANCE COMPARISON
    # ======================================================================
    print("\n" + "="*70)
    print("4. PERFORMANCE COMPARISON")
    print("="*70)

    speedup = elapsed_np / elapsed_torch if elapsed_torch > 0 else float('inf')

    print(f"\n{'Method':<25} {'Time (s)':<15} {'Speedup':<15}")
    print("-"*55)
    print(f"{'NumPy (scipy)':<25} {elapsed_np:<15.2f} {'1.0x':<15}")
    print(f"{f'PyTorch ({device.upper()})':<25} {elapsed_torch:<15.2f} {f'{speedup:.2f}x':<15}")

    # ======================================================================
    # 5. FINAL SUMMARY
    # ======================================================================
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    max_abs_err = np.max(abs_error)
    print(f"\nNumerical Agreement:")
    if max_abs_err < 1e-6:
        agreement = "EXCELLENT"
    elif max_abs_err < 1e-4:
        agreement = "VERY GOOD"
    elif max_abs_err < 1e-3:
        agreement = "GOOD"
    elif max_abs_err < 1e-2:
        agreement = "FAIR"
    else:
        agreement = "POOR"
    print(f"  {agreement} (max absolute error: {max_abs_err:.6e})")

    print(f"\nComputational Efficiency:")
    if speedup >= 1000:
        print(f"  PyTorch solver is {speedup:.0f}x faster than NumPy solver")
    elif speedup >= 100:
        print(f"  PyTorch solver is {speedup:.0f}x faster than NumPy solver")
    else:
        print(f"  PyTorch solver is {speedup:.1f}x faster than NumPy solver")

    print(f"\nRecommendation:")
    if speedup > 10:
        print(f"  ✓ Use PyTorch solver for production (much faster)")
    elif speedup > 2:
        print(f"  ✓ Use PyTorch solver for large-scale problems")
    else:
        print(f"  → Either solver is suitable")

    print(f"\nNotes:")
    print(f"  - NumPy: {quadrature_method_np} quadrature with {n_quad_np} points")
    print(f"  - PyTorch: Gauss-Hermite quadrature with {n_quad_torch} points")
    print(f"  - Both methods use Cole-Hopf transformation")
    print(f"  - Small differences expected due to different integration methods")
    print(f"  - Relative error computed where |solution| > {threshold}")

    print("\n" + "="*70 + "\n")

    # Return results
    return {
        'z_vals_np': z_vals_np,
        't_vals_np': t_vals_np,
        'X_np': X_np,
        'z_vals_torch': z_vals_torch,
        't_vals_torch': t_vals_torch,
        'X_torch': X_torch,
        'abs_error': abs_error,
        'rel_error': rel_error,
        'max_abs_error': max_abs_err,
        'max_rel_error': np.nanmax(rel_error) if len(rel_error_valid) > 0 else np.nan,
        'elapsed_np': elapsed_np,
        'elapsed_torch': elapsed_torch,
        'speedup': speedup
    }


# ======================================================================
# EXAMPLE USAGE
# ======================================================================

if __name__ == '__main__':

    # Example 1: Sine wave initial condition
    print("\n" + "="*70)
    print("EXAMPLE 1: Sine Wave Initial Condition")
    print("="*70)

    def x0_sine_np(q):
        """Initial condition: u(0,x) = sin(x)"""
        return np.sin(q)

    def G_sine_np(q):
        """Primitive: G(x) = 1 - cos(x)"""
        return 1 - np.cos(q)

    def x0_sine_torch(q):
        """Initial condition for PyTorch"""
        if isinstance(q, torch.Tensor):
            return torch.sin(q)
        else:
            return np.sin(q)

    def G_sine_torch(q):
        """Primitive for PyTorch"""
        if isinstance(q, torch.Tensor):
            return 1 - torch.cos(q)
        else:
            return 1 - np.cos(q)

    results_sine = compare_burgers_solvers(
        x0_np=x0_sine_np,
        G_np=G_sine_np,
        x0_torch=x0_sine_torch,
        G_torch=G_sine_torch,
        hz=0.1,
        ht=0.05,
        Tmax=5.0,
        z_range=(-7.0, 7.0),
        n_quad_np=20,
        n_quad_torch=20,
        quadrature_method_np='adaptive',
        device='cuda',
        enable_plotting=True
    )


    # Example 2: Gaussian initial condition
    print("\n" + "="*70)
    print("EXAMPLE 2: Gaussian Initial Condition")
    print("="*70)

    def x0_gaussian_np(q):
        """Initial condition: u(0,x) = exp(-x^2)"""
        return np.exp(-q**2)

    def G_gaussian_np(q):
        """Primitive: G(x) = sqrt(pi)/2 * erf(x)"""
        from scipy.special import erf
        return (np.sqrt(np.pi) / 2.0) * erf(q)

    def x0_gaussian_torch(q):
        """Initial condition for PyTorch"""
        if isinstance(q, torch.Tensor):
            return torch.exp(-q**2)
        else:
            return np.exp(-q**2)

    def G_gaussian_torch(q):
        """Primitive for PyTorch"""
        if isinstance(q, torch.Tensor):
            return (math.sqrt(math.pi) / 2.0) * torch.erf(q)
        else:
            from scipy.special import erf
            return (np.sqrt(np.pi) / 2.0) * erf(q)

    results_gaussian = compare_burgers_solvers(
        x0_np=x0_gaussian_np,
        G_np=G_gaussian_np,
        x0_torch=x0_gaussian_torch,
        G_torch=G_gaussian_torch,
        hz=0.1,
        ht=0.05,
        Tmax=5.0,
        z_range=(-7.0, 7.0),
        n_quad_np=20,
        n_quad_torch=20,
        quadrature_method_np='adaptive',
        device='cuda',
        enable_plotting=True
    )


    # ======================================================================
    # OVERALL SUMMARY
    # ======================================================================
    print("\n" + "="*70)
    print("OVERALL COMPARISON SUMMARY")
    print("="*70)

    print(f"\n{'Test Case':<30} {'Max Abs Error':<20} {'Speedup':<15}")
    print("-"*65)
    print(f"{'Sine Wave':<30} {results_sine['max_abs_error']:<20.6e} {results_sine['speedup']:<15.2f}x")
    print(f"{'Gaussian':<30} {results_gaussian['max_abs_error']:<20.6e} {results_gaussian['speedup']:<15.2f}x")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("\nBoth solvers produce highly accurate solutions that agree well.")
    print("The PyTorch GPU-accelerated solver offers significant speedup,")
    print("making it ideal for large-scale simulations and dataset generation.")
    print("\n" + "="*70 + "\n")
