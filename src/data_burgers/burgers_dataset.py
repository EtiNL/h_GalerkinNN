"""
Burgers Equation Dataset Generator

This submodule generates datasets for different neural network architectures
from the analytic/numerical solution of the viscous Burgers equation.

The Burgers equation:
    ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²

Uses the Cole-Hopf transformation for the analytic solution approximation.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Callable, Tuple, Optional, Dict, Any, Generator
from dataclasses import dataclass
import sys
import os
from scipy.special import eval_hermite, factorial
from pde_dataset import create_time_projected_dataset, TimeProjectedDataset

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Burgers solver from existing module
from burgers_analytic import num_approx_burgers

# Import from pde_dataset package (adjust path as needed)
try:
    from pde_dataset import (
        DomainBounds,
        SamplingDistributions,
        Normalizer,
        # PINN
        PINNDatasetConfig,
        PINNDataset,
        create_pinn_dataset,
        # Sequential
        SequentialDatasetConfig,
        SequentialPDEDataset,
        create_sequential_dataset_from_array,
        # Galerkin
        GalerkinDatasetConfig,
        GalerkinDataset,
        create_galerkin_dataset,
        # Time_sampled_projection
        create_time_projected_dataset,
        TimeProjectedDataset,
        # IO
        save_dataset,
        load_dataset,
        DatasetMetadata,
    )
except ImportError:
    raise Exception("Coudn't load pde_dataset submodules")


# =============================================================================
# Predefined Initial Conditions
# =============================================================================

@dataclass
class BurgersInitialCondition:
    """Container for Burgers initial condition and its primitive."""
    name: str
    x0: Callable[[float], float]  # Initial condition
    G: Callable[[float], float]   # Primitive (∫x0 dq)
    description: str = ""


class BurgersInitialConditions:
    """Collection of standard initial conditions for Burgers equation."""
    
    @staticmethod
    def sine() -> BurgersInitialCondition:
        """Sinusoidal initial condition: u(0,x) = sin(x)"""
        return BurgersInitialCondition(
            name="sine",
            x0=lambda q: np.sin(q),
            G=lambda q: 1 - np.cos(q),
            description="Sinusoidal: u(0,x) = sin(x)"
        )
    
    @staticmethod
    def gaussian(amplitude: float = 1.0, center: float = 0.0, 
                 width: float = 1.0) -> BurgersInitialCondition:
        """Gaussian initial condition."""
        def x0(q):
            return amplitude * np.exp(-((q - center) / width) ** 2)
        
        def G(q):
            # Approximate integral of Gaussian
            from scipy.special import erf
            return amplitude * width * np.sqrt(np.pi) / 2 * (
                erf((q - center) / width) + 1
            )
        
        return BurgersInitialCondition(
            name=f"gaussian_a{amplitude}_c{center}_w{width}",
            x0=x0,
            G=G,
            description=f"Gaussian: A={amplitude}, center={center}, width={width}"
        )
    
    @staticmethod
    def step(x_left: float = 1.0, x_right: float = 0.0, 
             transition: float = 0.0) -> BurgersInitialCondition:
        """Smoothed step function (tanh)."""
        steepness = 10.0
        
        def x0(q):
            return 0.5 * (x_left + x_right) + 0.5 * (x_left - x_right) * np.tanh(
                -steepness * (q - transition)
            )
        
        def G(q):
            return 0.5 * (x_left + x_right) * q - 0.5 * (x_left - x_right) / steepness * np.log(
                np.cosh(steepness * (q - transition))
            )
        
        return BurgersInitialCondition(
            name=f"step_{x_left}_{x_right}",
            x0=x0,
            G=G,
            description=f"Step: left={x_left}, right={x_right}"
        )
    
    @staticmethod
    def n_wave(amplitude: float = 1.0) -> BurgersInitialCondition:
        """N-wave initial condition (antisymmetric)."""
        def x0(q):
            return amplitude * q * np.exp(-q**2)
        
        def G(q):
            return -amplitude * 0.5 * np.exp(-q**2)
        
        return BurgersInitialCondition(
            name=f"n_wave_a{amplitude}",
            x0=x0,
            G=G,
            description=f"N-wave: amplitude={amplitude}"
        )
    
    @staticmethod
    def custom(x0: Callable, G: Callable, name: str = "custom") -> BurgersInitialCondition:
        """Create custom initial condition."""
        return BurgersInitialCondition(
            name=name,
            x0=x0,
            G=G,
            description=f"Custom initial condition: {name}"
        )


# =============================================================================
# Burgers Solution Wrapper
# =============================================================================

class BurgersSolution:
    """
    Wrapper class for Burgers equation solution.
    
    Provides interpolated access to the solution for arbitrary (t, x) points.
    Uses num_approx_burgers from src/burgers_analytic.py
    """
    
    def __init__(
        self,
        initial_condition: BurgersInitialCondition,
        hz: float = 0.1,
        ht: float = 0.05,
        Tmax: float = 5.0,
        z_range: Tuple[float, float] = (-7.0, 7.0),
        L: float = 6.0,
        compute_on_init: bool = True
    ):
        """
        Initialize Burgers solution.
        
        Args:
            initial_condition: Initial condition specification
            hz: Spatial step size
            ht: Time step size
            Tmax: Final time
            z_range: Spatial domain (used for bounds, actual grid is from burgers_analytic)
            L: Integration limit for Cole-Hopf quadrature
            compute_on_init: Whether to compute solution immediately
        """
        self.ic = initial_condition
        self.hz = hz
        self.ht = ht
        self.Tmax = Tmax
        self.z_range = z_range
        self.L = L
        
        self._z_vals = None
        self._t_vals = None
        self._U = None
        self._interpolator = None
        
        if compute_on_init:
            self.compute()
    
    def compute(self) -> 'BurgersSolution':
        """Compute the solution on the grid using burgers_analytic.num_approx_burgers."""
        self._z_vals, self._t_vals, self._U = num_approx_burgers(
            x0=self.ic.x0,
            G=self.ic.G,
            hz=self.hz,
            ht=self.ht,
            Tmax=self.Tmax,
            L=self.L
        )
        
        # Update z_range from actual computed values
        self.z_range = (self._z_vals.min(), self._z_vals.max())
        
        # Create interpolator for arbitrary (t, x) evaluation
        self._interpolator = RegularGridInterpolator(
            (self._t_vals, self._z_vals),
            self._U,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        
        return self
    
    @property
    def z_vals(self) -> np.ndarray:
        if self._z_vals is None:
            self.compute()
        return self._z_vals
    
    @property
    def t_vals(self) -> np.ndarray:
        if self._t_vals is None:
            self.compute()
        return self._t_vals
    
    @property
    def U(self) -> np.ndarray:
        """Solution array U[i,j] = u(t_vals[i], z_vals[j])"""
        if self._U is None:
            self.compute()
        return self._U
    
    @property
    def bounds(self) -> DomainBounds:
        """Get domain bounds."""
        return DomainBounds(
            t_min=0.0,
            t_max=self.Tmax,
            x_min=self.z_range[0],
            x_max=self.z_range[1],
            spatial_dim=1
        )
    
    def __call__(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Evaluate solution at arbitrary (t, x) points.
        
        Args:
            t: Time coordinates (can be scalar or array)
            x: Spatial coordinates (can be scalar or array)
            
        Returns:
            Solution values u(t, x)
        """
        if self._interpolator is None:
            self.compute()
        
        t = np.atleast_1d(t)
        x = np.atleast_1d(x)
        
        # Handle broadcasting
        if t.shape != x.shape:
            t, x = np.broadcast_arrays(t, x)
        
        points = np.column_stack([t.flatten(), x.flatten()])
        result = self._interpolator(points)
        
        return result.reshape(t.shape)
    
    def get_meshgrid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get solution as meshgrid arrays (T, X, U)."""
        T, X = np.meshgrid(self.t_vals, self.z_vals, indexing='ij')
        return T, X, self.U
    
    def generator(self) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Yield solution as a generator (for compatibility with pde_dataset).
        
        Yields:
            Tuples of (t, x, u) for each time step
        """
        for i, t in enumerate(self.t_vals):
            yield (
                np.full_like(self.z_vals, t),
                self.z_vals.copy(),
                self.U[i, :].copy()
            )


# =============================================================================
# Dataset Factories for Each Architecture
# =============================================================================

def create_burgers_pinn_dataset(
    initial_condition: BurgersInitialCondition = None,
    hz: float = 0.1,
    ht: float = 0.05,
    Tmax: float = 5.0,
    L: float = 6.0,
    n_collocation: int = 10000,
    n_boundary: int = 2000,
    n_initial: int = 2000,
    t_distribution: Callable = None,
    x_distribution: Callable = None,
    device: str = 'cpu',
    seed: Optional[int] = None,
    precomputed_solution: Optional[BurgersSolution] = None
) -> PINNDataset:
    """
    Create a PINN dataset from Burgers equation solution.
    
    Args:
        initial_condition: Initial condition (default: sine)
        hz, ht: Grid spacing for solution computation
        Tmax: Final time
        L: Integration limit for Cole-Hopf quadrature
        n_collocation: Number of interior collocation points
        n_boundary: Number of boundary points
        n_initial: Number of initial condition points
        t_distribution: Sampling distribution for time
        x_distribution: Sampling distribution for space
        device: Device for tensors
        seed: Random seed
        precomputed_solution: Use existing solution instead of computing
        
    Returns:
        PINNDataset configured for Burgers equation
    """
    if initial_condition is None:
        initial_condition = BurgersInitialConditions.sine()
    
    # Get or compute solution
    if precomputed_solution is not None:
        solution = precomputed_solution
    else:
        solution = BurgersSolution(
            initial_condition=initial_condition,
            hz=hz, ht=ht, Tmax=Tmax, L=L
        )
    
    # Create dataset
    return create_pinn_dataset(
        solution_function=solution,
        bounds=solution.bounds,
        n_collocation=n_collocation,
        n_boundary=n_boundary,
        n_initial=n_initial,
        t_distribution=t_distribution,
        x_distribution=x_distribution,
        device=device,
        normalize=True,
        seed=seed
    )


def create_burgers_sequential_dataset(
    initial_condition: BurgersInitialCondition = None,
    hz: float = 0.1,
    ht: float = 0.05,
    Tmax: float = 5.0,
    L: float = 6.0,
    seq_length: int = 10,
    pred_length: int = 1,
    stride: int = 1,
    device: str = 'cpu',
    normalize: bool = True,
    precomputed_solution: Optional[BurgersSolution] = None
) -> SequentialPDEDataset:
    """
    Create a sequential (LSTM/RNN) dataset from Burgers equation solution.
    
    Args:
        initial_condition: Initial condition (default: sine)
        hz, ht: Grid spacing for solution computation
        Tmax: Final time
        L: Integration limit for Cole-Hopf quadrature
        seq_length: Number of input time steps
        pred_length: Number of output time steps to predict
        stride: Stride between sequences
        device: Device for tensors
        normalize: Whether to normalize data
        precomputed_solution: Use existing solution instead of computing
        
    Returns:
        SequentialPDEDataset configured for Burgers equation
    """
    if initial_condition is None:
        initial_condition = BurgersInitialConditions.sine()
    
    # Get or compute solution
    if precomputed_solution is not None:
        solution = precomputed_solution
    else:
        solution = BurgersSolution(
            initial_condition=initial_condition,
            hz=hz, ht=ht, Tmax=Tmax, L=L
        )
    
    # Get meshgrid data
    T, X, U = solution.get_meshgrid()
    
    # Create dataset
    return create_sequential_dataset_from_array(
        T=T,
        X=X,
        U=U,
        seq_length=seq_length,
        pred_length=pred_length,
        device=device,
        normalize=normalize
    )


def create_burgers_galerkin_dataset(
    initial_condition: BurgersInitialCondition = None,
    hz: float = 0.1,
    ht: float = 0.05,
    Tmax: float = 5.0,
    L: float = 6.0,
    t_snapshot: float = None,
    n_quadrature: int = 512,
    quadrature_type: str = 'gauss-legendre',
    device: str = 'cpu',
    precomputed_solution: Optional[BurgersSolution] = None
) -> GalerkinDataset:
    """
    Create a Galerkin NN dataset from Burgers equation solution.
    
    For Galerkin methods, we typically work with a single time snapshot
    or use the steady-state solution. This creates quadrature points
    for the spatial domain at a specific time.
    
    Args:
        initial_condition: Initial condition (default: sine)
        hz, ht: Grid spacing for solution computation
        Tmax: Final time
        L: Integration limit for Cole-Hopf quadrature
        t_snapshot: Time at which to take snapshot (default: Tmax/2)
        n_quadrature: Number of quadrature points
        quadrature_type: Type of quadrature rule
        device: Device for tensors
        precomputed_solution: Use existing solution instead of computing
        
    Returns:
        GalerkinDataset configured for Burgers equation
    """
    if initial_condition is None:
        initial_condition = BurgersInitialConditions.sine()
    
    if t_snapshot is None:
        t_snapshot = Tmax / 2
    
    # Get or compute solution
    if precomputed_solution is not None:
        solution = precomputed_solution
    else:
        solution = BurgersSolution(
            initial_condition=initial_condition,
            hz=hz, ht=ht, Tmax=Tmax, L=L
        )
    
    # Create solution function at specific time
    def solution_at_t(x: np.ndarray) -> np.ndarray:
        t = np.full_like(x, t_snapshot)
        return solution(t, x)
    
    # For Burgers, the RHS depends on the specific form
    # For viscous Burgers: -u*u_x + nu*u_xx
    # Here we provide the solution for validation
    
    bounds = DomainBounds(
        t_min=0.0,
        t_max=Tmax,
        x_min=solution.z_range[0],
        x_max=solution.z_range[1],
        spatial_dim=1
    )
    
    return create_galerkin_dataset(
        bounds=bounds,
        solution_function=solution_at_t,
        n_quadrature=n_quadrature,
        quadrature_type=quadrature_type,
        problem_order=2,  # Second-order PDE
        device=device
    )

def hermit(k, y):
    return (1.0 / np.sqrt((2.0**k) * factorial(k) * np.sqrt(np.pi))
            * np.exp(-y**2 / 2.0) * eval_hermite(k, y))

def make_hermite_basis_eval(n_basis: int, scale: float, shift: float = 0.0):
    """
    Returns basis_eval(x_grid) -> Phi (K,nx) with
      y=(x-shift)/scale
      phi_k^x(x)=(1/sqrt(scale))*phi_k(y)
    """
    def basis_eval(x_grid):
        x_grid = np.asarray(x_grid, dtype=float)
        y = (x_grid - shift) / scale
        Phi = np.stack([hermit(k, y) for k in range(n_basis)], axis=0)  # (K,nx)
        return Phi / np.sqrt(scale)
    return basis_eval

def create_burgers_hermite_time_dataset(
    initial_condition: BurgersInitialCondition = None,
    hz: float = 0.1,
    ht: float = 0.05,
    Tmax: float = 5.0,
    L: float = 6.0,
    n_basis: int = 32,
    n_time_samples: int = 200,
    t_sampling: str = "grid",   # "grid" or "random"
    hermite_scale: float = None,
    hermite_shift: float = 0.0,
    device: str = "cpu",
    dtype: Any = None,          # torch dtype if you want, else default float32
    normalize_t: bool = False,
    normalize_c: bool = False,
    return_k_coords: bool = False,
    seed: Optional[int] = None,
    precomputed_solution: Optional[BurgersSolution] = None,
) -> TimeProjectedDataset:
    if initial_condition is None:
        initial_condition = BurgersInitialConditions.sine()

    sol = precomputed_solution or BurgersSolution(
        initial_condition=initial_condition,
        hz=hz, ht=ht, Tmax=Tmax, L=L
    )

    x_grid = sol.z_vals  # (nx,)

    if hermite_scale is None:
        # heuristic: make y roughly in [-3,3] over the x-range
        hermite_scale = (float(x_grid.max()) - float(x_grid.min())) / 6.0

    basis_eval = make_hermite_basis_eval(
        n_basis=n_basis,
        scale=hermite_scale,
        shift=hermite_shift
    )

    import torch
    if dtype is None:
        dtype = torch.float32

    # sol is callable: sol(t,x)->u
    return create_time_projected_dataset(
        solution_function=sol,
        x_grid=x_grid,
        t_min=0.0,
        t_max=float(sol.Tmax),
        basis_eval=basis_eval,
        n_time_samples=n_time_samples,
        t_sampling=t_sampling,
        seed=seed,
        weights=None,  # trapezoid default
        device=device,
        dtype=dtype,
        normalize_t=normalize_t,
        normalize_c=normalize_c,
        return_k_coords=return_k_coords,
        pde_name="burgers",
    )



# =============================================================================
# Batch Dataset Generation
# =============================================================================

def generate_all_burgers_datasets(
    initial_condition: BurgersInitialCondition = None,
    hz: float = 0.1,
    ht: float = 0.05,
    Tmax: float = 5.0,
    L: float = 6.0,
    device: str = 'cpu',
    save_path: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate datasets for all architectures from a single Burgers solution.
    
    Args:
        initial_condition: Initial condition
        hz, ht, Tmax: Solution parameters
        L: Integration limit for Cole-Hopf quadrature
        device: Device for tensors
        save_path: Optional path prefix for saving datasets
        **kwargs: Additional arguments passed to individual factories
        
    Returns:
        Dictionary with 'pinn', 'sequential', 'galerkin' datasets and 'solution'
    """
    if initial_condition is None:
        initial_condition = BurgersInitialConditions.sine()
    
    # Compute solution once
    print(f"Computing Burgers solution: {initial_condition.description}")
    solution = BurgersSolution(
        initial_condition=initial_condition,
        hz=hz, ht=ht, Tmax=Tmax, L=L
    )
    
    # Create all datasets
    print("Creating PINN dataset...")
    pinn_kwargs = {k: v for k, v in kwargs.items() 
                   if k in ['n_collocation', 'n_boundary', 'n_initial', 
                           't_distribution', 'x_distribution', 'seed']}
    pinn_dataset = create_burgers_pinn_dataset(
        precomputed_solution=solution,
        device=device,
        **pinn_kwargs
    )
    
    print("Creating Sequential dataset...")
    seq_kwargs = {k: v for k, v in kwargs.items()
                  if k in ['seq_length', 'pred_length', 'stride', 'normalize']}
    sequential_dataset = create_burgers_sequential_dataset(
        precomputed_solution=solution,
        device=device,
        **seq_kwargs
    )
    
    print("Creating Galerkin dataset...")
    gal_kwargs = {k: v for k, v in kwargs.items()
                  if k in ['t_snapshot', 'n_quadrature', 'quadrature_type']}
    galerkin_dataset = create_burgers_galerkin_dataset(
        precomputed_solution=solution,
        device=device,
        **gal_kwargs
    )
    
    result = {
        'solution': solution,
        'pinn': pinn_dataset,
        'sequential': sequential_dataset,
        'galerkin': galerkin_dataset,
    }
    
    # Save if requested
    if save_path:
        print(f"Saving datasets to {save_path}...")
        pinn_dataset.save(f"{save_path}_pinn.npz")
        sequential_dataset.save(f"{save_path}_sequential.npz")
        galerkin_dataset.save(f"{save_path}_galerkin.npz")
        print("Done!")
    
    return result


# =============================================================================
# Convenience Functions
# =============================================================================

def get_burgers_solution_function(
    initial_condition: BurgersInitialCondition = None,
    hz: float = 0.1,
    ht: float = 0.05,
    Tmax: float = 5.0,
    L: float = 6.0
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Get a callable solution function for Burgers equation.
    
    Returns:
        Function (t, x) -> u that can be used with pde_dataset factories
    """
    if initial_condition is None:
        initial_condition = BurgersInitialConditions.sine()
    
    solution = BurgersSolution(
        initial_condition=initial_condition,
        hz=hz, ht=ht, Tmax=Tmax, L=L
    )
    
    return solution


def get_burgers_generator(
    initial_condition: BurgersInitialCondition = None,
    hz: float = 0.1,
    ht: float = 0.05,
    Tmax: float = 5.0,
    L: float = 6.0
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    """
    Get a generator yielding (t, x, u) tuples for Burgers equation.
    
    Returns:
        Generator compatible with pde_dataset generator-based factories
    """
    if initial_condition is None:
        initial_condition = BurgersInitialConditions.sine()
    
    solution = BurgersSolution(
        initial_condition=initial_condition,
        hz=hz, ht=ht, Tmax=Tmax, L=L
    )
    
    return solution.generator()

if __name__ == '__main__':

    # Generate all datasets at once
    datasets = generate_all_burgers_datasets(
        Tmax=2.0,
        device='cuda',
        save_path='./burgers_data'
    )
