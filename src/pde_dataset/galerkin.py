"""
Submodule 4: Dataset adapted for Galerkin Neural Network architectures.

Based on the paper "Galerkin Neural Networks: A Framework for Approximating 
Variational Equations with Error Control" by Ainsworth & Dong.

Galerkin NN methods require:
- Quadrature points for computing inner products/integrals
- Points for evaluating the bilinear form a(u,v)
- Points for evaluating the linear functional L(v)
- Boundary quadrature points (for Robin/mixed BCs)

The approach adaptively constructs basis functions from neural networks.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import (
    Callable, Generator, Tuple, Optional, Dict, Any,
    Union, List
)
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from .utils import (
    DomainBounds, SamplingDistribution, SamplingDistributions,
    to_tensor, Normalizer, PDESolutionGenerator, Device
)
from .io import save_dataset, DatasetMetadata


# =============================================================================
# Galerkin Dataset Configuration
# =============================================================================

@dataclass
class GalerkinDatasetConfig:
    """Configuration for Galerkin NN dataset generation."""
    # Quadrature settings
    n_quadrature_interior: int = 512  # Interior quadrature points
    n_quadrature_boundary: int = 256  # Boundary quadrature points
    quadrature_type: str = 'gauss-legendre'  # 'gauss-legendre', 'gauss-lobatto', 'uniform'
    
    # Domain bounds
    bounds: DomainBounds = field(default_factory=DomainBounds)
    
    # Problem order (for Sobolev space selection)
    problem_order: int = 2  # 2 for Poisson, 4 for biharmonic
    
    # Penalization parameter for boundary conditions
    epsilon: float = 1e-4
    
    # Data options
    normalize: bool = False  # Usually not needed for Galerkin methods
    include_solution: bool = True  # Include reference solution if available
    
    # Random seed
    seed: Optional[int] = None


# =============================================================================
# Quadrature Rules
# =============================================================================

class QuadratureRule:
    """Base class for quadrature rules."""
    
    @staticmethod
    def gauss_legendre_1d(n: int, a: float = 0.0, b: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gauss-Legendre quadrature on [a, b].
        
        Returns:
            Tuple of (nodes, weights)
        """
        # Get nodes and weights on [-1, 1]
        nodes, weights = np.polynomial.legendre.leggauss(n)
        
        # Transform to [a, b]
        nodes = 0.5 * (b - a) * nodes + 0.5 * (b + a)
        weights = 0.5 * (b - a) * weights
        
        return nodes, weights
    
    @staticmethod
    def gauss_lobatto_1d(n: int, a: float = 0.0, b: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gauss-Lobatto quadrature on [a, b] (includes endpoints).
        
        Returns:
            Tuple of (nodes, weights)
        """
        if n < 2:
            raise ValueError("Gauss-Lobatto requires n >= 2")
        
        # Interior nodes are roots of P'_{n-1}
        inner_nodes, _ = np.polynomial.legendre.leggauss(n - 2)
        
        # Add endpoints
        nodes = np.concatenate([[-1.0], inner_nodes, [1.0]])
        nodes = np.sort(nodes)
        
        # Compute weights using Lagrange interpolation
        weights = np.zeros(n)
        for i in range(n):
            # Lagrange basis polynomial at node i
            li = np.ones(n)
            for j in range(n):
                if i != j:
                    li *= (nodes - nodes[j]) / (nodes[i] - nodes[j])
            weights[i] = np.sum(li) * 2 / (n - 1)
        
        # Transform to [a, b]
        nodes = 0.5 * (b - a) * nodes + 0.5 * (b + a)
        weights = 0.5 * (b - a) * weights
        
        return nodes, weights
    
    @staticmethod
    def uniform_1d(n: int, a: float = 0.0, b: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uniform (midpoint) quadrature on [a, b].
        
        Returns:
            Tuple of (nodes, weights)
        """
        h = (b - a) / n
        nodes = np.linspace(a + h/2, b - h/2, n)
        weights = np.full(n, h)
        return nodes, weights
    
    @staticmethod
    def tensor_product_2d(
        rule_1d: Callable[[int, float, float], Tuple[np.ndarray, np.ndarray]],
        nx: int, ny: int,
        x_bounds: Tuple[float, float],
        y_bounds: Tuple[float, float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tensor product quadrature in 2D.
        
        Returns:
            Tuple of (x_nodes, y_nodes, weights) where x,y are meshgridded
        """
        x_nodes, x_weights = rule_1d(nx, x_bounds[0], x_bounds[1])
        y_nodes, y_weights = rule_1d(ny, y_bounds[0], y_bounds[1])
        
        X, Y = np.meshgrid(x_nodes, y_nodes, indexing='ij')
        Wx, Wy = np.meshgrid(x_weights, y_weights, indexing='ij')
        
        return X.flatten(), Y.flatten(), (Wx * Wy).flatten()
    
    @staticmethod
    def get_rule(rule_type: str) -> Callable:
        """Get quadrature rule by name."""
        rules = {
            'gauss-legendre': QuadratureRule.gauss_legendre_1d,
            'gauss-lobatto': QuadratureRule.gauss_lobatto_1d,
            'uniform': QuadratureRule.uniform_1d,
        }
        return rules.get(rule_type, QuadratureRule.gauss_legendre_1d)


# =============================================================================
# Bilinear Form Definitions
# =============================================================================

class BilinearForm(ABC):
    """Abstract base class for bilinear forms."""
    
    @abstractmethod
    def __call__(
        self, 
        u: torch.Tensor, 
        v: torch.Tensor,
        du: Optional[Dict[str, torch.Tensor]] = None,
        dv: Optional[Dict[str, torch.Tensor]] = None,
        x: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Evaluate a(u, v)."""
        pass


class L2InnerProduct(BilinearForm):
    """L2 inner product: a(u,v) = ∫ u*v dx"""
    
    def __call__(
        self, 
        u: torch.Tensor, 
        v: torch.Tensor,
        du: Optional[Dict[str, torch.Tensor]] = None,
        dv: Optional[Dict[str, torch.Tensor]] = None,
        x: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if weights is None:
            return torch.sum(u * v)
        return torch.sum(weights * u * v)


class H1InnerProduct(BilinearForm):
    """
    H1 inner product: a(u,v) = ∫ ∇u·∇v dx + ε^{-1} ∫_∂Ω u*v ds
    
    For Poisson-type problems with Robin BC.
    """
    
    def __init__(self, epsilon: float = 1e-4):
        self.epsilon = epsilon
    
    def __call__(
        self, 
        u: torch.Tensor, 
        v: torch.Tensor,
        du: Optional[Dict[str, torch.Tensor]] = None,
        dv: Optional[Dict[str, torch.Tensor]] = None,
        x: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        boundary_u: Optional[torch.Tensor] = None,
        boundary_v: Optional[torch.Tensor] = None,
        boundary_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Interior gradient term
        if du is not None and dv is not None:
            grad_term = du['dx'] * dv['dx']
            if 'dy' in du:
                grad_term = grad_term + du['dy'] * dv['dy']
        else:
            grad_term = torch.zeros(1, device=u.device)
        
        if weights is not None:
            interior = torch.sum(weights * grad_term)
        else:
            interior = torch.sum(grad_term)
        
        # Boundary term
        boundary = torch.tensor(0.0, device=u.device)
        if boundary_u is not None and boundary_v is not None:
            if boundary_weights is not None:
                boundary = (1.0 / self.epsilon) * torch.sum(
                    boundary_weights * boundary_u * boundary_v
                )
            else:
                boundary = (1.0 / self.epsilon) * torch.sum(
                    boundary_u * boundary_v
                )
        
        return interior + boundary


class H2InnerProduct(BilinearForm):
    """
    H2 inner product: a(u,v) = ∫ Δu·Δv dx + boundary terms
    
    For biharmonic-type problems.
    """
    
    def __init__(self, epsilon1: float = 1e-4, epsilon2: float = 1e-4):
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
    
    def __call__(
        self, 
        u: torch.Tensor, 
        v: torch.Tensor,
        du: Optional[Dict[str, torch.Tensor]] = None,
        dv: Optional[Dict[str, torch.Tensor]] = None,
        x: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **boundary_kwargs
    ) -> torch.Tensor:
        # Laplacian term
        if du is not None and dv is not None and 'laplacian' in du:
            lap_term = du['laplacian'] * dv['laplacian']
        else:
            lap_term = torch.zeros(1, device=u.device)
        
        if weights is not None:
            interior = torch.sum(weights * lap_term)
        else:
            interior = torch.sum(lap_term)
        
        # Boundary terms would be added here
        return interior


# =============================================================================
# Galerkin Dataset Classes
# =============================================================================

class GalerkinDataset(Dataset):
    """
    PyTorch Dataset for Galerkin Neural Network methods.
    
    Provides quadrature points and weights for computing inner products
    in the variational formulation.
    """
    
    def __init__(
        self,
        config: GalerkinDatasetConfig,
        rhs_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        solution_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        bilinear_form: Optional[BilinearForm] = None,
        device: Device = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize Galerkin dataset.
        
        Args:
            config: Dataset configuration
            rhs_function: Right-hand side f in L(v) = ∫fv dx
            solution_function: Reference solution (for validation)
            bilinear_form: Bilinear form a(u,v) to use
            device: Device for tensors
            dtype: Data type for tensors
        """
        self.config = config
        self.rhs_function = rhs_function
        self.solution_function = solution_function
        self.bilinear_form = bilinear_form or L2InnerProduct()
        self.device = device
        self.dtype = dtype
        
        if config.seed is not None:
            np.random.seed(config.seed)
        
        # Generate quadrature points
        self._generate_quadrature()
        
        # Evaluate RHS and solution if provided
        self._evaluate_functions()
        
        # Convert to tensors
        self._to_tensors()
    
    def _generate_quadrature(self):
        """Generate quadrature points and weights."""
        bounds = self.config.bounds
        rule = QuadratureRule.get_rule(self.config.quadrature_type)
        
        if bounds.spatial_dim == 1:
            # 1D case
            self.x_interior, self.weights_interior = rule(
                self.config.n_quadrature_interior,
                bounds.x_min[0], bounds.x_max[0]
            )
            
            # Boundary points (just endpoints in 1D)
            self.x_boundary = np.array([bounds.x_min[0], bounds.x_max[0]])
            self.weights_boundary = np.ones(2)  # Point evaluation
            
        else:
            # 2D case (tensor product)
            n = int(np.sqrt(self.config.n_quadrature_interior))
            self.x_interior, self.y_interior, self.weights_interior = \
                QuadratureRule.tensor_product_2d(
                    rule, n, n,
                    (bounds.x_min[0], bounds.x_max[0]),
                    (bounds.x_min[1], bounds.x_max[1])
                )
            
            # Boundary quadrature (4 edges)
            self._generate_2d_boundary_quadrature()
    
    def _generate_2d_boundary_quadrature(self):
        """Generate boundary quadrature for 2D domains."""
        bounds = self.config.bounds
        n_per_edge = self.config.n_quadrature_boundary // 4
        rule = QuadratureRule.get_rule(self.config.quadrature_type)
        
        x_boundary = []
        y_boundary = []
        weights_boundary = []
        
        # Bottom edge: y = y_min
        x_nodes, weights = rule(n_per_edge, bounds.x_min[0], bounds.x_max[0])
        x_boundary.extend(x_nodes)
        y_boundary.extend([bounds.x_min[1]] * n_per_edge)
        weights_boundary.extend(weights)
        
        # Top edge: y = y_max
        x_boundary.extend(x_nodes)
        y_boundary.extend([bounds.x_max[1]] * n_per_edge)
        weights_boundary.extend(weights)
        
        # Left edge: x = x_min
        y_nodes, weights = rule(n_per_edge, bounds.x_min[1], bounds.x_max[1])
        x_boundary.extend([bounds.x_min[0]] * n_per_edge)
        y_boundary.extend(y_nodes)
        weights_boundary.extend(weights)
        
        # Right edge: x = x_max
        x_boundary.extend([bounds.x_max[0]] * n_per_edge)
        y_boundary.extend(y_nodes)
        weights_boundary.extend(weights)
        
        self.x_boundary = np.array(x_boundary)
        self.y_boundary = np.array(y_boundary)
        self.weights_boundary = np.array(weights_boundary)
    
    def _evaluate_functions(self):
        """Evaluate RHS and solution at quadrature points."""
        if self.config.bounds.spatial_dim == 1:
            if self.rhs_function is not None:
                self.f_interior = self.rhs_function(self.x_interior)
            else:
                self.f_interior = np.zeros_like(self.x_interior)
            
            if self.solution_function is not None:
                self.u_interior = self.solution_function(self.x_interior)
                self.u_boundary = self.solution_function(self.x_boundary)
            
        else:  # 2D
            coords = np.column_stack([self.x_interior, self.y_interior])
            if self.rhs_function is not None:
                self.f_interior = self.rhs_function(coords)
            else:
                self.f_interior = np.zeros(len(self.x_interior))
            
            if self.solution_function is not None:
                self.u_interior = self.solution_function(coords)
                boundary_coords = np.column_stack([self.x_boundary, self.y_boundary])
                self.u_boundary = self.solution_function(boundary_coords)
    
    def _to_tensors(self):
        """Convert arrays to tensors."""
        self.x_interior_tensor = to_tensor(
            self.x_interior, dtype=self.dtype, device=self.device
        )
        self.weights_interior_tensor = to_tensor(
            self.weights_interior, dtype=self.dtype, device=self.device
        )
        self.x_boundary_tensor = to_tensor(
            self.x_boundary, dtype=self.dtype, device=self.device
        )
        self.weights_boundary_tensor = to_tensor(
            self.weights_boundary, dtype=self.dtype, device=self.device
        )
        
        if hasattr(self, 'y_interior'):
            self.y_interior_tensor = to_tensor(
                self.y_interior, dtype=self.dtype, device=self.device
            )
        if hasattr(self, 'y_boundary'):
            self.y_boundary_tensor = to_tensor(
                self.y_boundary, dtype=self.dtype, device=self.device
            )
        
        if hasattr(self, 'f_interior'):
            self.f_interior_tensor = to_tensor(
                self.f_interior, dtype=self.dtype, device=self.device
            )
        
        if hasattr(self, 'u_interior'):
            self.u_interior_tensor = to_tensor(
                self.u_interior, dtype=self.dtype, device=self.device
            )
        if hasattr(self, 'u_boundary'):
            self.u_boundary_tensor = to_tensor(
                self.u_boundary, dtype=self.dtype, device=self.device
            )
    
    def __len__(self) -> int:
        return self.config.n_quadrature_interior
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single quadrature point."""
        if self.config.bounds.spatial_dim == 1:
            sample = {
                'x': self.x_interior_tensor[idx],
                'weight': self.weights_interior_tensor[idx],
            }
        else:
            sample = {
                'x': self.x_interior_tensor[idx],
                'y': self.y_interior_tensor[idx],
                'weight': self.weights_interior_tensor[idx],
            }
        
        if hasattr(self, 'f_interior_tensor'):
            sample['f'] = self.f_interior_tensor[idx]
        if hasattr(self, 'u_interior_tensor'):
            sample['u'] = self.u_interior_tensor[idx]
        
        return sample
    
    def get_interior_quadrature(self) -> Dict[str, torch.Tensor]:
        """Get all interior quadrature data."""
        result = {
            'x': self.x_interior_tensor,
            'weights': self.weights_interior_tensor,
        }
        
        if hasattr(self, 'y_interior_tensor'):
            result['y'] = self.y_interior_tensor
        
        if hasattr(self, 'f_interior_tensor'):
            result['f'] = self.f_interior_tensor
        if hasattr(self, 'u_interior_tensor'):
            result['u'] = self.u_interior_tensor
        
        return result
    
    def get_boundary_quadrature(self) -> Dict[str, torch.Tensor]:
        """Get all boundary quadrature data."""
        result = {
            'x': self.x_boundary_tensor,
            'weights': self.weights_boundary_tensor,
        }
        
        if hasattr(self, 'y_boundary_tensor'):
            result['y'] = self.y_boundary_tensor
        
        if hasattr(self, 'u_boundary_tensor'):
            result['u'] = self.u_boundary_tensor
        
        return result
    
    def compute_L_functional(
        self, 
        v: torch.Tensor,
        f: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the linear functional L(v) = ∫f·v dx.
        
        Args:
            v: Test function values at interior quadrature points
            f: RHS values (uses stored values if None)
            
        Returns:
            Scalar value of L(v)
        """
        if f is None:
            f = self.f_interior_tensor
        
        return torch.sum(self.weights_interior_tensor * f * v)
    
    def compute_inner_product(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        du: Optional[Dict[str, torch.Tensor]] = None,
        dv: Optional[Dict[str, torch.Tensor]] = None,
        u_boundary: Optional[torch.Tensor] = None,
        v_boundary: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the bilinear form a(u, v).
        
        Args:
            u, v: Function values at interior quadrature points
            du, dv: Derivative dictionaries (keys: 'dx', 'dy', 'laplacian')
            u_boundary, v_boundary: Boundary values
            
        Returns:
            Scalar value of a(u, v)
        """
        return self.bilinear_form(
            u, v, du, dv,
            weights=self.weights_interior_tensor,
            boundary_u=u_boundary,
            boundary_v=v_boundary,
            boundary_weights=self.weights_boundary_tensor
        )
    
    def compute_residual(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        du: Optional[Dict[str, torch.Tensor]] = None,
        dv: Optional[Dict[str, torch.Tensor]] = None,
        u_boundary: Optional[torch.Tensor] = None,
        v_boundary: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the residual r(u)(v) = L(v) - a(u, v).
        
        Args:
            u, v: Function values
            du, dv: Derivatives
            u_boundary, v_boundary: Boundary values
            
        Returns:
            Residual value
        """
        L_v = self.compute_L_functional(v)
        a_uv = self.compute_inner_product(u, v, du, dv, u_boundary, v_boundary)
        return L_v - a_uv
    
    def compute_energy_norm(
        self,
        u: torch.Tensor,
        du: Optional[Dict[str, torch.Tensor]] = None,
        u_boundary: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the energy norm |||u||| = sqrt(a(u, u)).
        
        Args:
            u: Function values
            du: Derivatives
            u_boundary: Boundary values
            
        Returns:
            Energy norm
        """
        a_uu = self.compute_inner_product(u, u, du, du, u_boundary, u_boundary)
        return torch.sqrt(torch.abs(a_uu))
    
    def to(self, device: Device) -> 'GalerkinDataset':
        """Move all data to specified device."""
        self.device = device
        self._to_tensors()
        return self
    
    def save(self, filepath: str, format: str = 'npz') -> None:
        """Save dataset to file."""
        data = {
            'x_interior': self.x_interior,
            'weights_interior': self.weights_interior,
            'x_boundary': self.x_boundary,
            'weights_boundary': self.weights_boundary,
        }
        
        if hasattr(self, 'y_interior'):
            data['y_interior'] = self.y_interior
        if hasattr(self, 'y_boundary'):
            data['y_boundary'] = self.y_boundary
        if hasattr(self, 'f_interior'):
            data['f_interior'] = self.f_interior
        if hasattr(self, 'u_interior'):
            data['u_interior'] = self.u_interior
        if hasattr(self, 'u_boundary'):
            data['u_boundary'] = self.u_boundary
        
        metadata = DatasetMetadata(
            dataset_type='galerkin',
            spatial_dim=self.config.bounds.spatial_dim,
            n_samples=self.config.n_quadrature_interior,
            bounds={
                'x_min': self.config.bounds.x_min.tolist(),
                'x_max': self.config.bounds.x_max.tolist(),
            },
            extra_info={
                'n_quadrature_interior': self.config.n_quadrature_interior,
                'n_quadrature_boundary': self.config.n_quadrature_boundary,
                'quadrature_type': self.config.quadrature_type,
                'problem_order': self.config.problem_order,
                'epsilon': self.config.epsilon,
            }
        )
        
        save_dataset(filepath, data, metadata, format=format)


class GalerkinBasisDataset(Dataset):
    """
    Dataset for storing and managing Galerkin basis functions.
    
    Following Algorithm 1 from the paper, this stores the sequence
    of basis functions {φ_i^NN} constructed during training.
    """
    
    def __init__(
        self,
        quadrature_dataset: GalerkinDataset,
        device: Device = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize basis dataset.
        
        Args:
            quadrature_dataset: Associated quadrature dataset
            device: Device for tensors
            dtype: Data type
        """
        self.quadrature = quadrature_dataset
        self.device = device
        self.dtype = dtype
        
        # Storage for basis functions (as networks or evaluated values)
        self.basis_networks: List[torch.nn.Module] = []
        self.basis_values_interior: List[torch.Tensor] = []
        self.basis_values_boundary: List[torch.Tensor] = []
        self.basis_derivatives: List[Dict[str, torch.Tensor]] = []
        
        # Galerkin matrix K and coefficients
        self.K_matrix: Optional[torch.Tensor] = None
        self.coefficients: Optional[torch.Tensor] = None
    
    def add_basis_function(
        self,
        network: Optional[torch.nn.Module] = None,
        values_interior: Optional[torch.Tensor] = None,
        values_boundary: Optional[torch.Tensor] = None,
        derivatives: Optional[Dict[str, torch.Tensor]] = None,
        normalize: bool = True
    ) -> int:
        """
        Add a new basis function to the set.
        
        Args:
            network: Neural network representing the basis function
            values_interior: Evaluated values at interior quadrature points
            values_boundary: Evaluated values at boundary points
            derivatives: Dictionary of derivative values
            normalize: Whether to normalize to unit energy norm
            
        Returns:
            Index of the new basis function
        """
        if network is not None:
            self.basis_networks.append(network)
        
        if values_interior is not None:
            if normalize:
                # Normalize by energy norm
                norm = self.quadrature.compute_energy_norm(
                    values_interior, derivatives, values_boundary
                )
                values_interior = values_interior / norm
                if values_boundary is not None:
                    values_boundary = values_boundary / norm
                if derivatives is not None:
                    derivatives = {k: v / norm for k, v in derivatives.items()}
            
            self.basis_values_interior.append(values_interior)
            self.basis_values_boundary.append(values_boundary)
            self.basis_derivatives.append(derivatives)
        
        # Update Galerkin matrix
        self._update_galerkin_matrix()
        
        return len(self.basis_values_interior) - 1
    
    def _update_galerkin_matrix(self):
        """Update the Galerkin matrix K with K_ij = a(φ_i, φ_j)."""
        n = len(self.basis_values_interior)
        if n == 0:
            return
        
        K = torch.zeros(n, n, device=self.device, dtype=self.dtype)
        
        for i in range(n):
            for j in range(i, n):
                a_ij = self.quadrature.compute_inner_product(
                    self.basis_values_interior[i],
                    self.basis_values_interior[j],
                    self.basis_derivatives[i],
                    self.basis_derivatives[j],
                    self.basis_values_boundary[i],
                    self.basis_values_boundary[j]
                )
                K[i, j] = a_ij
                K[j, i] = a_ij  # Symmetric
        
        self.K_matrix = K
    
    def solve_galerkin_system(self) -> torch.Tensor:
        """
        Solve the Galerkin system K·c = F for coefficients.
        
        Returns:
            Coefficient vector c
        """
        if self.K_matrix is None or len(self.basis_values_interior) == 0:
            raise ValueError("No basis functions available")
        
        n = len(self.basis_values_interior)
        F = torch.zeros(n, device=self.device, dtype=self.dtype)
        
        for i in range(n):
            F[i] = self.quadrature.compute_L_functional(
                self.basis_values_interior[i]
            )
        
        # Solve K·c = F
        self.coefficients = torch.linalg.solve(self.K_matrix, F)
        return self.coefficients
    
    def evaluate_approximation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the Galerkin approximation u_N = Σ c_i φ_i.
        
        Returns:
            Tuple of (interior values, boundary values)
        """
        if self.coefficients is None:
            self.solve_galerkin_system()
        
        u_interior = torch.zeros_like(self.basis_values_interior[0])
        u_boundary = torch.zeros_like(self.basis_values_boundary[0])
        
        for c, phi_int, phi_bnd in zip(
            self.coefficients,
            self.basis_values_interior,
            self.basis_values_boundary
        ):
            u_interior = u_interior + c * phi_int
            if phi_bnd is not None:
                u_boundary = u_boundary + c * phi_bnd
        
        return u_interior, u_boundary
    
    def compute_error_estimator(
        self,
        residual_maximizer: torch.Tensor,
        residual_maximizer_norm: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the error estimator η(u_{i-1}, φ_i^NN).
        
        Based on equation (2.14) from the paper:
        η(u_0, v) = <r(u_0), v> / |||v|||
        
        Args:
            residual_maximizer: The basis function maximizing the residual
            residual_maximizer_norm: Energy norm of the maximizer
            
        Returns:
            Error estimator value
        """
        # Get current approximation
        u_current, u_boundary = self.evaluate_approximation()
        
        # Compute residual
        residual = self.quadrature.compute_residual(
            u_current, residual_maximizer
        )
        
        return residual / residual_maximizer_norm
    
    def __len__(self) -> int:
        return len(self.basis_values_interior)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'values_interior': self.basis_values_interior[idx],
            'values_boundary': self.basis_values_boundary[idx],
            'derivatives': self.basis_derivatives[idx],
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_galerkin_dataset(
    bounds: DomainBounds,
    rhs_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    solution_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    n_quadrature: int = 512,
    quadrature_type: str = 'gauss-legendre',
    problem_order: int = 2,
    epsilon: float = 1e-4,
    device: Device = 'cpu'
) -> GalerkinDataset:
    """
    Factory function to create a Galerkin dataset.
    
    Args:
        bounds: Domain bounds
        rhs_function: Right-hand side function
        solution_function: Reference solution (for validation)
        n_quadrature: Number of interior quadrature points
        quadrature_type: Type of quadrature rule
        problem_order: Order of the PDE (2 for Poisson, 4 for biharmonic)
        epsilon: Penalization parameter
        device: Device for tensors
        
    Returns:
        Configured GalerkinDataset
    """
    config = GalerkinDatasetConfig(
        n_quadrature_interior=n_quadrature,
        bounds=bounds,
        quadrature_type=quadrature_type,
        problem_order=problem_order,
        epsilon=epsilon
    )
    
    # Select appropriate bilinear form
    if problem_order == 2:
        bilinear_form = H1InnerProduct(epsilon=epsilon)
    elif problem_order == 4:
        bilinear_form = H2InnerProduct(epsilon1=epsilon, epsilon2=epsilon)
    else:
        bilinear_form = L2InnerProduct()
    
    return GalerkinDataset(
        config=config,
        rhs_function=rhs_function,
        solution_function=solution_function,
        bilinear_form=bilinear_form,
        device=device
    )
