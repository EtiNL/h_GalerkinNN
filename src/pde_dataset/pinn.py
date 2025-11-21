"""
Submodule 2: Dataset adapted for PINN (Physics-Informed Neural Networks) architectures.

PINNs typically require:
- Collocation points in the domain interior for PDE residual
- Boundary condition points
- Initial condition points
- Optionally, sparse measurement/observation points

This module provides datasets that separate these different point types.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import (
    Callable, Generator, Tuple, Optional, Dict, Any,
    Union, List
)
from dataclasses import dataclass, field

from .utils import (
    DomainBounds, DomainType, SamplingConfig, SamplingDistribution,
    SamplingDistributions, sample_domain, to_tensor, Normalizer,
    PDESolutionGenerator, Device
)
from .io import save_dataset, DatasetMetadata


# =============================================================================
# PINN Dataset Configuration
# =============================================================================

@dataclass
class PINNDatasetConfig:
    """Configuration for PINN dataset generation."""
    # Number of points for each type
    n_collocation: int = 10000  # Interior points for PDE residual
    n_boundary: int = 2000      # Boundary condition points
    n_initial: int = 2000       # Initial condition points
    n_observation: int = 0      # Sparse observation/measurement points
    
    # Domain bounds
    bounds: DomainBounds = field(default_factory=DomainBounds)
    
    # Sampling distributions
    t_distribution: Optional[SamplingDistribution] = None
    x_distribution: Optional[SamplingDistribution] = None
    
    # Data options
    include_solution: bool = True  # Include u values (if available)
    include_derivatives: bool = False  # Include du/dt, du/dx (if available)
    normalize: bool = True
    
    # Random seed
    seed: Optional[int] = None


# =============================================================================
# PINN Dataset Classes
# =============================================================================

class PINNDataset(Dataset):
    """
    PyTorch Dataset for Physics-Informed Neural Networks.
    
    Provides separate access to collocation, boundary, and initial condition points.
    """
    
    def __init__(
        self,
        config: PINNDatasetConfig,
        solution_generator: Optional[Callable[..., PDESolutionGenerator]] = None,
        solution_function: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        device: Device = 'cpu',
        dtype: torch.dtype = torch.float32,
        **generator_kwargs
    ):
        """
        Initialize PINN dataset.
        
        Args:
            config: Dataset configuration
            solution_generator: Generator function yielding (t, x, u) tuples
            solution_function: Direct function u = f(t, x) for evaluation
            device: Device for tensors
            dtype: Data type for tensors
            **generator_kwargs: Arguments to pass to solution generator
        """
        self.config = config
        self.device = device
        self.dtype = dtype
        self.solution_generator = solution_generator
        self.solution_function = solution_function
        self.generator_kwargs = generator_kwargs
        
        if config.seed is not None:
            np.random.seed(config.seed)
        
        # Generate all points
        self._generate_points()
        
        # Compute solution values if possible
        if config.include_solution and (solution_generator or solution_function):
            self._compute_solution_values()
        
        # Normalize if requested
        self.normalizer = None
        if config.normalize:
            self._setup_normalizer()
        
        # Convert to tensors
        self._to_tensors()
    
    def _generate_points(self):
        """Generate all sampling points."""
        bounds = self.config.bounds
        t_dist = self.config.t_distribution
        x_dist = self.config.x_distribution
        
        # Collocation points (interior)
        self.t_collocation, self.x_collocation = sample_domain(
            bounds, self.config.n_collocation, DomainType.INTERIOR,
            t_dist=t_dist, x_dist=x_dist
        )
        
        # Boundary points
        if self.config.n_boundary > 0:
            self.t_boundary, self.x_boundary = sample_domain(
                bounds, self.config.n_boundary, DomainType.BOUNDARY,
                t_dist=t_dist, x_dist=x_dist
            )
        else:
            self.t_boundary = np.array([])
            self.x_boundary = np.array([])
        
        # Initial condition points
        if self.config.n_initial > 0:
            self.t_initial, self.x_initial = sample_domain(
                bounds, self.config.n_initial, DomainType.INITIAL,
                t_dist=t_dist, x_dist=x_dist
            )
        else:
            self.t_initial = np.array([])
            self.x_initial = np.array([])
        
        # Observation points (random in domain)
        if self.config.n_observation > 0:
            self.t_observation, self.x_observation = sample_domain(
                bounds, self.config.n_observation, DomainType.FULL,
                t_dist=t_dist, x_dist=x_dist
            )
        else:
            self.t_observation = np.array([])
            self.x_observation = np.array([])
    
    def _compute_solution_values(self):
        """Compute solution values at all points."""
        if self.solution_function is not None:
            # Direct function evaluation
            self.u_collocation = self.solution_function(
                self.t_collocation, self.x_collocation
            )
            if len(self.t_boundary) > 0:
                self.u_boundary = self.solution_function(
                    self.t_boundary, self.x_boundary
                )
            else:
                self.u_boundary = np.array([])
            
            if len(self.t_initial) > 0:
                self.u_initial = self.solution_function(
                    self.t_initial, self.x_initial
                )
            else:
                self.u_initial = np.array([])
            
            if len(self.t_observation) > 0:
                self.u_observation = self.solution_function(
                    self.t_observation, self.x_observation
                )
            else:
                self.u_observation = np.array([])
        
        elif self.solution_generator is not None:
            # Use interpolation from generator
            from .utils import evaluate_solution_generator
            
            self.u_collocation = evaluate_solution_generator(
                self.solution_generator,
                self.t_collocation, self.x_collocation,
                **self.generator_kwargs
            )
            if len(self.t_boundary) > 0:
                self.u_boundary = evaluate_solution_generator(
                    self.solution_generator,
                    self.t_boundary, self.x_boundary,
                    **self.generator_kwargs
                )
            else:
                self.u_boundary = np.array([])
            
            if len(self.t_initial) > 0:
                self.u_initial = evaluate_solution_generator(
                    self.solution_generator,
                    self.t_initial, self.x_initial,
                    **self.generator_kwargs
                )
            else:
                self.u_initial = np.array([])
            
            if len(self.t_observation) > 0:
                self.u_observation = evaluate_solution_generator(
                    self.solution_generator,
                    self.t_observation, self.x_observation,
                    **self.generator_kwargs
                )
            else:
                self.u_observation = np.array([])
        else:
            # No solution available - set to zeros (for unsupervised PINN)
            self.u_collocation = np.zeros_like(self.t_collocation)
            self.u_boundary = np.zeros_like(self.t_boundary)
            self.u_initial = np.zeros_like(self.t_initial)
            self.u_observation = np.zeros_like(self.t_observation)
    
    def _setup_normalizer(self):
        """Setup data normalizer."""
        # Combine all data for statistics
        all_t = np.concatenate([
            self.t_collocation,
            self.t_boundary if len(self.t_boundary) > 0 else np.array([]),
            self.t_initial if len(self.t_initial) > 0 else np.array([]),
            self.t_observation if len(self.t_observation) > 0 else np.array([])
        ])
        all_x = np.concatenate([
            self.x_collocation,
            self.x_boundary if len(self.x_boundary) > 0 else np.array([]),
            self.x_initial if len(self.x_initial) > 0 else np.array([]),
            self.x_observation if len(self.x_observation) > 0 else np.array([])
        ])
        
        if hasattr(self, 'u_collocation'):
            all_u = np.concatenate([
                self.u_collocation,
                self.u_boundary if len(self.u_boundary) > 0 else np.array([]),
                self.u_initial if len(self.u_initial) > 0 else np.array([]),
                self.u_observation if len(self.u_observation) > 0 else np.array([])
            ])
        else:
            all_u = np.zeros(1)
        
        self.normalizer = Normalizer.from_data(all_t, all_x, all_u)
    
    def _to_tensors(self):
        """Convert all arrays to tensors."""
        # Helper function
        def convert(arr):
            if len(arr) == 0:
                return torch.tensor([], dtype=self.dtype, device=self.device)
            return to_tensor(arr, dtype=self.dtype, device=self.device)
        
        # Collocation
        self.t_collocation_tensor = convert(self.t_collocation)
        self.x_collocation_tensor = convert(self.x_collocation)
        if hasattr(self, 'u_collocation'):
            self.u_collocation_tensor = convert(self.u_collocation)
        
        # Boundary
        self.t_boundary_tensor = convert(self.t_boundary)
        self.x_boundary_tensor = convert(self.x_boundary)
        if hasattr(self, 'u_boundary'):
            self.u_boundary_tensor = convert(self.u_boundary)
        
        # Initial
        self.t_initial_tensor = convert(self.t_initial)
        self.x_initial_tensor = convert(self.x_initial)
        if hasattr(self, 'u_initial'):
            self.u_initial_tensor = convert(self.u_initial)
        
        # Observation
        self.t_observation_tensor = convert(self.t_observation)
        self.x_observation_tensor = convert(self.x_observation)
        if hasattr(self, 'u_observation'):
            self.u_observation_tensor = convert(self.u_observation)
    
    def __len__(self) -> int:
        """Return total number of collocation points."""
        return self.config.n_collocation
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single collocation point."""
        sample = {
            't': self.t_collocation_tensor[idx],
            'x': self.x_collocation_tensor[idx],
        }
        if hasattr(self, 'u_collocation_tensor'):
            sample['u'] = self.u_collocation_tensor[idx]
        return sample
    
    def get_collocation_batch(
        self, 
        batch_size: Optional[int] = None,
        requires_grad: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Get a batch of collocation points for training.
        
        Args:
            batch_size: Number of points (None for all)
            requires_grad: Whether tensors require gradients
            
        Returns:
            Dictionary with 't', 'x', and optionally 'u'
        """
        if batch_size is None or batch_size >= self.config.n_collocation:
            indices = slice(None)
        else:
            indices = np.random.choice(
                self.config.n_collocation, batch_size, replace=False
            )
        
        t = self.t_collocation_tensor[indices].clone()
        x = self.x_collocation_tensor[indices].clone()
        
        if requires_grad:
            t.requires_grad_(True)
            x.requires_grad_(True)
        
        result = {'t': t, 'x': x}
        if hasattr(self, 'u_collocation_tensor'):
            result['u'] = self.u_collocation_tensor[indices]
        
        return result
    
    def get_boundary_batch(
        self, 
        batch_size: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Get a batch of boundary points."""
        if len(self.t_boundary_tensor) == 0:
            return {'t': torch.tensor([]), 'x': torch.tensor([])}
        
        if batch_size is None or batch_size >= len(self.t_boundary_tensor):
            indices = slice(None)
        else:
            indices = np.random.choice(
                len(self.t_boundary_tensor), batch_size, replace=False
            )
        
        result = {
            't': self.t_boundary_tensor[indices],
            'x': self.x_boundary_tensor[indices],
        }
        if hasattr(self, 'u_boundary_tensor'):
            result['u'] = self.u_boundary_tensor[indices]
        
        return result
    
    def get_initial_batch(
        self, 
        batch_size: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Get a batch of initial condition points."""
        if len(self.t_initial_tensor) == 0:
            return {'t': torch.tensor([]), 'x': torch.tensor([])}
        
        if batch_size is None or batch_size >= len(self.t_initial_tensor):
            indices = slice(None)
        else:
            indices = np.random.choice(
                len(self.t_initial_tensor), batch_size, replace=False
            )
        
        result = {
            't': self.t_initial_tensor[indices],
            'x': self.x_initial_tensor[indices],
        }
        if hasattr(self, 'u_initial_tensor'):
            result['u'] = self.u_initial_tensor[indices]
        
        return result
    
    def get_observation_batch(
        self, 
        batch_size: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Get a batch of observation/measurement points."""
        if len(self.t_observation_tensor) == 0:
            return {'t': torch.tensor([]), 'x': torch.tensor([])}
        
        if batch_size is None or batch_size >= len(self.t_observation_tensor):
            indices = slice(None)
        else:
            indices = np.random.choice(
                len(self.t_observation_tensor), batch_size, replace=False
            )
        
        result = {
            't': self.t_observation_tensor[indices],
            'x': self.x_observation_tensor[indices],
        }
        if hasattr(self, 'u_observation_tensor'):
            result['u'] = self.u_observation_tensor[indices]
        
        return result
    
    def get_training_batch(
        self,
        n_collocation: Optional[int] = None,
        n_boundary: Optional[int] = None,
        n_initial: Optional[int] = None,
        n_observation: Optional[int] = None,
        requires_grad: bool = True
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get a complete training batch with all point types.
        
        Returns:
            Dictionary with 'collocation', 'boundary', 'initial', 'observation' keys
        """
        return {
            'collocation': self.get_collocation_batch(n_collocation, requires_grad),
            'boundary': self.get_boundary_batch(n_boundary),
            'initial': self.get_initial_batch(n_initial),
            'observation': self.get_observation_batch(n_observation),
        }
    
    def to(self, device: Device) -> 'PINNDataset':
        """Move all data to specified device."""
        self.device = device
        self._to_tensors()  # Re-convert with new device
        return self
    
    def save(self, filepath: str, format: str = 'npz') -> None:
        """Save dataset to file."""
        data = {
            't_collocation': self.t_collocation,
            'x_collocation': self.x_collocation,
            't_boundary': self.t_boundary,
            'x_boundary': self.x_boundary,
            't_initial': self.t_initial,
            'x_initial': self.x_initial,
            't_observation': self.t_observation,
            'x_observation': self.x_observation,
        }
        
        if hasattr(self, 'u_collocation'):
            data['u_collocation'] = self.u_collocation
            data['u_boundary'] = self.u_boundary
            data['u_initial'] = self.u_initial
            data['u_observation'] = self.u_observation
        
        metadata = DatasetMetadata(
            dataset_type='pinn',
            spatial_dim=self.config.bounds.spatial_dim,
            n_samples=self.config.n_collocation,
            bounds={
                't_min': self.config.bounds.t_min,
                't_max': self.config.bounds.t_max,
                'x_min': self.config.bounds.x_min.tolist(),
                'x_max': self.config.bounds.x_max.tolist(),
            },
            normalizer=self.normalizer.to_dict() if self.normalizer else None,
        )
        
        save_dataset(filepath, data, metadata, format=format)


class PINNCombinedDataset(Dataset):
    """
    Alternative PINN dataset that combines all point types into a single dataset.
    
    Each sample includes a point type indicator.
    """
    
    COLLOCATION = 0
    BOUNDARY = 1
    INITIAL = 2
    OBSERVATION = 3
    
    def __init__(
        self,
        config: PINNDatasetConfig,
        solution_function: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        device: Device = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        """Initialize combined PINN dataset."""
        self.config = config
        self.device = device
        self.dtype = dtype
        
        # Generate base dataset
        self._base = PINNDataset(
            config=config,
            solution_function=solution_function,
            device=device,
            dtype=dtype
        )
        
        # Combine all points with type indicators
        self._combine_points()
    
    def _combine_points(self):
        """Combine all point types into single arrays."""
        t_list = [self._base.t_collocation_tensor]
        x_list = [self._base.x_collocation_tensor]
        type_list = [torch.full((len(self._base.t_collocation_tensor),), 
                                self.COLLOCATION, device=self.device)]
        
        if len(self._base.t_boundary_tensor) > 0:
            t_list.append(self._base.t_boundary_tensor)
            x_list.append(self._base.x_boundary_tensor)
            type_list.append(torch.full((len(self._base.t_boundary_tensor),),
                                        self.BOUNDARY, device=self.device))
        
        if len(self._base.t_initial_tensor) > 0:
            t_list.append(self._base.t_initial_tensor)
            x_list.append(self._base.x_initial_tensor)
            type_list.append(torch.full((len(self._base.t_initial_tensor),),
                                        self.INITIAL, device=self.device))
        
        if len(self._base.t_observation_tensor) > 0:
            t_list.append(self._base.t_observation_tensor)
            x_list.append(self._base.x_observation_tensor)
            type_list.append(torch.full((len(self._base.t_observation_tensor),),
                                        self.OBSERVATION, device=self.device))
        
        self.t = torch.cat(t_list)
        self.x = torch.cat(x_list)
        self.point_type = torch.cat(type_list).long()
        
        # Combine u values if available
        if hasattr(self._base, 'u_collocation_tensor'):
            u_list = [self._base.u_collocation_tensor]
            if len(self._base.u_boundary_tensor) > 0:
                u_list.append(self._base.u_boundary_tensor)
            if len(self._base.u_initial_tensor) > 0:
                u_list.append(self._base.u_initial_tensor)
            if len(self._base.u_observation_tensor) > 0:
                u_list.append(self._base.u_observation_tensor)
            self.u = torch.cat(u_list)
    
    def __len__(self) -> int:
        return len(self.t)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            't': self.t[idx],
            'x': self.x[idx],
            'point_type': self.point_type[idx],
        }
        if hasattr(self, 'u'):
            sample['u'] = self.u[idx]
        return sample


# =============================================================================
# Factory Functions
# =============================================================================

def create_pinn_dataset(
    solution_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    bounds: DomainBounds,
    n_collocation: int = 10000,
    n_boundary: int = 2000,
    n_initial: int = 2000,
    t_distribution: Optional[SamplingDistribution] = None,
    x_distribution: Optional[SamplingDistribution] = None,
    device: Device = 'cpu',
    normalize: bool = True,
    seed: Optional[int] = None
) -> PINNDataset:
    """
    Factory function to create a PINN dataset.
    
    Args:
        solution_function: Function u = f(t, x)
        bounds: Domain bounds
        n_collocation: Number of collocation points
        n_boundary: Number of boundary points
        n_initial: Number of initial condition points
        t_distribution: Temporal sampling distribution
        x_distribution: Spatial sampling distribution
        device: Device for tensors
        normalize: Whether to normalize data
        seed: Random seed
        
    Returns:
        Configured PINNDataset
    """
    config = PINNDatasetConfig(
        n_collocation=n_collocation,
        n_boundary=n_boundary,
        n_initial=n_initial,
        bounds=bounds,
        t_distribution=t_distribution,
        x_distribution=x_distribution,
        normalize=normalize,
        seed=seed
    )
    
    return PINNDataset(
        config=config,
        solution_function=solution_function,
        device=device
    )


def create_pinn_dataset_from_generator(
    solution_generator: Callable[..., PDESolutionGenerator],
    bounds: DomainBounds,
    n_collocation: int = 10000,
    n_boundary: int = 2000,
    n_initial: int = 2000,
    device: Device = 'cpu',
    seed: Optional[int] = None,
    **generator_kwargs
) -> PINNDataset:
    """
    Factory function to create a PINN dataset from a solution generator.
    
    Args:
        solution_generator: Generator yielding (t, x, u) tuples
        bounds: Domain bounds
        n_collocation: Number of collocation points
        n_boundary: Number of boundary points
        n_initial: Number of initial condition points
        device: Device for tensors
        seed: Random seed
        **generator_kwargs: Arguments for the generator
        
    Returns:
        Configured PINNDataset
    """
    config = PINNDatasetConfig(
        n_collocation=n_collocation,
        n_boundary=n_boundary,
        n_initial=n_initial,
        bounds=bounds,
        seed=seed
    )
    
    return PINNDataset(
        config=config,
        solution_generator=solution_generator,
        device=device,
        **generator_kwargs
    )
