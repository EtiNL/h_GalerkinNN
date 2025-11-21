"""
Submodule 5: Utilities for PDE dataset generation.

Contains sampling distributions, common types, and helper functions
used across all dataset submodules.
"""

import numpy as np
from typing import (
    Callable, Generator, Tuple, Optional, Dict, Any, 
    Union, List, Protocol, TypeVar, Iterator
)
from dataclasses import dataclass, field
from enum import Enum
import torch


# =============================================================================
# Type Definitions
# =============================================================================

# Solution/Simulation generator type: yields (t, x, u) tuples or arrays
PDESolutionGenerator = Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]

# Sampling distribution callable type
SamplingDistribution = Callable[[int], np.ndarray]

# Device type
Device = Union[str, torch.device]


class DomainType(Enum):
    """Type of domain for sampling."""
    INTERIOR = "interior"
    BOUNDARY = "boundary"
    INITIAL = "initial"
    FULL = "full"


@dataclass
class DomainBounds:
    """Defines the spatial and temporal bounds of the PDE domain."""
    t_min: float = 0.0
    t_max: float = 1.0
    x_min: Union[float, np.ndarray] = 0.0
    x_max: Union[float, np.ndarray] = 1.0
    spatial_dim: int = 1
    
    def __post_init__(self):
        # Convert to arrays for consistency
        if isinstance(self.x_min, (int, float)):
            self.x_min = np.array([self.x_min] * self.spatial_dim)
        if isinstance(self.x_max, (int, float)):
            self.x_max = np.array([self.x_max] * self.spatial_dim)
        self.x_min = np.asarray(self.x_min)
        self.x_max = np.asarray(self.x_max)


@dataclass
class SamplingConfig:
    """Configuration for sampling from PDE domain."""
    n_interior: int = 10000
    n_boundary: int = 1000
    n_initial: int = 1000
    t_distribution: Optional[SamplingDistribution] = None
    x_distribution: Optional[SamplingDistribution] = None
    seed: Optional[int] = None


# =============================================================================
# Sampling Distribution Factories
# =============================================================================

class SamplingDistributions:
    """
    Factory class for creating sampling distributions from np.random module.
    
    All methods return callables that take n_samples and return samples
    in the range [0, 1], which can then be scaled to the actual domain.
    """
    
    @staticmethod
    def uniform(low: float = 0.0, high: float = 1.0) -> SamplingDistribution:
        """Uniform distribution over [low, high]."""
        def sampler(n: int) -> np.ndarray:
            return np.random.uniform(low, high, n)
        return sampler
    
    @staticmethod
    def normal(mean: float = 0.5, std: float = 0.2, 
               clip_low: float = 0.0, clip_high: float = 1.0) -> SamplingDistribution:
        """Truncated normal distribution."""
        def sampler(n: int) -> np.ndarray:
            samples = np.random.normal(mean, std, n)
            return np.clip(samples, clip_low, clip_high)
        return sampler
    
    @staticmethod
    def beta(a: float = 2.0, b: float = 2.0) -> SamplingDistribution:
        """Beta distribution (good for concentrating samples in interior)."""
        def sampler(n: int) -> np.ndarray:
            return np.random.beta(a, b, n)
        return sampler
    
    @staticmethod
    def exponential(scale: float = 0.3, 
                    clip_high: float = 1.0) -> SamplingDistribution:
        """Exponential distribution (good for initial conditions)."""
        def sampler(n: int) -> np.ndarray:
            samples = np.random.exponential(scale, n)
            return np.clip(samples, 0.0, clip_high)
        return sampler
    
    @staticmethod
    def log_uniform(low: float = 1e-3, high: float = 1.0) -> SamplingDistribution:
        """Log-uniform distribution (good for multi-scale problems)."""
        def sampler(n: int) -> np.ndarray:
            log_low, log_high = np.log(low), np.log(high)
            return np.exp(np.random.uniform(log_low, log_high, n))
        return sampler
    
    @staticmethod
    def latin_hypercube(n_dims: int = 1) -> Callable[[int], np.ndarray]:
        """Latin Hypercube Sampling for better space coverage."""
        def sampler(n: int) -> np.ndarray:
            samples = np.zeros((n, n_dims))
            for dim in range(n_dims):
                perm = np.random.permutation(n)
                samples[:, dim] = (perm + np.random.uniform(size=n)) / n
            return samples if n_dims > 1 else samples.flatten()
        return sampler
    
    @staticmethod
    def sobol(n_dims: int = 1) -> Callable[[int], np.ndarray]:
        """
        Sobol sequence for quasi-random sampling.
        Falls back to uniform if scipy not available.
        """
        try:
            from scipy.stats import qmc
            def sampler(n: int) -> np.ndarray:
                sobol_engine = qmc.Sobol(d=n_dims, scramble=True)
                samples = sobol_engine.random(n)
                return samples if n_dims > 1 else samples.flatten()
            return sampler
        except ImportError:
            return SamplingDistributions.uniform()
    
    @staticmethod
    def mixture(distributions: List[Tuple[SamplingDistribution, float]]) -> SamplingDistribution:
        """
        Mixture of distributions with specified weights.
        
        Args:
            distributions: List of (distribution, weight) tuples
        """
        dists, weights = zip(*distributions)
        weights = np.array(weights) / np.sum(weights)
        
        def sampler(n: int) -> np.ndarray:
            # Determine how many samples from each distribution
            counts = np.random.multinomial(n, weights)
            samples = []
            for dist, count in zip(dists, counts):
                if count > 0:
                    samples.append(dist(count))
            return np.concatenate(samples) if samples else np.array([])
        return sampler
    
    @staticmethod
    def adaptive_refinement(base_dist: SamplingDistribution,
                           refinement_regions: List[Tuple[float, float, float]]) -> SamplingDistribution:
        """
        Adaptive sampling with refinement in specified regions.
        
        Args:
            base_dist: Base sampling distribution
            refinement_regions: List of (center, width, density_multiplier) tuples
        """
        def sampler(n: int) -> np.ndarray:
            base_samples = base_dist(n)
            
            # Add refined samples in specified regions
            refined_samples = []
            for center, width, multiplier in refinement_regions:
                n_refined = int(n * (multiplier - 1) * width)
                refined = np.random.uniform(
                    center - width/2, center + width/2, n_refined
                )
                refined_samples.append(refined)
            
            if refined_samples:
                all_samples = np.concatenate([base_samples] + refined_samples)
                # Subsample to get back to n samples
                indices = np.random.choice(len(all_samples), n, replace=False)
                return all_samples[indices]
            return base_samples
        return sampler


# =============================================================================
# Domain Sampling Utilities
# =============================================================================

def sample_domain(
    bounds: DomainBounds,
    n_samples: int,
    domain_type: DomainType = DomainType.INTERIOR,
    t_dist: Optional[SamplingDistribution] = None,
    x_dist: Optional[SamplingDistribution] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample points from the PDE domain.
    
    Args:
        bounds: Domain bounds
        n_samples: Number of samples
        domain_type: Type of domain region to sample
        t_dist: Distribution for temporal sampling (default: uniform)
        x_dist: Distribution for spatial sampling (default: uniform)
        seed: Random seed
        
    Returns:
        Tuple of (t_samples, x_samples) arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Default distributions
    if t_dist is None:
        t_dist = SamplingDistributions.uniform()
    if x_dist is None:
        x_dist = SamplingDistributions.uniform()
    
    if domain_type == DomainType.INITIAL:
        # t = t_min, x varies
        t_samples = np.full(n_samples, bounds.t_min)
        x_samples = _sample_spatial(x_dist, bounds, n_samples)
        
    elif domain_type == DomainType.BOUNDARY:
        # x on boundary, t varies
        t_samples = t_dist(n_samples) * (bounds.t_max - bounds.t_min) + bounds.t_min
        x_samples = _sample_boundary(bounds, n_samples)
        
    elif domain_type == DomainType.INTERIOR:
        # Both t and x in interior
        t_samples = t_dist(n_samples) * (bounds.t_max - bounds.t_min) + bounds.t_min
        x_samples = _sample_spatial(x_dist, bounds, n_samples)
        
    else:  # FULL
        t_samples = t_dist(n_samples) * (bounds.t_max - bounds.t_min) + bounds.t_min
        x_samples = _sample_spatial(x_dist, bounds, n_samples)
    
    return t_samples, x_samples


def _sample_spatial(
    x_dist: SamplingDistribution,
    bounds: DomainBounds,
    n_samples: int
) -> np.ndarray:
    """Sample spatial coordinates."""
    if bounds.spatial_dim == 1:
        samples = x_dist(n_samples)
        return samples * (bounds.x_max[0] - bounds.x_min[0]) + bounds.x_min[0]
    else:
        samples = np.zeros((n_samples, bounds.spatial_dim))
        for d in range(bounds.spatial_dim):
            s = x_dist(n_samples)
            samples[:, d] = s * (bounds.x_max[d] - bounds.x_min[d]) + bounds.x_min[d]
        return samples


def _sample_boundary(bounds: DomainBounds, n_samples: int) -> np.ndarray:
    """Sample points on the spatial boundary."""
    if bounds.spatial_dim == 1:
        # Randomly choose left or right boundary
        choices = np.random.choice([0, 1], n_samples)
        return np.where(choices == 0, bounds.x_min[0], bounds.x_max[0])
    else:
        # For higher dimensions, sample uniformly on faces
        samples = np.zeros((n_samples, bounds.spatial_dim))
        for i in range(n_samples):
            # Choose a random dimension and side
            dim = np.random.randint(bounds.spatial_dim)
            side = np.random.choice([0, 1])
            
            # Set the chosen dimension to boundary value
            for d in range(bounds.spatial_dim):
                if d == dim:
                    samples[i, d] = bounds.x_min[d] if side == 0 else bounds.x_max[d]
                else:
                    samples[i, d] = np.random.uniform(bounds.x_min[d], bounds.x_max[d])
        return samples


# =============================================================================
# Solution Generator Utilities
# =============================================================================

def evaluate_solution_generator(
    solution_gen: Callable[..., PDESolutionGenerator],
    t_points: np.ndarray,
    x_points: np.ndarray,
    **gen_kwargs
) -> np.ndarray:
    """
    Evaluate a solution generator at specified (t, x) points.
    
    Args:
        solution_gen: Generator function that yields (t, x, u) tuples
        t_points: Time points to evaluate
        x_points: Spatial points to evaluate
        **gen_kwargs: Additional arguments to pass to generator
        
    Returns:
        Array of solution values at the specified points
    """
    # Collect all generated data
    t_data, x_data, u_data = [], [], []
    for t, x, u in solution_gen(**gen_kwargs):
        t_data.append(t)
        x_data.append(x)
        u_data.append(u)
    
    t_data = np.concatenate([np.atleast_1d(t) for t in t_data])
    x_data = np.concatenate([np.atleast_1d(x) for x in x_data])
    u_data = np.concatenate([np.atleast_1d(u) for u in u_data])
    
    # Interpolate to requested points
    from scipy.interpolate import griddata
    
    points = np.column_stack([t_data.flatten(), x_data.flatten()])
    query_points = np.column_stack([t_points.flatten(), x_points.flatten()])
    
    u_interp = griddata(points, u_data.flatten(), query_points, method='linear')
    return u_interp.reshape(t_points.shape)


def grid_from_generator(
    solution_gen: Callable[..., PDESolutionGenerator],
    bounds: DomainBounds,
    nt: int,
    nx: int,
    **gen_kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a regular grid from a solution generator.
    
    Returns:
        Tuple of (T, X, U) meshgrid arrays
    """
    t_vals = np.linspace(bounds.t_min, bounds.t_max, nt)
    x_vals = np.linspace(bounds.x_min[0], bounds.x_max[0], nx)
    T, X = np.meshgrid(t_vals, x_vals, indexing='ij')
    
    U = evaluate_solution_generator(
        solution_gen, T.flatten(), X.flatten(), **gen_kwargs
    ).reshape(T.shape)
    
    return T, X, U


# =============================================================================
# Tensor Conversion Utilities
# =============================================================================

def to_tensor(
    array: np.ndarray,
    dtype: torch.dtype = torch.float32,
    device: Device = 'cpu',
    requires_grad: bool = False
) -> torch.Tensor:
    """Convert numpy array to torch tensor."""
    tensor = torch.tensor(array, dtype=dtype, device=device)
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    return tensor.detach().cpu().numpy()


def collate_pde_batch(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for PDE datasets.
    """
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch]) for key in keys}


# =============================================================================
# Normalization Utilities
# =============================================================================

@dataclass
class Normalizer:
    """Normalizer for PDE data."""
    t_mean: float = 0.0
    t_std: float = 1.0
    x_mean: Union[float, np.ndarray] = 0.0
    x_std: Union[float, np.ndarray] = 1.0
    u_mean: float = 0.0
    u_std: float = 1.0
    
    @classmethod
    def from_data(cls, t: np.ndarray, x: np.ndarray, u: np.ndarray) -> 'Normalizer':
        """Create normalizer from data statistics."""
        return cls(
            t_mean=float(np.mean(t)),
            t_std=float(np.std(t)) + 1e-8,
            x_mean=np.mean(x, axis=0) if x.ndim > 1 else float(np.mean(x)),
            x_std=np.std(x, axis=0) + 1e-8 if x.ndim > 1 else float(np.std(x)) + 1e-8,
            u_mean=float(np.mean(u)),
            u_std=float(np.std(u)) + 1e-8
        )
    
    def normalize_t(self, t: np.ndarray) -> np.ndarray:
        return (t - self.t_mean) / self.t_std
    
    def normalize_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_std
    
    def normalize_u(self, u: np.ndarray) -> np.ndarray:
        return (u - self.u_mean) / self.u_std
    
    def denormalize_t(self, t: np.ndarray) -> np.ndarray:
        return t * self.t_std + self.t_mean
    
    def denormalize_x(self, x: np.ndarray) -> np.ndarray:
        return x * self.x_std + self.x_mean
    
    def denormalize_u(self, u: np.ndarray) -> np.ndarray:
        return u * self.u_std + self.u_mean
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            't_mean': self.t_mean, 't_std': self.t_std,
            'x_mean': self.x_mean if isinstance(self.x_mean, float) else self.x_mean.tolist(),
            'x_std': self.x_std if isinstance(self.x_std, float) else self.x_std.tolist(),
            'u_mean': self.u_mean, 'u_std': self.u_std
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Normalizer':
        return cls(
            t_mean=d['t_mean'], t_std=d['t_std'],
            x_mean=np.array(d['x_mean']) if isinstance(d['x_mean'], list) else d['x_mean'],
            x_std=np.array(d['x_std']) if isinstance(d['x_std'], list) else d['x_std'],
            u_mean=d['u_mean'], u_std=d['u_std']
        )
