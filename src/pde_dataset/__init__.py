"""
PDE Dataset Generation Module

A modular PyTorch dataset generation system for PDE solutions and simulations.
Supports multiple neural network architectures:
- PINN (Physics-Informed Neural Networks)
- Sequential (LSTM/RNN/Transformer)
- Galerkin Neural Networks

Submodules:
-----------
- utils: Sampling distributions, common utilities, normalization
- io: Save/load compressed datasets to/from disk
- pinn: Datasets for PINN architectures
- sequential: Datasets for LSTM/RNN architectures
- galerkin: Datasets for Galerkin NN architectures

Example Usage:
--------------
```python
import numpy as np
from pde_dataset import (
    create_pinn_dataset,
    create_sequential_dataset,
    create_galerkin_dataset,
    DomainBounds,
    SamplingDistributions
)

# Define domain
bounds = DomainBounds(t_min=0, t_max=1, x_min=-1, x_max=1)

# Define solution (e.g., Burgers equation)
def solution(t, x):
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)

# Create PINN dataset
pinn_data = create_pinn_dataset(
    solution_function=solution,
    bounds=bounds,
    n_collocation=10000,
    t_distribution=SamplingDistributions.beta(2, 2),
    device='cuda'
)

# Create Sequential dataset
seq_data = create_sequential_dataset(
    solution_function=solution,
    bounds=bounds,
    seq_length=10,
    pred_length=1,
    nt=100,
    nx=100
)

# Save and load
pinn_data.save('pinn_burgers.npz')
from pde_dataset import load_dataset
data, metadata = load_dataset('pinn_burgers.npz', device='cuda')
```
"""

__version__ = '0.1.0'
__author__ = 'PDE Dataset Module'

# =============================================================================
# Import from submodules
# =============================================================================

# Utils
from .utils import (
    # Types
    DomainBounds,
    DomainType,
    SamplingConfig,
    Device,
    PDESolutionGenerator,
    SamplingDistribution,
    
    # Sampling
    SamplingDistributions,
    sample_domain,
    
    # Normalization
    Normalizer,
    
    # Tensor utilities
    to_tensor,
    to_numpy,
    collate_pde_batch,
    
    # Solution utilities
    evaluate_solution_generator,
    grid_from_generator,
)

# IO
from .io import (
    # Metadata
    DatasetMetadata,
    CompressionFormat,
    
    # Save functions
    save_dataset,
    save_dataset_npz,
    save_dataset_hdf5,
    save_dataset_torch,
    save_dataset_pickle,
    
    # Load functions
    load_dataset,
    load_dataset_npz,
    load_dataset_hdf5,
    load_dataset_torch,
    load_dataset_pickle,
    
    # Dataset wrappers
    LoadedPDEDataset,
    ChunkedDatasetLoader,
)

# PINN
from .pinn import (
    PINNDatasetConfig,
    PINNDataset,
    PINNCombinedDataset,
    create_pinn_dataset,
    create_pinn_dataset_from_generator,
)

# Sequential
from .sequential import (
    SequentialDatasetConfig,
    SequentialPDEDataset,
    MultiStepSequentialDataset,
    SlidingWindowDataset,
    create_sequential_dataset,
    create_sequential_dataset_from_generator,
    create_sequential_dataset_from_array,
)

# Galerkin
from .galerkin import (
    GalerkinDatasetConfig,
    GalerkinDataset,
    GalerkinBasisDataset,
    QuadratureRule,
    BilinearForm,
    L2InnerProduct,
    H1InnerProduct,
    H2InnerProduct,
    create_galerkin_dataset,
)

# add near the other imports
from .neural_galerkin_ode import (
    NeuralGalerkinDatasetConfig,
    NeuralGalerkinDataset,
    create_NeuralGalerkin_dataset,
)

# =============================================================================
# Package-level utilities
# =============================================================================

def create_dataset_from_simulation(
    simulation_generator,
    architecture: str = 'pinn',
    bounds: DomainBounds = None,
    device: str = 'cpu',
    **kwargs
):
    """
    Create a dataset from a simulation/solution generator.
    
    This is a high-level factory function that selects the appropriate
    dataset class based on the target architecture.
    
    Args:
        simulation_generator: Generator yielding (t, x, u) tuples
        architecture: Target architecture ('pinn', 'sequential', 'galerkin')
        bounds: Domain bounds (inferred from generator if None)
        device: Device for tensors
        **kwargs: Additional arguments for specific dataset types
        
    Returns:
        Appropriate dataset instance
    """
    if bounds is None:
        bounds = DomainBounds()
    
    if architecture.lower() == 'pinn':
        return create_pinn_dataset_from_generator(
            simulation_generator, bounds, device=device, **kwargs
        )
    elif architecture.lower() in ['sequential', 'lstm', 'rnn']:
        return create_sequential_dataset_from_generator(
            simulation_generator, bounds, device=device, **kwargs
        )
    elif architecture.lower() == 'galerkin':
        # Galerkin needs solution function, not generator
        raise ValueError(
            "Galerkin datasets require a direct solution function, "
            "not a generator. Use create_galerkin_dataset directly."
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def create_dataset_from_function(
    solution_function,
    architecture: str = 'pinn',
    bounds: DomainBounds = None,
    device: str = 'cpu',
    **kwargs
):
    """
    Create a dataset from a solution function u = f(t, x).
    
    Args:
        solution_function: Function (t, x) -> u
        architecture: Target architecture ('pinn', 'sequential', 'galerkin')
        bounds: Domain bounds
        device: Device for tensors
        **kwargs: Additional arguments for specific dataset types
        
    Returns:
        Appropriate dataset instance
    """
    if bounds is None:
        bounds = DomainBounds()
    
    if architecture.lower() == 'pinn':
        return create_pinn_dataset(
            solution_function, bounds, device=device, **kwargs
        )
    elif architecture.lower() in ['sequential', 'lstm', 'rnn']:
        return create_sequential_dataset(
            solution_function, bounds, device=device, **kwargs
        )
    elif architecture.lower() == 'galerkin':
        # For Galerkin, solution_function should only take x
        return create_galerkin_dataset(
            bounds, solution_function=solution_function, device=device, **kwargs
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Version
    '__version__',
    
    # Utils
    'DomainBounds',
    'DomainType',
    'SamplingConfig',
    'SamplingDistributions',
    'sample_domain',
    'Normalizer',
    'to_tensor',
    'to_numpy',
    'collate_pde_batch',
    'evaluate_solution_generator',
    'grid_from_generator',
    
    # IO
    'DatasetMetadata',
    'CompressionFormat',
    'save_dataset',
    'load_dataset',
    'save_dataset_npz',
    'save_dataset_hdf5',
    'save_dataset_torch',
    'save_dataset_pickle',
    'load_dataset_npz',
    'load_dataset_hdf5',
    'load_dataset_torch',
    'load_dataset_pickle',
    'LoadedPDEDataset',
    'ChunkedDatasetLoader',
    
    # PINN
    'PINNDatasetConfig',
    'PINNDataset',
    'PINNCombinedDataset',
    'create_pinn_dataset',
    'create_pinn_dataset_from_generator',
    
    # Sequential
    'SequentialDatasetConfig',
    'SequentialPDEDataset',
    'MultiStepSequentialDataset',
    'SlidingWindowDataset',
    'create_sequential_dataset',
    'create_sequential_dataset_from_generator',
    'create_sequential_dataset_from_array',
    
    # Galerkin
    'GalerkinDatasetConfig',
    'GalerkinDataset',
    'GalerkinBasisDataset',
    'QuadratureRule',
    'BilinearForm',
    'L2InnerProduct',
    'H1InnerProduct',
    'H2InnerProduct',
    'create_galerkin_dataset',

    # Time sampled projection
    "NeuralGalerkinDatasetConfig",
    "NeuralGalerkinDataset",
    "create_NeuralGalerkin_dataset",
    
    # High-level factories
    'create_dataset_from_simulation',
    'create_dataset_from_function',
]
