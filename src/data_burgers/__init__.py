"""
Burgers Equation Dataset Submodule

Generates PyTorch datasets for training neural networks on the Burgers equation
using different architectures (PINN, LSTM/RNN, Galerkin NN).

Example Usage:
--------------
```python
from data_burgers import (
    BurgersInitialConditions,
    create_burgers_pinn_dataset,
    create_burgers_sequential_dataset,
    create_burgers_galerkin_dataset,
    generate_all_burgers_datasets
)

# Use default sine initial condition
pinn_data = create_burgers_pinn_dataset(
    Tmax=5.0,
    n_collocation=10000,
    device='cuda'
)

# Use custom initial condition
ic = BurgersInitialConditions.gaussian(amplitude=2.0, width=0.5)
seq_data = create_burgers_sequential_dataset(
    initial_condition=ic,
    seq_length=20,
    pred_length=5,
    device='cuda'
)

# Generate all datasets at once
datasets = generate_all_burgers_datasets(
    Tmax=5.0,
    save_path='./burgers_data'
)
```
"""

from .burgers_dataset import (
    # Solution computation
    num_approx_burgers,
    BurgersSolution,
    
    # Initial conditions
    BurgersInitialCondition,
    BurgersInitialConditions,
    
    # Dataset factories
    create_burgers_pinn_dataset,
    create_burgers_sequential_dataset,
    create_burgers_galerkin_dataset,
    generate_all_burgers_datasets,
    
    # Utility functions
    get_burgers_solution_function,
    get_burgers_generator,
)

__all__ = [
    # Solution
    'num_approx_burgers',
    'BurgersSolution',
    
    # Initial conditions
    'BurgersInitialCondition',
    'BurgersInitialConditions',
    
    # Factories
    'create_burgers_pinn_dataset',
    'create_burgers_sequential_dataset',
    'create_burgers_galerkin_dataset',
    'generate_all_burgers_datasets',
    
    # Utilities
    'get_burgers_solution_function',
    'get_burgers_generator',
]
