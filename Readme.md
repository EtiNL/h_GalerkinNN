# Neural Galerkin ODE - Predict Function Documentation

## Overview

Handles prediction, reconstruction, error computation, and visualization.

## Key Features

### 1. **Prediction from Initial Conditions**
- Takes a trained ODE function and predicts coefficient trajectories
- Uses the same ODE solver (dopri5, rk4, etc.) as training
- Handles normalized/denormalized data automatically

### 2. **Spatial Reconstruction**
- Reconstructs full spatial solution `u(t,x)` from Galerkin coefficients `c_k(t)`
- Uses the basis functions stored in your dataset
- Returns solutions on the original spatial grid

### 3. **Error Computation**
- Computes MSE in coefficient space
- Computes MSE in spatial domain (if reconstruction enabled)
- Compares predictions against ground truth

### 4. **Visualization**
- **Plot 1**: Coefficient trajectories over time (first 5 modes)
- **Plot 2**: Spatial solution snapshots at multiple times
- **Plot 3**: Spatiotemporal heatmaps (prediction, ground truth, error)

## Function Signature

```python
predict(
    func: nn.Module,                    # Trained CoeffODEFunc model
    dataset,                            # NeuralGalerkinDataset
    ic_idx: int = 0,                   # Which initial condition to use
    t_eval: torch.Tensor | None = None, # Custom time points (optional)
    method: str = "dopri5",            # ODE solver method
    rtol: float = 1e-6,                # Relative tolerance
    atol: float = 1e-6,                # Absolute tolerance
    reconstruct: bool = True,          # Reconstruct u(t,x)?
    compare_ground_truth: bool = True, # Compute errors?
    plot: bool = True,                 # Create visualizations?
)
```

## Return Value

Returns a dictionary with:
```python
{
    't': time points (nT,),
    't_phys': denormalized time points (nT,),
    'c_pred': predicted coefficients (nT, K),
    'c_pred_phys': denormalized predicted coefficients (nT, K),
    'c_true': ground truth coefficients (nT, K),          # if compare_ground_truth=True
    'c_true_phys': denormalized ground truth (nT, K),     # if compare_ground_truth=True
    'u_pred': reconstructed spatial solution (nT, nx),    # if reconstruct=True
    'u_true': ground truth spatial solution (nT, nx),     # if reconstruct=True & compare_ground_truth=True
    'x_grid': spatial grid (nx,),                         # if reconstruct=True
    'mse_coeff': MSE in coefficient space,                # if compare_ground_truth=True
    'mse_spatial': MSE in spatial domain,                 # if reconstruct=True & compare_ground_truth=True
}
```

## Usage Examples

### Example 1: Basic Prediction
```python
from neural_galerkin import predict

# After training your model
func, info = train_neural_ode_with_val(ds, epochs=2000, ...)

# Predict and visualize
results = predict(
    func=func,
    dataset=ds,
    ic_idx=0,
    reconstruct=True,
    compare_ground_truth=True,
    plot=True
)

print(f"MSE in coefficient space: {results['mse_coeff']:.6e}")
print(f"MSE in spatial domain: {results['mse_spatial']:.6e}")
```

### Example 2: Custom Time Points
```python
# Predict at 200 evenly-spaced time points
t_custom = torch.linspace(0, 2.0, 200, device='cuda')

results = predict(
    func=func,
    dataset=ds,
    ic_idx=0,
    t_eval=t_custom,
    reconstruct=True,
    compare_ground_truth=False,  # No ground truth at these times
    plot=True
)
```

### Example 3: Batch Evaluation (No Plots)
```python
# Evaluate on multiple ICs without creating plots
mse_list = []

for ic_idx in range(len(ds)):
    results = predict(
        func=func,
        dataset=ds,
        ic_idx=ic_idx,
        reconstruct=True,
        compare_ground_truth=True,
        plot=False  # Disable plotting for batch processing
    )
    mse_list.append(results['mse_spatial'])

print(f"Average MSE: {sum(mse_list)/len(mse_list):.6e}")
```

### Example 4: Extrapolation Beyond Training Range
```python
# Test extrapolation capability
t_extend = torch.linspace(0, 3.0, 300, device='cuda')  # Beyond training range

results = predict(
    func=func,
    dataset=ds,
    ic_idx=0,
    t_eval=t_extend,
    reconstruct=True,
    compare_ground_truth=False,
    plot=True
)
```

### Example 5: Save Predictions
```python
import numpy as np

# Get predictions without plots
results = predict(
    func=func,
    dataset=ds,
    ic_idx=0,
    reconstruct=True,
    plot=False
)

# Save to file
np.savez(
    'predictions.npz',
    t=results['t_phys'].numpy(),
    u=results['u_pred'].numpy(),
    x=results['x_grid'],
    c=results['c_pred_phys'].numpy()
)
```

## What the Visualizations Show

### Plot 1: Coefficient Trajectories
- Shows first 5 Galerkin coefficients over time
- Solid lines = predictions
- Dashed lines = ground truth (if available)
- Helps verify ODE learning quality

### Plot 2: Spatial Solution Snapshots
- Shows `u(t,x)` at 6 different time snapshots
- Compares predicted vs ground truth spatial profiles
- Helps visualize how well the solution is captured

### Plot 3: Spatiotemporal Heatmaps
- **Left**: Prediction heatmap
- **Middle**: Ground truth heatmap
- **Right**: Absolute error heatmap
- Shows full evolution over space and time

## Tips

1. **For training data evaluation**: Use `compare_ground_truth=True`
2. **For new predictions**: Use `compare_ground_truth=False` and provide `t_eval`
3. **For batch processing**: Use `plot=False` to speed up evaluation
4. **For debugging**: Keep `plot=True` to visually inspect results
5. **For extrapolation testing**: Use `t_eval` beyond training range
