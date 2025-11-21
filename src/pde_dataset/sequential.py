"""
Submodule 3: Dataset adapted for LSTM/RNN-like architectures.

Sequential models for PDEs typically require:
- Sequences of spatial snapshots over time
- Input: u(t_i, x) for i = 1, ..., seq_len
- Output: u(t_{i+1}, x) or future time steps

This module provides datasets that structure PDE solutions as sequences.
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
    DomainBounds, SamplingDistribution, to_tensor, Normalizer,
    PDESolutionGenerator, Device, grid_from_generator
)
from .io import save_dataset, DatasetMetadata


# =============================================================================
# Sequential Dataset Configuration
# =============================================================================

@dataclass
class SequentialDatasetConfig:
    """Configuration for sequential (LSTM/RNN) dataset generation."""
    # Sequence parameters
    seq_length: int = 10        # Input sequence length
    pred_length: int = 1        # Prediction horizon (output length)
    stride: int = 1             # Stride between sequences
    
    # Grid resolution
    nt: int = 100               # Number of time steps
    nx: int = 100               # Number of spatial points (or nx * ny for 2D)
    
    # Domain bounds
    bounds: DomainBounds = field(default_factory=DomainBounds)
    
    # Data options
    normalize: bool = True
    return_coordinates: bool = True  # Include (t, x) coordinates
    
    # Random seed
    seed: Optional[int] = None


# =============================================================================
# Sequential Dataset Classes
# =============================================================================

class SequentialPDEDataset(Dataset):
    """
    PyTorch Dataset for sequence-to-sequence PDE learning (LSTM/RNN/Transformer).
    
    Creates sequences of spatial snapshots for temporal prediction.
    """
    
    def __init__(
        self,
        config: SequentialDatasetConfig,
        solution_generator: Optional[Callable[..., PDESolutionGenerator]] = None,
        solution_function: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        precomputed_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        device: Device = 'cpu',
        dtype: torch.dtype = torch.float32,
        **generator_kwargs
    ):
        """
        Initialize sequential dataset.
        
        Args:
            config: Dataset configuration
            solution_generator: Generator function yielding (t, x, u) tuples
            solution_function: Direct function u = f(t, x) for evaluation
            precomputed_data: Optional tuple of (T, X, U) meshgrid arrays
            device: Device for tensors
            dtype: Data type for tensors
            **generator_kwargs: Arguments to pass to solution generator
        """
        self.config = config
        self.device = device
        self.dtype = dtype
        
        if config.seed is not None:
            np.random.seed(config.seed)
        
        # Generate or use precomputed grid data
        if precomputed_data is not None:
            self.T, self.X, self.U = precomputed_data
        else:
            self._generate_grid_data(
                solution_generator, solution_function, **generator_kwargs
            )
        
        # Create sequences
        self._create_sequences()
        
        # Normalize if requested
        self.normalizer = None
        if config.normalize:
            self._setup_normalizer()
            self._apply_normalization()
        
        # Convert to tensors
        self._to_tensors()
    
    def _generate_grid_data(
        self,
        solution_generator: Optional[Callable[..., PDESolutionGenerator]],
        solution_function: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]],
        **generator_kwargs
    ):
        """Generate solution on a regular grid."""
        bounds = self.config.bounds
        
        # Create grid
        t_vals = np.linspace(bounds.t_min, bounds.t_max, self.config.nt)
        x_vals = np.linspace(bounds.x_min[0], bounds.x_max[0], self.config.nx)
        self.T, self.X = np.meshgrid(t_vals, x_vals, indexing='ij')
        
        if solution_function is not None:
            self.U = solution_function(self.T, self.X)
        elif solution_generator is not None:
            self.T, self.X, self.U = grid_from_generator(
                solution_generator, bounds,
                self.config.nt, self.config.nx,
                **generator_kwargs
            )
        else:
            raise ValueError("Either solution_function or solution_generator required")
    
    def _create_sequences(self):
        """Create input-output sequence pairs."""
        seq_len = self.config.seq_length
        pred_len = self.config.pred_length
        stride = self.config.stride
        
        # U shape: (nt, nx) - each row is a spatial snapshot at time t
        n_sequences = (self.config.nt - seq_len - pred_len) // stride + 1
        
        self.input_sequences = []
        self.output_sequences = []
        self.input_times = []
        self.output_times = []
        
        for i in range(0, self.config.nt - seq_len - pred_len + 1, stride):
            # Input: seq_len snapshots
            input_seq = self.U[i:i + seq_len, :]  # Shape: (seq_len, nx)
            # Output: pred_len snapshots
            output_seq = self.U[i + seq_len:i + seq_len + pred_len, :]  # Shape: (pred_len, nx)
            
            self.input_sequences.append(input_seq)
            self.output_sequences.append(output_seq)
            
            # Time coordinates
            self.input_times.append(self.T[i:i + seq_len, 0])
            self.output_times.append(self.T[i + seq_len:i + seq_len + pred_len, 0])
        
        self.input_sequences = np.array(self.input_sequences)
        self.output_sequences = np.array(self.output_sequences)
        self.input_times = np.array(self.input_times)
        self.output_times = np.array(self.output_times)
        
        # Spatial coordinates (same for all sequences)
        self.x_coords = self.X[0, :]  # Shape: (nx,)
    
    def _setup_normalizer(self):
        """Setup data normalizer."""
        self.normalizer = Normalizer.from_data(
            self.T.flatten(),
            self.X.flatten(),
            self.U.flatten()
        )
    
    def _apply_normalization(self):
        """Apply normalization to sequences."""
        if self.normalizer:
            self.input_sequences = self.normalizer.normalize_u(self.input_sequences)
            self.output_sequences = self.normalizer.normalize_u(self.output_sequences)
            self.input_times = self.normalizer.normalize_t(self.input_times)
            self.output_times = self.normalizer.normalize_t(self.output_times)
            self.x_coords = self.normalizer.normalize_x(self.x_coords)
    
    def _to_tensors(self):
        """Convert all arrays to tensors."""
        self.input_tensor = to_tensor(
            self.input_sequences, dtype=self.dtype, device=self.device
        )
        self.output_tensor = to_tensor(
            self.output_sequences, dtype=self.dtype, device=self.device
        )
        self.input_times_tensor = to_tensor(
            self.input_times, dtype=self.dtype, device=self.device
        )
        self.output_times_tensor = to_tensor(
            self.output_times, dtype=self.dtype, device=self.device
        )
        self.x_coords_tensor = to_tensor(
            self.x_coords, dtype=self.dtype, device=self.device
        )
    
    def __len__(self) -> int:
        return len(self.input_sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sequence pair.
        
        Returns:
            Dictionary with:
                - 'input': Input sequence (seq_len, nx)
                - 'output': Output sequence (pred_len, nx)
                - 't_input': Input time coordinates (seq_len,)
                - 't_output': Output time coordinates (pred_len,)
                - 'x': Spatial coordinates (nx,)
        """
        sample = {
            'input': self.input_tensor[idx],
            'output': self.output_tensor[idx],
        }
        
        if self.config.return_coordinates:
            sample['t_input'] = self.input_times_tensor[idx]
            sample['t_output'] = self.output_times_tensor[idx]
            sample['x'] = self.x_coords_tensor
        
        return sample
    
    def get_full_trajectory(self) -> Dict[str, torch.Tensor]:
        """Get the full solution trajectory."""
        return {
            'T': to_tensor(self.T, dtype=self.dtype, device=self.device),
            'X': to_tensor(self.X, dtype=self.dtype, device=self.device),
            'U': to_tensor(self.U, dtype=self.dtype, device=self.device),
        }
    
    def to(self, device: Device) -> 'SequentialPDEDataset':
        """Move all data to specified device."""
        self.device = device
        self._to_tensors()
        return self
    
    def save(self, filepath: str, format: str = 'npz') -> None:
        """Save dataset to file."""
        data = {
            'input_sequences': self.input_sequences,
            'output_sequences': self.output_sequences,
            'input_times': self.input_times,
            'output_times': self.output_times,
            'x_coords': self.x_coords,
            'T': self.T,
            'X': self.X,
            'U': self.U,
        }
        
        metadata = DatasetMetadata(
            dataset_type='sequential',
            spatial_dim=self.config.bounds.spatial_dim,
            n_samples=len(self),
            bounds={
                't_min': self.config.bounds.t_min,
                't_max': self.config.bounds.t_max,
                'x_min': self.config.bounds.x_min.tolist(),
                'x_max': self.config.bounds.x_max.tolist(),
            },
            normalizer=self.normalizer.to_dict() if self.normalizer else None,
            extra_info={
                'seq_length': self.config.seq_length,
                'pred_length': self.config.pred_length,
                'stride': self.config.stride,
                'nt': self.config.nt,
                'nx': self.config.nx,
            }
        )
        
        save_dataset(filepath, data, metadata, format=format)


class MultiStepSequentialDataset(Dataset):
    """
    Dataset for multi-step autoregressive prediction.
    
    Designed for training models that recursively predict multiple steps ahead.
    """
    
    def __init__(
        self,
        config: SequentialDatasetConfig,
        solution_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
        n_rollout_steps: int = 10,
        device: Device = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize multi-step dataset.
        
        Args:
            config: Dataset configuration
            solution_function: Function u = f(t, x)
            n_rollout_steps: Number of steps for rollout training
            device: Device for tensors
            dtype: Data type for tensors
        """
        self.config = config
        self.n_rollout_steps = n_rollout_steps
        self.device = device
        self.dtype = dtype
        
        # Create base sequential dataset
        self._base = SequentialPDEDataset(
            config=config,
            solution_function=solution_function,
            device=device,
            dtype=dtype
        )
        
        self.normalizer = self._base.normalizer
    
    def __len__(self) -> int:
        # Adjust length to ensure we have enough steps for rollout
        return max(0, len(self._base) - self.n_rollout_steps + 1)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a multi-step sequence.
        
        Returns:
            Dictionary with:
                - 'input': Initial input sequence (seq_len, nx)
                - 'targets': Target sequences for each rollout step (n_rollout, nx)
        """
        # Get initial input
        base_sample = self._base[idx]
        
        # Collect all target steps
        targets = [self._base.output_tensor[idx]]
        for step in range(1, self.n_rollout_steps):
            if idx + step < len(self._base):
                targets.append(self._base.output_tensor[idx + step])
        
        targets = torch.stack(targets)
        
        return {
            'input': base_sample['input'],
            'targets': targets,
            't_input': base_sample.get('t_input'),
            'x': base_sample.get('x'),
        }


class SlidingWindowDataset(Dataset):
    """
    Sliding window dataset for local spatiotemporal prediction.
    
    Creates samples with spatial windows for each time step.
    """
    
    def __init__(
        self,
        config: SequentialDatasetConfig,
        solution_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
        spatial_window: int = 5,
        device: Device = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize sliding window dataset.
        
        Args:
            config: Dataset configuration
            solution_function: Function u = f(t, x)
            spatial_window: Size of spatial window (each side)
            device: Device for tensors
            dtype: Data type for tensors
        """
        self.config = config
        self.spatial_window = spatial_window
        self.device = device
        self.dtype = dtype
        
        if config.seed is not None:
            np.random.seed(config.seed)
        
        # Generate grid data
        bounds = config.bounds
        t_vals = np.linspace(bounds.t_min, bounds.t_max, config.nt)
        x_vals = np.linspace(bounds.x_min[0], bounds.x_max[0], config.nx)
        self.T, self.X = np.meshgrid(t_vals, x_vals, indexing='ij')
        self.U = solution_function(self.T, self.X)
        
        # Create windowed samples
        self._create_windows()
        
        # Normalize
        self.normalizer = None
        if config.normalize:
            self.normalizer = Normalizer.from_data(
                self.T.flatten(), self.X.flatten(), self.U.flatten()
            )
        
        self._to_tensors()
    
    def _create_windows(self):
        """Create spatiotemporal windows."""
        seq_len = self.config.seq_length
        window = self.spatial_window
        
        self.windows = []
        self.targets = []
        self.positions = []
        
        # Pad spatial dimension for boundary handling
        U_padded = np.pad(
            self.U, 
            ((0, 0), (window, window)), 
            mode='reflect'
        )
        
        for t in range(seq_len, self.config.nt - 1):
            for x in range(self.config.nx):
                # Input: temporal sequence at spatial window
                input_window = U_padded[t-seq_len:t, x:x+2*window+1]
                # Target: next time step at center point
                target = self.U[t+1, x]
                
                self.windows.append(input_window)
                self.targets.append(target)
                self.positions.append((t, x))
        
        self.windows = np.array(self.windows)
        self.targets = np.array(self.targets)
    
    def _to_tensors(self):
        """Convert to tensors."""
        windows = self.windows
        targets = self.targets
        
        if self.normalizer:
            windows = self.normalizer.normalize_u(windows)
            targets = self.normalizer.normalize_u(targets)
        
        self.windows_tensor = to_tensor(windows, dtype=self.dtype, device=self.device)
        self.targets_tensor = to_tensor(targets, dtype=self.dtype, device=self.device)
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input': self.windows_tensor[idx],
            'target': self.targets_tensor[idx],
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_sequential_dataset(
    solution_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    bounds: DomainBounds,
    seq_length: int = 10,
    pred_length: int = 1,
    nt: int = 100,
    nx: int = 100,
    device: Device = 'cpu',
    normalize: bool = True,
    seed: Optional[int] = None
) -> SequentialPDEDataset:
    """
    Factory function to create a sequential dataset.
    
    Args:
        solution_function: Function u = f(t, x)
        bounds: Domain bounds
        seq_length: Input sequence length
        pred_length: Prediction horizon
        nt: Number of time steps
        nx: Number of spatial points
        device: Device for tensors
        normalize: Whether to normalize data
        seed: Random seed
        
    Returns:
        Configured SequentialPDEDataset
    """
    config = SequentialDatasetConfig(
        seq_length=seq_length,
        pred_length=pred_length,
        nt=nt,
        nx=nx,
        bounds=bounds,
        normalize=normalize,
        seed=seed
    )
    
    return SequentialPDEDataset(
        config=config,
        solution_function=solution_function,
        device=device
    )


def create_sequential_dataset_from_generator(
    solution_generator: Callable[..., PDESolutionGenerator],
    bounds: DomainBounds,
    seq_length: int = 10,
    pred_length: int = 1,
    nt: int = 100,
    nx: int = 100,
    device: Device = 'cpu',
    seed: Optional[int] = None,
    **generator_kwargs
) -> SequentialPDEDataset:
    """
    Factory function to create a sequential dataset from a generator.
    
    Args:
        solution_generator: Generator yielding (t, x, u) tuples
        bounds: Domain bounds
        seq_length: Input sequence length
        pred_length: Prediction horizon
        nt: Number of time steps
        nx: Number of spatial points
        device: Device for tensors
        seed: Random seed
        **generator_kwargs: Arguments for the generator
        
    Returns:
        Configured SequentialPDEDataset
    """
    config = SequentialDatasetConfig(
        seq_length=seq_length,
        pred_length=pred_length,
        nt=nt,
        nx=nx,
        bounds=bounds,
        seed=seed
    )
    
    return SequentialPDEDataset(
        config=config,
        solution_generator=solution_generator,
        device=device,
        **generator_kwargs
    )


def create_sequential_dataset_from_array(
    T: np.ndarray,
    X: np.ndarray,
    U: np.ndarray,
    seq_length: int = 10,
    pred_length: int = 1,
    device: Device = 'cpu',
    normalize: bool = True
) -> SequentialPDEDataset:
    """
    Create sequential dataset from precomputed arrays.
    
    Args:
        T: Time meshgrid (nt, nx)
        X: Spatial meshgrid (nt, nx)
        U: Solution values (nt, nx)
        seq_length: Input sequence length
        pred_length: Prediction horizon
        device: Device for tensors
        normalize: Whether to normalize data
        
    Returns:
        SequentialPDEDataset
    """
    nt, nx = U.shape
    bounds = DomainBounds(
        t_min=T.min(),
        t_max=T.max(),
        x_min=X.min(),
        x_max=X.max()
    )
    
    config = SequentialDatasetConfig(
        seq_length=seq_length,
        pred_length=pred_length,
        nt=nt,
        nx=nx,
        bounds=bounds,
        normalize=normalize
    )
    
    return SequentialPDEDataset(
        config=config,
        precomputed_data=(T, X, U),
        device=device
    )
