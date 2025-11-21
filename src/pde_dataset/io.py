"""
Submodule 1: Save and Load compressed PDE datasets.

Provides functions for serializing datasets to compressed formats
and loading them to specified devices (CPU/GPU).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
from dataclasses import dataclass, asdict
import json
import gzip
import pickle
import h5py
import io
import os

from .utils import Device, Normalizer, DomainBounds


# =============================================================================
# Compression Formats
# =============================================================================

class CompressionFormat:
    """Supported compression formats."""
    GZIP = 'gzip'
    LZ4 = 'lz4'
    ZSTD = 'zstd'
    NPZ = 'npz'
    HDF5 = 'hdf5'
    TORCH = 'torch'


@dataclass
class DatasetMetadata:
    """Metadata for a saved PDE dataset."""
    dataset_type: str  # 'pinn', 'sequential', 'galerkin'
    pde_name: str = "unknown"
    spatial_dim: int = 1
    n_samples: int = 0
    bounds: Optional[Dict[str, float]] = None
    normalizer: Optional[Dict[str, Any]] = None
    extra_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DatasetMetadata':
        return cls(**d)


# =============================================================================
# Save Functions
# =============================================================================

def save_dataset_npz(
    filepath: Union[str, Path],
    data: Dict[str, np.ndarray],
    metadata: DatasetMetadata,
    compress: bool = True
) -> None:
    """
    Save dataset in compressed NPZ format.
    
    Args:
        filepath: Path to save file
        data: Dictionary of numpy arrays to save
        metadata: Dataset metadata
        compress: Whether to use compression
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata as JSON string
    data_with_meta = data.copy()
    data_with_meta['__metadata__'] = np.array([json.dumps(metadata.to_dict())])
    
    if compress:
        np.savez_compressed(filepath, **data_with_meta)
    else:
        np.savez(filepath, **data_with_meta)


def save_dataset_hdf5(
    filepath: Union[str, Path],
    data: Dict[str, np.ndarray],
    metadata: DatasetMetadata,
    compression: str = 'gzip',
    compression_opts: int = 4
) -> None:
    """
    Save dataset in HDF5 format with compression.
    
    Args:
        filepath: Path to save file
        data: Dictionary of numpy arrays to save
        metadata: Dataset metadata
        compression: Compression algorithm ('gzip', 'lzf', 'szip')
        compression_opts: Compression level (0-9 for gzip)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        # Save data arrays
        for key, value in data.items():
            f.create_dataset(
                key, data=value,
                compression=compression,
                compression_opts=compression_opts
            )
        
        # Save metadata as attributes
        f.attrs['metadata'] = json.dumps(metadata.to_dict())


def save_dataset_torch(
    filepath: Union[str, Path],
    data: Dict[str, torch.Tensor],
    metadata: DatasetMetadata,
    compress: bool = True
) -> None:
    """
    Save dataset in PyTorch format with optional compression.
    
    Args:
        filepath: Path to save file
        data: Dictionary of tensors to save
        metadata: Dataset metadata
        compress: Whether to use gzip compression
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'data': {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                 for k, v in data.items()},
        'metadata': metadata.to_dict()
    }
    
    if compress:
        # Serialize to bytes, then compress
        buffer = io.BytesIO()
        torch.save(save_dict, buffer)
        buffer.seek(0)
        
        with gzip.open(filepath.with_suffix('.pt.gz'), 'wb') as f:
            f.write(buffer.read())
    else:
        torch.save(save_dict, filepath)


def save_dataset_pickle(
    filepath: Union[str, Path],
    data: Dict[str, Any],
    metadata: DatasetMetadata,
    compression: str = 'gzip'
) -> None:
    """
    Save dataset in compressed pickle format.
    
    Args:
        filepath: Path to save file
        data: Dictionary of data to save
        metadata: Dataset metadata
        compression: Compression type ('gzip', 'lz4', 'zstd')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'data': data,
        'metadata': metadata.to_dict()
    }
    
    if compression == 'gzip':
        with gzip.open(filepath.with_suffix('.pkl.gz'), 'wb') as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif compression == 'lz4':
        try:
            import lz4.frame
            with lz4.frame.open(filepath.with_suffix('.pkl.lz4'), 'wb') as f:
                pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        except ImportError:
            raise ImportError("lz4 not installed. Use 'pip install lz4'")
    elif compression == 'zstd':
        try:
            import zstandard as zstd
            cctx = zstd.ZstdCompressor()
            with open(filepath.with_suffix('.pkl.zst'), 'wb') as f:
                with cctx.stream_writer(f) as compressor:
                    pickle.dump(save_dict, compressor, protocol=pickle.HIGHEST_PROTOCOL)
        except ImportError:
            raise ImportError("zstandard not installed. Use 'pip install zstandard'")
    else:
        with open(filepath.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


# =============================================================================
# Load Functions
# =============================================================================

def load_dataset_npz(
    filepath: Union[str, Path],
    device: Device = 'cpu',
    dtype: torch.dtype = torch.float32
) -> Tuple[Dict[str, torch.Tensor], DatasetMetadata]:
    """
    Load dataset from NPZ format.
    
    Args:
        filepath: Path to load file
        device: Device to load tensors to
        dtype: Data type for tensors
        
    Returns:
        Tuple of (data dict, metadata)
    """
    with np.load(filepath, allow_pickle=True) as npz:
        # Extract metadata
        metadata_json = str(npz['__metadata__'][0])
        metadata = DatasetMetadata.from_dict(json.loads(metadata_json))
        
        # Convert arrays to tensors
        data = {}
        for key in npz.files:
            if key != '__metadata__':
                arr = npz[key]
                data[key] = torch.tensor(arr, dtype=dtype, device=device)
    
    return data, metadata


def load_dataset_hdf5(
    filepath: Union[str, Path],
    device: Device = 'cpu',
    dtype: torch.dtype = torch.float32,
    keys: Optional[List[str]] = None
) -> Tuple[Dict[str, torch.Tensor], DatasetMetadata]:
    """
    Load dataset from HDF5 format.
    
    Args:
        filepath: Path to load file
        device: Device to load tensors to
        dtype: Data type for tensors
        keys: Specific keys to load (None for all)
        
    Returns:
        Tuple of (data dict, metadata)
    """
    with h5py.File(filepath, 'r') as f:
        # Extract metadata
        metadata = DatasetMetadata.from_dict(json.loads(f.attrs['metadata']))
        
        # Load data
        data = {}
        load_keys = keys if keys else list(f.keys())
        for key in load_keys:
            if key in f:
                arr = f[key][:]
                data[key] = torch.tensor(arr, dtype=dtype, device=device)
    
    return data, metadata


def load_dataset_torch(
    filepath: Union[str, Path],
    device: Device = 'cpu',
    dtype: Optional[torch.dtype] = None
) -> Tuple[Dict[str, torch.Tensor], DatasetMetadata]:
    """
    Load dataset from PyTorch format.
    
    Args:
        filepath: Path to load file
        device: Device to load tensors to
        dtype: Data type for tensors (None to keep original)
        
    Returns:
        Tuple of (data dict, metadata)
    """
    filepath = Path(filepath)
    
    # Check if compressed
    if filepath.suffix == '.gz' or filepath.suffixes == ['.pt', '.gz']:
        with gzip.open(filepath, 'rb') as f:
            buffer = io.BytesIO(f.read())
            save_dict = torch.load(buffer, map_location=device)
    else:
        save_dict = torch.load(filepath, map_location=device)
    
    metadata = DatasetMetadata.from_dict(save_dict['metadata'])
    
    # Move data to device and optionally convert dtype
    data = {}
    for key, value in save_dict['data'].items():
        if isinstance(value, torch.Tensor):
            tensor = value.to(device)
            if dtype is not None:
                tensor = tensor.to(dtype)
            data[key] = tensor
        elif isinstance(value, np.ndarray):
            data[key] = torch.tensor(value, dtype=dtype or torch.float32, device=device)
        else:
            data[key] = value
    
    return data, metadata


def load_dataset_pickle(
    filepath: Union[str, Path],
    device: Device = 'cpu',
    dtype: torch.dtype = torch.float32
) -> Tuple[Dict[str, torch.Tensor], DatasetMetadata]:
    """
    Load dataset from compressed pickle format.
    
    Args:
        filepath: Path to load file
        device: Device to load tensors to
        dtype: Data type for tensors
        
    Returns:
        Tuple of (data dict, metadata)
    """
    filepath = Path(filepath)
    suffix = filepath.suffix
    
    if suffix == '.gz' or '.pkl.gz' in str(filepath):
        with gzip.open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
    elif suffix == '.lz4':
        import lz4.frame
        with lz4.frame.open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
    elif suffix == '.zst':
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        with open(filepath, 'rb') as f:
            with dctx.stream_reader(f) as reader:
                save_dict = pickle.load(reader)
    else:
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
    
    metadata = DatasetMetadata.from_dict(save_dict['metadata'])
    
    # Convert to tensors
    data = {}
    for key, value in save_dict['data'].items():
        if isinstance(value, np.ndarray):
            data[key] = torch.tensor(value, dtype=dtype, device=device)
        elif isinstance(value, torch.Tensor):
            data[key] = value.to(dtype=dtype, device=device)
        else:
            data[key] = value
    
    return data, metadata


# =============================================================================
# Auto-detect Format
# =============================================================================

def save_dataset(
    filepath: Union[str, Path],
    data: Dict[str, Union[np.ndarray, torch.Tensor]],
    metadata: DatasetMetadata,
    format: str = 'auto',
    **kwargs
) -> Path:
    """
    Save dataset with automatic format detection.
    
    Args:
        filepath: Path to save file
        data: Dictionary of arrays/tensors to save
        metadata: Dataset metadata
        format: Format ('auto', 'npz', 'hdf5', 'torch', 'pickle')
        **kwargs: Additional arguments for specific format
        
    Returns:
        Actual path where file was saved
    """
    filepath = Path(filepath)
    
    # Auto-detect format from extension
    if format == 'auto':
        suffix = filepath.suffix.lower()
        if suffix == '.npz':
            format = 'npz'
        elif suffix in ['.h5', '.hdf5']:
            format = 'hdf5'
        elif suffix in ['.pt', '.pth']:
            format = 'torch'
        elif suffix == '.pkl':
            format = 'pickle'
        else:
            format = 'npz'  # Default
    
    # Convert data to appropriate format
    if format in ['npz', 'hdf5']:
        np_data = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                   for k, v in data.items()}
        if format == 'npz':
            save_dataset_npz(filepath, np_data, metadata, **kwargs)
        else:
            save_dataset_hdf5(filepath, np_data, metadata, **kwargs)
    elif format == 'torch':
        torch_data = {k: torch.tensor(v) if isinstance(v, np.ndarray) else v 
                      for k, v in data.items()}
        save_dataset_torch(filepath, torch_data, metadata, **kwargs)
    elif format == 'pickle':
        save_dataset_pickle(filepath, data, metadata, **kwargs)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return filepath


def load_dataset(
    filepath: Union[str, Path],
    device: Device = 'cpu',
    dtype: torch.dtype = torch.float32,
    **kwargs
) -> Tuple[Dict[str, torch.Tensor], DatasetMetadata]:
    """
    Load dataset with automatic format detection.
    
    Args:
        filepath: Path to load file
        device: Device to load tensors to
        dtype: Data type for tensors
        **kwargs: Additional arguments for specific format
        
    Returns:
        Tuple of (data dict, metadata)
    """
    filepath = Path(filepath)
    
    # Detect format from extension
    suffixes = filepath.suffixes
    name = filepath.name.lower()
    
    if '.npz' in name:
        return load_dataset_npz(filepath, device, dtype, **kwargs)
    elif '.h5' in name or '.hdf5' in name:
        return load_dataset_hdf5(filepath, device, dtype, **kwargs)
    elif '.pt' in name or '.pth' in name:
        return load_dataset_torch(filepath, device, dtype, **kwargs)
    elif '.pkl' in name:
        return load_dataset_pickle(filepath, device, dtype, **kwargs)
    else:
        # Try each format
        for loader in [load_dataset_npz, load_dataset_torch, 
                      load_dataset_hdf5, load_dataset_pickle]:
            try:
                return loader(filepath, device, dtype)
            except Exception:
                continue
        raise ValueError(f"Could not determine format for {filepath}")


# =============================================================================
# Dataset Wrapper for Loading
# =============================================================================

class LoadedPDEDataset(Dataset):
    """
    PyTorch Dataset wrapper for loaded PDE data.
    
    Provides a unified interface for accessing loaded data regardless
    of the original format.
    """
    
    def __init__(
        self,
        filepath: Union[str, Path],
        device: Device = 'cpu',
        dtype: torch.dtype = torch.float32,
        transform: Optional[callable] = None,
        **load_kwargs
    ):
        """
        Initialize dataset by loading from file.
        
        Args:
            filepath: Path to dataset file
            device: Device to load data to
            dtype: Data type for tensors
            transform: Optional transform to apply to samples
            **load_kwargs: Additional arguments for loader
        """
        self.data, self.metadata = load_dataset(
            filepath, device=device, dtype=dtype, **load_kwargs
        )
        self.device = device
        self.dtype = dtype
        self.transform = transform
        
        # Determine length from first array
        self._length = None
        for key, value in self.data.items():
            if isinstance(value, torch.Tensor):
                self._length = len(value)
                break
        
        if self._length is None:
            self._length = 0
    
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {key: value[idx] for key, value in self.data.items()
                  if isinstance(value, torch.Tensor) and len(value) == self._length}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def to(self, device: Device) -> 'LoadedPDEDataset':
        """Move all data to specified device."""
        self.device = device
        self.data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in self.data.items()}
        return self
    
    def get_all(self) -> Dict[str, torch.Tensor]:
        """Get all data as a dictionary."""
        return self.data.copy()


# =============================================================================
# Chunked Loading for Large Datasets
# =============================================================================

class ChunkedDatasetLoader:
    """
    Iterator for loading large datasets in chunks.
    
    Useful when dataset doesn't fit in memory.
    """
    
    def __init__(
        self,
        filepath: Union[str, Path],
        chunk_size: int = 10000,
        device: Device = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize chunked loader.
        
        Args:
            filepath: Path to HDF5 dataset file
            chunk_size: Number of samples per chunk
            device: Device to load chunks to
            dtype: Data type for tensors
        """
        self.filepath = Path(filepath)
        self.chunk_size = chunk_size
        self.device = device
        self.dtype = dtype
        
        # Get dataset info
        with h5py.File(self.filepath, 'r') as f:
            self.metadata = DatasetMetadata.from_dict(
                json.loads(f.attrs['metadata'])
            )
            self.keys = list(f.keys())
            self.total_samples = len(f[self.keys[0]])
    
    def __len__(self) -> int:
        return (self.total_samples + self.chunk_size - 1) // self.chunk_size
    
    def __iter__(self):
        self._current_chunk = 0
        return self
    
    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._current_chunk >= len(self):
            raise StopIteration
        
        start_idx = self._current_chunk * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_samples)
        
        with h5py.File(self.filepath, 'r') as f:
            data = {}
            for key in self.keys:
                arr = f[key][start_idx:end_idx]
                data[key] = torch.tensor(arr, dtype=self.dtype, device=self.device)
        
        self._current_chunk += 1
        return data
    
    def get_chunk(self, chunk_idx: int) -> Dict[str, torch.Tensor]:
        """Get a specific chunk by index."""
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_samples)
        
        with h5py.File(self.filepath, 'r') as f:
            data = {}
            for key in self.keys:
                arr = f[key][start_idx:end_idx]
                data[key] = torch.tensor(arr, dtype=self.dtype, device=self.device)
        
        return data
