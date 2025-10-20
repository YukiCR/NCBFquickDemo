"""
Minimal utilities for pseudo-negative data enhancement.
Basic I/O operations shared between methods 2 and 3.
"""

import numpy as np
import h5py
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def load_hdf5_dataset(file_path: str) -> Dict[str, np.ndarray]:
    """
    Load dataset from HDF5 file in standard format.

    Args:
        file_path: Path to HDF5 dataset file

    Returns:
        Dictionary with 'states', 'labels', 'actions' (optional), 'metadata'
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with h5py.File(file_path, 'r') as hf:
        # Load required data
        states = hf['data/states'][:]
        labels = hf['data/labels'][:]

        # Load optional actions
        actions = None
        if 'actions' in hf['data']:
            actions = hf['data/actions'][:]

        # Load metadata attributes
        metadata = dict(hf['metadata'].attrs)

        return {
            'states': states,
            'labels': labels,
            'actions': actions,
            'metadata': metadata
        }


def save_hdf5_dataset(file_path: str, data: Dict[str, np.ndarray],
                     metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save dataset to HDF5 file in standard format.

    Args:
        file_path: Output file path
        data: Dictionary with 'states', 'labels', 'actions' (optional)
        metadata: Optional metadata to include
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract data
    states = data['states']
    labels = data['labels']
    actions = data.get('actions')

    with h5py.File(file_path, 'w') as hf:
        # Create data group
        data_group = hf.create_group('data')
        data_group.create_dataset('states', data=states, compression='gzip', compression_opts=4)
        data_group.create_dataset('labels', data=labels, compression='gzip', compression_opts=4)

        # Add actions if present
        if actions is not None:
            data_group.create_dataset('actions', data=actions, compression='gzip', compression_opts=4)

        # Create metadata group
        meta_group = hf.create_group('metadata')

        # Basic statistics
        num_samples = len(states)
        num_safe = int(np.sum(labels.flatten() == 1))
        num_unsafe = int(np.sum(labels.flatten() == 0))

        meta_group.attrs['num_samples'] = num_samples
        meta_group.attrs['num_safe'] = num_safe
        meta_group.attrs['num_unsafe'] = num_unsafe
        meta_group.attrs['obstacle_focus_ratio'] = metadata.get('obstacle_focus_ratio', 0.3)
        meta_group.attrs['seed'] = metadata.get('seed', -1)
        meta_group.attrs['generation_timestamp'] = metadata.get('generation_timestamp', '')
        meta_group.attrs['map_info_json'] = json.dumps(metadata.get('map_info', {}))

        # Add provided metadata
        if metadata:
            for key, value in metadata.items():
                if key in ['map_info', 'config_params']:
                    # JSON-encode complex objects
                    meta_group.attrs[f"{key}_json"] = json.dumps(value)
                else:
                    meta_group.attrs[key] = value


def get_workspace_bounds(states: np.ndarray, padding: float = 0.5) -> Tuple[float, float]:
    """
    Get workspace bounds from state data with padding.

    Args:
        states: State vectors [N, state_dim]
        padding: Padding to add to bounds

    Returns:
        Tuple of (min_bound, max_bound) for workspace
    """
    # Use x, y coordinates (first two dimensions) for workspace bounds
    positions = states[:, :2] if states.shape[1] >= 2 else states

    min_pos = np.min(positions)
    max_pos = np.max(positions)

    # Add padding
    min_bound = min_pos - padding
    max_bound = max_pos + padding

    # Ensure non-negative bounds (typical for our workspace)
    min_bound = max(0.0, min_bound)

    return float(min_bound), float(max_bound)