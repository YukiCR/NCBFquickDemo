"""
Map filtering utilities for Neural Control Barrier Functions.

This module provides NCBFMapFilter class for filtering training data
to create subsets like safe-only datasets. This is a standalone class
that reuses data loading/saving functionality from map_manager without
inheritance complications.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any
import json
from datetime import datetime
import h5py  # For HDF5 file operations


class NCBFMapFilter:
    """
    Standalone filter for training data that creates subsets like safe-only datasets.

    This class provides data filtering capabilities for HDF5 training data files,
    allowing creation of safe-only datasets, regional subsets, and other
    filtered versions while maintaining compatibility with the existing training pipeline.
    """

    def __init__(self, training_data_path: Optional[Union[str, Path]] = None):
        """
        Initialize filter with optional training data path.

        Args:
            training_data_path: Path to training data HDF5 file (optional)
        """
        self.training_data_path = None
        self.training_data = None

        # Load training data if path provided
        if training_data_path:
            self.load_training_data_file(training_data_path)

    def load_training_data_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load training data from HDF5 file and store internally.

        Args:
            file_path: Path to training data HDF5 file

        Returns:
            Dictionary containing training data and metadata
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Training data file not found: {file_path}")

        print(f"📂 Loading training data from: {file_path}")

        # Load HDF5 file - replicate functionality from map_manager
        with h5py.File(file_path, 'r') as hf:
            # Load data group
            data_group = hf['data']
            states = data_group['states'][:]
            labels = data_group['labels'][:]

            # Load metadata group
            meta_group = hf['metadata']
            num_samples = meta_group.attrs['num_samples']
            num_safe = meta_group.attrs['num_safe']
            num_unsafe = meta_group.attrs['num_unsafe']
            obstacle_focus_ratio = meta_group.attrs['obstacle_focus_ratio']
            seed = meta_group.attrs['seed']
            generation_timestamp = meta_group.attrs['generation_timestamp']
            map_info_json = meta_group.attrs['map_info_json']

            # Reconstruct dictionary
            self.training_data = {
                'states': states,
                'labels': labels,
                'num_samples': num_samples,
                'num_safe': num_safe,
                'num_unsafe': num_unsafe,
                'obstacle_focus_ratio': obstacle_focus_ratio,
                'seed': seed if seed != -1 else None,
                'map_info': json.loads(map_info_json),
                'generation_timestamp': generation_timestamp
            }

        self.training_data_path = file_path

        print(f"✅ Training data loaded: {self.training_data['num_samples']} samples")
        print(f"   Safe samples: {self.training_data['num_safe']}")
        print(f"   Unsafe samples: {self.training_data['num_unsafe']}")

        return self.training_data

    def filter_safe_only(
        self,
        num_samples: Optional[int] = None,
        save_path: Optional[Union[str, Path]] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Filter dataset to include only safe states (label=1).

        Args:
            num_samples: Maximum number of safe samples to include (None = all)
            save_path: Optional path to save filtered dataset
            seed: Random seed for reproducible subsampling

        Returns:
            Dictionary containing filtered training data

        Raises:
            ValueError: If no training data is loaded
        """
        if self.training_data is None:
            raise ValueError("No training data loaded. Use load_training_data_file() first.")

        print(f"🔄 Filtering to safe-only dataset...")

        # Extract safe samples
        safe_mask = self.training_data['labels'].flatten() == 1
        safe_states = self.training_data['states'][safe_mask]
        safe_labels = self.training_data['labels'][safe_mask]

        original_safe_count = len(safe_states)
        print(f"📊 Found {original_safe_count} safe samples in original dataset")

        # Handle subsampling if requested
        if num_samples is not None and num_samples < original_safe_count:
            if seed is not None:
                np.random.seed(seed)

            # Random subsampling
            indices = np.random.choice(original_safe_count, num_samples, replace=False)
            safe_states = safe_states[indices]
            safe_labels = safe_labels[indices]

            print(f"📉 Subsampled to {num_samples} safe samples")
        elif num_samples is not None and num_samples > original_safe_count:
            print(f"⚠️  Requested {num_samples} samples but only {original_safe_count} safe samples available")
            print(f"📊 Using all {original_safe_count} safe samples")

        # Create filtered dataset preserving original structure
        filtered_data = {
            'states': safe_states,
            'labels': safe_labels,
            'num_samples': len(safe_states),
            'num_safe': len(safe_states),
            'num_unsafe': 0,
            'actual_safe_ratio': 1.0,
            'actual_unsafe_ratio': 0.0,
            'obstacle_focus_ratio': self.training_data.get('obstacle_focus_ratio', 0.3),
            'seed': seed,
            'filter_type': 'safe_only',
            'original_file': str(self.training_data_path) if self.training_data_path else None,
            'filter_timestamp': datetime.now().isoformat(),
            'sampling_breakdown': {
                'original_safe_count': original_safe_count,
                'final_safe_count': len(safe_states),
                'subsampled': num_samples is not None and num_samples < original_safe_count
            },
            'config_params': self.training_data.get('config_params', {}),
            'map_info': self.training_data.get('map_info', {}),
            'generation_timestamp': self.training_data.get('generation_timestamp', ''),
            'filtered_timestamp': datetime.now().isoformat()
        }

        print(f"✅ Filtered dataset created:")
        print(f"   Total samples: {filtered_data['num_samples']}")
        print(f"   Safe samples: {filtered_data['num_safe']} (100%)")
        print(f"   Unsafe samples: {filtered_data['num_unsafe']} (0%)")

        # Save if path provided
        if save_path:
            self.save_filtered_data(filtered_data, save_path)

        return filtered_data

    def save_filtered_data(self, filtered_data: Dict[str, Any], save_path: Union[str, Path]):
        """
        Save filtered training data to HDF5 file.

        Args:
            filtered_data: Dictionary containing filtered training data
            save_path: Path to save the filtered dataset
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"💾 Saving filtered training data to: {save_path}")

        # Save as HDF5 format - replicate functionality from map_manager
        with h5py.File(save_path, 'w') as hf:
            # Create main data group
            data_group = hf.create_group('data')
            data_group.create_dataset('states', data=filtered_data['states'], compression='gzip', compression_opts=4)
            data_group.create_dataset('labels', data=filtered_data['labels'], compression='gzip', compression_opts=4)

            # Create metadata group
            meta_group = hf.create_group('metadata')
            meta_group.attrs['num_samples'] = filtered_data['num_samples']
            meta_group.attrs['num_safe'] = filtered_data['num_safe']
            meta_group.attrs['num_unsafe'] = filtered_data['num_unsafe']
            meta_group.attrs['obstacle_focus_ratio'] = filtered_data['obstacle_focus_ratio']
            meta_group.attrs['seed'] = filtered_data['seed'] if filtered_data['seed'] is not None else -1
            meta_group.attrs['generation_timestamp'] = filtered_data.get('generation_timestamp', '')
            meta_group.attrs['map_info_json'] = json.dumps(filtered_data.get('map_info', {}))

        print(f"✅ Filtered data saved successfully!")
        print(f"📁 File size: {save_path.stat().st_size / 1024:.1f} KB")

    def get_filter_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current training data and filtering options.

        Returns:
            Dictionary containing data statistics and filter information
        """
        if self.training_data is None:
            return {"status": "no_data_loaded"}

        stats = {
            "status": "data_loaded",
            "original_file": str(self.training_data_path) if self.training_data_path else None,
            "total_samples": self.training_data['num_samples'],
            "safe_samples": self.training_data['num_safe'],
            "unsafe_samples": self.training_data['num_unsafe'],
            "safe_ratio": self.training_data['num_safe'] / self.training_data['num_samples'],
            "filter_options": {
                "can_filter_safe_only": self.training_data['num_safe'] > 0,
                "can_filter_unsafe_only": self.training_data['num_unsafe'] > 0,
                "max_safe_samples": self.training_data['num_safe'],
                "max_unsafe_samples": self.training_data['num_unsafe']
            }
        }

        return stats

    def __repr__(self) -> str:
        """String representation of the filter."""
        if self.training_data is None:
            return f"NCBFMapFilter(no_data_loaded)"

        stats = self.get_filter_statistics()
        return (f"NCBFMapFilter(data_loaded: {stats['total_samples']} samples, "
                f"safe: {stats['safe_samples']}, unsafe: {stats['unsafe_samples']}, "
                f"safe_ratio: {stats['safe_ratio']:.1%})")


# Convenience function for quick safe-only filtering
def create_safe_only_dataset(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    num_samples: Optional[int] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function to quickly create safe-only dataset from existing training data.

    Args:
        input_path: Path to input training data HDF5 file
        output_path: Path to output safe-only dataset
        num_samples: Maximum number of samples (None = all safe samples)
        seed: Random seed for reproducible subsampling

    Returns:
        Filtered dataset dictionary

    Example:
        >>> create_safe_only_dataset(
        ...     "map_files/map1/training_data_large.h5",
        ...     "map_files/map1/training_data_safe_only.h5",
        ...     num_samples=5000
        ... )
    """
    filter = NCBFMapFilter(training_data_path=input_path)
    return filter.filter_safe_only(num_samples=num_samples, save_path=output_path, seed=seed)