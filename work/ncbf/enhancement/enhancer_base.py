"""
Abstract base class for pseudo-negative data enhancement.
Minimal interface for maximum extensibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from pathlib import Path


class PseudoNegativeEnhancer(ABC):
    """
    Abstract base class for pseudo-negative data enhancement.
    Minimal interface for maximum extensibility.
    """

    def __init__(self, config: 'EnhancerConfig'):
        """
        Initialize enhancer with configuration.

        Args:
            config: Configuration instance with enhancement parameters
        """
        self.config = config
        self.is_fitted = False

    @abstractmethod
    def fit(self, **kwargs) -> None:
        """
        Fit enhancer to prepare for negative sample generation.
        Loads dataset and learns method-specific models.

        Args:
            **kwargs: Method-specific parameters
                - iDBF: dynamics_model (required)
                - Others: no parameters needed
        """
        pass

    @abstractmethod
    def generate_pseudo_negatives(self, num_samples: int) -> np.ndarray:
        """
        Generate pseudo-negative states.

        Args:
            num_samples: Number of negative samples to generate

        Returns:
            Negative states [num_samples, state_dim]
        """
        pass

    @abstractmethod
    def load_dataset(self, dataset_path: str) -> Dict[str, np.ndarray]:
        """
        Load dataset from file.

        Args:
            dataset_path: Path to HDF5 dataset file

        Returns:
            Dictionary with 'states', 'labels', 'actions' (optional)
        """
        pass

    @abstractmethod
    def save_dataset(self, dataset_path: str, data: Dict[str, np.ndarray]) -> None:
        """
        Save dataset to file.

        Args:
            dataset_path: Output file path
            data: Dictionary with dataset contents
        """
        pass

    @abstractmethod
    def enhance_dataset(self, output_path: str, **kwargs) -> str:
        """
        Complete enhancement pipeline.

        Args:
            output_path: Where to save enhanced dataset
            **kwargs: Method-specific parameters for fit()

        Returns:
            Path to enhanced dataset file
        """
        pass


class EnhancerFactory:
    """Factory class for creating appropriate enhancer instances."""

    @staticmethod
    def create_enhancer(config: 'EnhancerConfig') -> 'PseudoNegativeEnhancer':
        """
        Create enhancer based on configuration.

        Args:
            config: Enhancer configuration

        Returns:
            Appropriate enhancer instance

        Raises:
            ValueError: If method is unknown or config is invalid
        """
        if config.method == 'complement':
            from .complement_enhancer import ComplementEnhancer
            return ComplementEnhancer(config)
        elif config.method == 'idbf':
            from .idbf_enhancer import iDBFEnhancer
            return iDBFEnhancer(config)
        elif config.method == 'ad':
            from .ad_enhancer import ADEnhancer
            # Ensure we have ADConfig for AD method
            if not hasattr(config, 'ad_method'):
                from .enhancer_config import ADConfig
                # Convert base config to AD config if needed
                ad_config = ADConfig(
                    input_dataset_path=config.input_dataset_path,
                    target_ratio=config.target_ratio,
                    workspace_padding=config.workspace_padding,
                    random_seed=config.random_seed
                )
                return ADEnhancer(ad_config)
            return ADEnhancer(config)
        else:
            raise ValueError(f"Unknown enhancement method: {config.method}")