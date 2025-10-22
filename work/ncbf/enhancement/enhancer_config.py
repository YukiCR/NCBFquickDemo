"""
Configuration classes for pseudo-negative data enhancement.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class EnhancerConfig:
    """
    Base configuration for pseudo-negative data enhancement.

    Attributes:
        input_dataset_path: Path to input safe-only dataset (required)
        method: Enhancement method ('complement', 'idbf', 'ad')
        target_ratio: Ratio of pseudo-negative to safe samples
        workspace_padding: Padding beyond data bounds for sampling
        random_seed: Random seed for reproducibility
    """
    input_dataset_path: str
    method: str = 'ad'
    target_ratio: float = 0.3
    workspace_padding: float = 0.0
    random_seed: int = 42

    def __post_init__(self):
        """Validate configuration parameters."""
        if not self.input_dataset_path:
            raise ValueError("input_dataset_path is required")

        if self.method not in ['complement', 'idbf', 'ad']:
            raise ValueError(f"Unknown method: {self.method}. Must be 'complement', 'idbf', or 'ad'")

        if self.target_ratio <= 0:
            raise ValueError("target_ratio must be positive")

        if self.workspace_padding < 0:
            raise ValueError("workspace_padding must be non-negative")


@dataclass
class ADConfig(EnhancerConfig):
    """
    Configuration for anomaly detection-based enhancement.

    Attributes:
        ad_method: Specific AD algorithm ('ocsvm', 'isolation_forest', 'local_outlier_factor', 'autoencoder')
        kernel: Kernel type for OneClassSVM
        nu: Expected proportion of outliers for OneClassSVM
        gamma: Kernel coefficient for OneClassSVM
        threshold_quantile: Quantile for OOD threshold determination
        use_full_state: Whether to use full state vs position-only for OOD detection
        contamination: Expected proportion of outliers for IsolationForest
        n_neighbors: Number of neighbors for LocalOutlierFactor
    """
    ad_method: str = 'ocsvm'
    kernel: str = 'rbf'
    nu: float = 0.001 # all data is normal
    gamma: str = 8.0 # 'scale' or 'auto' or float value
    threshold_quantile: float = 0.00005
    use_full_state: bool = False
    # placeholders for other AD methods, currently we only use OCSVM
    contamination: float = 0.1
    n_neighbors: int = 20

    def __post_init__(self):
        """Validate AD-specific parameters."""
        super().__post_init__()  # Call parent validation

        if self.method != 'ad':
            raise ValueError("ADConfig can only be used with method='ad'")

        if self.ad_method not in ['ocsvm', 'isolation_forest', 'local_outlier_factor', 'autoencoder']:
            raise ValueError(f"Unknown AD method: {self.ad_method}")

        if self.nu <= 0 or self.nu >= 1:
            raise ValueError("nu must be in (0, 1)")

        if self.threshold_quantile <= 0 or self.threshold_quantile >= 1:
            raise ValueError("threshold_quantile must be in (0, 1)")

        if self.contamination <= 0 or self.contamination >= 1:
            raise ValueError("contamination must be in (0, 1)")