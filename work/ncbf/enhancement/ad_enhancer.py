"""
Anomaly Detection-based pseudo-negative data enhancement.
Method 2: Use in-distribution classifier to identify unsafe states.
"""

import numpy as np
from typing import Dict, Any, Union
from enhancer_base import PseudoNegativeEnhancer
from enhancer_utils import load_hdf5_dataset, save_hdf5_dataset, get_workspace_bounds


class ADEnhancer(PseudoNegativeEnhancer):
    """Anomaly detection method using in-distribution vs out-of-distribution classification."""

    def __init__(self, config: 'ADConfig'):
        """
        Initialize AD enhancer with configuration.

        Args:
            config: AD configuration with method-specific parameters
        """
        super().__init__(config)
        self.ad_method = config.ad_method
        self.ad_model = None
        self.decision_threshold = None
        self.safe_states = None  # Store for workspace bounds calculation

    def fit(self, **kwargs) -> None:
        """
        Fit AD enhancer by loading dataset and training anomaly detector.

        Args:
            **kwargs: Method-specific parameters that can override config
                - ad_method: Override AD method selection
                - nu: Override nu parameter for OneClassSVM
                - kernel: Override kernel parameter for OneClassSVM
                - threshold_quantile: Override threshold quantile
                - use_full_state: Override state representation choice
        """
        # Load dataset using utils
        data = load_hdf5_dataset(self.config.input_dataset_path)
        safe_states = data['states']  # All states are safe in safe-only dataset
        self.safe_states = safe_states  # Store for bounds calculation

        # Method selection hierarchy: kwargs > config > default
        method = kwargs.get('ad_method', self.ad_method)

        # Parameter override via kwargs (for experimentation)
        if 'nu' in kwargs:
            self.config.nu = kwargs['nu']
        if 'kernel' in kwargs:
            self.config.kernel = kwargs['kernel']
        if 'threshold_quantile' in kwargs:
            self.config.threshold_quantile = kwargs['threshold_quantile']
        if 'use_full_state' in kwargs:
            self.config.use_full_state = kwargs['use_full_state']

        # Train appropriate AD model based on method
        self.ad_model = self._train_ad_model(safe_states, method)

        # Determine decision threshold for OOD detection
        self.decision_threshold = self._determine_threshold(safe_states, method)

        self.is_fitted = True

    def _train_ad_model(self, safe_states: np.ndarray, method: str) -> Any:
        """
        Train specific AD model based on method.

        Args:
            safe_states: Safe state vectors [N, state_dim]
            method: AD method name

        Returns:
            Trained AD model
        """
        # State representation selection
        if self.config.use_full_state:
            training_data = safe_states
        else:
            training_data = safe_states[:, :2]  # Position only (default)

        if method == 'ocsvm':
            return self._train_ocsvm(training_data)
        elif method == 'isolation_forest':
            return self._train_isolation_forest(training_data)  # Placeholder
        elif method == 'local_outlier_factor':
            return self._train_lof(training_data)  # Placeholder
        elif method == 'autoencoder':
            return self._train_autoencoder(training_data)  # Placeholder
        else:
            raise ValueError(f"Unknown AD method: {method}")

    def _train_ocsvm(self, training_data: np.ndarray):
        """
        Train OneClassSVM on safe states.

        Args:
            training_data: Training data for OneClassSVM

        Returns:
            Trained OneClassSVM model
        """
        from sklearn.svm import OneClassSVM

        model = OneClassSVM(
            kernel=self.config.kernel,
            nu=self.config.nu,
            gamma=self.config.gamma
        )

        model.fit(training_data)
        return model

    def _train_isolation_forest(self, training_data: np.ndarray):
        """Placeholder for IsolationForest training."""
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(
            contamination=self.config.contamination,
            random_state=self.config.random_seed
        )

        model.fit(training_data)
        return model

    def _train_lof(self, training_data: np.ndarray):
        """Placeholder for Local Outlier Factor training."""
        from sklearn.neighbors import LocalOutlierFactor

        model = LocalOutlierFactor(
            n_neighbors=self.config.n_neighbors,
            contamination=self.config.contamination
        )

        # Note: LOF doesn't have decision_function, uses negative_outlier_factor_
        model.fit(training_data)
        return model

    def _train_autoencoder(self, training_data: np.ndarray):
        """Placeholder for Autoencoder training."""
        # This would require PyTorch/TensorFlow implementation
        # For now, fall back to OneClassSVM
        print(f"Autoencoder not implemented yet, falling back to OneClassSVM")
        return self._train_ocsvm(training_data)

    def _determine_threshold(self, safe_states: np.ndarray, method: str) -> float:
        """
        Determine decision threshold for OOD detection.

        Args:
            safe_states: Safe state vectors for threshold calculation
            method: AD method name

        Returns:
            Threshold value for OOD classification
        """
        # State representation selection
        if self.config.use_full_state:
            detection_data = safe_states
        else:
            detection_data = safe_states[:, :2]  # Position only

        # Get decision values on safe training data
        decision_values = self.ad_model.decision_function(detection_data)

        # Use lower quantile as threshold (conservative approach)
        # Lower quantile = more states classified as OOD
        threshold_quantile = self.config.threshold_quantile
        return float(np.percentile(decision_values, threshold_quantile * 100))

    def generate_pseudo_negatives(self, num_samples: int) -> np.ndarray:
        """
        Generate pseudo-negative states using anomaly detection.

        Args:
            num_samples: Number of negative samples to generate

        Returns:
            Negative states [num_samples, state_dim]
        """
        # Get workspace bounds for spatial sampling
        min_bound, max_bound = get_workspace_bounds(self.safe_states, self.config.workspace_padding)

        # Rejection sampling: generate candidates until we have enough OOD states
        negative_states = []
        attempts = 0
        max_attempts = num_samples * 10  # Prevent infinite loops

        while len(negative_states) < num_samples and attempts < max_attempts:
            # Sample random state in workspace
            candidate = self._sample_random_state(min_bound, max_bound, self.safe_states.shape[1])

            # Check if it's OOD according to AD model
            if self._is_out_of_distribution(candidate):
                negative_states.append(candidate)

            attempts += 1

        return np.array(negative_states[:num_samples])

    def _is_out_of_distribution(self, state: np.ndarray) -> bool:
        """
        Check if state is out-of-distribution using trained AD model.

        Args:
            state: State vector to check

        Returns:
            True if state is OOD, False otherwise
        """
        # State representation selection
        if self.config.use_full_state:
            detection_data = state.reshape(1, -1)
        else:
            detection_data = state[:2].reshape(1, -1)  # Position only

        # Get decision function value
        if self.ad_method == 'lof':
            # LOF uses negative_outlier_factor_ instead of decision_function
            # For now, fall back to decision_function (works in newer sklearn versions)
            try:
                decision_value = self.ad_model.decision_function(detection_data)[0]
            except AttributeError:
                # Fallback: use score_samples if available
                decision_value = self.ad_model.score_samples(detection_data)[0]
        else:
            decision_value = self.ad_model.decision_function(detection_data)[0]

        # OOD if decision value is below threshold
        return decision_value < self.decision_threshold

    def _sample_random_state(self, min_bound: float, max_bound: float, state_dim: int) -> np.ndarray:
        """
        Sample random state within workspace bounds.

        Args:
            min_bound: Minimum bound for sampling
            max_bound: Maximum bound for sampling
            state_dim: Dimension of state vector

        Returns:
            Random state vector
        """
        # Sample position coordinates
        if state_dim >= 3:  # Unicycle: [x, y, theta]
            x = np.random.uniform(min_bound, max_bound)
            y = np.random.uniform(min_bound, max_bound)
            theta = np.random.uniform(-np.pi, np.pi)
            return np.array([x, y, theta])
        else:  # Lower dimensional
            return np.random.uniform(min_bound, max_bound, size=state_dim)

    def load_dataset(self, dataset_path: str) -> Dict[str, np.ndarray]:
        """Load dataset from HDF5 file using utils."""
        return load_hdf5_dataset(dataset_path)

    def save_dataset(self, dataset_path: str, data: Dict[str, np.ndarray]) -> None:
        """Save dataset to HDF5 file using utils."""
        save_hdf5_dataset(dataset_path, data)

    def enhance_dataset(self, output_path: str, **kwargs) -> str:
        """
        Complete enhancement pipeline for AD method.

        Args:
            output_path: Where to save enhanced dataset
            **kwargs: Method-specific parameters for fit()

        Returns:
            Path to enhanced dataset file
        """
        # Fit enhancer (with method-specific parameters)
        self.fit(**kwargs)

        # Generate pseudo-negatives
        num_safe = len(self.safe_states)
        num_negative = int(num_safe * self.config.target_ratio)
        negative_states = self.generate_pseudo_negatives(num_negative)

        # Create enhanced dataset
        all_states = np.vstack([self.safe_states, negative_states])
        all_labels = np.hstack([np.ones(num_safe), np.zeros(len(negative_states))])

        # Shuffle dataset
        indices = np.random.permutation(len(all_states))
        enhanced_data = {
            'states': all_states[indices],
            'labels': all_labels[indices]
        }

        # Add enhancement metadata
        metadata = {
            'enhancement_method': 'anomaly_detection',
            'enhancement_ratio': self.config.target_ratio,
            'ad_method': self.ad_method,
            'enhancement_timestamp': str(np.datetime64('now')),
            'obstacle_focus_ratio': 0.3,  # Required by visualization tool
            'seed': self.config.random_seed,
            'generation_timestamp': str(np.datetime64('now')),
            'map_info': {}  # Will be populated if available from original dataset
        }

        # Save enhanced dataset with metadata
        save_hdf5_dataset(output_path, enhanced_data, metadata)
        return output_path