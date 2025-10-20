"""
Configuration class for Neural Control Barrier Functions (NCBF).

This module defines the configuration parameters for NCBF neural networks,
including architecture specifications, training parameters, and network settings.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class NCBFConfig:
    """
    Configuration for Neural Control Barrier Function.

    This class manages all parameters needed for NCBF neural network architecture
    and training. Note: alpha (CBF parameter) is managed by the unicycle model,
    not by the NCBF configuration.
    """

    # Neural network architecture parameters
    input_dim: int = 3  # State dimension [x, y, theta] for unicycle
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64, 32])  # Hidden layer sizes
    activation: str = 'tanh'  # Activation function ('tanh', 'relu', 'sigmoid', 'leaky_relu')
    output_dim: int = 1  # Output dimension (scalar h(x))

    # Training parameters
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    dropout_rate: float = 0.1
    batch_size: int = 256
    max_epochs: int = 3000

    # Architecture options
    use_batch_norm: bool = False
    use_residual: bool = False
    use_dropout: bool = True

    # Loss function weights
    classification_weight: float = 1.0  # Weight for classification loss
    barrier_weight: float = 0.1  # Weight for barrier derivative loss
    regularization_weight: float = 0.02  # Weight for L2 regularization
    margin: float = 0.2  # Margin for hinge loss in classification

    # Training data parameters
    min_unsafe_ratio: float = 0.3  # Minimum ratio of unsafe samples
    obstacle_focus_ratio: float = 0.3  # Ratio of samples focused around obstacles

    # Model saving/loading
    checkpoint_dir: str = "/home/chengrui/wk/NCBFquickDemo/work/ncbf/weights/checkpoints"
    model_name: str = "ncbf_model"
    save_frequency: int = 100  # Save every N epochs

    # Validation parameters
    validation_split: float = 0.2  # Fraction of data for validation
    early_stopping_patience: int = 50  # Epochs to wait before early stopping

    # Optimization parameters
    optimizer: str = 'adam'  # 'adam', 'sgd', 'rmsprop'
    scheduler: str = 'step'  # 'step', 'cosine', 'plateau'
    scheduler_step_size: int = 200
    scheduler_gamma: float = 0.1

    # GPU and performance parameters
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'cuda:0', etc.
    num_workers: int = 4  # DataLoader workers
    pin_memory: bool = True  # GPU memory pinning
    mixed_precision: bool = True  # Use AMP for faster training

    # Training monitoring
    log_frequency: int = 10  # Log every N batches
    plot_frequency: int = 50  # Plot every N epochs
    save_best_only: bool = True  # Only save best validation model
    show_training_plots: bool = True  # Display plots during training

    # Contour visualization parameters
    contour_resolution: int = 100  # Grid resolution for contour plots
    contour_levels: int = 20  # Number of contour levels
    evaluation_theta: float = 0.0  # Fixed theta for 2D contour visualization

    # Conservative NCBF parameters (new)
    conservative_weight: float = 0.1  # Weight for conservative loss
    temperature: float = 0.1  # Temperature parameter for log-sum-exp
    num_random_controls: int = 10  # Number of random controls to sample
    enable_pretraining: bool = False  # Enable two-phase training
    pretrain_epochs: int = 0  # Number of pretraining epochs
    pretrain_lr: float = 0.001  # Learning rate for pretraining

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not self.hidden_dims:
            raise ValueError("hidden_dims cannot be empty")
        if any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("All hidden dimensions must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if not 0 <= self.validation_split <= 1:
            raise ValueError("validation_split must be between 0 and 1")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")

    def get_activation_function(self) -> str:
        """Get the activation function name."""
        return self.activation.lower()

    def get_network_architecture(self) -> List[int]:
        """Get the complete network architecture including input and output dimensions."""
        return [self.input_dim] + self.hidden_dims + [self.output_dim]

    def __str__(self) -> str:
        """String representation of the configuration."""
        return (
            f"NCBFConfig(architecture={self.get_network_architecture()}, "
            f"activation={self.activation}, "
            f"lr={self.learning_rate}, batch_size={self.batch_size})"
        )


def create_default_ncbf_config() -> NCBFConfig:
    """Create a default NCBF configuration for quick start."""
    return NCBFConfig()


def create_large_ncbf_config() -> NCBFConfig:
    """Create a large NCBF configuration for serious learning with intensive training."""
    return NCBFConfig(
        # Large network architecture for serious learning
        hidden_dims=[256, 256, 128, 64, 32],
        activation='relu',

        # Aggressive training parameters
        batch_size=1024,
        max_epochs=500,  # More epochs for serious learning
        learning_rate=0.005,

        # Robust network features
        use_batch_norm=True,
        dropout_rate=0.3,

        # Intensive loss weights for safety enforcement
        classification_weight=5.0,
        barrier_weight=2.0,
        regularization_weight=0.5,
        margin=0.5,

        # Performance settings
        mixed_precision=True,
        num_workers=8,

        # High-resolution evaluation
        contour_resolution=150,

        # Frequent monitoring
        plot_frequency=25,
        save_frequency=25
    )


def create_small_ncbf_config() -> NCBFConfig:
    """Create a small NCBF configuration for fast training."""
    return NCBFConfig(
        hidden_dims=[32, 16],
        activation='tanh',
        batch_size=128,
        max_epochs=500,
        learning_rate=5e-3
    )