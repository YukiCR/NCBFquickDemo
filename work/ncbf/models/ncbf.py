"""
Neural Control Barrier Function (NCBF) implementation.

This module implements a learnable Control Barrier Function using a Multi-Layer Perceptron (MLP)
neural network. The NCBF inherits from both CBFFunction (for CBF interface) and nn.Module
(for PyTorch functionality), providing a flexible architecture for learning safety certificates.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Union, List, Optional
import os
from pathlib import Path

# Import the CBF function base class
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from safe_control.cbf_function import CBFFunction

# Import configuration
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from ncbf.configs.ncbf_config import NCBFConfig


class NCBF(CBFFunction, nn.Module):
    """
    Neural Control Barrier Function implemented as a Multi-Layer Perceptron.

    This class implements a learnable barrier function h(x) using a neural network,
    where h(x) >= 0 indicates safe states and h(x) < 0 indicates unsafe states.

    The implementation supports:
    - Configurable MLP architecture (hidden layers, sizes, activation functions)
    - Both numpy array and PyTorch tensor inputs
    - Automatic differentiation for gradient computation
    - Model saving/loading functionality
    - Integration with existing CBF infrastructure

    Inherits from:
    - CBFFunction: Provides the CBF interface (h(x), grad_h(x), etc.)
    - nn.Module: Provides PyTorch neural network functionality
    """

    def __init__(self, config: NCBFConfig):
        """
        Initialize the Neural Control Barrier Function.

        Args:
            config: NCBFConfig instance containing all configuration parameters

        Raises:
            ValueError: If config parameters are invalid
            TypeError: If config is not an NCBFConfig instance
        """
        # Validate config type (allow duck typing)
        if not hasattr(config, 'input_dim') or not hasattr(config, 'hidden_dims'):
            raise TypeError(f"config must be NCBFConfig-like object with required attributes, got {type(config)}")

        # Initialize CBFFunction parent (sets state_dim and alpha)
        # Note: alpha will be provided by the unicycle model when used in CBF filter
        super().__init__(state_dim=config.input_dim, alpha=1.0)  # Default alpha, will be overridden

        # Initialize nn.Module parent
        super(CBFFunction, self).__init__()

        self.config = config

        # Build the neural network architecture
        self.network = self._build_network()

        # Initialize weights using appropriate initialization
        self._initialize_weights()

    def _build_network(self) -> nn.Sequential:
        """Build the MLP network architecture based on configuration."""
        layers = []

        # Get layer dimensions
        layer_dims = self.config.get_network_architecture()

        # Build hidden layers
        for i in range(len(layer_dims) - 2):
            input_dim = layer_dims[i]
            output_dim = layer_dims[i + 1]

            # Linear layer
            layers.append(nn.Linear(input_dim, output_dim))

            # Batch normalization (if enabled) - only add if we have reasonable batch sizes
            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm1d(output_dim))

            # Activation function
            layers.append(self._get_activation_function())

            # Dropout (if enabled and not last hidden layer)
            if self.config.use_dropout and self.config.dropout_rate > 0 and i < len(layer_dims) - 3:
                layers.append(nn.Dropout(self.config.dropout_rate))

        # Output layer (no activation - raw h(x) values)
        layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))

        return nn.Sequential(*layers)

    def _get_activation_function(self) -> nn.Module:
        """Get the activation function module based on configuration."""
        activation_map = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(negative_slope=0.01),
            'elu': nn.ELU(),
            'swish': nn.SiLU(),  # Swish is same as SiLU
        }

        activation = self.config.get_activation_function()
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation function: {activation}. "
                           f"Supported: {list(activation_map.keys())}")

        return activation_map[activation]

    def _initialize_weights(self):
        """Initialize network weights using appropriate initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                # Batch norm initialization
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Output tensor [batch_size, output_dim] with h(x) values
        """
        return self.network(x)

    def h(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute the barrier function value h(x) using the neural network.

        This is the primary method that must be implemented for the CBF interface.

        Args:
            x: State vector [state_dim] or [batch_size, state_dim]

        Returns:
            h(x) values (scalar or array matching input shape)

        Note:
            h(x) >= 0 indicates safe state
            h(x) < 0 indicates unsafe state
        """
        # Handle input type conversion
        use_numpy = isinstance(x, np.ndarray)
        original_shape = x.shape if hasattr(x, 'shape') else None

        if use_numpy:
            x = torch.tensor(x, dtype=torch.float32)

        # Ensure proper batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension: [state_dim] -> [1, state_dim]
            single_sample = True
        else:
            single_sample = False

        # Compute h(x) using neural network
        h_val = self.forward(x)  # Returns [batch_size, 1]

        # Squeeze the output dimension (since we have output_dim=1)
        h_val = h_val.squeeze(-1)  # [batch_size, 1] -> [batch_size]

        # Handle single sample case
        if single_sample:
            h_val = h_val.squeeze(0)  # [1] -> scalar or [] depending on tensor type

        # Convert back to numpy if needed
        if use_numpy:
            result = h_val.detach().cpu().numpy()
            # Ensure proper scalar/array format
            if single_sample and result.ndim == 0:
                return float(result)  # Return scalar for single input
            return result
        else:
            if single_sample and h_val.dim() == 0:
                return h_val.item()  # Return Python scalar for single input
            return h_val

    def grad_h(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute gradient ∇h(x) using automatic differentiation.

        This method uses PyTorch's autograd to compute gradients of the neural network.

        Args:
            x: State vector [state_dim] or [batch_size, state_dim]

        Returns:
            ∇h(x) gradients [state_dim] or [batch_size, state_dim]

        Note:
            The gradient is computed with respect to the full state vector,
            including orientation θ even though h(x) typically only depends on position.
        """
        # Handle input type conversion
        use_numpy = isinstance(x, np.ndarray)

        if use_numpy:
            x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        else:
            x = x.detach().requires_grad_(True)

        # Ensure proper batch dimension
        original_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            single_sample = True
        else:
            single_sample = False

        # Compute h(x) with gradient tracking
        h_val = self.forward(x)

        # Compute gradient using automatic differentiation
        if x.shape[0] == 1:
            # Single state - sum to get scalar for gradient computation
            grad_h = torch.autograd.grad(h_val.sum(), x, create_graph=False)[0]
            if single_sample:
                grad_h = grad_h.squeeze(0)  # Remove batch dimension
        else:
            # Batch of states - sum across batch for gradient computation
            grad_h = torch.autograd.grad(h_val.sum(), x, create_graph=False)[0]

        # Convert back to numpy if needed
        if use_numpy:
            return grad_h.detach().cpu().numpy()
        else:
            return grad_h.detach()

    def save_model(self, filepath: Union[str, Path]):
        """
        Save the NCBF model to a file.

        Args:
            filepath: Path to save the model (supports both .pt and .pth extensions)
        """
        filepath = Path(filepath)

        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model state dict and config
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_class': self.__class__.__name__,
        }, filepath)

        print(f"✅ NCBF model saved to: {filepath}")

    def load_model(self, filepath: Union[str, Path], map_location: Optional[str] = None):
        """
        Load the NCBF model from a file.

        Args:
            filepath: Path to the saved model
            map_location: Device to map the model to (e.g., 'cpu', 'cuda:0')

        Returns:
            Loaded NCBF instance
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)

        # Verify this is an NCBF model
        if 'config' not in checkpoint or 'model_state_dict' not in checkpoint:
            raise ValueError("Invalid NCBF model file - missing required keys")

        # Load state dict
        self.load_state_dict(checkpoint['model_state_dict'])

        print(f"✅ NCBF model loaded from: {filepath}")
        return self

    def get_model_info(self) -> dict:
        """Get information about the NCBF model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'architecture': self.config.get_network_architecture(),
            'activation': self.config.get_activation_function(),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Approximate size in MB
        }

    def __str__(self) -> str:
        """String representation of the NCBF model."""
        info = self.get_model_info()
        return (f"NCBF(architecture={info['architecture']}, "
                f"activation={info['activation']}, "
                f"params={info['total_parameters']})")

    def __repr__(self) -> str:
        """Detailed representation of the NCBF model."""
        return self.__str__()