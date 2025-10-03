"""
Abstract base class for Control Barrier Functions (CBFs).

This module defines the minimal abstract interface for CBF implementations.
Subclasses can implement gradient computation via closed-form, symbolic,
automatic differentiation, or any other method.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List, Optional


class CBFFunction(ABC):
    """
    Abstract base class for Control Barrier Functions.

    A Control Barrier Function h(x) satisfies:
    - h(x) ≥ 0 for safe states
    - h(x) < 0 for unsafe states
    - dh/dt ≥ -αh for safety (forward invariance)
    """

    def __init__(self, state_dim: int, alpha: float = 1.0):
        """
        Initialize CBF function.

        Args:
            state_dim: Dimension of state space
            alpha: CBF parameter for safety condition (dh/dt ≥ -αh)
        """
        self.state_dim = state_dim
        self.alpha = alpha

        # Basic validation
        if state_dim <= 0:
            raise ValueError("State dimension must be positive")
        if alpha <= 0:
            raise ValueError("Alpha must be positive")

    @abstractmethod
    def h(self, x: Union[np.ndarray, 'torch.Tensor']) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Compute the barrier function value h(x).

        Args:
            x: State vector [state_dim] or [batch_size, state_dim]

        Returns:
            h(x) values (scalar or array matching input shape)

        Note:
            h(x) ≥ 0 indicates safe state
            h(x) < 0 indicates unsafe state
        """
        pass

    @abstractmethod
    def grad_h(self, x: Union[np.ndarray, 'torch.Tensor']) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Compute gradient ∇h(x).

        Args:
            x: State vector [state_dim] or [batch_size, state_dim]

        Returns:
            ∇h(x) gradients [state_dim] or [batch_size, state_dim]

        Note:
            Implementation left to subclass - can use closed-form,
            symbolic computation, automatic differentiation, etc.
        """
        pass

    def is_safe(self, x: Union[np.ndarray, 'torch.Tensor'],
                margin: float = 0.0) -> Union[bool, np.ndarray]:
        """
        Check if state(s) are safe.

        Args:
            x: State vector(s)
            margin: Safety margin (h(x) ≥ margin for safety)

        Returns:
            Boolean or boolean array indicating safety
        """
        h_vals = self.h(x)

        if hasattr(h_vals, 'numpy'):  # torch.Tensor
            return h_vals.detach().cpu().numpy() >= margin
        elif hasattr(h_vals, 'detach'):  # torch.Tensor scalar
            return h_vals.detach().cpu().numpy() >= margin
        else:  # numpy array or scalar
            return h_vals >= margin

    def get_safety_level(self, x: Union[np.ndarray, 'torch.Tensor']) -> Union[float, np.ndarray]:
        """
        Get safety level (h(x) values).

        Args:
            x: State vector(s)

        Returns:
            h(x) values
        """
        return self.h(x)


# Export the abstract base class
__all__ = ['CBFFunction']