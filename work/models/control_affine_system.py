"""
Base class for control-affine dynamical systems.

A control-affine system has the form:
    x_dot = f(x) + g(x) * u
where x is the state, u is the control input, f(x) is the drift dynamics,
and g(x) is the control input matrix.
"""

from abc import ABC, abstractmethod
import numpy as np


class ControlAffineSystem(ABC):
    """Abstract base class for control-affine dynamical systems."""

    @abstractmethod
    def f(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the drift dynamics f(x).

        Args:
            x: State vector

        Returns:
            f(x): Drift dynamics vector
        """
        pass

    @abstractmethod
    def g(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the control input matrix g(x).

        Args:
            x: State vector

        Returns:
            g(x): Control input matrix
        """
        pass

    @abstractmethod
    def state_dim(self) -> int:
        """Return the dimension of the state space."""
        pass

    @abstractmethod
    def control_dim(self) -> int:
        """Return the dimension of the control input space."""
        pass

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute the full dynamics x_dot = f(x) + g(x) * u.

        Args:
            x: State vector
            u: Control vector

        Returns:
            x_dot: Time derivative of state
        """
        return self.f(x) + np.dot(self.g(x), u)

    def __str__(self) -> str:
        """String representation of the system."""
        return f"{self.__class__.__name__}(state_dim={self.state_dim()}, control_dim={self.control_dim()})"

    def __repr__(self) -> str:
        """Detailed string representation of the system."""
        return self.__str__()