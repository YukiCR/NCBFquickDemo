"""
Configuration file for unicycle model parameters.

This file contains all the parameters needed for the unicycle model,
including physical parameters, control constraints, controller gains,
and visualization settings.
"""

from dataclasses import dataclass
from typing import Union, Tuple


@dataclass
class UnicycleConfig:
    """Configuration class for unicycle model parameters."""

    # Physical parameters
    robot_radius: float = 0.2          # Robot radius (m)
    D: float = 0.05                  # Distance for control point (m) - needed for M matrix

    # Control constraints - norm constraint for explicit optimal control in loss computation
    max_control_norm: float = 2.0      # Control norm constraint sqrt(v² + ω²)

    # PD controller gains - basic control only (from MATLAB getPcontrol)
    kp_linear_basic: float = 0.8       # Basic PD: linear velocity gain
    kp_angular_basic: float = 1.2      # Basic PD: angular velocity gain

    # Proportional controller gain
    kp_proportional: float = 0.15       # Proportional control velocity scaling factor (tuned for stability)

    # CBF parameters
    cbf_alpha: float = 0.5             # Barrier function parameter
    safety_radius: float = 0.2        # Minimum safe distance from obstacles (for CBF constraints)

    # Obstacle parameters
    min_obstacle_radius: float = 0.1   # Minimum obstacle radius for visualization scaling

    # Simulation parameters
    dt: float = 0.05                    # Time step (s)
    max_history: int = 1000            # Maximum history buffer size

    # Visualization parameters - support RGB hex codes and alpha
    car_color: Union[str, Tuple[float, float, float]] = '#0066CC'      # Blue in hex
    obstacle_color: Union[str, Tuple[float, float, float]] = '#CC0000'  # Red in hex
    trajectory_color: Union[str, Tuple[float, float, float]] = '#00CC00'  # Green in hex
    visualization_alpha: float = 0.5   # Alpha for transparency
    car_size: int = 20                 # Size of car marker in plots

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.robot_radius <= 0:
            raise ValueError("Robot radius must be positive")
        if self.D <= 0:
            raise ValueError("D parameter must be positive")
        if self.max_control_norm <= 0:
            raise ValueError("Max control norm must be positive")
        if self.cbf_alpha <= 0:
            raise ValueError("CBF alpha must be positive")
        if self.safety_radius <= 0:
            raise ValueError("Safety radius must be positive")
        if self.min_obstacle_radius <= 0:
            raise ValueError("Minimum obstacle radius must be positive")
        if self.dt <= 0:
            raise ValueError("Time step must be positive")
        if not (0 <= self.visualization_alpha <= 1):
            raise ValueError("Visualization alpha must be between 0 and 1")