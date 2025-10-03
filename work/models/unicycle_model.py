"""
Unicycle model implementation with integrated PD controller and visualization.

This module implements a unicycle robot model that extends the control-affine
system base class. It includes the basic PD control strategy from the MATLAB
carModel.m implementation and integrated visualization capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Union
import math

from .control_affine_system import ControlAffineSystem
from configs.unicycle_config import UnicycleConfig


class UnicycleModel(ControlAffineSystem):
    """
    Unicycle model with integrated PD controller and visualization.

    State: x = [px, py, theta] (position and orientation)
    Control: u = [v, omega] (linear and angular velocities)
    Dynamics: px_dot = v * cos(theta), py_dot = v * sin(theta), theta_dot = omega
    """

    def __init__(self, config: UnicycleConfig, initial_state: Optional[np.ndarray] = None,
                 target: Optional[np.ndarray] = None):
        """
        Initialize unicycle model.

        Args:
            config: Configuration object with model parameters
            initial_state: Initial state [px, py, theta]. If None, starts at origin.
            target: Target position [px_target, py_target]. If None, set to origin.
        """
        self.config = config

        # Initialize state
        if initial_state is None:
            self.state = np.array([0.0, 0.0, 0.0])  # [px, py, theta]
        else:
            self.state = np.array(initial_state)
            if len(self.state) != 3:
                raise ValueError("State must be 3D: [px, py, theta]")

        # Initialize target
        if target is None:
            self.target = np.array([0.0, 0.0])  # Default target at origin
        else:
            self.target = np.array(target)
            if len(self.target) != 2:
                raise ValueError("Target must be 2D: [px_target, py_target]")

        # History for visualization
        self.history = [self.state.copy()]
        self.control_history = []

    def f(self, x: np.ndarray) -> np.ndarray:
        """
        Compute drift dynamics f(x) (no drift when u=0).

        Args:
            x: State vector [px, py, theta]

        Returns:
            f(x): Drift dynamics [0, 0, 0]
        """
        return np.zeros(3)

    def g(self, x: np.ndarray) -> np.ndarray:
        """
        Compute control input matrix g(x).

        Args:
            x: State vector [px, py, theta]

        Returns:
            g(x): Control input matrix 3x2
        """
        theta = x[2]
        return np.array([
            [np.cos(theta), 0],      # px_dot = v * cos(theta)
            [np.sin(theta), 0],      # py_dot = v * sin(theta)
            [0, 1]                     # theta_dot = omega
        ])

    def state_dim(self) -> int:
        """Return state dimension."""
        return 3

    def control_dim(self) -> int:
        """Return control dimension."""
        return 2

    def get_state(self) -> np.ndarray:
        """Get current state."""
        return self.state.copy()

    def get_target(self) -> np.ndarray:
        """Get current target."""
        return self.target.copy()

    def set_target(self, target: np.ndarray):
        """Set target position."""
        self.target = np.array(target)
        if len(self.target) != 2:
            raise ValueError("Target must be 2D: [px_target, py_target]")

    def set_state(self, state: np.ndarray):
        """Set current state."""
        self.state = np.array(state)
        if len(self.state) != 3:
            raise ValueError("State must be 3D: [px, py, theta]")

    def update_state(self, control: np.ndarray, dt: Optional[float] = None):
        """
        Update state using Euler integration with control constraints.

        Args:
            control: Control input [v, omega]
            dt: Time step. If None, uses config.dt
        """
        if dt is None:
            dt = self.config.dt

        # Apply control constraints
        constrained_control = self._apply_control_constraints(control)

        # Update state using dynamics
        state_dot = self.dynamics(self.state, constrained_control)
        self.state = self.state + state_dot * dt

        # Normalize angle to [-pi, pi]
        self.state[2] = self._normalize_angle(self.state[2])

        # Record history
        self.history.append(self.state.copy())
        self.control_history.append(constrained_control.copy())

        # Limit history size
        if len(self.history) > self.config.max_history:
            self.history.pop(0)
            if self.control_history:
                self.control_history.pop(0)

    def _apply_control_constraints(self, control: np.ndarray) -> np.ndarray:
        """
        Apply control constraints including norm limits.

        Args:
            control: Raw control input [v, omega]

        Returns:
            Constrained control input
        """
        v, omega = control

        # Norm constraint: sqrt(v² + ω²) <= max_control_norm
        norm = np.sqrt(v**2 + omega**2)
        if norm > self.config.max_control_norm:
            scale = self.config.max_control_norm / norm
            v *= scale
            omega *= scale

        return np.array([v, omega])

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    def get_transformation_matrix(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get transformation matrix for converting underactuated control to virtual fully actuated system.

        For CBF applications with underactuated unicycle, we need to transform the constraint
        from control space [v, omega] to virtual velocity space [vx, vy, vomega].

        Following the MATLAB carModel.m implementation, we use a control point ahead of the robot
        to create a virtually fully actuated system. The transformation matrix M converts:
        [vx; vy; vomega] = M * [v; omega]

        Args:
            x: State vector [px, py, theta]. If None, uses current state.

        Returns:
            M: Transformation matrix 3x2 that makes system virtually fully actuated
        """
        if x is None:
            theta = self.state[2]
        else:
            theta = x[2]

        # Use the same D parameter as in the proportional control
        D = self.config.D

        # Transformation matrix from MATLAB carModel.m
        # This creates a virtually fully actuated system
        M = np.array([
            [np.cos(theta), -D * np.sin(theta)],      # vx = v*cos(theta) - D*omega*sin(theta)
            [np.sin(theta),  D * np.cos(theta)],      # vy = v*sin(theta) + D*omega*cos(theta)
            [0, 1]                                       # vomega = omega
        ])

        return M

    def get_virtual_dynamics_matrix(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get the virtual dynamics matrix for CBF constraint computation.

        For underactuated systems, CBF constraints should be applied to the virtual
        fully actuated dynamics. This method returns the matrix that should be used
        instead of g(x) for CBF Lie derivative computation.

        The virtual dynamics matrix transforms the gradient to the virtual control space:
        L_g_virtual h = ∇h * M where M is the transformation matrix.

        Args:
            x: State vector [px, py, theta]. If None, uses current state.

        Returns:
            Virtual dynamics matrix for CBF computation (same as transformation matrix)
        """
        import warnings
        warnings.warn(
            "Using virtual fully actuated dynamics for CBF computation. "
            "This treats the underactuated unicycle as virtually fully actuated "
            "by converting constraints from [v, omega] space to [vx, vy, vomega] space.",
            UserWarning
        )
        return self.get_transformation_matrix(x)

    # ===== PD Controllers (from MATLAB carModel.m) =====

    def pd_control_basic(self) -> np.ndarray:
        """
        Basic PD control from MATLAB getPcontrol().
        Uses the internal target property.

        Returns:
            Control input [v, omega]
        """
        px, py, theta = self.state
        target_pos = self.target
        current_pos = np.array([px, py])

        # Position error vector
        error_vec = target_pos - current_pos

        # Distance error (projection along heading direction)
        distance_error = np.dot(error_vec, np.array([np.cos(theta), np.sin(theta)]))

        # Angle error (heading toward target)
        target_angle = np.arctan2(error_vec[1], error_vec[0])
        angle_error = self._normalize_angle(target_angle - theta)

        # PD control law
        v = self.config.kp_linear_basic * distance_error
        omega = self.config.kp_angular_basic * angle_error

        return np.array([v, omega])

    def pd_control_proportional(self) -> np.ndarray:
        """
        Proportional control from MATLAB getPorpotionalcontrol().
        Uses transformation matrix M for point stabilization.
        Uses the internal target property.

        Returns:
            Control input [v, omega]
        """
        px, py, theta = self.state
        target_pos = self.target

        # Control point (ahead of robot by distance D) - from MATLAB implementation
        D = self.config.D
        Pa = np.array([px + D * np.cos(theta), py + D * np.sin(theta)])

        # Transformation matrix (from MATLAB implementation)
        M = np.array([
            [np.cos(theta), -D * np.sin(theta)],
            [np.sin(theta),  D * np.cos(theta)]
        ])

        # Desired velocity at control point (from MATLAB: V = (goal - Pa) * kp_proportional)
        V = self.config.kp_proportional * (target_pos - Pa)

        # Solve for control inputs: M * [v; omega] = V
        try:
            vs = np.linalg.solve(M, V)
            return vs
        except np.linalg.LinAlgError:
            # Fallback to basic control if matrix is singular
            return self.pd_control_basic()

    # ===== Visualization Methods =====

    def plot_trajectory(self, obstacles: Optional[List[np.ndarray]] = None,
                       show_direction: bool = True, show_target: bool = True,
                       figsize: Tuple[int, int] = (10, 8)):
        """
        Plot trajectory with optional obstacles and target.

        Args:
            obstacles: List of obstacle arrays [x, y, radius] or [x, y] (uses min_obstacle_radius)
            show_direction: Whether to show direction arrows
            show_target: Whether to show target position
            figsize: Figure size
        """
        if not self.history:
            print("No trajectory data available")
            return

        plt.figure(figsize=figsize)

        # Convert history to arrays
        history_array = np.array(self.history)
        px, py, theta = history_array[:, 0], history_array[:, 1], history_array[:, 2]

        # Plot trajectory
        plt.plot(px, py, color=self.config.trajectory_color, linewidth=2,
                label='Trajectory', marker='o', markersize=3, alpha=self.config.visualization_alpha)

        # Plot robot positions with direction
        if show_direction:
            step_interval = max(1, len(history_array) // 20)  # Show ~20 arrows
            for i in range(0, len(history_array), step_interval):
                x, y, ang = px[i], py[i], theta[i]
                dx = 0.3 * np.cos(ang)
                dy = 0.3 * np.sin(ang)
                plt.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1,
                         fc=self.config.car_color, ec=self.config.car_color, alpha=self.config.visualization_alpha)

        # Plot obstacles (support both [x,y,radius] and [x,y] formats)
        if obstacles:
            for i, obs in enumerate(obstacles):
                # Handle both [x, y, radius] and [x, y] formats
                if len(obs) == 3:
                    # [x, y, radius] format
                    x, y, radius = obs[0], obs[1], obs[2]
                elif len(obs) == 2:
                    # [x, y] format - use minimum obstacle radius
                    x, y = obs[0], obs[1]
                    radius = self.config.min_obstacle_radius
                else:
                    raise ValueError(f"Obstacle must be [x,y] or [x,y,radius], got {obs}")

                # Draw obstacle circle
                circle = plt.Circle((x, y), radius,
                                  color=self.config.obstacle_color, alpha=self.config.visualization_alpha * 0.5)
                plt.gca().add_patch(circle)
                plt.plot(x, y, 'o', color=self.config.obstacle_color,
                        markersize=8, label='Obstacle' if i == 0 else "")

        # Plot start and end
        plt.plot(px[0], py[0], 'go', markersize=10, label='Start')
        plt.plot(px[-1], py[-1], 'ro', markersize=10, label='End')

        # Plot target
        if show_target:
            plt.plot(self.target[0], self.target[1], 'm*', markersize=15, label='Target')

        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Unicycle Trajectory')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_current_state(self, obstacles: Optional[List[np.ndarray]] = None,
                          show_trajectory: bool = True, show_target: bool = True):
        """
        Plot current state with robot representation and obstacles.

        Args:
            obstacles: List of obstacle arrays [x, y, radius] or [x, y] (uses min_obstacle_radius)
            show_trajectory: Whether to show trajectory history
            show_target: Whether to show target position
        """
        plt.figure(figsize=(10, 8))

        px, py, theta = self.state

        # Plot trajectory history
        if show_trajectory and len(self.history) > 1:
            history_array = np.array(self.history)
            plt.plot(history_array[:, 0], history_array[:, 1],
                    color=self.config.trajectory_color, alpha=self.config.visualization_alpha * 0.7,
                    linewidth=1, label='Trajectory')

        # Plot robot as circle with orientation arrow
        robot_circle = plt.Circle((px, py), self.config.robot_radius,
                                 color=self.config.car_color, alpha=self.config.visualization_alpha)
        plt.gca().add_patch(robot_circle)

        # Plot orientation arrow
        arrow_length = self.config.robot_radius * 1.5
        dx = arrow_length * np.cos(theta)
        dy = arrow_length * np.sin(theta)
        plt.arrow(px, py, dx, dy, head_width=0.1, head_length=0.1,
                 fc='black', ec='black', linewidth=2)

        # Plot obstacles (support both [x,y,radius] and [x,y] formats)
        if obstacles:
            for i, obs in enumerate(obstacles):
                # Handle both [x, y, radius] and [x, y] formats
                if len(obs) == 3:
                    # [x, y, radius] format
                    x, y, radius = obs[0], obs[1], obs[2]
                elif len(obs) == 2:
                    # [x, y] format - use minimum obstacle radius
                    x, y = obs[0], obs[1]
                    radius = self.config.min_obstacle_radius
                else:
                    raise ValueError(f"Obstacle must be [x,y] or [x,y,radius], got {obs}")

                # Draw obstacle circle
                circle = plt.Circle((x, y), radius,
                                  color=self.config.obstacle_color, alpha=self.config.visualization_alpha * 0.5)
                plt.gca().add_patch(circle)
                plt.plot(x, y, 'o', color=self.config.obstacle_color,
                        markersize=8, label='Obstacle' if i == 0 else "")

        # Plot target
        if show_target:
            plt.plot(self.target[0], self.target[1], 'm*', markersize=15, label='Target')

        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title(f'Current State: ({px:.2f}, {py:.2f}, {theta:.2f} rad)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def clear_history(self):
        """Clear trajectory and control history."""
        self.history = [self.state.copy()]
        self.control_history = []

    def get_history(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get trajectory and control history.

        Returns:
            (trajectory, controls): Tuple of history arrays
        """
        trajectory = np.array(self.history)
        controls = np.array(self.control_history) if self.control_history else np.array([])
        return trajectory, controls

    # Override string representations from base class
    def __str__(self) -> str:
        """String representation of the unicycle model."""
        px, py, theta = self.state
        tx, ty = self.target
        return f"UnicycleModel(state=[{px:.2f}, {py:.2f}, {theta:.2f}], target=[{tx:.2f}, {ty:.2f}])"

    def __repr__(self) -> str:
        """Detailed string representation of the unicycle model."""
        return (f"UnicycleModel(config={self.config.__class__.__name__}, "
                f"initial_state={self.get_state().tolist()}, "
                f"target={self.get_target().tolist()})")