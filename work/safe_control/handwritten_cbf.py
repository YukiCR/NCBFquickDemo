"""
Handwritten Control Barrier Function implementations.

This module provides concrete implementations of CBF functions for obstacle avoidance,
supporting both single and multiple obstacles with proper gradient computation.
"""

import numpy as np
import torch
from typing import Union, List, Optional
from .cbf_function import CBFFunction


class CBFsingleobs(CBFFunction):
    """
    CBF for single circular obstacle avoidance.

    Uses the barrier function: h(x) = ||x - x_obs|| - (safety_radius + obs_radius)
    where x = [px, py, theta] is the robot state, x_obs = [x_obs, y_obs] is obstacle center.
    """

    def __init__(self, config, obstacle: Union[np.ndarray, torch.Tensor]):
        """
        Initialize single obstacle CBF.

        Args:
            config: Configuration object containing safety_radius and other parameters
            obstacle: Obstacle definition as [x, y] or [x, y, radius] array
        """
        super().__init__(state_dim=3, alpha=getattr(config, 'alpha', 1.0))

        self.config = config
        self.safety_radius = getattr(config, 'safety_radius', 0.3)

        # Parse obstacle format
        if isinstance(obstacle, np.ndarray):
            obstacle = torch.tensor(obstacle, dtype=torch.float32)
        elif isinstance(obstacle, torch.Tensor):
            obstacle = obstacle.float()

        if len(obstacle) == 2:
            # [x, y] format - use default obstacle radius
            self.obstacle_center = obstacle[:2]
            self.obstacle_radius = getattr(config, 'min_obstacle_radius', 0.1)
        elif len(obstacle) == 3:
            # [x, y, radius] format
            self.obstacle_center = obstacle[:2]
            self.obstacle_radius = obstacle[2]
        else:
            raise ValueError("Obstacle must be [x, y] or [x, y, radius]")

        self.safety_distance = self.safety_radius + self.obstacle_radius

    def h(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute barrier function value h(x) = ||x_robot - x_obs|| - safety_distance.

        Args:
            x: State vector [px, py, theta] or [batch_size, 3]

        Returns:
            h(x) values (scalar or array matching input shape)
        """
        use_numpy = isinstance(x, np.ndarray)

        if use_numpy:
            x = torch.tensor(x, dtype=torch.float32)

        # Extract robot position (first 2 elements)
        if x.dim() == 1:
            robot_pos = x[:2]  # [px, py]
        else:
            robot_pos = x[:, :2]  # [batch_size, 2]

        # Compute distance to obstacle
        distance = torch.norm(robot_pos - self.obstacle_center, dim=-1 if x.dim() > 1 else 0)

        # Compute barrier function
        h_val = distance - self.safety_distance

        if use_numpy:
            return h_val.detach().cpu().numpy()
        else:
            return h_val

    def grad_h(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute gradient ∇h(x) using closed-form expression.

        For h(x) = ||x_robot - x_obs|| - safety_distance:
        ∇h(x) = [(x_robot - x_obs)/||x_robot - x_obs||, 0] (gradient w.r.t. [px, py, theta])

        Args:
            x: State vector [px, py, theta] or [batch_size, 3]

        Returns:
            ∇h(x) gradients [3] or [batch_size, 3]
        """
        use_numpy = isinstance(x, np.ndarray)

        if use_numpy:
            x = torch.tensor(x, dtype=torch.float32)

        # Extract robot position
        if x.dim() == 1:
            robot_pos = x[:2]  # [px, py]
            batch_mode = False
        else:
            robot_pos = x[:, :2]  # [batch_size, 2]
            batch_mode = True

        # Compute vector from obstacle to robot
        vec_robot_to_obs = robot_pos - self.obstacle_center

        # Compute distance
        distance = torch.norm(vec_robot_to_obs, dim=-1 if batch_mode else 0, keepdim=True if batch_mode else False)

        # Avoid division by zero
        epsilon = 1e-8
        if batch_mode:
            distance = distance.squeeze(-1)
            safe_distance = torch.clamp(distance, min=epsilon)
            grad_xy = vec_robot_to_obs / safe_distance.unsqueeze(-1)
            grad_theta = torch.zeros(x.shape[0], 1, dtype=torch.float32)
            grad_h = torch.cat([grad_xy, grad_theta], dim=-1)  # [batch_size, 3]
        else:
            safe_distance = torch.clamp(distance, min=epsilon)
            grad_xy = vec_robot_to_obs / safe_distance
            grad_theta = torch.tensor([0.0], dtype=torch.float32)
            grad_h = torch.cat([grad_xy, grad_theta])  # [3]

        if use_numpy:
            return grad_h.detach().cpu().numpy()
        else:
            return grad_h


class CBFmultipleobs(CBFFunction):
    """
    CBF for multiple circular obstacles using soft-min (log-sum-exp) formulation.

    Uses the barrier function: h(x) = -log(sum(exp(-α * h_i(x)))) / α
    where h_i(x) are individual obstacle barrier functions and α > 0 controls smoothness.
    """

    def __init__(self, config, obstacles: List[Union[np.ndarray, torch.Tensor]], alpha_softmin: float = 10.0):
        """
        Initialize multiple obstacles CBF.

        Args:
            config: Configuration object containing safety_radius and other parameters
            obstacles: List of obstacle definitions as [x, y] or [x, y, radius] arrays
            alpha_softmin: Smoothing parameter for soft-min (larger = closer to true min)
        """
        super().__init__(state_dim=3, alpha=getattr(config, 'alpha', 1.0))

        self.config = config
        self.safety_radius = getattr(config, 'safety_radius', 0.3)
        self.alpha_softmin = alpha_softmin

        # Parse all obstacles
        self.obstacle_centers = []
        self.obstacle_radii = []

        for obstacle in obstacles:
            if isinstance(obstacle, np.ndarray):
                obstacle = torch.tensor(obstacle, dtype=torch.float32)
            elif isinstance(obstacle, torch.Tensor):
                obstacle = obstacle.float()

            if len(obstacle) == 2:
                # [x, y] format
                self.obstacle_centers.append(obstacle[:2])
                self.obstacle_radii.append(getattr(config, 'min_obstacle_radius', 0.1))
            elif len(obstacle) == 3:
                # [x, y, radius] format
                self.obstacle_centers.append(obstacle[:2])
                self.obstacle_radii.append(obstacle[2])
            else:
                raise ValueError("Each obstacle must be [x, y] or [x, y, radius]")

        self.num_obstacles = len(self.obstacle_centers)
        self.safety_distances = [self.safety_radius + radius for radius in self.obstacle_radii]

    def _compute_individual_h(self, robot_pos: torch.Tensor, obstacle_idx: int) -> torch.Tensor:
        """
        Compute individual barrier function for a single obstacle.

        Args:
            robot_pos: Robot position [px, py] or [batch_size, 2]
            obstacle_idx: Index of the obstacle

        Returns:
            Individual h_i(x) values
        """
        obstacle_center = self.obstacle_centers[obstacle_idx]
        safety_distance = self.safety_distances[obstacle_idx]

        # Compute distance to obstacle
        distance = torch.norm(robot_pos - obstacle_center, dim=-1 if robot_pos.dim() > 1 else 0)

        # Individual barrier function: h_i(x) = distance - safety_distance
        return distance - safety_distance

    def h(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute barrier function value using soft-min formulation.

        h(x) = -log(sum(exp(-α * h_i(x)))) / α

        Args:
            x: State vector [px, py, theta] or [batch_size, 3]

        Returns:
            h(x) values (scalar or array matching input shape)
        """
        use_numpy = isinstance(x, np.ndarray)

        if use_numpy:
            x = torch.tensor(x, dtype=torch.float32)

        # Extract robot position
        if x.dim() == 1:
            robot_pos = x[:2]  # [px, py]
            batch_mode = False
        else:
            robot_pos = x[:, :2]  # [batch_size, 2]
            batch_mode = True

        # Compute individual barrier functions for all obstacles
        individual_h_values = []
        for i in range(self.num_obstacles):
            h_i = self._compute_individual_h(robot_pos, i)
            individual_h_values.append(h_i)

        # Stack individual h values
        if batch_mode:
            h_matrix = torch.stack(individual_h_values, dim=1)  # [batch_size, num_obstacles]
        else:
            h_matrix = torch.stack(individual_h_values, dim=0)  # [num_obstacles]

        # Apply soft-min: h(x) = -log(sum(exp(-α * h_i))) / α
        # This approximates min(h_i) when α is large
        exp_neg_alpha_h = torch.exp(-self.alpha_softmin * h_matrix)
        sum_exp = torch.sum(exp_neg_alpha_h, dim=-1)  # Sum over obstacles

        # Compute soft-min
        h_val = -torch.log(sum_exp) / self.alpha_softmin

        if use_numpy:
            return h_val.detach().cpu().numpy()
        else:
            return h_val

    def grad_h(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute gradient ∇h(x) using automatic differentiation.

        For the soft-min formulation, the gradient is:
        ∇h(x) = sum(w_i * ∇h_i(x)) where w_i = exp(-α * h_i) / sum(exp(-α * h_j))

        Args:
            x: State vector [px, py, theta] or [batch_size, 3]

        Returns:
            ∇h(x) gradients [3] or [batch_size, 3]
        """
        use_numpy = isinstance(x, np.ndarray)

        if use_numpy:
            x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        else:
            x = x.detach().requires_grad_(True)

        # Compute h(x) with gradient tracking
        h_val = self.h(x)

        # Compute gradient using automatic differentiation
        if x.dim() == 1:
            # Single state
            grad_h = torch.autograd.grad(h_val, x, create_graph=False)[0]
        else:
            # Batch of states
            grad_h = torch.autograd.grad(h_val.sum(), x, create_graph=False)[0]

        if use_numpy:
            return grad_h.detach().cpu().numpy()
        else:
            return grad_h.detach()