"""
CBF Safety Filter with Lie Derivatives.

Simple and robust implementation with analytical solution for single constraint case.
Implements the CBF-QP formulation with only CBF constraints.
"""

import numpy as np
import torch
from typing import Optional, Union, Tuple

# Import CBF and system interfaces
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from safe_control.cbf_function import CBFFunction
from models.control_affine_system import ControlAffineSystem
from models.unicycle_model import UnicycleModel


class CBFFilter:
    """
    Simple CBF Safety Filter for computing safe control inputs.

    Implements the CBF-QP formulation:
    minimize    ||u - u_nom||²
    subject to  L_f h(x) + L_g h(x)·u ≥ -αh(x)

    where L_f h = ∇h·f(x) and L_g h = ∇h·g(x) are Lie derivatives.
    Uses analytical solution for single constraint (much faster than QP solver).
    """

    def __init__(self, cbf_function: CBFFunction, control_affine_system: ControlAffineSystem,
                 alpha: Optional[float] = None):
        """
        Initialize CBF safety filter.

        Args:
            cbf_function: CBF implementation providing h(x) and ∇h(x)
            control_affine_system: Control-affine system with f(x) and g(x)
            alpha: CBF parameter for safety condition (dh/dt ≥ -αh).
                   If None, uses default from config or can be set per unicycle model.
        """
        self.cbf = cbf_function
        self.system = control_affine_system
        self.alpha = alpha

        # Validate compatibility
        if self.cbf.state_dim != self.system.state_dim():
            raise ValueError(f"CBF state dim ({self.cbf.state_dim}) != System state dim ({self.system.state_dim()})")

        # Set alpha based on system type if not provided
        if self.alpha is None:
            self.alpha = self._get_default_alpha()

    def _get_default_alpha(self) -> float:
        """Get default alpha based on system configuration."""
        # For unicycle model, use cbf_alpha from config
        if isinstance(self.system, UnicycleModel):
            return getattr(self.system.config, 'cbf_alpha', 1.0)
        else:
            return 1.0  # Default value

    def set_alpha(self, alpha: float) -> None:
        """Set the CBF alpha parameter."""
        self.alpha = alpha

    def compute_lie_derivatives(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute Lie derivatives L_f h(x) and L_g h(x).

        For underactuated systems like unicycle, uses virtual fully actuated dynamics
        to properly handle CBF constraints.

        Args:
            x: State vector [state_dim] as numpy array

        Returns:
            (L_f h, L_g h) where L_f h is scalar and L_g h is [control_dim] vector
        """
        # Convert to tensor for system calls, then back to numpy
        x_tensor = torch.tensor(x, dtype=torch.float32)

        # Get gradients and system dynamics
        grad_h = self.cbf.grad_h(x)  # Should return numpy array
        f_x = self.system.f(x_tensor)  # Returns numpy array

        # For unicycle (underactuated system), use virtual dynamics matrix
        if isinstance(self.system, UnicycleModel):
            # Use virtual dynamics matrix for CBF constraint computation
            g_virtual = self.system.get_virtual_dynamics_matrix(x)
            g_x = g_virtual
        else:
            # Use standard dynamics for fully actuated systems
            g_x = self.system.g(x_tensor)

        # Ensure all are numpy arrays
        if isinstance(grad_h, torch.Tensor):
            grad_h = grad_h.numpy()
        if isinstance(f_x, torch.Tensor):
            f_x = f_x.numpy()
        if isinstance(g_x, torch.Tensor):
            g_x = g_x.numpy()

        # Compute Lie derivatives using numpy operations
        L_f_h = float(np.dot(grad_h, f_x))  # ∇h·f
        L_g_h = np.dot(g_x.T, grad_h)       # ∇h·g

        return L_f_h, L_g_h

    def compute_safe_control(self, x: np.ndarray, u_nominal: np.ndarray,
                           alpha: Optional[float] = None) -> np.ndarray:
        """
        Compute safe control input using analytical CBF solution.

        For single constraint: minimize ||u - u_nom||² subject to a·u ≥ b
        Analytical solution: u = u_nom + λa where λ = max(0, (b - a·u_nom)/||a||²)

        For underactuated systems with virtual dynamics, solves the QP in virtual control space
        then transforms back to original control space.

        Args:
            x: Current state [state_dim] as numpy array
            u_nominal: Nominal control input [control_dim] as numpy array
            alpha: CBF parameter (uses self.alpha if None)

        Returns:
            Safe control input as numpy array
        """
        alpha_val = alpha if alpha is not None else self.alpha

        # Compute Lie derivatives
        L_f_h, L_g_h = self.compute_lie_derivatives(x)

        # Compute h(x)
        h_val = float(self.cbf.h(x))

        # CBF constraint: L_f h + L_g h·u ≥ -αh
        # Rearranged: L_g h·u ≥ -αh - L_f h
        constraint_value = -alpha_val * h_val - L_f_h

        # For underactuated systems with virtual dynamics, use special analytical solution
        if isinstance(self.system, UnicycleModel):
            u_safe = self._analytical_solution_virtual(u_nominal, L_g_h, constraint_value, x)
        else:
            # Standard analytical solution for fully actuated systems
            u_safe = self._analytical_solution(u_nominal, L_g_h, constraint_value)

        return u_safe

    def _analytical_solution_virtual(self, u_nominal: np.ndarray, L_g_h: np.ndarray,
                                   constraint_value: float, x: np.ndarray) -> np.ndarray:
        """
        Analytical solution for QP with virtual dynamics (underactuated systems).

        Following MATLAB implementation for unicycle with transformation matrix M:
        minimize    ||u - u_nom||²_M where ||u||²_M = uᵀMᵀMu
        subject to  a·u ≥ b

        Uses only the first two rows of the transformation matrix for [vx, vy] components.
        """
        # Get transformation matrix (3x2) but use only first two rows for virtual velocities
        M_full = self.system.get_transformation_matrix(x)
        M = M_full[:2, :]  # Use only first two rows for [vx, vy] components

        # Compute weighted norm matrix: MᵀM for virtual velocity norm
        MTM = np.dot(M.T, M)

        # Compute dot product
        a_dot_u_nom = np.dot(L_g_h, u_nominal)

        # Check if constraint is already satisfied
        if a_dot_u_nom >= constraint_value - 1e-8:
            return u_nominal

        # Compute Lagrange multiplier with weighted norm
        # For weighted norm: λ = (b - a·u_nom) / (aᵀ(MᵀM)⁻¹a)
        try:
            # Solve (MᵀM)x = L_g_h for x, then compute L_g_h·x
            MTM_inv_a = np.linalg.solve(MTM, L_g_h)
            a_norm_sq_weighted = np.dot(L_g_h, MTM_inv_a)

            if a_norm_sq_weighted < 1e-12:  # Avoid division by zero
                return u_nominal

            lambda_val = (constraint_value - a_dot_u_nom) / a_norm_sq_weighted
            lambda_val = max(0.0, lambda_val)  # Ensure λ ≥ 0

            # Optimal solution: u = u_nom + λ(MᵀM)⁻¹a
            u_safe = u_nominal + lambda_val * MTM_inv_a

            return u_safe

        except np.linalg.LinAlgError:
            # Fallback to standard solution if matrix is singular
            return self._analytical_solution(u_nominal, L_g_h, constraint_value)

    def _analytical_solution(self, u_nominal: np.ndarray, L_g_h: np.ndarray, constraint_value: float) -> np.ndarray:
        """
        Analytical solution for QP with single constraint.

        For: minimize ||u - u_nom||² subject to a·u ≥ b
        Solution: u = u_nom + λa where λ = max(0, (b - a·u_nom)/||a||²)
        """
        # Compute dot product a·u_nom
        a_dot_u_nom = np.dot(L_g_h, u_nominal)

        # Check if constraint is already satisfied
        if a_dot_u_nom >= constraint_value - 1e-8:
            return u_nominal

        # Compute Lagrange multiplier
        a_norm_sq = np.dot(L_g_h, L_g_h)
        if a_norm_sq < 1e-12:  # Avoid division by zero
            return u_nominal

        lambda_val = (constraint_value - a_dot_u_nom) / a_norm_sq
        lambda_val = max(0.0, lambda_val)  # Ensure λ ≥ 0

        # Optimal solution: u = u_nom + λa
        return u_nominal + lambda_val * L_g_h

    def check_safety(self, x: np.ndarray, u: np.ndarray, alpha: Optional[float] = None) -> bool:
        """
        Check if a control input satisfies the CBF safety condition.

        Args:
            x: State vector [state_dim] as numpy array
            u: Control input [control_dim] as numpy array
            alpha: CBF parameter (uses self.alpha if None)

        Returns:
            True if control is safe, False otherwise
        """
        alpha_val = alpha if alpha is not None else self.alpha

        # Compute Lie derivatives
        L_f_h, L_g_h = self.compute_lie_derivatives(x)

        # Compute h(x)
        h_val = float(self.cbf.h(x))

        # Safety condition: L_f h + L_g h·u ≥ -αh
        safety_check = L_f_h + np.dot(L_g_h, u) >= -alpha_val * h_val - 1e-8
        return bool(safety_check)

    def get_safety_margin(self, x: np.ndarray, u: np.ndarray, alpha: Optional[float] = None) -> float:
        """
        Compute the safety margin for a given state and control.

        Args:
            x: State vector [state_dim] as numpy array
            u: Control input [control_dim] as numpy array
            alpha: CBF parameter (uses self.alpha if None)

        Returns:
            Safety margin = L_f h + L_g h·u + αh (should be ≥ 0 for safety)
        """
        alpha_val = alpha if alpha is not None else self.alpha

        # Compute Lie derivatives
        L_f_h, L_g_h = self.compute_lie_derivatives(x)

        # Compute h(x)
        h_val = float(self.cbf.h(x))

        # Safety margin: L_f h + L_g h·u + αh
        margin = L_f_h + np.dot(L_g_h, u) + alpha_val * h_val
        return float(margin)


def create_cbf_filter(cbf_type: str, config, obstacles: list, system: ControlAffineSystem,
                     alpha: Optional[float] = None, **kwargs) -> 'CBFFilter':
    """
    Factory function for creating CBF filters.

    Args:
        cbf_type: 'single' or 'multiple'
        config: Configuration object
        obstacles: List of obstacle definitions
        system: Control-affine system
        alpha: CBF parameter (optional)
        **kwargs: Additional arguments for CBF construction

    Returns:
        Configured CBF filter instance
    """
    from safe_control.handwritten_cbf import CBFsingleobs, CBFmultipleobs

    if cbf_type == 'single':
        if len(obstacles) != 1:
            raise ValueError("Single CBF requires exactly one obstacle")
        cbf = CBFsingleobs(config, obstacles[0])
    elif cbf_type == 'multiple':
        cbf = CBFmultipleobs(config, obstacles, **kwargs)
    else:
        raise ValueError(f"Unknown CBF type: {cbf_type}")

    return CBFFilter(cbf, system, alpha)