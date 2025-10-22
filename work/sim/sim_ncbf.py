#!/usr/bin/env python3
"""
NCBF Control Simulation for Unicycle Navigation

This script simulates unicycle navigation using trained Neural Control Barrier Function
for safety filtering. It demonstrates the integration of learned safety certificates
with control systems.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import json
import torch
import argparse

# Add necessary paths - be more specific about the paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our implemented components
from models.unicycle_model import UnicycleModel
from configs.unicycle_config import UnicycleConfig
from safe_control.cbf_filter import CBFFilter
from ncbf.models.ncbf import NCBF
from ncbf.configs.ncbf_config import NCBFConfig
from ncbf.maps.map_manager import load_map, NCBFMap


class NCBFSimulation:
    """
    Simulation class for unicycle navigation with NCBF safety filtering.
    """

    def __init__(self,
                 map_file: str = None,
                 ncbf_weights: str = None,
                 ncbf_config: str = None):
        """
        Initialize NCBF simulation with trained model and map.

        Args:
            map_file: Path to map JSON file
            ncbf_weights: Path to trained NCBF weights
            ncbf_config: Path to NCBF configuration
        """
        print("🚀 Initializing NCBF Simulation")
        print("=" * 60)

        # Set default paths if not provided
        if map_file is None:
            map_file = "/home/chengrui/wk/NCBFquickDemo/work/ncbf/maps/map_files/map1/map1.json"
        if ncbf_weights is None:
            ncbf_weights = "/home/chengrui/wk/NCBFquickDemo/work/ncbf/weights/ad_enhanced_training/best_model.pt"
        if ncbf_config is None:
            ncbf_config = "/home/chengrui/wk/NCBFquickDemo/work/ncbf/weights/ad_enhanced_training/training_config.json"

        # Load map
        print(f"Loading map from: {map_file}")
        self.map_data = load_map(map_file)
        print(f"✅ Map loaded: {len(self.map_data)} obstacles")

        # Load NCBF configuration
        print(f"Loading NCBF config from: {ncbf_config}")
        with open(ncbf_config, 'r') as f:
            config_dict = json.load(f)
        self.ncbf_config = NCBFConfig(**config_dict)
        print(f"✅ NCBF config loaded: {self.ncbf_config.hidden_dims}")

        # Initialize NCBF model
        print(f"Loading NCBF weights from: {ncbf_weights}")
        self.ncbf = NCBF(self.ncbf_config)
        checkpoint = torch.load(ncbf_weights, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.ncbf.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.ncbf.load_state_dict(checkpoint['state_dict'])
        else:
            self.ncbf.load_state_dict(checkpoint)
        self.ncbf.eval()
        print(f"✅ NCBF loaded: {sum(p.numel() for p in self.ncbf.parameters()):,} parameters")

        print("✅ NCBF Simulation initialized successfully!")

    def simulate(self,
                 initial_state: np.ndarray,
                 goal: np.ndarray,
                 unicycle_config: Optional[UnicycleConfig] = None,
                 max_sim_time: float = 30.0,
                 verbose: bool = True) -> dict:
        """
        Run simulation with NCBF safety filtering.

        Args:
            initial_state: Initial state [px, py, theta]
            goal: Goal position [px_goal, py_goal]
            unicycle_config: Optional unicycle config (uses defaults if None)
            max_sim_time: Maximum simulation time in seconds
            verbose: Print simulation progress

        Returns:
            Simulation results dictionary with trajectory, controls, and metrics
        """
        if verbose:
            print(f"\n🎯 Starting NCBF Simulation")
            print(f"Initial state: {initial_state}")
            print(f"Goal: {goal}")

        # Create unicycle config if not provided
        if unicycle_config is None:
            unicycle_config = UnicycleConfig()

        # Create unicycle model with NCBF integration
        print("Creating unicycle model with NCBF safety filter...")
        unicycle_model = UnicycleModel(
            config=unicycle_config,
            initial_state=initial_state,
            target=goal
        )

        # Initialize CBF safety filter with trained NCBF
        safety_filter = CBFFilter(
            cbf_function=self.ncbf,
            control_affine_system=unicycle_model,
            alpha=unicycle_config.cbf_alpha
        )

        print("✅ Unicycle model and safety filter initialized")
        print(f"Time step: {unicycle_config.dt}s")

        # Initialize tracking variables
        simulation_time = 0.0
        goal_reached = False
        collision = False
        min_distances = []
        h_values = []

        if verbose:
            print(f"Max simulation time: {max_sim_time}s")

        # Simulation loop
        step = 0
        while simulation_time < max_sim_time:
            # Get current state
            current_state = unicycle_model.state.copy()

            # Compute nominal control (PD controller)
            if hasattr(unicycle_model, 'pd_control_proportional'):
                nominal_control = unicycle_model.pd_control_proportional()
            else:
                nominal_control = unicycle_model.pd_control_basic()

            # Apply safety filtering using NCBF
            safe_control = safety_filter.compute_safe_control(current_state, nominal_control)

            # Update state using unicycle dynamics (handles constraints automatically)
            unicycle_model.update_state(safe_control)

            # Compute metrics
            new_state = unicycle_model.state
            min_dist = self._compute_min_distance_to_obstacles(new_state, unicycle_config)
            h_val = float(self.ncbf.h(new_state))

            # Track metrics
            min_distances.append(min_dist)
            h_values.append(h_val)

            # Check conditions
            goal_distance = np.linalg.norm(new_state[:2] - goal)
            goal_reached = goal_distance < 0.2  # 20cm tolerance
            collision = min_dist < 0  # Negative distance means collision

            if verbose and step % 50 == 0:
                print(f"Step {step}: Time={simulation_time:.2f}s, Pos=({new_state[0]:.2f}, {new_state[1]:.2f}), "
                      f"Goal dist={goal_distance:.2f}, Min obs dist={min_dist:.2f}, h={h_val:.3f}")

            # Check termination
            if goal_reached or collision:
                break

            # Update simulation time
            simulation_time += unicycle_config.dt
            step += 1

        # Compile results
        results = {
            'trajectory': np.array(unicycle_model.history),
            'controls': np.array(unicycle_model.control_history),
            'distances_to_obstacles': np.array(min_distances),
            'h_values': np.array(h_values),
            'goal_reached': goal_reached,
            'collision': collision,
            'simulation_time': simulation_time,
            'num_steps': step,
            'unicycle_model': unicycle_model,
            'map_data': self.map_data,
            'safety_filter': safety_filter,
            'unicycle_config': unicycle_config
        }

        if verbose:
            print(f"\n📊 Simulation Complete")
            print(f"Goal reached: {goal_reached}")
            print(f"Collision: {collision}")
            print(f"Total time: {simulation_time:.2f}s")
            print(f"Steps: {step}")
            print(f"Min obstacle distance: {min(min_distances):.3f}")
            print(f"Final h value: {h_values[-1]:.3f}")

        return results

    def _compute_min_distance_to_obstacles(self, state: np.ndarray, config: UnicycleConfig) -> float:
        """
        Compute minimum distance from robot center to obstacle boundaries.

        Args:
            state: Robot state [px, py, theta]
            config: Unicycle configuration with robot radius and safety parameters

        Returns:
            Minimum distance from robot center to obstacle boundaries
        """
        robot_pos = state[:2]
        min_distance = float('inf')

        for obstacle in self.map_data.obstacles:
            obs_x, obs_y, obs_radius = obstacle
            # Distance from robot center to obstacle center minus obstacle radius
            center_distance = np.linalg.norm(robot_pos - np.array([obs_x, obs_y]))
            # Distance from robot center to obstacle boundary
            boundary_distance = center_distance - obs_radius
            min_distance = min(min_distance, boundary_distance)

        return min_distance

    def visualize_paper_ready(self, results: dict, save_path: Optional[str] = None, show_plot: bool = True):
        """
        Create paper-ready visualization focusing on trajectory and safety analysis.

        Args:
            results: Simulation results from simulate()
            save_path: Path to save figure (optional)
            show_plot: Whether to show the plot
        """
        print("\n🎨 Creating paper-ready visualization...")

        trajectory = results['trajectory']
        controls = results['controls']
        distances = results['distances_to_obstacles']
        h_values = results['h_values']
        map_data = results['map_data']
        unicycle_model = results['unicycle_model']
        unicycle_config = results['unicycle_config']

        print(f"Debug: trajectory shape {trajectory.shape}")
        print(f"Debug: controls shape {controls.shape}")
        print(f"Debug: simulation_time {results['simulation_time']}")

        # Create figure with two subplots of equal size
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=False)

        # 1. Trajectory plot with speed reflection
        self._plot_trajectory_paper_ready(ax1, trajectory, controls, map_data, unicycle_config, unicycle_model)

        # 2. Combined distance and CBF analysis
        sim_time = results['simulation_time']
        self._plot_combined_safety_analysis(ax2, distances, h_values, sim_time, unicycle_config)

        # Clean layout for paper
        plt.tight_layout()

        if save_path:
            # Use PNG format instead of PDF
            if not save_path.endswith('.png'):
                save_path = save_path.replace('.pdf', '.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
            print(f"✅ Paper-ready visualization saved to: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def _plot_trajectory_paper_ready(self, ax, trajectory, controls, map_data, config, unicycle_model):
        """
        Plot trajectory with speed reflection (line width/color based on speed).
        """
        # Calculate speed (magnitude of velocity) for each point
        speeds = np.sqrt(controls[:, 0]**2 + controls[:, 1]**2)

        # Normalize speeds for color mapping
        norm_speeds = (speeds - speeds.min()) / (speeds.max() - speeds.min() + 1e-8)

        # Plot trajectory with color based on speed
        for i in range(len(trajectory) - 1):
            ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1],
                   color=plt.cm.viridis(norm_speeds[i]),
                   linewidth=1.5 + 2 * norm_speeds[i],  # Line width reflects speed
                   alpha=0.8)

        # Plot obstacles - using the same style as original visualization
        for obs in map_data.obstacles:
            obs_x, obs_y, obs_radius = obs
            # Safety region (larger circle)
            safety_circle = plt.Circle((obs_x, obs_y), obs_radius + config.safety_radius,
                                     facecolor='orange', alpha=0.3, fill=True,
                                     edgecolor='orange', linewidth=1, linestyle='--')
            ax.add_patch(safety_circle)
            # Obstacle (smaller circle)
            obstacle_circle = plt.Circle((obs_x, obs_y), obs_radius,
                                       facecolor='red', alpha=0.8, fill=True,
                                       edgecolor='darkred', linewidth=2)
            ax.add_patch(obstacle_circle)
            ax.plot(obs_x, obs_y, 'ko', markersize=3, alpha=0.9)

        # Plot start and end points
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=6, label='Start')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=6, label='End')

        # Add goal with asterisk
        ax.plot(unicycle_model.target[0], unicycle_model.target[1], 'g*', markersize=12, label='Goal')

        # Add colorbar for speed
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=speeds.min(), vmax=speeds.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.05)
        cbar.set_label('Control Norm')

        ax.set_xlim(0, map_data.workspace_size)
        ax.set_ylim(0, map_data.workspace_size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_xlabel('X [m]', fontsize=12)
        ax.set_ylabel('Y [m]', fontsize=12)
        ax.set_title('Robot Trajectory with Speed Visualization', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', framealpha=0.9)

    def _plot_combined_safety_analysis(self, ax, distances, h_values, total_time, config):
        """
        Plot combined distance to obstacles (minus safety distance) and NCBF values.
        Uses natural matplotlib scaling without forced alignment for better readability.
        """
        time_array = np.linspace(0, total_time, len(distances))

        # Calculate effective distance (distance - safety_distance)
        effective_distances = distances - config.safety_radius

        # Create twin axis for the second y-axis
        ax2 = ax.twinx()

        # Plot effective distance on left axis
        line1 = ax.plot(time_array, effective_distances, 'b-', linewidth=2,
                       label='Distance to obstacles (minus safety radius)')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Safety boundary (distance=0)')

        # Plot NCBF values on right axis
        line2 = ax2.plot(time_array, h_values, 'g-', linewidth=2,
                        label='NCBF value h(x)')
        ax2.axhline(y=0, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Safety boundary (h=0)')

        # Let matplotlib handle the scaling naturally (no forced alignment)
        # Just ensure reasonable margins for readability
        ax.margins(y=0.1)
        ax2.margins(y=0.1)

        # Styling
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('Distance [m]', fontsize=12, color='b')
        ax2.set_ylabel('NCBF value h(x)', fontsize=12, color='g')
        ax.set_title('Safety Analysis: Distance and NCBF Values', fontsize=14, fontweight='bold')

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right', framealpha=0.9)

        # Grid and limits
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_xlim(0, total_time)

        # Color coordination
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='g')

    def visualize(self, results: dict, save_path: Optional[str] = None, show_plot: bool = True):
        """
        Create comprehensive visualization of simulation results.

        Args:
            results: Simulation results from simulate()
            save_path: Path to save figure (optional)
            show_plot: Whether to show the plot
        """
        print("\n🎨 Creating visualization...")

        trajectory = results['trajectory']
        controls = results['controls']
        distances = results['distances_to_obstacles']
        h_values = results['h_values']
        map_data = results['map_data']
        unicycle_model = results['unicycle_model']
        unicycle_config = results['unicycle_config']

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))

        # 1. Trajectory plot with obstacles
        ax1 = plt.subplot(2, 3, 1)
        self._plot_trajectory(ax1, trajectory, map_data, unicycle_model, unicycle_config)

        # 2. Distance to obstacles over time
        ax2 = plt.subplot(2, 3, 2)
        self._plot_distances(ax2, distances, results['simulation_time'], unicycle_config)

        # 3. NCBF values over time
        ax3 = plt.subplot(2, 3, 3)
        self._plot_h_values(ax3, h_values, results['simulation_time'])

        # 4. Control inputs over time
        ax4 = plt.subplot(2, 3, 4)
        self._plot_controls(ax4, controls, results['simulation_time'], unicycle_config)

        # 5. Safety analysis
        ax5 = plt.subplot(2, 3, 5)
        self._plot_safety_analysis(ax5, h_values, distances, unicycle_config)

        # 6. Final state analysis
        ax6 = plt.subplot(2, 3, 6)
        self._plot_final_state(ax6, trajectory, h_values, distances, unicycle_config)

        # Overall title
        goal_status = "GOAL REACHED" if results['goal_reached'] else "GOAL NOT REACHED"
        safety_status = "NO COLLISION" if not results['collision'] else "COLLISION"
        title = f"NCBF Control Simulation - {goal_status} - {safety_status}"
        fig.suptitle(title, fontsize=16, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def _plot_trajectory(self, ax, trajectory, map_data, unicycle_model, config):
        """Plot trajectory with obstacles and safety regions."""
        # Plot obstacles
        for obs in map_data.obstacles:
            obs_x, obs_y, obs_radius = obs
            # Safety region (larger circle)
            safety_circle = plt.Circle((obs_x, obs_y), obs_radius + config.safety_radius,
                                     facecolor='orange', alpha=0.3, fill=True,
                                     edgecolor='orange', linewidth=1, linestyle='--')
            ax.add_patch(safety_circle)
            # Obstacle (smaller circle)
            obstacle_circle = plt.Circle((obs_x, obs_y), obs_radius,
                                       facecolor='red', alpha=0.8, fill=True,
                                       edgecolor='darkred', linewidth=2)
            ax.add_patch(obstacle_circle)
            ax.plot(obs_x, obs_y, 'ko', markersize=3, alpha=0.9)

        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory')
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8, label='Start')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'rs', markersize=8, label='End')

        # Plot target
        ax.plot(unicycle_model.target[0], unicycle_model.target[1], 'g*', markersize=12, label='Goal')

        # Plot orientation arrows every few steps
        step_size = max(1, len(trajectory) // 20)
        for i in range(0, len(trajectory), step_size):
            x, y, theta = trajectory[i]
            dx = 0.2 * np.cos(theta)
            dy = 0.2 * np.sin(theta)
            ax.arrow(x, y, dx, dy, head_width=0.05, head_length=0.05,
                    fc='blue', ec='blue', alpha=0.7)

        ax.set_xlim(0, map_data.workspace_size)
        ax.set_ylim(0, map_data.workspace_size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Trajectory with Obstacles')
        ax.legend()

    def _plot_distances(self, ax, distances, total_time, config):
        """Plot minimum distance to obstacles over time."""
        time_array = np.linspace(0, total_time, len(distances))

        ax.plot(time_array, distances, 'b-', linewidth=2, label='Min distance to obstacles')
        ax.axhline(y=config.safety_radius, color='r', linestyle='--',
                  label=f'Safety radius ({config.safety_radius}m)')
        ax.axhline(y=0, color='red', linestyle='-', alpha=0.5, label='Collision threshold')

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Distance [m]')
        ax.set_title('Minimum Distance to Obstacles')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_h_values(self, ax, h_values, total_time):
        """Plot NCBF values over time."""
        time_array = np.linspace(0, total_time, len(h_values))

        ax.plot(time_array, h_values, 'g-', linewidth=2, label='h(x) value')
        ax.axhline(y=0, color='black', linestyle='--', label='Safety boundary (h=0)')
        ax.axhline(y=0.1, color='orange', linestyle=':', alpha=0.7, label='Safety margin')
        ax.fill_between(time_array, h_values, 0, where=(h_values < 0),
                       color='red', alpha=0.3, label='Unsafe region')

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('h(x)')
        ax.set_title('NCBF Safety Values')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_controls(self, ax, controls, total_time, config):
        """Plot control inputs over time."""
        time_array = np.linspace(0, total_time, len(controls))

        ax.plot(time_array, controls[:, 0], 'r-', linewidth=2, label='Linear velocity (v)')
        ax.plot(time_array, controls[:, 1], 'b-', linewidth=2, label='Angular velocity (ω)')
        ax.axhline(y=config.max_control_norm, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=-config.max_control_norm, color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Control Input')
        ax.set_title('Control Inputs')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_safety_analysis(self, ax, h_values, distances, config):
        """Plot safety analysis scatter plot."""
        ax.scatter(distances, h_values, c=h_values, cmap='RdYlGn', alpha=0.7, s=20)
        ax.axhline(y=0, color='black', linestyle='--', label='Safety boundary')
        ax.axvline(x=config.safety_radius, color='orange', linestyle='--',
                  label='Safety radius')

        ax.set_xlabel('Distance to Obstacles [m]')
        ax.set_ylabel('h(x) Value')
        ax.set_title('Safety Analysis: h(x) vs Distance')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_final_state(self, ax, trajectory, h_values, distances, config):
        """Plot final state with detailed information."""
        final_state = trajectory[-1]
        final_distance = distances[-1]
        final_h = h_values[-1]

        # Simple text-based info
        info_text = f"""Final State Analysis:
Position: ({final_state[0]:.2f}, {final_state[1]:.2f})
Orientation: {np.degrees(final_state[2]):.1f}°
Distance to obstacles: {final_distance:.3f}m
NCBF value: {final_h:.3f}
Safety status: {'SAFE' if final_h >= 0 else 'UNSAFE'}
Robot radius: {config.robot_radius}m
Safety radius: {config.safety_radius}m
"""

        ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Final State Analysis')


def main():
    """Main function for running NCBF simulation."""
    parser = argparse.ArgumentParser(description="NCBF Control Simulation for Unicycle Navigation")
    parser.add_argument('--initial-x', type=float, default=0.5, help='Initial x position')
    parser.add_argument('--initial-y', type=float, default=0.5, help='Initial y position')
    parser.add_argument('--initial-theta', type=float, default=0.0, help='Initial orientation (radians)')
    parser.add_argument('--goal-x', type=float, default=7.0, help='Goal x position')
    parser.add_argument('--goal-y', type=float, default=7.0, help='Goal y position')
    parser.add_argument('--save-path', type=str, default=None, help='Path to save visualization')
    parser.add_argument('--no-display', action='store_true', help='Do not show plot')
    parser.add_argument('--paper-ready', action='store_true', help='Use paper-ready visualization format (2 subfigures: trajectory with speed + combined safety analysis)')

    args = parser.parse_args()

    # Create simulation
    sim = NCBFSimulation()

    # Set initial state and goal
    initial_state = np.array([args.initial_x, args.initial_y, args.initial_theta])
    goal = np.array([args.goal_x, args.goal_y])

    # Run simulation
    results = sim.simulate(initial_state, goal, verbose=True)

    # Visualize results
    save_path = args.save_path if args.save_path else '/home/chengrui/wk/NCBFquickDemo/work/sim/results/ncbf_simulation_result.png'

    if args.paper_ready:
        paper_save_path = save_path.replace('.png', '_paper.png') if save_path.endswith('.png') else save_path.replace('.pdf', '_paper.png')
        sim.visualize_paper_ready(results, save_path=paper_save_path, show_plot=not args.no_display)
    else:
        sim.visualize(results, save_path=save_path, show_plot=not args.no_display)

    print(f"\n✅ NCBF simulation completed!")
    if args.paper_ready:
        print(f"Paper-ready visualization saved to: {paper_save_path}")
    else:
        print(f"Results visualization saved to: {save_path}")


if __name__ == "__main__":
    main()