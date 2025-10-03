#!/usr/bin/env python3
"""
NCBF Visualization Tool - Command Line Interface

A reusable command-line tool for loading and visualizing trained Neural CBF models.
This tool provides flexible visualization options and can work with any trained
NCBF checkpoint and configuration.

Usage:
    python ncbf_visualization_tool.py --checkpoint <checkpoint_file> --config <config_file>

    # With custom output and options
    python ncbf_visualization_tool.py --checkpoint best_model.pt --config training_config.json \
        --output my_visualization.png --resolution 100 --theta 0.5
"""

import argparse
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# Add necessary paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ncbf.models.ncbf import NCBF
from ncbf.configs.ncbf_config import NCBFConfig
from ncbf.maps.map_manager import load_map, NCBFMap


class NCBFVisualizationTool:
    """A reusable tool for loading and visualizing NCBF models."""

    def __init__(self, checkpoint_path, config_path, device='auto', map_path=None, safety_radius=None):
        """
        Initialize the visualization tool.

        Args:
            checkpoint_path: Path to the model checkpoint (.pt file)
            config_path: Path to the training configuration (.json file)
            device: Device to use ('auto', 'cpu', 'cuda:0', etc.)
            map_path: Optional path to map JSON file
            safety_radius: Optional safety radius for visualization (in meters)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.map_path = Path(map_path) if map_path else None
        self.safety_radius = safety_radius
        self.device = self._get_device(device)
        self.ncbf = None
        self.config = None
        self.map_data = None

    def _get_device(self, device):
        """Get the appropriate device."""
        if device == 'auto':
            return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    def load_model(self):
        """Load the NCBF model with configuration."""
        print(f"Loading NCBF model...")
        print(f"  Checkpoint: {self.checkpoint_path}")
        print(f"  Config: {self.config_path}")
        print(f"  Device: {self.device}")

        # Load configuration
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config_dict = json.load(f)
        self.config = NCBFConfig(**config_dict)

        print(f"  Model architecture: {self.config.hidden_dims}")

        # Initialize model
        self.ncbf = NCBF(self.config)

        # Load weights
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.ncbf.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.ncbf.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume the checkpoint IS the state dict
            self.ncbf.load_state_dict(checkpoint)

        # Set model to evaluation mode
        self.ncbf.eval()
        self.ncbf.to(self.device)

        print(f"✅ Model loaded successfully")
        print(f"  Total parameters: {sum(p.numel() for p in self.ncbf.parameters()):,}")

        return self.ncbf, self.config

    def load_map(self):
        """Load map data if map_path is provided."""
        if self.map_path is None:
            return None

        if not self.map_path.exists():
            print(f"⚠️  Map file not found: {self.map_path}")
            return None

        try:
            self.map_data = load_map(self.map_path)
            print(f"✅ Map loaded from: {self.map_path}")
            print(f"   Workspace size: {self.map_data.workspace_size}m × {self.map_data.workspace_size}m")
            print(f"   Obstacles: {len(self.map_data)}")

            # Get obstacle info
            info = self.map_data.get_info()
            print(f"   Radius range: {info['min_radius']:.3f} - {info['max_radius']:.3f}m")

            return self.map_data
        except Exception as e:
            print(f"⚠️  Error loading map: {e}")
            return None

    def test_basic_functionality(self):
        """Test basic NCBF functionality."""
        if self.ncbf is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        print("\nTesting basic functionality...")

        # Test single point - ensure it's on the correct device
        test_state = np.array([2.0, 2.0, 0.0])

        # Convert to tensor and move to device
        test_tensor = torch.tensor(test_state, dtype=torch.float32, device=self.device)
        h_val = self.ncbf.h(test_tensor)
        grad_h = self.ncbf.grad_h(test_tensor)

        print(f"Test state: {test_state}")
        print(f"  h(x) = {h_val:.4f}")
        print(f"  ∇h(x) = {grad_h}")
        # Convert CUDA tensor to numpy for norm calculation
        grad_h_numpy = grad_h.cpu().numpy() if hasattr(grad_h, 'cpu') else np.array(grad_h)
        print(f"  ||∇h(x)|| = {np.linalg.norm(grad_h_numpy):.4f}")
        print(f"  Safety: {'SAFE' if h_val >= 0 else 'UNSAFE'}")

        return h_val, grad_h

    def create_contour_visualization(self, x_range=None, y_range=None,
                                   resolution=50, theta=0.0, obstacles=None,
                                   save_path=None, show_plot=True):
        """
        Create contour visualization of the NCBF.

        Args:
            x_range: Tuple of (x_min, x_max), or None to use map boundaries
            y_range: Tuple of (y_min, y_max), or None to use map boundaries
            resolution: Grid resolution
            theta: Fixed orientation angle
            obstacles: List of obstacles [(x, y, radius), ...], or None to use map obstacles
            save_path: Path to save the figure
            show_plot: Whether to show the plot

        Returns:
            fig: Matplotlib figure
            Z: Grid values
        """
        if self.ncbf is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        print(f"\nCreating contour visualization...")

        # Use map boundaries if available and no custom range specified
        if self.map_data is not None:
            workspace_size = self.map_data.workspace_size
            if x_range is None:
                x_range = (0, workspace_size)
            if y_range is None:
                y_range = (0, workspace_size)
            if obstacles is None:
                obstacles = self.map_data.get_obs()

        # Default ranges if no map and no custom ranges
        if x_range is None:
            x_range = (-1, 6)
        if y_range is None:
            y_range = (-1, 6)

        print(f"  Grid: {x_range} x {y_range} with resolution {resolution}")
        print(f"  Orientation: θ = {theta:.2f}")

        # Create grid
        x_vals = np.linspace(x_range[0], x_range[1], resolution)
        y_vals = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x_vals, y_vals)

        # Create state vectors [x, y, theta] for each grid point
        states = np.zeros((X.size, 3))
        states[:, 0] = X.flatten()
        states[:, 1] = Y.flatten()
        states[:, 2] = theta

        # Evaluate NCBF
        print("  Evaluating NCBF on grid...")
        Z = np.zeros(X.size)

        # For gradient visualization, use a coarser grid to avoid overcrowding
        quiver_resolution = max(10, resolution // 3)  # Coarser grid for quivers
        x_quiver = np.linspace(x_range[0], x_range[1], quiver_resolution)
        y_quiver = np.linspace(y_range[0], y_range[1], quiver_resolution)
        X_quiver, Y_quiver = np.meshgrid(x_quiver, y_quiver)

        # Arrays for gradient components (only x,y components for 2D quiver)
        grad_x = np.zeros((quiver_resolution, quiver_resolution))
        grad_y = np.zeros((quiver_resolution, quiver_resolution))
        grad_magnitude = np.zeros((quiver_resolution, quiver_resolution))

        # Batch evaluation for efficiency
        batch_size = min(1000, len(states))
        for i in range(0, len(states), batch_size):
            batch = states[i:i+batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                batch_h = self.ncbf(batch_tensor)
                # Handle different output shapes - squeeze if needed
                if batch_h.dim() > 1 and batch_h.shape[1] == 1:
                    batch_h = batch_h.squeeze(1)
            Z[i:i+batch_size] = batch_h.cpu().numpy()

        Z = Z.reshape(X.shape)

        # Compute gradients for flow map visualization
        print("  Computing gradient field for flow map...")
        quiver_states = np.zeros((X_quiver.size, 3))
        quiver_states[:, 0] = X_quiver.flatten()
        quiver_states[:, 1] = Y_quiver.flatten()
        quiver_states[:, 2] = theta

        # Compute gradients at quiver points
        for i in range(0, len(quiver_states), batch_size):
            batch = quiver_states[i:i+batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32, device=self.device)

            # Compute gradients using grad_h method
            batch_grads = []
            for state in batch:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                grad = self.ncbf.grad_h(state_tensor)
                # Take only x and y components for 2D quiver, convert to numpy
                grad_numpy = grad.cpu().numpy() if hasattr(grad, 'cpu') else np.array(grad)
                batch_grads.append([grad_numpy[0], grad_numpy[1]])

            batch_grads = np.array(batch_grads)

            # Store gradient components
            start_idx = i
            end_idx = min(i + batch_size, len(quiver_states))

            for j in range(start_idx, end_idx):
                idx = j - start_idx
                # Convert flat index to 2D indices
                row = j // quiver_resolution
                col = j % quiver_resolution
                grad_x[row, col] = batch_grads[idx, 0]
                grad_y[row, col] = batch_grads[idx, 1]
                grad_magnitude[row, col] = np.sqrt(batch_grads[idx, 0]**2 + batch_grads[idx, 1]**2)

        # Gradient arrays are already in correct 2D shape from direct assignment

        # Create figure with 3 subplots - ensure equal sizing with GridSpec for precise control
        fig = plt.figure(figsize=(12, 4))
        fig.suptitle(f'Neural CBF Visualization (θ = {theta:.2f})', fontsize=12)

        # Use GridSpec to ensure equal widths for all subplots
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

        # Plot 1: Contour plot
        ax = axes[0]
        contour = ax.contour(X, Y, Z, levels=15, cmap='RdYlBu', linewidths=0.8)
        ax.contourf(X, Y, Z, levels=15, cmap='RdYlBu', alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=7, fmt='%.2f')

        # Note: Obstacles are NOT drawn in the left figure (contours only)
        # This allows clear visualization of the learned CBF structure

        ax.set_title('NCBF h(x) Contours', fontsize=10)
        ax.set_xlabel('X [m]', fontsize=9)
        ax.set_ylabel('Y [m]', fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        # Ensure consistent axis limits with other subplots
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])

        # Plot 2: Safety boundary and comparison with real obstacles
        ax = axes[1]

        # Safety boundary (h=0) from NCBF
        contour_zero = ax.contour(X, Y, Z, levels=[0], colors='blue', linewidths=2, linestyles='-')

        # Add some contour levels to show structure
        levels = [-1, -0.5, 0, 0.5, 1]
        contour2 = ax.contour(X, Y, Z, levels=levels, colors='gray', linewidths=0.8, alpha=0.7)
        ax.clabel(contour2, inline=True, fontsize=7, fmt='%.1f')

        # Add obstacles and safety regions if provided
        if obstacles:
            for i, obs in enumerate(obstacles):
                x, y, radius = obs

                # Draw safety region first (if safety_radius is provided) - larger, lighter circle
                if self.safety_radius is not None and self.safety_radius > 0:
                    safety_radius_total = radius + self.safety_radius
                    safety_circle = plt.Circle((x, y), safety_radius_total,
                                             facecolor='orange', alpha=0.3, fill=True,
                                             edgecolor='orange', linewidth=1.5, linestyle='--',
                                             label='Safety Region' if i == 0 else "")
                    ax.add_patch(safety_circle)

                # Draw obstacle on top (smaller, darker circle)
                obstacle_circle = plt.Circle((x, y), radius,
                                           facecolor='red', alpha=0.8, fill=True,
                                           edgecolor='darkred', linewidth=1.5,
                                           label='Obstacle' if i == 0 else "")
                ax.add_patch(obstacle_circle)

                # Mark center
                ax.plot(x, y, 'ko', markersize=3, alpha=0.9)

        ax.set_title('NCBF Safety Boundary vs Real Obstacles', fontsize=10)
        ax.set_xlabel('X [m]', fontsize=9)
        ax.set_ylabel('Y [m]', fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Add legend for the second subplot
        if obstacles:
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], color='blue', linewidth=3, linestyle='-', label='Learned h=0'),
                Patch(facecolor='red', alpha=0.8, label='Obstacles'),
            ]

            # Add safety region to legend if safety_radius is provided
            if self.safety_radius is not None and self.safety_radius > 0:
                legend_elements.insert(1, Patch(facecolor='orange', alpha=0.3,
                                              label=f'Safety Region (+{self.safety_radius}m)'))

            ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

        # Plot 3: Gradient flow map visualization
        ax = axes[2]

        # Create flow map (streamplot) showing gradient field
        # Use the gradient components directly for natural flow visualization
        U = grad_x  # x-components of gradient
        V = grad_y  # y-components of gradient

        # Create streamplot - this shows the flow lines of the gradient field
        # The density parameter controls how many streamlines are drawn
        streamplot = ax.streamplot(X_quiver, Y_quiver, U, V,
                                  color=grad_magnitude,
                                  cmap='plasma',
                                  density=2.0,
                                  linewidth=1.0,
                                  arrowsize=1.5,
                                  arrowstyle='->',
                                  minlength=0.1)

        # Add colorbar for gradient magnitude - use external axis to avoid size distortion
        from matplotlib.colorbar import Colorbar
        # Create a dedicated axis for colorbar that doesn't affect main subplot size
        cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(streamplot.lines, cax=cbar_ax)
        cbar.set_label('||∇h||', fontsize=8, labelpad=2)

        # Add obstacles to gradient plot for reference (without safety regions to avoid clutter)
        if obstacles:
            for obs in obstacles:
                x, y, radius = obs
                # Draw obstacle outline for reference
                obstacle_circle = plt.Circle((x, y), radius, fill=False,
                                           edgecolor='red', linewidth=1.5, alpha=0.8)
                ax.add_patch(obstacle_circle)

        ax.set_title('NCBF Gradient Flow Map ∇h(x)', fontsize=10)
        ax.set_xlabel('X [m]', fontsize=9)
        ax.set_ylabel('Y [m]', fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Set consistent axis limits with other subplots
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])

        # Add text info about gradient flow map
        grad_info = f"Streamlines: ∇h flow\nColor: ||∇h|| magnitude\nDensity: {2.0}"
        ax.text(0.02, 0.98, grad_info, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Contour plot saved to: {save_path}")

        if show_plot:
            plt.show()

        return fig, Z, grad_magnitude

    def print_statistics(self, Z, grad_magnitude=None):
        """Print statistics about the NCBF values and gradient field."""
        print(f"\n📊 NCBF Statistics:")
        print(f"   h(x) range: [{Z.min():.3f}, {Z.max():.3f}]")
        print(f"   Mean h(x): {Z.mean():.3f}")
        print(f"   Std h(x): {Z.std():.3f}")
        print(f"   Safe points (h>0): {np.sum(Z > 0):,} / {Z.size:,}")
        print(f"   Unsafe points (h<0): {np.sum(Z < 0):,} / {Z.size:,}")
        print(f"   Boundary points (|h|<0.1): {np.sum(np.abs(Z) < 0.1):,} / {Z.size:,}")

        # Add gradient statistics if available
        if grad_magnitude is not None:
            print(f"\n📊 Gradient Statistics:")
            print(f"   ||∇h|| range: [{grad_magnitude.min():.3f}, {grad_magnitude.max():.3f}]")
            print(f"   Mean ||∇h||: {grad_magnitude.mean():.3f}")
            print(f"   Std ||∇h||: {grad_magnitude.std():.3f}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="NCBF Visualization Tool - Load and visualize trained Neural CBF models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python ncbf_visualization_tool.py --checkpoint best_model.pt --config training_config.json

    # With map file for obstacle visualization and correct boundaries
    python ncbf_visualization_tool.py --checkpoint best_model.pt --config training_config.json \
        --map /home/chengrui/wk/NCBFquickDemo/work/ncbf/maps/map_files/map1/map1.json

    # With safety radius visualization
    python ncbf_visualization_tool.py --checkpoint best_model.pt --config training_config.json \
        --map /home/chengrui/wk/NCBFquickDemo/work/ncbf/maps/map_files/map1/map1.json --safety-radius 0.3

    # Custom output path
    python ncbf_visualization_tool.py --checkpoint best_model.pt --config training_config.json \
        --output my_visualization.png

    # Custom visualization parameters
    python ncbf_visualization_tool.py --checkpoint best_model.pt --config training_config.json \
        --resolution 100 --theta 0.5 --xmin -2 --xmax 8 --ymin -2 --ymax 8

    # No display (save only)
    python ncbf_visualization_tool.py --checkpoint best_model.pt --config training_config.json \
        --output result.png --no-display
        """
    )

    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint file (.pt)'
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training config file (.json)'
    )

    # Optional arguments
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for visualization (default: same directory as checkpoint)'
    )

    parser.add_argument(
        '--resolution',
        type=int,
        default=50,
        help='Grid resolution (default: 50)'
    )

    parser.add_argument(
        '--theta',
        type=float,
        default=0.0,
        help='Fixed orientation angle in radians (default: 0.0)'
    )

    parser.add_argument(
        '--xmin',
        type=float,
        default=-1.0,
        help='Minimum x value (default: -1.0)'
    )

    parser.add_argument(
        '--xmax',
        type=float,
        default=6.0,
        help='Maximum x value (default: 6.0)'
    )

    parser.add_argument(
        '--ymin',
        type=float,
        default=-1.0,
        help='Minimum y value (default: -1.0)'
    )

    parser.add_argument(
        '--ymax',
        type=float,
        default=6.0,
        help='Maximum y value (default: 6.0)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto, cpu, cuda:0, etc.) (default: auto)'
    )

    parser.add_argument(
        '--map',
        type=str,
        default=None,
        help='Path to map JSON file (optional, for obstacle visualization and correct boundaries)'
    )

    parser.add_argument(
        '--safety-radius',
        type=float,
        default=None,
        help='Safety radius for visualization (in meters, optional, for showing safety regions around obstacles)'
    )

    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not display plot (save only)'
    )

    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='Do not print statistics'
    )

    args = parser.parse_args()

    print("🚀 NCBF Visualization Tool")
    print("="*60)

    try:
        # Create visualization tool
        tool = NCBFVisualizationTool(args.checkpoint, args.config, args.device, args.map, args.safety_radius)

        # Load model
        ncbf, config = tool.load_model()

        # Load map if provided
        if args.map:
            tool.load_map()

        # Test basic functionality
        tool.test_basic_functionality()

        # Set output path if not provided
        if args.output is None:
            checkpoint_dir = Path(args.checkpoint).parent
            output_path = checkpoint_dir / 'ncbf_visualization.png'
        else:
            output_path = Path(args.output)

        # Create visualization
        # Use custom ranges if provided, otherwise let the tool decide (map boundaries or defaults)
        x_range = (args.xmin, args.xmax) if args.xmin != -1.0 or args.xmax != 6.0 else None
        y_range = (args.ymin, args.ymax) if args.ymin != -1.0 or args.ymax != 6.0 else None

        fig, Z, grad_magnitude = tool.create_contour_visualization(
            x_range=x_range,
            y_range=y_range,
            resolution=args.resolution,
            theta=args.theta,
            save_path=output_path,
            show_plot=not args.no_display
        )

        # Print statistics
        if not args.no_stats:
            tool.print_statistics(Z, grad_magnitude)

        print(f"\n✅ Visualization completed successfully!")
        print(f"   Output saved to: {output_path}")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())