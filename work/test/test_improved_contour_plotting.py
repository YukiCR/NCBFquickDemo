#!/usr/bin/env python3
"""
Test improved contour plotting with intensive_training_2 weights.
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add necessary paths
sys.path.insert(0, '/home/chengrui/wk/NCBFquickDemo/work')
sys.path.insert(0, '/home/chengrui/wk/NCBFquickDemo/work/ncbf')
sys.path.insert(0, '/home/chengrui/wk/NCBFquickDemo/work/ncbf/training')
sys.path.insert(0, '/home/chengrui/wk/NCBFquickDemo/work/ncbf/models')
sys.path.insert(0, '/home/chengrui/wk/NCBFquickDemo/work/ncbf/configs')
sys.path.insert(0, '/home/chengrui/wk/NCBFquickDemo/work/ncbf/maps')

# Add the parent directory to sys.path to handle the ncbf package imports
sys.path.insert(0, '/home/chengrui/wk/NCBFquickDemo/work/ncbf')

def test_improved_contour_plotting():
    """Test the improved contour plotting functionality."""
    print("🎨 Testing Improved Contour Plotting")
    print("="*60)

    try:
        # Import required modules
        import h5py

        # Import NCBF components using direct imports
        sys.path.insert(0, '/home/chengrui/wk/NCBFquickDemo/work')

        # Import the NCBF model directly
        from ncbf.models.ncbf import NCBF
        from ncbf.configs.ncbf_config import create_large_ncbf_config
        from ncbf.maps.map_manager import load_map

        # Load the best model from intensive_training_2
        model_path = '/home/chengrui/wk/NCBFquickDemo/work/ncbf/weights/intensive_training_2/best_model.pt'
        config = create_large_ncbf_config()

        print(f"✅ Loading model from: {model_path}")
        print(f"   Architecture: {config.hidden_dims}")
        print(f"   Activation: {config.activation}")

        # Create and load model
        model = NCBF(config)
        model.load_model(model_path)
        print(f"✅ Model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters")

        # Load map data for obstacles
        map_path = '/home/chengrui/wk/NCBFquickDemo/work/ncbf/maps/map_files/map1/map1.json'
        ncbf_map = load_map(map_path)
        obstacles = ncbf_map.obstacles
        print(f"✅ Map loaded with {len(obstacles)} obstacles")

        # Load training data for overlay
        data_path = '/home/chengrui/wk/NCBFquickDemo/work/ncbf/maps/map_files/map1/training_data_new.h5'
        with h5py.File(data_path, 'r') as f:
            states = f['data']['states'][:]
            labels = f['data']['labels'][:]
        print(f"✅ Training data loaded: {len(states):,} samples")

        # Create evaluation grid
        resolution = 100
        theta_fixed = 0.0
        x_range = np.linspace(0, 8, resolution)
        y_range = np.linspace(0, 8, resolution)
        X, Y = np.meshgrid(x_range, y_range)

        # Evaluate h(x) on grid
        grid_states = np.column_stack([
            X.ravel(), Y.ravel(),
            np.full(X.size, theta_fixed)
        ])

        # Convert to tensor
        grid_states_tensor = torch.tensor(grid_states, dtype=torch.float32)

        with torch.no_grad():
            h_values = model.h(grid_states_tensor).cpu().numpy().reshape(X.shape)

        print(f"✅ h(x) evaluated on grid: shape {h_values.shape}")

        # Create improved visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Contour plot with zero line - IMPROVED
        contour = axes[0, 0].contour(X, Y, h_values, levels=20, cmap='RdBu_r', alpha=0.8)
        axes[0, 0].clabel(contour, inline=True, fontsize=8)
        zero_contour = axes[0, 0].contour(X, Y, h_values, levels=[0], colors='black', linewidths=3)
        axes[0, 0].set_title('NCBF h(x) Contours with Safety Boundary', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('X Position (m)', fontsize=10)
        axes[0, 0].set_ylabel('Y Position (m)', fontsize=10)
        axes[0, 0].set_aspect('equal')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Enhanced safety boundary with obstacles - IMPROVED
        # Draw obstacles with better visibility
        for obs in obstacles:
            circle = plt.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.4,
                              fill=True, edgecolor='darkred', linewidth=2)
            axes[0, 1].add_patch(circle)
            # Add obstacle centers
            axes[0, 1].plot(obs[0], obs[1], 'ro', markersize=6, alpha=0.8)

        # Enhanced safety boundary visualization
        zero_contour = axes[0, 1].contour(X, Y, h_values, levels=[0],
                                         colors=['black'], linewidths=4, alpha=0.9)

        # Add safety regions
        safe_region = axes[0, 1].contourf(X, Y, h_values, levels=[0, np.max(h_values)],
                                         colors=['lightgreen'], alpha=0.3)
        unsafe_region = axes[0, 1].contourf(X, Y, h_values, levels=[np.min(h_values), 0],
                                           colors=['lightcoral'], alpha=0.3)

        axes[0, 1].set_title('Safety Boundary with Obstacles', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('X Position (m)', fontsize=10)
        axes[0, 1].set_ylabel('Y Position (m)', fontsize=10)
        axes[0, 1].set_aspect('equal')
        axes[0, 1].grid(True, alpha=0.3)

        # Add legend for safety regions
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', alpha=0.3, label='Safe Region (h>0)'),
            Patch(facecolor='lightcoral', alpha=0.3, label='Unsafe Region (h<0)'),
            plt.Line2D([0], [0], color='black', linewidth=4, label='Safety Boundary (h=0)'),
            plt.Line2D([0], [0], marker='o', color='red', markersize=8, alpha=0.8, label='Obstacle Centers')
        ]
        axes[0, 1].legend(handles=legend_elements, loc='upper right', fontsize=9)

        # 3. Gradient magnitude - IMPROVED
        grad_x, grad_y = np.gradient(h_values, x_range[1] - x_range[0], y_range[1] - y_range[0])
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        im = axes[1, 0].imshow(grad_magnitude, extent=[0, 8, 0, 8], origin='lower', cmap='viridis', aspect='equal')
        axes[1, 0].set_title('Gradient Magnitude ||∇h(x)||', fontsize=12, fontweight='bold')
        cbar = plt.colorbar(im, ax=axes[1, 0], shrink=0.8)
        cbar.set_label('Gradient Magnitude', fontsize=9)
        axes[1, 0].set_xlabel('X Position (m)', fontsize=10)
        axes[1, 0].set_ylabel('Y Position (m)', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Training data overlay - IMPROVED
        # Sample training data for visualization
        sample_size = min(1000, len(states))
        indices = np.random.choice(len(states), sample_size, replace=False)

        sample_states = states[indices]
        sample_labels = labels[indices]

        safe_sample_mask = sample_labels.flatten() == 1
        unsafe_sample_mask = sample_labels.flatten() == 0

        axes[1, 1].scatter(sample_states[safe_sample_mask, 0],
                          sample_states[safe_sample_mask, 1],
                          c='blue', s=3, alpha=0.7, label=f'Safe ({np.sum(safe_sample_mask)})',
                          edgecolors='none')
        axes[1, 1].scatter(sample_states[unsafe_sample_mask, 0],
                          sample_states[unsafe_sample_mask, 1],
                          c='red', s=3, alpha=0.7, label=f'Unsafe ({np.sum(unsafe_sample_mask)})',
                          edgecolors='none')

        axes[1, 1].set_title('Training Data Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=9, loc='upper right')
        axes[1, 1].set_xlabel('X Position (m)', fontsize=10)
        axes[1, 1].set_ylabel('Y Position (m)', fontsize=10)
        axes[1, 1].set_aspect('equal')
        axes[1, 1].grid(True, alpha=0.3)

        # Global improvements
        for ax in axes.flat:
            ax.set_xlim(0, 8)
            ax.set_ylim(0, 8)
            ax.tick_params(labelsize=9)

        plt.tight_layout()

        # Save the improved plot
        output_path = '/home/chengrui/wk/NCBFquickDemo/work/ncbf/weights/intensive_training_2/improved_contour_evaluation.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Improved contour plot saved to: {output_path}")

        # Also show the plot
        plt.show()

        # Print analysis
        print(f"\n📊 Contour Analysis:")
        print(f"   h(x) range: [{np.min(h_values):.3f}, {np.max(h_values):.3f}]")
        print(f"   Safety boundary length: {len(zero_contour.collections[0].get_paths())} segments")
        print(f"   Average gradient magnitude: {np.mean(grad_magnitude):.3f}")

        return True

    except Exception as e:
        print(f"❌ Error in contour plotting: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 Testing Improved Contour Plotting")
    print("="*60)

    success = test_improved_contour_plotting()

    if success:
        print("\n✅ Improved contour plotting test completed successfully!")
    else:
        print("\n❌ Improved contour plotting test failed.")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())