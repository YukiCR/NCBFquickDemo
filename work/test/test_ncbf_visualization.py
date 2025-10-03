#!/usr/bin/env python3
"""
Test script for loading and visualizing trained Neural CBF (NCBF) models.

This script:
1. Loads a pretrained NCBF model from weights directory
2. Initializes it with the training configuration
3. Visualizes the learned CBF as contour plots
4. Compares with handwritten CBF for validation
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# Add the work directory to Python path
sys.path.insert(0, '/home/chengrui/wk/NCBFquickDemo/work')

from ncbf.models.ncbf import NCBF
from ncbf.configs.ncbf_config import NCBFConfig
from safe_control.cbf_function import CBFConfig
from safe_control.handwritten_cbf import CBFmultipleobs
from models.unicycle_model import UnicycleConfig

def load_training_config(config_path):
    """Load training configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return NCBFConfig(**config_dict)

def load_ncbf_model(weights_path, config_path):
    """Load pretrained NCBF model with configuration."""
    # Load configuration
    config = load_training_config(config_path)

    # Initialize model
    ncbf = NCBF(config)

    # Load weights
    ncbf.load_model(weights_path)

    # Set model to evaluation mode
    ncbf.eval()

    return ncbf, config

def create_test_obstacles():
    """Create test obstacles for visualization."""
    obstacles = [
        np.array([1.0, 1.0, 0.2]),    # [x, y, radius]
        np.array([3.0, 1.5, 0.3]),
        np.array([2.0, 3.0, 0.25]),
        np.array([4.0, 2.5, 0.2]),
        np.array([1.5, 4.0, 0.15]),
    ]
    return obstacles

def visualize_ncbf_contour(ncbf, obstacles, theta=0.0, save_path=None):
    """Visualize NCBF as contour plot."""
    # Create grid
    x_range = np.linspace(-1, 6, 100)
    y_range = np.linspace(-1, 6, 100)
    X, Y = np.meshgrid(x_range, y_range)

    # Create state vectors [x, y, theta] for each grid point
    states = np.zeros((X.size, 3))
    states[:, 0] = X.flatten()
    states[:, 1] = Y.flatten()
    states[:, 2] = theta  # Fixed orientation

    # Evaluate NCBF
    print("Evaluating NCBF on grid...")
    Z_ncbf = np.zeros(X.size)

    # Batch evaluation for efficiency
    batch_size = 1000
    for i in range(0, len(states), batch_size):
        batch = states[i:i+batch_size]
        batch_tensor = torch.tensor(batch, dtype=torch.float32)
        with torch.no_grad():
            batch_h = ncbf(batch_tensor)
        Z_ncbf[i:i+batch_size] = batch_h.numpy()

    Z_ncbf = Z_ncbf.reshape(X.shape)

    # Create handwritten CBF for comparison
    config = CBFConfig(safety_radius=0.3)
    handwritten_cbf = CBFmultipleobs(config, obstacles, alpha_softmin=10.0)

    # Evaluate handwritten CBF
    print("Evaluating handwritten CBF on grid...")
    Z_handwritten = np.zeros(X.size)
    for i in range(len(states)):
        Z_handwritten[i] = handwritten_cbf.h(states[i])
    Z_handwritten = Z_handwritten.reshape(X.shape)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Neural CBF vs Handwritten CBF (θ = {theta:.2f})', fontsize=16)

    # Plot 1: NCBF contour
    ax = axes[0, 0]
    contour1 = ax.contour(X, Y, Z_ncbf, levels=20, cmap='RdYlBu', linewidths=1.0)
    ax.contourf(X, Y, Z_ncbf, levels=20, cmap='RdYlBu', alpha=0.6)
    ax.clabel(contour1, inline=True, fontsize=8, fmt='%.2f')
    ax.set_title('Neural CBF Contours')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot obstacles
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.7)
        ax.add_patch(circle)

    # Plot 2: Handwritten CBF contour
    ax = axes[0, 1]
    contour2 = ax.contour(X, Y, Z_handwritten, levels=20, cmap='RdYlBu', linewidths=1.0)
    ax.contourf(X, Y, Z_handwritten, levels=20, cmap='RdYlBu', alpha=0.6)
    ax.clabel(contour2, inline=True, fontsize=8, fmt='%.2f')
    ax.set_title('Handwritten CBF Contours')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot obstacles
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.7)
        ax.add_patch(circle)

    # Plot 3: Safety boundaries (h=0)
    ax = axes[1, 0]
    ax.contour(X, Y, Z_ncbf, levels=[0], colors='blue', linewidths=2, linestyles='-', label='NCBF h=0')
    ax.contour(X, Y, Z_handwritten, levels=[0], colors='orange', linewidths=2, linestyles='--', label='Handwritten h=0')
    ax.set_title('Safety Boundaries Comparison')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot obstacles
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.7)
        ax.add_patch(circle)

    # Plot 4: Difference between NCBF and handwritten CBF
    ax = axes[1, 1]
    diff = Z_ncbf - Z_handwritten
    contour4 = ax.contour(X, Y, diff, levels=20, cmap='RdBu', linewidths=1.0)
    ax.contourf(X, Y, diff, levels=20, cmap='RdBu', alpha=0.6)
    ax.clabel(contour4, inline=True, fontsize=8, fmt='%.2f')
    ax.set_title('Difference (NCBF - Handwritten)')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot obstacles
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.7)
        ax.add_patch(circle)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Contour plot saved to: {save_path}")
    else:
        plt.show()

    return fig, Z_ncbf, Z_handwritten

def analyze_ncbf_quality(ncbf, obstacles, theta_values=[0.0, np.pi/4, np.pi/2]):
    """Analyze NCBF quality across different orientations."""
    print("\n" + "="*60)
    print("NCBF QUALITY ANALYSIS")
    print("="*60)

    # Create handwritten CBF for comparison
    config = CBFConfig(safety_radius=0.3)
    handwritten_cbf = CBFmultipleobs(config, obstacles, alpha_softmin=10.0)

    # Test points
    test_points = [
        np.array([0.5, 0.5]),  # Safe point
        np.array([1.0, 1.0]),  # Near obstacle
        np.array([2.0, 2.0]),  # Another obstacle
        np.array([5.0, 5.0]),  # Far away
    ]

    for theta in theta_values:
        print(f"\n--- Orientation θ = {theta:.2f} rad ---")

        for i, point in enumerate(test_points):
            state = np.array([point[0], point[1], theta])

            # Evaluate NCBF
            h_ncbf = ncbf.h(state)

            # Evaluate handwritten CBF
            h_handwritten = handwritten_cbf.h(state)

            # Evaluate gradients
            grad_ncbf = ncbf.grad_h(state)
            grad_handwritten = handwritten_cbf.grad_h(state)

            # Compute gradient magnitudes
            grad_mag_ncbf = np.linalg.norm(grad_ncbf[:2])  # Only position components
            grad_mag_handwritten = np.linalg.norm(grad_handwritten[:2])

            print(f"Point {i+1} ({point[0]:.1f}, {point[1]:.1f}):")
            print(f"  h_ncbf: {h_ncbf:.3f}, h_handwritten: {h_handwritten:.3f}")
            print(f"  ∇h_mag_ncbf: {grad_mag_ncbf:.3f}, ∇h_mag_handwritten: {grad_mag_handwritten:.3f}")
            print(f"  Safety: {'SAFE' if h_ncbf >= 0 else 'UNSAFE'}")

def main():
    """Main function to test NCBF visualization."""
    print("🚀 Starting NCBF Visualization Test")
    print("="*50)

    # Configuration
    weights_path = "/home/chengrui/wk/NCBFquickDemo/work/ncbf/weights/checkpoints/best_model.pt"
    config_path = "/home/chengrui/wk/NCBFquickDemo/work/ncbf/weights/final_models/training_config.json"
    save_dir = "/home/chengrui/wk/NCBFquickDemo/test/ncbf_visualizations"

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    try:
        # Load NCBF model
        print("Loading NCBF model...")
        ncbf, config = load_ncbf_model(weights_path, config_path)
        print(f"✅ Model loaded successfully")
        print(f"Model info: {ncbf.get_model_info()}")

        # Create test obstacles
        obstacles = create_test_obstacles()
        print(f"Created {len(obstacles)} test obstacles")

        # Visualize at different orientations
        theta_values = [0.0, np.pi/4, np.pi/2]

        for theta in theta_values:
            print(f"\n📊 Visualizing NCBF at θ = {theta:.2f}...")
            save_path = os.path.join(save_dir, f"ncbf_contour_theta_{theta:.2f}.png")

            fig, Z_ncbf, Z_handwritten = visualize_ncbf_contour(
                ncbf, obstacles, theta=theta, save_path=save_path
            )

            # Compute some statistics
            print(f"NCBF range: [{Z_ncbf.min():.3f}, {Z_ncbf.max():.3f}]")
            print(f"Handwritten CBF range: [{Z_handwritten.min():.3f}, {Z_handwritten.max():.3f}]")

            # Close figure to free memory
            plt.close(fig)

        # Analyze NCBF quality
        analyze_ncbf_quality(ncbf, obstacles, theta_values)

        print(f"\n✅ All visualizations saved to: {save_dir}")

    except Exception as e:
        print(f"❌ Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 NCBF visualization test completed successfully!")
    else:
        print("\n❌ NCBF visualization test failed!")
        sys.exit(1)