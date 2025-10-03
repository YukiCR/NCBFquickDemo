#!/usr/bin/env python3
"""
Simple test script for loading and visualizing trained Neural CBF (NCBF) models.

This script performs two basic tests:
1. Load a pretrained NCBF model with correct configuration
2. Draw a simple contour map to verify the learned CBF makes sense
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

def load_ncbf_simple(weights_path, config_path):
    """Load pretrained NCBF model with configuration."""
    print(f"Loading config from: {config_path}")

    # Load configuration
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    config = NCBFConfig(**config_dict)

    print(f"Model architecture: {config.hidden_dims}")
    print(f"Input dim: {config.input_dim}, Output dim: {config.output_dim}")

    # Initialize model
    ncbf = NCBF(config)
    print(f"✅ Model initialized successfully")

    # Load weights
    ncbf.load_model(weights_path)
    print(f"✅ Model weights loaded from: {weights_path}")

    # Set model to evaluation mode
    ncbf.eval()

    return ncbf, config

def create_simple_contour(ncbf, theta=0.0, save_path=None):
    """Create a simple contour plot of the NCBF."""
    print("Creating simple contour visualization...")

    # Create grid
    x_range = np.linspace(-1, 6, 50)  # Smaller grid for quick testing
    y_range = np.linspace(-1, 6, 50)
    X, Y = np.meshgrid(x_range, y_range)

    # Create state vectors [x, y, theta] for each grid point
    states = np.zeros((X.size, 3))
    states[:, 0] = X.flatten()
    states[:, 1] = Y.flatten()
    states[:, 2] = theta  # Fixed orientation

    # Evaluate NCF
    print("Evaluating NCBF on grid...")
    Z = np.zeros(X.size)

    # Batch evaluation for efficiency
    batch_size = 500
    for i in range(0, len(states), batch_size):
        batch = states[i:i+batch_size]
        batch_tensor = torch.tensor(batch, dtype=torch.float32)
        with torch.no_grad():
            batch_h = ncbf(batch_tensor)
            # Handle different output shapes - squeeze if needed
            if batch_h.dim() > 1 and batch_h.shape[1] == 1:
                batch_h = batch_h.squeeze(1)
        Z[i:i+batch_size] = batch_h.numpy()

    Z = Z.reshape(X.shape)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Neural CBF Visualization (θ = {theta:.2f})', fontsize=14)

    # Plot 1: Contour plot
    ax = axes[0]
    contour = ax.contour(X, Y, Z, levels=15, cmap='RdYlBu', linewidths=1.0)
    ax.contourf(X, Y, Z, levels=15, cmap='RdYlBu', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
    ax.set_title('NCBF h(x) Contours')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot 2: Safety boundary (h=0) and regions
    ax = axes[1]
    ax.contour(X, Y, Z, levels=[0], colors='red', linewidths=2, linestyles='-', label='h=0')

    # Add some contour levels to show structure
    levels = [-1, -0.5, 0, 0.5, 1]
    contour2 = ax.contour(X, Y, Z, levels=levels, colors='gray', linewidths=0.8, alpha=0.7)
    ax.clabel(contour2, inline=True, fontsize=8, fmt='%.1f')

    ax.set_title('Safety Boundary (h=0)')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Simple contour plot saved to: {save_path}")
    else:
        plt.show()

    return fig, Z

def test_ncbf_basic(ncbf, config):
    """Basic test of NCBF functionality."""
    print("\n" + "="*50)
    print("BASIC NCBF TESTS")
    print("="*50)

    # Test single point
    test_state = np.array([2.0, 2.0, 0.0])
    h_val = ncbf.h(test_state)
    grad_h = ncbf.grad_h(test_state)

    print(f"Test state: {test_state}")
    print(f"h(x) = {h_val:.4f}")
    print(f"∇h(x) = {grad_h}")
    print(f"||∇h(x)|| = {np.linalg.norm(grad_h):.4f}")
    print(f"Safety: {'SAFE' if h_val >= 0 else 'UNSAFE'}")

    # Test batch of points
    test_states = np.array([
        [0.5, 0.5, 0.0],  # Should be safe
        [2.0, 2.0, 0.0],  # Test point
        [4.0, 4.0, 0.0],  # Another test point
    ])

    h_vals = ncbf.h(test_states)
    print(f"\nBatch test:")
    for i, (state, h_val) in enumerate(zip(test_states, h_vals)):
        print(f"  State {i+1}: {state} -> h={h_val:.4f} ({'safe' if h_val >= 0 else 'unsafe'})")

def main():
    """Main function for simple NCBF test."""
    print("🚀 Simple NCBF Test - Load Model & Visualize")
    print("="*60)

    # Configuration - using the intensive training results
    weights_path = "/home/chengrui/wk/NCBFquickDemo/work/ncbf/weights/intensive_large_1000ep/best_model.pt"
    config_path = "/home/chengrui/wk/NCBFquickDemo/work/ncbf/weights/intensive_large_1000ep/training_config.json"
    save_dir = "/home/chengrui/wk/NCBFquickDemo/test/ncbf_simple_test"

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    try:
        # Load NCBF model
        print("1. Loading NCBF model...")
        ncbf, config = load_ncbf_simple(weights_path, config_path)

        # Get model info
        model_info = ncbf.get_model_info()
        print(f"Model info: {model_info}")

        # Basic functionality tests
        print("\n2. Running basic tests...")
        test_ncbf_basic(ncbf, config)

        # Simple visualization
        print("\n3. Creating simple contour visualization...")
        save_path = os.path.join(save_dir, "simple_ncbf_contour.png")
        fig, Z = create_simple_contour(ncbf, theta=0.0, save_path=save_path)

        # Print some statistics
        print(f"\n📊 Contour Statistics:")
        print(f"   h(x) range: [{Z.min():.3f}, {Z.max():.3f}]")
        print(f"   Mean h(x): {Z.mean():.3f}")
        print(f"   Std h(x): {Z.std():.3f}")
        print(f"   Safe points (h>0): {np.sum(Z > 0):,} / {Z.size:,}")
        print(f"   Unsafe points (h<0): {np.sum(Z < 0):,} / {Z.size:,}")
        print(f"   Boundary points (|h|<0.1): {np.sum(np.abs(Z) < 0.1):,} / {Z.size:,}")

        plt.close(fig)

        print(f"\n✅ Simple test completed successfully!")
        print(f"   Results saved to: {save_dir}")

        return True

    except Exception as e:
        print(f"\n❌ Error during simple test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Simple NCBF test completed successfully!")
    else:
        print("\n❌ Simple NCBF test failed!")
        sys.exit(1)