"""
Map generation for NCBF training data.

Creates obstacle maps for training the neural network to learn safety certificates.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import random


def generate_moderate_map(seed: int = 42, workspace_size: float = 8.0) -> List[np.ndarray]:
    """
    Generate a moderate density obstacle map for NCBF training.

    Args:
        seed: Random seed for reproducibility
        workspace_size: Size of the square workspace

    Returns:
        List of obstacle arrays [x, y, radius]
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Moderate density: 8 obstacles
    num_obstacles = 8
    min_radius = 0.15
    max_radius = 0.30
    min_spacing = 0.8  # Minimum distance between obstacles

    obstacles = []

    for i in range(num_obstacles):
        # Random radius
        radius = random.uniform(min_radius, max_radius)

        # Find valid position (not too close to edges or other obstacles)
        max_attempts = 100
        for attempt in range(max_attempts):
            # Random position with margin from edges
            margin = radius + 0.5
            x = random.uniform(margin, workspace_size - margin)
            y = random.uniform(margin, workspace_size - margin)

            # Check distance from other obstacles
            too_close = False
            for existing_obs in obstacles:
                existing_x, existing_y, existing_r = existing_obs
                dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                if dist < min_spacing + existing_r + radius:
                    too_close = True
                    break

            if not too_close:
                obstacles.append(np.array([x, y, radius]))
                break

        # If can't find valid position, use a simpler placement
        if attempt == max_attempts - 1:
            # Place in a grid-like pattern as fallback
            grid_x = (i % 3) * workspace_size / 3 + workspace_size / 6
            grid_y = (i // 3) * workspace_size / 3 + workspace_size / 6

            # Add some random offset
            offset = 0.5
            x = grid_x + random.uniform(-offset, offset)
            y = grid_y + random.uniform(-offset, offset)

            # Ensure within bounds
            x = max(radius + 0.5, min(workspace_size - radius - 0.5, x))
            y = max(radius + 0.5, min(workspace_size - radius - 0.5, y))

            obstacles.append(np.array([x, y, radius]))

    return obstacles


def visualize_map(obstacles: List[np.ndarray], workspace_size: float = 8.0, save_path: str = None):
    """
    Visualize the obstacle map.

    Args:
        obstacles: List of obstacle arrays [x, y, radius]
        workspace_size: Size of the workspace
        save_path: Optional path to save the visualization
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot obstacles
    for obs in obstacles:
        x, y, radius = obs
        circle = plt.Circle((x, y), radius, color='red', alpha=0.6)
        ax.add_patch(circle)
        ax.plot(x, y, 'ro', markersize=4)

    # Set plot properties
    ax.set_xlim(0, workspace_size)
    ax.set_ylim(0, workspace_size)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('NCBF Training Map - Moderate Density')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Add text info
    info_text = f"Obstacles: {len(obstacles)}\nWorkspace: {workspace_size}×{workspace_size}m\nDensity: Moderate"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Map visualization saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def generate_test_map() -> List[np.ndarray]:
    """
    Generate a simple test map for quick verification.

    Returns:
        List of obstacle arrays for testing
    """
    # Simple test map with 4 obstacles
    obstacles = [
        np.array([2.0, 2.0, 0.25]),
        np.array([6.0, 2.0, 0.20]),
        np.array([2.0, 6.0, 0.30]),
        np.array([6.0, 6.0, 0.22])
    ]
    return obstacles


if __name__ == "__main__":
    # Generate and visualize a moderate map
    obstacles = generate_moderate_map()
    print(f"Generated {len(obstacles)} obstacles:")
    for i, obs in enumerate(obstacles):
        print(f"  Obstacle {i+1}: position=({obs[0]:.2f}, {obs[1]:.2f}), radius={obs[2]:.2f}m")

    # Visualize the map
    visualize_map(obstacles, save_path="/home/chengrui/wk/NCBFquickDemo/work/ncbf/maps/moderate_map.png")