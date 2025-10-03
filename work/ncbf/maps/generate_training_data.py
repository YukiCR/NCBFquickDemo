#!/usr/bin/env python3
"""
Command-line tool for generating NCBF training data.

This tool generates labeled training data for neural control barrier function training
using obstacle maps and configurable safety parameters.
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'work'))

from ncbf.maps import load_map
from configs.unicycle_config import UnicycleConfig


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate labeled training data for NCBF neural network training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate data for map1 with default parameters
  python generate_training_data.py --map map1/map1.json --output map1/training_data.h5

  # Generate with custom parameters
  python generate_training_data.py --map map1/map1.json --output map1/custom_data.h5 --samples 5000 --min-unsafe-ratio 0.4

  # Generate with seed for reproducibility
  python generate_training_data.py --map map1/map1.json --output map1/reproducible_data.h5 --seed 42
        """
    )

    parser.add_argument(
        '--map', '-m',
        type=str,
        required=True,
        help='Path to map JSON file (relative to map_files/ or absolute path)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output file path for training data (HDF5 format)'
    )

    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=10000,
        help='Number of training samples to generate (default: 10000)'
    )

    parser.add_argument(
        '--min-unsafe-ratio',
        type=float,
        default=0.3,
        help='Minimum ratio of unsafe samples (default: 0.3, range: 0.1-0.9)'
    )

    parser.add_argument(
        '--obstacle-focus-ratio',
        type=float,
        default=0.3,
        help='Ratio of samples focused around obstacles (default: 0.3, range: 0.0-1.0)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible data generation (default: None for random)'
    )

    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Generate visualization of the training data'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output'
    )

    return parser.parse_args()

def validate_arguments(args):
    """Validate input arguments."""
    if args.samples < 100:
        raise ValueError("Number of samples must be at least 100")

    if not (0.1 <= args.min_unsafe_ratio <= 0.9):
        raise ValueError("Minimum unsafe ratio must be between 0.1 and 0.9")

    if not (0.0 <= args.obstacle_focus_ratio <= 1.0):
        raise ValueError("Obstacle focus ratio must be between 0.0 and 1.0")

def resolve_map_path(map_path: str) -> Path:
    """Resolve map file path."""
    map_file = Path(map_path)

    if map_file.is_absolute():
        return map_file
    else:
        # Try relative to map_files directory
        map_files_dir = Path(__file__).parent / 'map_files'
        full_path = map_files_dir / map_path
        if full_path.exists():
            return full_path
        else:
            raise FileNotFoundError(f"Map file not found: {map_path}")

def resolve_output_path(output_path: str) -> Path:
    """Resolve output file path."""
    output_file = Path(output_path)

    if output_file.is_absolute():
        return output_file
    else:
        # Create relative to map_files directory
        map_files_dir = Path(__file__).parent / 'map_files'
        full_path = map_files_dir / output_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        return full_path

def generate_training_data_cli():
    """Main function for command-line data generation."""
    try:
        # Parse arguments
        args = parse_arguments()
        validate_arguments(args)

        # Resolve file paths
        map_path = resolve_map_path(args.map)
        output_path = resolve_output_path(args.output)

        # Print header
        if not args.quiet:
            print("🎯 NCBF Training Data Generator")
            print("=" * 60)
            print(f"📍 Map file: {map_path}")
            print(f"📁 Output file: {output_path}")
            print(f"📊 Samples: {args.samples:,}")
            print(f"🔒 Min unsafe ratio: {args.min_unsafe_ratio:.1f}")
            print(f"🎯 Obstacle focus ratio: {args.obstacle_focus_ratio:.1f}")
            print(f"🎲 Seed: {args.seed if args.seed is not None else 'random'}")
            print("=" * 60)

        # Load map
        if not args.quiet:
            print("🗺️  Loading map...")
        ncbf_map = load_map(map_path)

        if not args.quiet:
            print(f"✅ Map loaded: {ncbf_map}")

        # Create UnicycleConfig
        if not args.quiet:
            print("🔧 Creating configuration...")
        config = UnicycleConfig()

        if not args.quiet:
            print(f"✅ Configuration created:")
            print(f"   Safety radius: {config.safety_radius}m")
            print(f"   Robot radius: {config.robot_radius}m")

        # Generate training data
        if not args.quiet:
            print(f"\n🧪 Generating {args.samples:,} training samples...")

        data_result = ncbf_map.generate_training_data(
            judge_safety_func=ncbf_map.judge_unicycle_safety,
            config=config,
            num_samples=args.samples,
            seed=args.seed,
            obstacle_focus_ratio=args.obstacle_focus_ratio,
            min_unsafe_ratio=args.min_unsafe_ratio,
            save_path=output_path
        )

        # Print results
        if not args.quiet:
            print(f"\n📊 Generation complete!")
            print(f"✅ Total samples: {data_result['num_samples']:,}")
            print(f"✅ Safe samples: {data_result['num_safe']:,} ({100*data_result['actual_safe_ratio']:.1f}%)")
            print(f"✅ Unsafe samples: {data_result['num_unsafe']:,} ({100*data_result['actual_unsafe_ratio']:.1f}%)")
            print(f"✅ Data saved to: {output_path}")
            print(f"📁 File size: {output_path.stat().st_size / 1024:.1f} KB")

        # Generate visualization if requested
        if args.visualize:
            if not args.quiet:
                print(f"\n🎨 Generating visualization...")
            generate_visualization(data_result, config, ncbf_map, output_path)

        # Success message
        if not args.quiet:
            print(f"\n" + "=" * 60)
            print("🎉 Training data generation complete!")
            print(f"✅ Ready for NCBF neural network training!")
            print(f"\nTo use this data, load it with:")
            print(f"  from ncbf.maps import load_map, NCBFMap")
            print(f"  ncbf_map = load_map('{map_path}')")
            print(f"  training_data = ncbf_map.load_training_data('{output_path}')")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1

def generate_visualization(data_dict, config, ncbf_map, output_path):
    """Generate visualization of the training data."""
    import matplotlib.pyplot as plt

    states = data_dict['states']
    labels = data_dict['labels'].flatten()

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Main scatter plot with safety boundaries
    ax1 = axes[0, 0]

    # Plot obstacles and safety boundaries
    for obs in ncbf_map.obstacles:
        x, y, radius = obs
        # Obstacle boundary (red)
        circle = plt.Circle((x, y), radius, color='red', alpha=0.4, linewidth=2)
        ax1.add_patch(circle)

        # Safety boundary (orange dashed) - CORRECTED: only obstacle + safety radius
        safety_boundary = radius + config.safety_radius
        safety_circle = plt.Circle((x, y), safety_boundary, color='orange',
                                  alpha=0.2, linewidth=2, linestyle='--')
        ax1.add_patch(safety_circle)

    # Plot safe and unsafe points
    safe_mask = labels == 1
    unsafe_mask = labels == 0

    # Use BLUE for safe states, RED for unsafe states
    ax1.scatter(states[safe_mask, 0], states[safe_mask, 1], c='blue', s=6, alpha=0.7,
               label=f'Safe ({np.sum(safe_mask):,})', edgecolors='darkblue', linewidth=0.5)
    ax1.scatter(states[unsafe_mask, 0], states[unsafe_mask, 1], c='red', s=6, alpha=0.7,
               label=f'Unsafe ({np.sum(unsafe_mask):,})', edgecolors='darkred', linewidth=0.5)

    ax1.set_xlim(0, ncbf_map.workspace_size)
    ax1.set_ylim(0, ncbf_map.workspace_size)
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title(f'Training Data Distribution - {data_dict["num_samples"]:,} samples', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Add info text
    info_text = (f"Training Data\n"
                f"Total: {data_dict['num_samples']:,}\n"
                f"Safe: {data_dict['num_safe']:,} ({100*data_dict['actual_safe_ratio']:.1f}%)\n"
                f"Unsafe: {data_dict['num_unsafe']:,} ({100*data_dict['actual_unsafe_ratio']:.1f}%)\n"
                f"Safety boundary: obstacle + safety radius")

    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5',
            facecolor='lightblue', alpha=0.9, edgecolor='blue'))

    # 2. Sampling breakdown
    ax2 = axes[0, 1]
    if 'sampling_breakdown' in data_dict:
        sampling_data = data_dict['sampling_breakdown']
        categories = ['Boundary', 'Uniform', 'Obstacle']
        safe_counts = [sampling_data['boundary']['safe'], sampling_data['uniform']['safe'], sampling_data['obstacle']['safe']]
        unsafe_counts = [sampling_data['boundary']['unsafe'], sampling_data['uniform']['unsafe'], sampling_data['obstacle']['unsafe']]

        x = np.arange(len(categories))
        width = 0.35

        ax2.bar(x - width/2, safe_counts, width, label='Safe', color='blue', alpha=0.7)
        ax2.bar(x + width/2, unsafe_counts, width, label='Unsafe', color='red', alpha=0.7)

        ax2.set_xlabel('Sampling Strategy', fontsize=12)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.set_title('Sampling Strategy Breakdown', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

    # 3. Distance distribution
    ax3 = axes[1, 0]
    distances_to_nearest = []
    for state in states:
        min_dist = float('inf')
        for obs in ncbf_map.obstacles:
            obs_x, obs_y, obs_radius = obs
            distance_to_center = np.linalg.norm(state[:2] - np.array([obs_x, obs_y]))
            min_dist = min(min_dist, distance_to_center)
        distances_to_nearest.append(min_dist)

    distances_to_nearest = np.array(distances_to_nearest)
    safe_distances = distances_to_nearest[safe_mask]
    unsafe_distances = distances_to_nearest[unsafe_mask]

    ax3.hist(safe_distances, bins=30, alpha=0.7, color='blue', label='Safe', edgecolor='darkblue')
    ax3.hist(unsafe_distances, bins=30, alpha=0.7, color='red', label='Unsafe', edgecolor='darkred')

    ax3.set_xlabel('Distance to Nearest Obstacle (m)', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Distance to Nearest Obstacle Distribution', fontsize=14)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    # 4. Heading angle distribution
    ax4 = axes[1, 1]
    ax4.hist(states[safe_mask, 2], bins=30, alpha=0.7, color='blue', label='Safe', edgecolor='darkblue')
    ax4.hist(states[unsafe_mask, 2], bins=30, alpha=0.7, color='red', label='Unsafe', edgecolor='darkred')

    ax4.set_xlabel('Heading Angle (rad)', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Heading Angle Distribution', fontsize=14)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save visualization
    viz_path = output_path.with_suffix('.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"📸 Visualization saved to: {viz_path}")

def main():
    """Main entry point."""
    sys.exit(generate_training_data_cli())

if __name__ == "__main__":
    main()