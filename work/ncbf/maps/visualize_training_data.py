#!/usr/bin/env python3
"""
Command-line tool for visualizing NCBF training data.

This tool loads HDF5 training data files and creates visualizations
for NCBF training analysis with configurable display options.
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'work'))

from ncbf.maps import load_map, NCBFMap
from configs.unicycle_config import UnicycleConfig
import matplotlib.pyplot as plt


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize NCBF training data from HDF5 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize training data with default map
  python visualize_training_data.py --training_data map1/training_data.h5

  # Visualize with specific map file
  python visualize_training_data.py --training_data map1/training_data.h5 --map map1/map1.json

  # Save visualization without displaying window
  python visualize_training_data.py --training_data map1/training_data.h5 --save_only

  # Quiet mode (minimal output)
  python visualize_training_data.py --training_data map1/training_data.h5 --quiet
        """
    )

    parser.add_argument(
        '--training_data', '-t',
        type=str,
        required=True,
        help='Path to training data HDF5 file (relative to map_files/ or absolute path)'
    )

    parser.add_argument(
        '--map', '-m',
        type=str,
        default=None,
        help='Path to map JSON file (optional, will try to infer from data if not provided)'
    )

    parser.add_argument(
        '--save_only',
        action='store_true',
        help='Save visualization without displaying matplotlib window'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output filename for visualization (default: auto-generated based on input)'
    )

    return parser.parse_args()


def resolve_file_path(file_path: str, base_dir: Path) -> Path:
    """Resolve file path relative to base directory or absolute."""
    path = Path(file_path)

    if path.is_absolute():
        return path
    else:
        # Try relative to base directory
        full_path = base_dir / path
        if full_path.exists():
            return full_path
        else:
            raise FileNotFoundError(f"File not found: {file_path}")


def visualize_training_data_cli(training_data_path: Path, map_path: Optional[Path] = None,
                               save_only: bool = False, quiet: bool = False,
                               output_path: Optional[Path] = None):
    """
    Visualize training data with 3-panel display.

    Args:
        training_data_path: Path to HDF5 training data file
        map_path: Optional path to map JSON file
        save_only: Whether to save without displaying window
        quiet: Whether to suppress detailed output
        output_path: Optional custom output path
    """
    try:
        if not quiet:
            print("🎨 NCBF Training Data Visualizer")
            print("=" * 50)
            print(f"📊 Training data: {training_data_path}")
            print(f"🗺️  Map file: {map_path if map_path else 'Auto-detect'}")

        # Load training data
        if not quiet:
            print("📂 Loading training data...")

        # Create temporary map instance to load data - use minimal valid configuration
        temp_map = NCBFMap(obstacles=[np.array([1.0, 1.0, 0.1])], workspace_size=8.0)  # Minimal valid obstacle
        training_data = temp_map.load_training_data(training_data_path)

        if not quiet:
            print(f"✅ Training data loaded successfully!")

        # Extract data
        states = training_data['states']
        labels = training_data['labels'].flatten()
        num_samples = training_data['num_samples']
        num_safe = training_data['num_safe']
        num_unsafe = training_data['num_unsafe']

        # Calculate ratios
        actual_safe_ratio = num_safe / num_samples
        actual_unsafe_ratio = num_unsafe / num_samples

        if not quiet:
            print(f"📊 Data Statistics:")
            print(f"   Total samples: {num_samples:,}")
            print(f"   Safe samples: {num_safe:,} ({100*actual_safe_ratio:.1f}%)")
            print(f"   Unsafe samples: {num_unsafe:,} ({100*actual_unsafe_ratio:.1f}%)")

        # Load map if provided, otherwise try to infer from data or use default
        if map_path:
            if not quiet:
                print("🗺️  Loading specified map...")
            ncbf_map = load_map(map_path)
        else:
            # Try to find a corresponding map file in the same directory
            potential_map_path = training_data_path.parent / f"{training_data_path.stem.split('_')[0]}.json"
            if potential_map_path.exists():
                if not quiet:
                    print(f"🗺️  Found corresponding map file: {potential_map_path}")
                ncbf_map = load_map(potential_map_path)
            else:
                if not quiet:
                    print("⚠️  No map file found, creating visualization without obstacles...")
                # Create minimal workspace for basic visualization
                workspace_size = 8.0
                if 'map_info' in training_data and 'workspace_size' in training_data['map_info']:
                    workspace_size = training_data['map_info']['workspace_size']
                ncbf_map = NCBFMap(obstacles=[np.array([workspace_size/2, workspace_size/2, 0.1])], workspace_size=workspace_size)

        if not quiet:
            print(f"✅ Map loaded: {ncbf_map}")

        # Create 3-panel visualization - more compact for paper-ready display
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # 1. Data scatter plot with safety boundaries
        ax1 = axes[0]

        # Plot obstacles and safety boundaries if map has obstacles
        if ncbf_map.obstacles:
            config = UnicycleConfig()
            for obs in ncbf_map.obstacles:
                x, y, radius = obs
                # Obstacle boundary (red)
                circle = plt.Circle((x, y), radius, color='red', alpha=0.4, linewidth=2)
                ax1.add_patch(circle)

                # Safety boundary (orange dashed)
                safety_boundary = radius + config.safety_radius
                safety_circle = plt.Circle((x, y), safety_boundary, color='orange',
                                          alpha=0.2, linewidth=2, linestyle='--')
                ax1.add_patch(safety_circle)

        # Plot safe and unsafe points
        safe_mask = labels == 1
        unsafe_mask = labels == 0

        # Use BLUE for safe states, RED for unsafe states
        ax1.scatter(states[safe_mask, 0], states[safe_mask, 1], c='blue', s=8, alpha=0.7,
                   label=f'Safe ({np.sum(safe_mask):,})', edgecolors='darkblue', linewidth=0.5)
        ax1.scatter(states[unsafe_mask, 0], states[unsafe_mask, 1], c='red', s=8, alpha=0.7,
                   label=f'Unsafe ({np.sum(unsafe_mask):,})', edgecolors='darkred', linewidth=0.5)

        ax1.set_xlim(0, ncbf_map.workspace_size)
        ax1.set_ylim(0, ncbf_map.workspace_size)
        ax1.set_xlabel('X Position (m)', fontsize=10)
        ax1.set_ylabel('Y Position (m)', fontsize=10)
        ax1.set_title('Training Data Distribution', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # Add statistics info
        info_text = (f"Training Data\n"
                    f"Total: {num_samples:,}\n"
                    f"Safe: {num_safe:,} ({100*actual_safe_ratio:.1f}%)\n"
                    f"Unsafe: {num_unsafe:,} ({100*actual_unsafe_ratio:.1f}%)")

        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3',
                facecolor='lightblue', alpha=0.8, edgecolor='blue'))

        # 2. Ratio comparison bar chart
        ax2 = axes[1]

        categories = ['Safe', 'Unsafe']
        counts = [num_safe, num_unsafe]
        colors = ['blue', 'red']

        bars = ax2.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Number of Samples', fontsize=10)
        ax2.set_title('Safe vs Unsafe Ratio', fontsize=11)
        ax2.grid(True, alpha=0.3)

        # Add percentage labels on bars (only percentage, no sample count)
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            percentage = (count / num_samples) * 100
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{percentage:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 3. Data density heatmap
        ax3 = axes[2]

        # Create 2D histogram for density
        h = ax3.hist2d(states[:, 0], states[:, 1], bins=20, cmap='YlOrRd', alpha=0.8)
        ax3.set_xlabel('X Position (m)', fontsize=10)
        ax3.set_ylabel('Y Position (m)', fontsize=10)
        ax3.set_title('Data Density Heatmap', fontsize=11)

        # Add colorbar
        cbar = plt.colorbar(h[3], ax=ax3, shrink=0.8)
        cbar.set_label('Sample Count', fontsize=9)

        plt.tight_layout()

        # Handle output - only save if explicitly requested
        if output_path is not None or save_only:
            # Save visualization if output path specified or save_only mode
            if output_path is None:
                output_path = training_data_path.parent / f"{training_data_path.stem}_viz.png"

            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            if not quiet:
                print(f"📸 Visualization saved to: {output_path}")

        # Display the plot unless save_only is specified
        if not save_only:
            if not quiet:
                print("🎨 Displaying visualization window...")
            plt.show(block=True)
        else:
            plt.close(fig)

        if not quiet:
            print("\n" + "=" * 50)
            print("🎉 Visualization Complete!")
            print("✅ Training data successfully visualized")
            print("✅ Ready for NCBF neural network training!")

        return 0

    except Exception as e:
        print(f"\n❌ Visualization failed with error: {e}")
        if not quiet:
            import traceback
            traceback.print_exc()
        return 1


def main():
    """Main entry point."""
    args = parse_arguments()

    try:
        # Resolve paths
        base_dir = Path(__file__).parent / 'map_files'
        training_data_path = resolve_file_path(args.training_data, base_dir)

        map_path = None
        if args.map:
            map_path = resolve_file_path(args.map, base_dir)

        # Run visualization
        return visualize_training_data_cli(
            training_data_path=training_data_path,
            map_path=map_path,
            save_only=args.save_only,
            quiet=args.quiet,
            output_path=Path(args.output) if args.output else None
        )

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())