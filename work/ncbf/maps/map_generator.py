"""
Interactive map generation sequence for NCBF training.

This module provides an interactive workflow for generating and approving
obstacle maps for neural control barrier function training.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
import json
from datetime import datetime

from .map_generation import generate_moderate_map, visualize_map
from .map_manager import NCBFMap


class MapGenerator:
    """
    Interactive map generator with user approval workflow.

    Provides a simple interface for generating obstacle maps,
    visualizing them, and getting user approval before saving.
    """

    def __init__(self, map_storage_dir: str = "/home/chengrui/wk/NCBFquickDemo/work/ncbf/maps/map_files"):
        """
        Initialize map generator.

        Args:
            map_storage_dir: Directory to store approved map files
        """
        self.map_storage_dir = Path(map_storage_dir)
        self.map_storage_dir.mkdir(parents=True, exist_ok=True)

    def generate_and_approve(
        self,
        workspace_size: float = 8.0,
        seed: Optional[int] = None,
        max_attempts: int = 10,
        auto_approve: bool = False
    ) -> Tuple[NCBFMap, Path]:
        """
        Generate maps until user approves one.

        Args:
            workspace_size: Size of the workspace
            seed: Random seed (None for random)
            max_attempts: Maximum number of generation attempts
            auto_approve: If True, automatically approve first map (for testing)

        Returns:
            Tuple of (approved NCBFMap, path to saved JSON file)

        Raises:
            RuntimeError: If max_attempts reached without approval
        """
        print(f"\n{'='*60}")
        print("NCBF Map Generation - Interactive Approval")
        print(f"{'='*60}")
        print(f"Workspace size: {workspace_size}m × {workspace_size}m")
        print(f"Map storage: {self.map_storage_dir}")
        print(f"Commands: 'y' = accept, 'n' = reject & regenerate, 'q' = quit")
        print(f"{'='*60}")

        attempt = 0
        current_seed = seed

        while attempt < max_attempts:
            attempt += 1
            print(f"\n--- Generation Attempt {attempt} ---")

            # Generate new map with different seed if needed
            if current_seed is not None:
                current_seed += attempt  # Change seed for each attempt

            # Generate obstacles
            obstacles = generate_moderate_map(
                seed=current_seed,
                workspace_size=workspace_size
            )

            # Create NCBFMap
            ncbf_map = NCBFMap(obstacles=obstacles, workspace_size=workspace_size)

            # Visualize the map
            print(f"Generated map with {len(ncbf_map)} obstacles")
            print(f"Radius range: {ncbf_map.get_info()['min_radius']:.2f} - {ncbf_map.get_info()['max_radius']:.2f}m")

            # Show visualization
            ncbf_map.visualize(
                title=f"Generated Map (Attempt {attempt})",
                show=True
            )

            if auto_approve:
                print("Auto-approve enabled - accepting map")
                approval = 'y'
            else:
                approval = input("Accept this map? (y/n/q): ").strip().lower()

            if approval == 'y':
                # Save the approved map
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ncbf_map_{timestamp}.json"
                save_path = self.map_storage_dir / filename

                ncbf_map.save(save_path)
                print(f"✓ Map saved to: {save_path}")
                print(f"✓ Map approved after {attempt} attempts")

                return ncbf_map, save_path

            elif approval == 'q':
                print("Map generation cancelled by user")
                raise KeyboardInterrupt("User cancelled map generation")

            elif approval == 'n':
                print("Map rejected - generating new one...")
                continue

            else:
                print("Invalid input - treating as rejection")
                continue

        # If we get here, max attempts reached
        raise RuntimeError(f"Maximum attempts ({max_attempts}) reached without user approval")

    def quick_generate(
        self,
        workspace_size: float = 8.0,
        seed: int = 42,
        save: bool = True
    ) -> NCBFMap:
        """
        Quickly generate a map without user interaction.

        Args:
            workspace_size: Size of the workspace
            seed: Random seed
            save: Whether to automatically save the map

        Returns:
            Generated NCBFMap
        """
        print(f"Quickly generating map (seed={seed}, workspace={workspace_size}m)...")

        obstacles = generate_moderate_map(seed=seed, workspace_size=workspace_size)
        ncbf_map = NCBFMap(obstacles=obstacles, workspace_size=workspace_size)

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ncbf_map_quick_{timestamp}.json"
            save_path = self.map_storage_dir / filename
            ncbf_map.save(save_path)
            print(f"✓ Map saved to: {save_path}")

        print(f"✓ Generated map with {len(ncbf_map)} obstacles")
        return ncbf_map

    def list_saved_maps(self) -> list:
        """
        List all saved map files.

        Returns:
            List of Path objects to saved map files
        """
        json_files = list(self.map_storage_dir.glob("*.json"))
        return sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True)

    def load_map(self, map_name: str) -> NCBFMap:
        """
        Load a specific map by name.

        Args:
            map_name: Name of the map file (with or without .json extension)

        Returns:
            Loaded NCBFMap

        Raises:
            FileNotFoundError: If map file doesn't exist
        """
        if not map_name.endswith('.json'):
            map_name += '.json'

        map_path = self.map_storage_dir / map_name
        if not map_path.exists():
            raise FileNotFoundError(f"Map file not found: {map_path}")

        return load_map(map_path)

    def show_map_info(self, map_file: str):
        """
        Show information about a saved map.

        Args:
            map_file: Path to map file (relative to storage dir or absolute)
        """
        try:
            if Path(map_file).is_absolute():
                map_path = Path(map_file)
            else:
                map_path = self.map_storage_dir / map_file

            ncbf_map = load_map(map_path)
            info = ncbf_map.get_info()

            print(f"\nMap Information: {map_path.name}")
            print(f"  Obstacles: {info['num_obstacles']}")
            print(f"  Workspace: {info['workspace_size']}m × {info['workspace_size']}m")
            print(f"  Radius range: {info['min_radius']:.2f} - {info['max_radius']:.2f}m")
            print(f"  Average radius: {info['avg_radius']:.2f}m")

            # Show visualization
            ncbf_map.visualize(title=f"Map: {map_path.stem}", show=True)

        except Exception as e:
            print(f"Error loading map: {e}")


def interactive_map_generation():
    """
    Main interactive function for map generation.

    This function can be called directly to start the interactive
    map generation process.
    """
    generator = MapGenerator()

    try:
        print("Welcome to NCBF Map Generator!")
        print("This tool helps you create obstacle maps for neural CBF training.")

        # Get parameters
        workspace_size = 8.0
        try:
            user_size = input(f"Workspace size (default: {workspace_size}m): ").strip()
            if user_size:
                workspace_size = float(user_size)
        except ValueError:
            print(f"Using default workspace size: {workspace_size}m")

        seed = None
        user_seed = input("Random seed (empty for random): ").strip()
        if user_seed:
            try:
                seed = int(user_seed)
            except ValueError:
                print("Invalid seed - using random generation")

        # Generate and approve map
        approved_map, save_path = generator.generate_and_approve(
            workspace_size=workspace_size,
            seed=seed
        )

        print(f"\n🎉 Success! Map saved to: {save_path}")
        print(f"Map ready for NCBF training with {len(approved_map)} obstacles")

        return approved_map, save_path

    except KeyboardInterrupt:
        print("\nMap generation cancelled.")
        return None, None
    except Exception as e:
        print(f"Error during map generation: {e}")
        return None, None


def main():
    """Main entry point for standalone execution."""
    try:
        interactive_map_generation()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()