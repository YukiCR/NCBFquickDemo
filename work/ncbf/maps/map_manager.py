"""
Map management for Neural Control Barrier Functions.

This module provides the NCBFMap class for managing obstacle maps with
persistent storage and CBF compatibility.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Callable
from pathlib import Path
from datetime import datetime
import h5py  # For efficient binary data storage


class NCBFMap:
    """
    Neural Control Barrier Function Map - manages obstacle data with storage capabilities.

    This class provides a unified interface for obstacle map management, supporting
    both in-memory obstacle lists and persistent JSON storage. It's designed to be
    compatible with CBF implementations that expect obstacles as lists of numpy arrays.
    """

    def __init__(
        self,
        obstacles: Optional[List[np.ndarray]] = None,
        map_file: Optional[Union[str, Path]] = None,
        workspace_size: Optional[float] = None
    ):
        """
        Initialize NCBF Map with either obstacles list or JSON file.

        Args:
            obstacles: List of obstacle arrays [x, y, radius] (optional)
            map_file: Path to JSON map file (optional)
            workspace_size: Size of square workspace (optional, defaults to 8.0 for obstacles, loaded from JSON for files)

        Raises:
            ValueError: If neither obstacles nor map_file is provided, or if workspace_size not provided for obstacles
            FileNotFoundError: If map_file doesn't exist
            json.JSONDecodeError: If map_file is invalid JSON
        """
        self.workspace_size: float = 8.0  # Default value
        self.obstacles: List[np.ndarray] = []

        # Validate input parameters
        if obstacles is not None and map_file is not None:
            raise ValueError("Cannot specify both obstacles and map_file. Choose one.")

        if obstacles is None and map_file is None:
            raise ValueError("Must provide either obstacles or map_file.")

        # Initialize from obstacles list
        if obstacles is not None:
            if workspace_size is None:
                raise ValueError("workspace_size must be provided when initializing from obstacles")
            self.from_obstacles(obstacles, workspace_size)

        # Initialize from JSON file
        if map_file is not None:
            self.from_json(map_file)

    def from_obstacles(self, obstacles: List[np.ndarray], workspace_size: float = 8.0):
        """
        Initialize map from list of obstacle arrays.

        Args:
            obstacles: List of numpy arrays [x, y, radius]
            workspace_size: Size of square workspace

        Raises:
            ValueError: If obstacles format is invalid
        """
        if not obstacles:
            raise ValueError("Obstacles list cannot be empty")

        # Validate obstacle format
        for i, obs in enumerate(obstacles):
            if not isinstance(obs, np.ndarray):
                raise ValueError(f"Obstacle {i} must be numpy array, got {type(obs)}")
            if obs.shape != (3,):
                raise ValueError(f"Obstacle {i} must have shape (3,), got {obs.shape}")
            if obs[2] <= 0:  # radius must be positive
                raise ValueError(f"Obstacle {i} radius must be positive, got {obs[2]}")

        self.obstacles = [obs.copy() for obs in obstacles]  # Deep copy
        self.workspace_size = workspace_size

    def from_json(self, file_path: Union[str, Path]):
        """
        Initialize map from JSON file.

        Args:
            file_path: Path to JSON map file

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is invalid JSON
            ValueError: If JSON structure is invalid
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Map file not found: {file_path}")

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in map file: {file_path}", e.doc, e.pos)

        # Validate JSON structure
        if 'obstacles' not in data:
            raise ValueError("JSON must contain 'obstacles' key")

        if not isinstance(data['obstacles'], list):
            raise ValueError("'obstacles' must be a list")

        # Parse obstacles
        obstacles = []
        for i, obs_data in enumerate(data['obstacles']):
            if not isinstance(obs_data, dict):
                raise ValueError(f"Obstacle {i} must be a dictionary")

            required_keys = ['x', 'y', 'radius']
            if not all(key in obs_data for key in required_keys):
                raise ValueError(f"Obstacle {i} missing required keys: {required_keys}")

            try:
                x = float(obs_data['x'])
                y = float(obs_data['y'])
                radius = float(obs_data['radius'])

                if radius <= 0:
                    raise ValueError(f"Obstacle {i} radius must be positive")

                obstacles.append(np.array([x, y, radius]))
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid obstacle {i} data: {e}")

        self.obstacles = obstacles
        # Use workspace_size from JSON file if available, otherwise keep default
        self.workspace_size = float(data.get('workspace_size', 8.0))

    def get_obs(self) -> List[np.ndarray]:
        """
        Get obstacles in CBF-compatible format.

        Returns:
            List of obstacle arrays [x, y, radius] compatible with CBF implementations
        """
        return self.obstacles.copy()  # Return copy to prevent external modification

    def save(self, file_path: Union[str, Path]):
        """
        Save map to JSON file.

        Args:
            file_path: Path to save JSON file

        Raises:
            IOError: If file cannot be written
        """
        file_path = Path(file_path)

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for JSON serialization
        data = {
            'workspace_size': self.workspace_size,
            'obstacles': []
        }

        for obs in self.obstacles:
            data['obstacles'].append({
                'x': float(obs[0]),
                'y': float(obs[1]),
                'radius': float(obs[2])
            })

        # Write to file
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            raise IOError(f"Cannot write map file: {file_path}") from e

    def visualize(
        self,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        title: str = "NCBF Map",
        figsize: tuple = (8, 8)
    ):
        """
        Visualize the obstacle map.

        Args:
            save_path: Optional path to save visualization
            show: Whether to display the plot
            title: Plot title
            figsize: Figure size (width, height)
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot obstacles
        for obs in self.obstacles:
            x, y, radius = obs
            circle = plt.Circle((x, y), radius, color='red', alpha=0.6, label='Obstacle' if obs is self.obstacles[0] else "")
            ax.add_patch(circle)
            ax.plot(x, y, 'ro', markersize=4)

        # Set plot properties
        ax.set_xlim(0, self.workspace_size)
        ax.set_ylim(0, self.workspace_size)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'{title} - {len(self.obstacles)} Obstacles')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Add legend (only show once)
        if self.obstacles:
            ax.legend()

        # Add text info
        info_text = f"Workspace: {self.workspace_size}×{self.workspace_size}m\nObstacles: {len(self.obstacles)}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Save or show
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Map visualization saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def get_info(self) -> dict:
        """
        Get map information.

        Returns:
            Dictionary with map metadata
        """
        if not self.obstacles:
            return {
                'num_obstacles': 0,
                'workspace_size': self.workspace_size,
                'min_radius': 0.0,
                'max_radius': 0.0,
                'avg_radius': 0.0
            }

        radii = [obs[2] for obs in self.obstacles]
        return {
            'num_obstacles': len(self.obstacles),
            'workspace_size': self.workspace_size,
            'min_radius': min(radii),
            'max_radius': max(radii),
            'avg_radius': np.mean(radii)
        }

    def __len__(self) -> int:
        """Return number of obstacles."""
        return len(self.obstacles)

    def __repr__(self) -> str:
        """String representation."""
        info = self.get_info()
        return (f"NCBFMap(obstacles={info['num_obstacles']}, "
                f"workspace={info['workspace_size']}m, "
                f"radius_range=[{info['min_radius']:.2f}, {info['max_radius']:.2f}])")

    def judge_unicycle_safety(self, state: np.ndarray, unicycle_config) -> int:
        """
        Judge safety of a state based on UnicycleConfig parameters.

        Safety rule: A state is safe if distance to all obstacles is larger than
        unicycle_config.safety_radius + obstacle_radius. No extra safe margin added.

        Args:
            state: State vector [x, y, theta] or just [x, y]
            unicycle_config: UnicycleConfig instance with safety parameters

        Returns:
            1 if safe, 0 if unsafe
        """
        # Extract position (first 2 elements if 3D state)
        if len(state) >= 2:
            robot_pos = state[:2]
        else:
            raise ValueError("State must have at least 2 dimensions [x, y]")

        # Check distance to all obstacles
        for obs in self.obstacles:
            obs_x, obs_y, obs_radius = obs
            obs_center = np.array([obs_x, obs_y])

            # Distance from robot to obstacle center
            distance_to_center = np.linalg.norm(robot_pos - obs_center)

            # Safety threshold: safety_radius + obstacle_radius (no extra margin)
            safety_threshold = unicycle_config.safety_radius + obs_radius

            # If within safety threshold, state is unsafe
            if distance_to_center <= safety_threshold:
                return 0  # Unsafe

        return 1  # Safe (no obstacles within safety threshold)

    def generate_training_data(self, judge_safety_func: Callable, config, num_samples: int = 10000,
                              seed: Optional[int] = None, obstacle_focus_ratio: float = 0.3,
                              min_unsafe_ratio: float = 0.2, save_path: Optional[Union[str, Path]] = None) -> dict:
        """
        Generate labeled training data using safety judgment function with improved balance.

        This method implements a multi-stage sampling strategy:
        1. Boundary-focused sampling (40%): Samples concentrated around safety boundaries
        2. Uniform sampling (30%): Global coverage of the workspace
        3. Obstacle-focused sampling (30%): Detailed sampling around obstacles

        Args:
            judge_safety_func: Function that takes (state, config) and returns safety label
            config: Configuration object (e.g., UnicycleConfig) with safety parameters
            num_samples: Number of data points to generate (default: 10000)
            seed: Random seed for reproducibility
            obstacle_focus_ratio: Ratio of samples focused around obstacles (0-1)
            min_unsafe_ratio: Minimum ratio of unsafe samples to ensure balance (default: 0.2)
            save_path: Optional path to save the generated data

        Returns:
            Dictionary containing generated data and metadata
        """
        if seed is not None:
            np.random.seed(seed)

        print(f"🧪 Generating {num_samples} training data points...")
        print(f"📊 Obstacle focus ratio: {obstacle_focus_ratio:.1f}")
        print(f"📊 Minimum unsafe ratio: {min_unsafe_ratio:.1f}")
        print(f"🎯 Safety judgment: {judge_safety_func.__name__}")

        # Calculate theoretical safety area ratio for better sampling balance
        workspace_area = self.workspace_size ** 2
        total_safety_area = 0
        for obs in self.obstacles:
            _, _, obs_radius = obs
            safety_area = np.pi * (obs_radius + config.robot_radius + config.safety_radius) ** 2
            total_safety_area += safety_area

        theoretical_unsafe_ratio = min(total_safety_area / workspace_area, 0.5)  # Cap at 50%
        target_unsafe_ratio = max(theoretical_unsafe_ratio, min_unsafe_ratio)
        target_safe_ratio = 1.0 - target_unsafe_ratio

        print(f"📊 Theoretical unsafe ratio: {theoretical_unsafe_ratio:.1f}")
        print(f"📊 Target safe ratio: {target_safe_ratio:.1f}")
        print(f"📊 Target unsafe ratio: {target_unsafe_ratio:.1f}")

        # Initialize data arrays
        states = np.zeros((num_samples, 3))  # [x, y, theta]
        labels = np.zeros((num_samples, 1), dtype=np.int32)  # 1=safe, 0=unsafe

        # Multi-stage sampling strategy for better balance
        num_boundary_samples = int(num_samples * 0.4)  # 40% focused on boundaries
        num_uniform_samples = int(num_samples * 0.3)   # 30% uniform
        num_obstacle_samples = int(num_samples * 0.3)  # 30% around obstacles

        print(f"📍 Boundary-focused sampling: {num_boundary_samples} points")
        print(f"📍 Uniform sampling: {num_uniform_samples} points")
        print(f"📍 Obstacle-focused sampling: {num_obstacle_samples} points")

        current_idx = 0

        # 1. Boundary-focused sampling (most important for learning)
        print("🎯 Generating boundary-focused samples...")
        boundary_count = 0
        safe_boundary_count = 0
        unsafe_boundary_count = 0

        while boundary_count < num_boundary_samples:
            # Select random obstacle
            obs_idx = np.random.randint(0, len(self.obstacles))
            obs_x, obs_y, obs_radius = self.obstacles[obs_idx]

            # Sample around safety boundary with Gaussian distribution
            safety_threshold = obs_radius + config.robot_radius + config.safety_radius
            boundary_scale = safety_threshold * 0.3  # Narrow band around boundary

            # Sample in a ring around the safety boundary
            angle = np.random.uniform(0, 2*np.pi)
            distance_from_boundary = np.random.normal(0, boundary_scale)
            distance_from_center = safety_threshold + distance_from_boundary

            x = obs_x + distance_from_center * np.cos(angle)
            y = obs_y + distance_from_center * np.sin(angle)
            theta = np.random.uniform(-np.pi, np.pi)

            # Ensure within workspace bounds
            x = np.clip(x, 0, self.workspace_size)
            y = np.clip(y, 0, self.workspace_size)

            state = np.array([x, y, theta])
            safety_label = judge_safety_func(state, config)

            # Accept samples near boundaries (both safe and unsafe)
            actual_distance = np.linalg.norm(state[:2] - np.array([obs_x, obs_y]))
            distance_to_boundary = abs(actual_distance - safety_threshold)

            if distance_to_boundary < boundary_scale * 2:  # Accept samples near boundary
                states[current_idx] = state
                labels[current_idx] = safety_label
                current_idx += 1
                boundary_count += 1

                if safety_label == 1:
                    safe_boundary_count += 1
                else:
                    unsafe_boundary_count += 1

        print(f"✅ Boundary samples: {boundary_count} (safe: {safe_boundary_count}, unsafe: {unsafe_boundary_count})")

        # 2. Uniform sampling (for global coverage)
        print("🌍 Generating uniform samples...")
        uniform_count = 0
        safe_uniform_count = 0
        unsafe_uniform_count = 0

        while uniform_count < num_uniform_samples:
            x = np.random.uniform(0, self.workspace_size)
            y = np.random.uniform(0, self.workspace_size)
            theta = np.random.uniform(-np.pi, np.pi)

            state = np.array([x, y, theta])
            safety_label = judge_safety_func(state, config)

            states[current_idx] = state
            labels[current_idx] = safety_label
            current_idx += 1
            uniform_count += 1

            if safety_label == 1:
                safe_uniform_count += 1
            else:
                unsafe_uniform_count += 1

        print(f"✅ Uniform samples: {uniform_count} (safe: {safe_uniform_count}, unsafe: {unsafe_uniform_count})")

        # 3. Obstacle-focused sampling (for detailed obstacle learning)
        print("🎯 Generating obstacle-focused samples...")
        obstacle_count = 0
        safe_obstacle_count = 0
        unsafe_obstacle_count = 0

        while obstacle_count < num_obstacle_samples:
            # Select random obstacle
            obs_idx = np.random.randint(0, len(self.obstacles))
            obs_x, obs_y, obs_radius = self.obstacles[obs_idx]

            # Sample around obstacle with Gaussian distribution
            scale = obs_radius * 1.5  # Wider spread for obstacle focus
            x = np.random.normal(obs_x, scale)
            y = np.random.normal(obs_y, scale)
            theta = np.random.uniform(-np.pi, np.pi)

            # Ensure within workspace bounds
            x = np.clip(x, 0, self.workspace_size)
            y = np.clip(y, 0, self.workspace_size)

            state = np.array([x, y, theta])
            safety_label = judge_safety_func(state, config)

            states[current_idx] = state
            labels[current_idx] = safety_label
            current_idx += 1
            obstacle_count += 1

            if safety_label == 1:
                safe_obstacle_count += 1
            else:
                unsafe_obstacle_count += 1

        print(f"✅ Obstacle samples: {obstacle_count} (safe: {safe_obstacle_count}, unsafe: {unsafe_obstacle_count})")

        # Final statistics
        total_safe = safe_boundary_count + safe_uniform_count + safe_obstacle_count
        total_unsafe = unsafe_boundary_count + unsafe_uniform_count + unsafe_obstacle_count
        actual_safe_ratio = total_safe / num_samples
        actual_unsafe_ratio = total_unsafe / num_samples

        print(f"\n📊 Final statistics:")
        print(f"   Total samples: {num_samples}")
        print(f"   Safe samples: {total_safe} ({100*actual_safe_ratio:.1f}%)")
        print(f"   Unsafe samples: {total_unsafe} ({100*actual_unsafe_ratio:.1f}%)")
        print(f"   Actual unsafe ratio: {actual_unsafe_ratio:.3f} (target: {target_unsafe_ratio:.3f})")

        # Create result dictionary with detailed breakdown
        result = {
            'states': states,
            'labels': labels,
            'num_samples': num_samples,
            'num_safe': total_safe,
            'num_unsafe': total_unsafe,
            'actual_safe_ratio': actual_safe_ratio,
            'actual_unsafe_ratio': actual_unsafe_ratio,
            'obstacle_focus_ratio': obstacle_focus_ratio,
            'seed': seed,
            'sampling_breakdown': {
                'boundary': {'total': boundary_count, 'safe': safe_boundary_count, 'unsafe': unsafe_boundary_count},
                'uniform': {'total': uniform_count, 'safe': safe_uniform_count, 'unsafe': unsafe_uniform_count},
                'obstacle': {'total': obstacle_count, 'safe': safe_obstacle_count, 'unsafe': unsafe_obstacle_count}
            },
            'config_params': {
                'safety_radius': config.safety_radius,
                'robot_radius': config.robot_radius
            },
            'map_info': self.get_info(),
            'generation_timestamp': datetime.now().isoformat()
        }

        # Save if path provided
        if save_path:
            self.save_training_data(result, save_path)

        return result

    def save_training_data(self, data_dict: dict, file_path: Union[str, Path]):
        """
        Save training data to HDF5 format for efficient storage and loading.

        Args:
            data_dict: Dictionary containing training data and metadata
            file_path: Path to save the data file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"💾 Saving training data to: {file_path}")

        # Save as HDF5 format for efficient storage
        with h5py.File(file_path, 'w') as hf:
            # Create main data group
            data_group = hf.create_group('data')
            data_group.create_dataset('states', data=data_dict['states'], compression='gzip', compression_opts=4)
            data_group.create_dataset('labels', data=data_dict['labels'], compression='gzip', compression_opts=4)

            # Create metadata group
            meta_group = hf.create_group('metadata')
            meta_group.attrs['num_samples'] = data_dict['num_samples']
            meta_group.attrs['num_safe'] = data_dict['num_safe']
            meta_group.attrs['num_unsafe'] = data_dict['num_unsafe']
            meta_group.attrs['obstacle_focus_ratio'] = data_dict['obstacle_focus_ratio']
            meta_group.attrs['seed'] = data_dict['seed'] if data_dict['seed'] is not None else -1
            meta_group.attrs['generation_timestamp'] = data_dict['generation_timestamp']
            meta_group.attrs['map_info_json'] = json.dumps(data_dict['map_info'])

        print(f"✅ Training data saved successfully!")
        print(f"📁 File size: {file_path.stat().st_size / 1024:.1f} KB")

    def load_training_data(self, file_path: Union[str, Path]) -> dict:
        """
        Load training data from HDF5 format.

        Args:
            file_path: Path to the data file

        Returns:
            Dictionary containing training data and metadata
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Training data file not found: {file_path}")

        print(f"📂 Loading training data from: {file_path}")

        # Load HDF5 file
        with h5py.File(file_path, 'r') as hf:
            # Load data group
            data_group = hf['data']
            states = data_group['states'][:]
            labels = data_group['labels'][:]

            # Load metadata group
            meta_group = hf['metadata']
            num_samples = meta_group.attrs['num_samples']
            num_safe = meta_group.attrs['num_safe']
            num_unsafe = meta_group.attrs['num_unsafe']
            obstacle_focus_ratio = meta_group.attrs['obstacle_focus_ratio']
            seed = meta_group.attrs['seed']
            generation_timestamp = meta_group.attrs['generation_timestamp']
            map_info_json = meta_group.attrs['map_info_json']

            # Reconstruct dictionary
            result = {
                'states': states,
                'labels': labels,
                'num_samples': num_samples,
                'num_safe': num_safe,
                'num_unsafe': num_unsafe,
                'obstacle_focus_ratio': obstacle_focus_ratio,
                'seed': seed if seed != -1 else None,
                'map_info': json.loads(map_info_json),
                'generation_timestamp': generation_timestamp
            }

        print(f"✅ Training data loaded successfully!")
        print(f"📊 States shape: {result['states'].shape}")
        print(f"📊 Labels shape: {result['labels'].shape}")

        return result


def create_moderate_map(seed: int = 42, workspace_size: float = 8.0) -> NCBFMap:
    """
    Create a moderate density map using the existing generation function.

    Args:
        seed: Random seed for reproducibility
        workspace_size: Size of square workspace

    Returns:
        NCBFMap instance with generated obstacles
    """
    # Import here to avoid circular imports
    from .map_generation import generate_moderate_map

    obstacles = generate_moderate_map(seed=seed, workspace_size=workspace_size)
    return NCBFMap(obstacles=obstacles, workspace_size=workspace_size)


def load_map(file_path: Union[str, Path]) -> NCBFMap:
    """
    Load map from JSON file.

    Args:
        file_path: Path to JSON map file

    Returns:
        NCBFMap instance
    """
    return NCBFMap(map_file=file_path)