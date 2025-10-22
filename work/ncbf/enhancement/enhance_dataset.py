#!/usr/bin/env python3
"""
Command-line interface for dataset enhancement using pseudo-negative data generation.

This script provides a unified interface for enhancing safe-only datasets with
pseudo-negative samples using three methods:
- Anomaly Detection (AD): Use state distribution to identify out-of-distribution states
- Complement: Use geometric distance from safe states to generate negatives
- iDBF: Use control distribution and forward simulation to generate unsafe states

All methods follow the same enhancement pipeline:
load safe-only dataset → fit enhancer → generate pseudo-negatives → create balanced dataset
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ncbf.enhancement.enhancer_base import EnhancerFactory
from ncbf.enhancement.enhancer_config import EnhancerConfig, ADConfig
from ncbf.enhancement import enhancer_utils as utils


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhance safe-only datasets with pseudo-negative samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic AD enhancement with default parameters
  python enhance_dataset.py --input-data map1/training_data_safe_only.h5 --method ad

  # AD enhancement with custom parameters
  python enhance_dataset.py --input-data map1/training_data_safe_only.h5 \
                           --method ad --target-ratio 0.4 \
                           --ad-method ocsvm --nu 0.1 --kernel rbf \
                           --threshold-quantile 0.05

  # Complement method with geometric parameters
  python enhance_dataset.py --input-data map1/training_data_safe_only.h5 \
                           --method complement --target-ratio 0.3 \
                           --workspace-padding 0.5

  # iDBF method with behavior cloning parameters
  python enhance_dataset.py --input-data map1/training_data_safe_only.h5 \
                           --method idbf --target-ratio 0.3 \
                           --bc-epochs 100 --ood-sigma 2.0

  # Custom output path
  python enhance_dataset.py --input-data map1/training_data_safe_only.h5 \
                           --method ad --output my_enhanced_dataset.h5 \
                           --seed 42
        """
    )

    # Required parameters
    parser.add_argument(
        '--input-data', '-i',
        type=str,
        required=True,
        help='Path to input safe-only dataset HDF5 file (relative to map_files/ or absolute)'
    )

    parser.add_argument(
        '--method', '-m',
        type=str,
        choices=['ad', 'complement', 'idbf'],
        required=True,
        default=None,
        help='Enhancement method (overrides config default)'
    )

    # Core enhancement parameters
    parser.add_argument(
        '--target-ratio', '-r',
        type=float,
        default=None,
        help='Target ratio of pseudo-negative to safe samples (overrides config default)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for enhanced dataset (default: input_data_enhanced.h5)'
    )

    parser.add_argument(
        '--workspace-padding',
        type=float,
        default=None,
        help='Padding beyond data bounds for sampling (overrides config default)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (overrides config default)'
    )

    # Anomaly Detection (AD) specific parameters
    ad_group = parser.add_argument_group('AD Method Parameters')
    ad_group.add_argument(
        '--ad-method',
        type=str,
        choices=['ocsvm', 'isolation_forest', 'local_outlier_factor', 'autoencoder'],
        default=None,
        help='Specific AD algorithm (overrides config default)'
    )

    ad_group.add_argument(
        '--kernel',
        type=str,
        choices=['rbf', 'linear', 'poly', 'sigmoid'],
        default=None,
        help='Kernel type for OneClassSVM (overrides config default)'
    )

    ad_group.add_argument(
        '--nu',
        type=float,
        default=None,
        help='Expected proportion of outliers for OneClassSVM (overrides config default)'
    )

    ad_group.add_argument(
        '--gamma',
        type=str,
        default=None,
        help='Kernel coefficient for OneClassSVM (overrides config default)'
    )

    ad_group.add_argument(
        '--threshold-quantile',
        type=float,
        default=None,
        help='Quantile for OOD threshold determination (overrides config default)'
    )

    ad_group.add_argument(
        '--use-full-state',
        action='store_true',
        help='Use full state vs position-only for OOD detection'
    )

    ad_group.add_argument(
        '--contamination',
        type=float,
        default=None,
        help='Expected proportion of outliers for IsolationForest (overrides config default)'
    )

    ad_group.add_argument(
        '--n-neighbors',
        type=int,
        default=None,
        help='Number of neighbors for LocalOutlierFactor (overrides config default)'
    )

    # Complement method specific parameters
    complement_group = parser.add_argument_group('Complement Method Parameters')
    complement_group.add_argument(
        '--distance-threshold',
        type=float,
        default=None,
        help='Distance threshold for initial unsafe labeling (default: auto-compute)'
    )

    complement_group.add_argument(
        '--inflation-radius',
        type=float,
        default=None,
        help='Radius for unsafe set inflation (default: auto-compute)'
    )

    # iDBF method specific parameters
    idbf_group = parser.add_argument_group('iDBF Method Parameters')
    idbf_group.add_argument(
        '--bc-epochs',
        type=int,
        default=None,
        help='Behavior cloning training epochs (overrides config default)'
    )

    idbf_group.add_argument(
        '--ood-sigma',
        type=float,
        default=None,
        help='Standard deviation multiplier for OOD control sampling (overrides config default)'
    )

    idbf_group.add_argument(
        '--forward-steps',
        type=int,
        default=None,
        help='Number of forward simulation steps (overrides config default)'
    )

    # Output options
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip saving the enhanced dataset (for testing)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output'
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


def create_config_from_args(args) -> EnhancerConfig:
    """Create enhancer configuration from command-line arguments."""
    # Create base configuration with only non-None values to preserve config defaults
    base_config = {
        'input_dataset_path': str(args.input_data),
    }

    # Only override config defaults if explicitly provided
    if args.method is not None:
        base_config['method'] = args.method
    if args.target_ratio is not None:
        base_config['target_ratio'] = args.target_ratio
    if args.workspace_padding is not None:
        base_config['workspace_padding'] = args.workspace_padding
    if args.seed is not None:
        base_config['random_seed'] = args.seed

    # Method-specific configuration
    if args.method == 'ad':
        config = ADConfig(**base_config)

        # Override AD-specific parameters only if explicitly provided
        if args.ad_method is not None:
            config.ad_method = args.ad_method
        if args.kernel is not None:
            config.kernel = args.kernel
        if args.nu is not None:
            config.nu = args.nu
        if args.gamma is not None:
            gamma_lower = args.gamma.lower()
            if gamma_lower in ['scale', 'auto']:
                config.gamma = gamma_lower
            else:
                try:
                    config.gamma = float(args.gamma)
                except ValueError:
                    raise ValueError(
                        f"Invalid gamma value: {args.gamma}. "
                        "Must be 'scale', 'auto' or a float value."
                    )
        if args.threshold_quantile is not None:
            config.threshold_quantile = args.threshold_quantile
        if args.use_full_state:
            config.use_full_state = args.use_full_state
        if args.contamination is not None:
            config.contamination = args.contamination
        if args.n_neighbors is not None:
            config.n_neighbors = args.n_neighbors

    elif args.method == 'complement':
        # For now, use base config - complement-specific parameters will be added
        # when ComplementEnhancer is implemented
        config = EnhancerConfig(**base_config)

    elif args.method == 'idbf':
        # For now, use base config - iDBF-specific parameters will be added
        # when iDBFEnhancer is implemented
        config = EnhancerConfig(**base_config)

        # Override iDBF-specific parameters only if explicitly provided
        if args.bc_epochs is not None:
            # This will be added when iDBFEnhancer is implemented
            pass
        if args.ood_sigma is not None:
            # This will be added when iDBFEnhancer is implemented
            pass
        if args.forward_steps is not None:
            # This will be added when iDBFEnhancer is implemented
            pass

    else:
        config = EnhancerConfig(**base_config)

    return config


def main():
    """Main enhancement function."""
    try:
        # Parse arguments
        args = parse_arguments()

        # Setup logging
        if args.quiet:
            logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        logger = logging.getLogger(__name__)

        # Welcome message
        if not args.quiet:
            print("🔧 Dataset Enhancement Tool")
            print("=" * 60)

        # Resolve file paths
        base_dir = Path(__file__).parent.parent.parent / 'map_files'
        input_path = resolve_file_path(args.input_data, base_dir)

        # Create configuration early to use config defaults
        config = create_config_from_args(args)

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            # Generate default output name
            input_name = input_path.stem
            output_path = input_path.parent / f"{input_name}_enhanced.h5"

        if not args.quiet:
            print(f"Input dataset: {input_path}")
            print(f"Enhancement method: {config.method.upper()}")
            print(f"Target ratio: {config.target_ratio}")
            print(f"Output path: {output_path}")
            print("-" * 60)

        if not args.quiet:
            print("Configuration created successfully")

        # Create enhancer using factory
        factory = EnhancerFactory()
        enhancer = factory.create_enhancer(config)

        if not args.quiet:
            print(f"Created {config.method.upper()} enhancer")

        # Load original dataset for visualization
        original_data = utils.load_hdf5_dataset(str(input_path))

        # Run enhancement pipeline
        if not args.quiet:
            print("Fitting enhancer to safe data...")
        enhancer.fit()

        if not args.quiet:
            print("Generating pseudo-negative samples...")

        # Determine number of pseudo-negatives to generate
        num_safe = np.sum(original_data['labels'] >= 0)
        num_pseudo_negatives = int(num_safe * config.target_ratio)

        if not args.quiet:
            print(f"Generating {num_pseudo_negatives} pseudo-negative samples...")

        # Generate pseudo-negatives
        pseudo_negatives = enhancer.generate_pseudo_negatives(num_pseudo_negatives)

        if not args.quiet:
            print(f"Generated {len(pseudo_negatives)} pseudo-negative samples")

        # Create enhanced dataset
        if not args.quiet:
            print("Creating enhanced dataset...")

        enhanced_dataset_path = enhancer.enhance_dataset(str(output_path))

        if not args.quiet:
            print(f"Enhanced dataset saved to: {enhanced_dataset_path}")

        # Load enhanced dataset for statistics
        enhanced_data = utils.load_hdf5_dataset(enhanced_dataset_path)

        # Print statistics
        if not args.quiet:
            original_safe = np.sum(original_data['labels'] >= 0)
            enhanced_safe = np.sum(enhanced_data['labels'] == 1)
            enhanced_unsafe = np.sum(enhanced_data['labels'] == 0)

            print("\nEnhancement Statistics:")
            print(f"  Original safe samples: {original_safe}")
            print(f"  Enhanced safe samples: {enhanced_safe}")
            print(f"  Generated unsafe samples: {enhanced_unsafe}")
            print(f"  Total enhanced samples: {len(enhanced_data['labels'])}")
            print(f"  Actual ratio: {enhanced_unsafe / enhanced_safe:.3f}")

        # Visualization recommendation
        if not args.quiet:
            print("\n✅ Enhancement completed successfully!")
            print("\n💡 To visualize the enhanced dataset, run:")
            print(f"   python work/ncbf/maps/visualize_training_data.py --training_data {enhanced_dataset_path}")
            print("=" * 60)

    except Exception as e:
        logger.error(f"Enhancement failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()