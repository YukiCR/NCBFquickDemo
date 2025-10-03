#!/usr/bin/env python3
"""
Command-line interface for training Neural Control Barrier Functions (NCBF).

This script provides a comprehensive interface for training NCBF models with:
- Configurable training parameters
- GPU acceleration
- Real-time monitoring
- Checkpoint management
- Contour visualization
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ncbf.training.ncbf_trainer import NCBFTrainer
from ncbf.configs.ncbf_config import NCBFConfig, create_default_ncbf_config, create_large_ncbf_config, create_small_ncbf_config
from configs.unicycle_config import UnicycleConfig
from ncbf.maps import load_map


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Neural Control Barrier Functions (NCBF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default configuration
  python train_ncbf.py --data map1/training_data_large.h5

  # Train with custom parameters
  python train_ncbf.py --data map1/training_data_large.h5 \
                       --epochs 100 --batch_size 256 --lr 1e-3

  # Train with large model configuration
  python train_ncbf.py --data map1/training_data_large.h5 --config large

  # Resume training from checkpoint
  python train_ncbf.py --data map1/training_data_large.h5 \
                       --resume checkpoints/ncbf/checkpoint_epoch_50.pt

  # Train with visualization
  python train_ncbf.py --data map1/training_data_large.h5 --visualize

  # Quick training with small model
  python train_ncbf.py --data map1/training_data_large.h5 --config small \
                       --epochs 10 --no-visualize
        """
    )

    # Data parameters
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to training data HDF5 file (relative to map_files/ or absolute)'
    )

    parser.add_argument(
        '--map', '-m',
        type=str,
        default=None,
        help='Path to map JSON file (optional, for visualization)'
    )

    # Model configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        choices=['default', 'large', 'small'],
        default='default',
        help='Model configuration preset (default: default)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint file for resuming training'
    )

    # Training parameters
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )

    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=None,
        help='Training batch size (overrides config)'
    )

    parser.add_argument(
        '--lr', '-l',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )

    parser.add_argument(
        '--validation_split',
        type=float,
        default=None,
        help='Fraction of data for validation (overrides config)'
    )

    # Performance parameters
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'cuda:0'],
        help='Device to use for training (default: auto)'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of DataLoader workers (overrides config)'
    )

    parser.add_argument(
        '--no_mixed_precision',
        action='store_true',
        help='Disable mixed precision training'
    )

    # Monitoring parameters
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Show training plots and contour visualization'
    )

    parser.add_argument(
        '--no_visualize',
        action='store_true',
        help='Suppress training plots (useful for remote training)'
    )

    parser.add_argument(
        '--log_frequency',
        type=int,
        default=None,
        help='Log every N batches (overrides config)'
    )

    parser.add_argument(
        '--plot_frequency',
        type=int,
        default=None,
        help='Plot every N epochs (overrides config)'
    )

    parser.add_argument(
        '--save_frequency',
        type=int,
        default=None,
        help='Save checkpoint every N epochs (overrides config)'
    )

    # Loss function parameters
    parser.add_argument(
        '--classification_weight',
        type=float,
        default=None,
        help='Weight for classification loss (overrides config)'
    )

    parser.add_argument(
        '--barrier_weight',
        type=float,
        default=None,
        help='Weight for barrier loss (overrides config)'
    )

    parser.add_argument(
        '--margin',
        type=float,
        default=None,
        help='Margin for hinge loss (overrides config)'
    )

    # Output parameters
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=None,
        help='Output directory for models and logs (overrides config)'
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


def create_config_from_args(args) -> NCBFConfig:
    """Create NCBFConfig from command-line arguments."""
    # Start with base configuration
    if args.config == 'default':
        config = create_default_ncbf_config()
    elif args.config == 'large':
        config = create_large_ncbf_config()
    elif args.config == 'small':
        config = create_small_ncbf_config()
    else:
        config = create_default_ncbf_config()

    # Override with command-line arguments
    if args.epochs is not None:
        config.max_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.validation_split is not None:
        config.validation_split = args.validation_split
    if args.device is not None:
        config.device = args.device
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.no_mixed_precision:
        config.mixed_precision = False
    if args.log_frequency is not None:
        config.log_frequency = args.log_frequency
    if args.plot_frequency is not None:
        config.plot_frequency = args.plot_frequency
    if args.save_frequency is not None:
        config.save_frequency = args.save_frequency
    if args.classification_weight is not None:
        config.classification_weight = args.classification_weight
    if args.barrier_weight is not None:
        config.barrier_weight = args.barrier_weight
    if args.margin is not None:
        config.margin = args.margin
    if args.output_dir is not None:
        config.checkpoint_dir = args.output_dir
    if args.no_visualize:
        config.show_training_plots = False

    return config


def main():
    """Main training function."""
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
            print("🚀 NCBF Training Script")
            print("=" * 60)

        # Resolve file paths
        base_dir = Path(__file__).parent.parent / 'map_files'
        data_path = resolve_file_path(args.data, base_dir)

        if args.map:
            map_path = resolve_file_path(args.map, base_dir)
        else:
            # Try to find corresponding map file
            map_name = data_path.stem.split('_')[0]  # Extract map name from data file
            potential_map_path = data_path.parent / f"{map_name}.json"
            map_path = potential_map_path if potential_map_path.exists() else None

        # Create configuration
        config = create_config_from_args(args)

        if not args.quiet:
            print(f"📊 Data file: {data_path}")
            print(f"🗺️  Map file: {map_path}")
            print(f"⚙️  Configuration: {config}")

        # Load unicycle model for dynamics (needed for barrier loss)
        if map_path:
            logger.info("Loading unicycle model with map data...")
            ncbf_map = load_map(map_path)
            # Create UnicycleConfig with map parameters
            unicycle_config = UnicycleConfig(
                safety_radius=0.2,  # Default safety radius
                cbf_alpha=0.6,      # Default CBF alpha
                max_control_norm=2.0  # Default max control
            )
            # Create UnicycleModel from config for proper barrier loss computation
            try:
                # Try importing from current script location first
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from models.unicycle_model import UnicycleModel
                unicycle_model = UnicycleModel(config=unicycle_config)
                logger.info(f"✅ Unicycle model loaded with {len(ncbf_map.obstacles)} obstacles")
            except ImportError as e1:
                try:
                    # Fallback: try importing from work directory
                    sys.path.insert(0, '/home/chengrui/wk/NCBFquickDemo/work')
                    from models.unicycle_model import UnicycleModel
                    unicycle_model = UnicycleModel(config=unicycle_config)
                    logger.info(f"✅ Unicycle model loaded with {len(ncbf_map.obstacles)} obstacles (fallback import)")
                except ImportError as e2:
                    logger.error(f"Failed to import UnicycleModel: {e1} and {e2}")
                    logger.warning("Barrier loss will be disabled due to import failure")
                    unicycle_model = None
        else:
            logger.warning("No map file provided - barrier loss will be simplified")
            unicycle_model = None

        # Initialize trainer
        trainer = NCBFTrainer(config, unicycle_model=unicycle_model)

        # Start training
        logger.info("🚀 Starting NCBF training...")
        training_history = trainer.train(
            data_path=str(data_path),
            resume_from=args.resume
        )

        # Generate final visualization
        if config.show_training_plots:
            logger.info("Generating final contour visualization...")
            trainer.visualize_contours()

        # Save final report
        logger.info("✅ Training completed successfully!")

        if not args.quiet:
            final_train_loss = training_history['train_loss'][-1]
            final_val_loss = training_history['val_loss'][-1]
            print(f"\n📈 Final Results:")
            print(f"   Training Loss: {final_train_loss:.6f}")
            print(f"   Validation Loss: {final_val_loss:.6f}")
            print(f"   Models saved to: {config.checkpoint_dir}")

        return 0

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())