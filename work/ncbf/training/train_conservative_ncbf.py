#!/usr/bin/env python3
"""
Command-line interface for training Conservative Neural Control Barrier Functions (NCBF).

This script provides a comprehensive interface for training conservative NCBF models with:
- Conservative loss using log-sum-exp approximation
- Two-phase training (pretraining + main training)
- Random control sampling for proceeding states
- Integration with existing unicycle model dynamics

Based on: Tabbara et al. 2025 - Learning Neural Control Barrier Functions from Offline Safe Demonstrations
"""

import sys
from pathlib import Path

# Add work directory for unicycle model imports (go up 3 levels from training/ncbf/training)
work_dir = Path(__file__).parent.parent.parent  # /home/chengrui/wk/NCBFquickDemo/work
sys.path.insert(0, str(work_dir))

# Add parent directory for ncbf package imports
parent_dir = work_dir.parent  # /home/chengrui/wk/NCBFquickDemo
sys.path.insert(0, str(parent_dir))

import argparse
import json
import logging
from typing import Optional

from ncbf.training.conservative_ncbf_trainer import ConservativeNCBFTrainer
from ncbf.training.ncbf_trainer import NCBFTrainer  # For baseline comparison
from ncbf.configs.ncbf_config import NCBFConfig, create_default_ncbf_config, create_large_ncbf_config, create_small_ncbf_config
from ncbf.maps import load_map

# Import unicycle model components (now that work directory is in path)
try:
    from configs.unicycle_config import UnicycleConfig
    from models.unicycle_model import UnicycleModel
    UNICYCLLE_IMPORT_SUCCESS = True
except ImportError as e:
    # Fallback with warning
    import warnings
    warnings.warn(f"Could not import unicycle model components: {e}. Conservative loss will be limited.")
    UnicycleConfig = None
    UnicycleModel = None
    UNICYCLLE_IMPORT_SUCCESS = False


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Conservative Neural Control Barrier Functions (NCBF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Conservative training with default parameters
  python train_conservative_ncbf.py --data map1/training_data_safe_only.h5 --config large

  # Conservative training with custom parameters
  python train_conservative_ncbf.py --data map1/training_data_safe_only.h5 \
                                    --conservative-weight 2.0 --temperature 0.05 \
                                    --num-random-controls 15 --config large

  # Two-phase training with pretraining
  python train_conservative_ncbf.py --data map1/training_data_safe_only.h5 \
                                    --enable-pretraining --pretrain-epochs 50 \
                                    --config large

  # Baseline comparison (no conservative loss)
  python train_conservative_ncbf.py --data map1/training_data_safe_only.h5 \
                                    --no-conservative --config large

  # Quick training with visualization
  python train_conservative_ncbf.py --data map1/training_data_safe_only.h5 \
                                    --config small --epochs 100 --visualize

    Note:
    - The map file is NOT optional for conservative training. You should always provide a map file
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
        help='Path to map JSON file (optional, for visualization and dynamics)'
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

    # Conservative NCBF parameters
    parser.add_argument(
        '--conservative-weight',
        type=float,
        default=None,
        help='Weight for conservative loss (overrides config)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help='Temperature parameter for log-sum-exp (overrides config)'
    )

    parser.add_argument(
        '--num-random-controls',
        type=int,
        default=None,
        help='Number of random controls to sample (overrides config)'
    )

    parser.add_argument(
        '--enable-pretraining',
        action='store_true',
        help='Enable two-phase training with pretraining'
    )

    parser.add_argument(
        '--pretrain-epochs',
        type=int,
        default=None,
        help='Number of pretraining epochs (overrides config)'
    )

    parser.add_argument(
        '--no-conservative',
        action='store_true',
        help='Disable conservative loss (baseline comparison)'
    )

    parser.add_argument(
        '--pretraining-only',
        action='store_true',
        help='Run only pretraining phase (no main training)'
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
        '--pretrain-lr',
        type=float,
        default=None,
        help='Pretraining learning rate (overrides config)'
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
    """Create NCBFConfig from command-line arguments with conservative parameters."""
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

    # Conservative NCBF parameters
    if args.conservative_weight is not None:
        config.conservative_weight = args.conservative_weight
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.num_random_controls is not None:
        config.num_random_controls = args.num_random_controls
    if args.enable_pretraining:
        config.enable_pretraining = True
    if args.pretrain_epochs is not None:
        config.pretrain_epochs = args.pretrain_epochs
    if args.pretrain_lr is not None:
        config.pretrain_lr = args.pretrain_lr

    return config


def main():
    """Main training function for conservative NCBF."""
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
            print("🛡️ Conservative NCBF Training Script")
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

        # Load unicycle model for dynamics (needed for conservative loss)
        if map_path and UnicycleModel is not None and UnicycleConfig is not None:
            logger.info("Loading unicycle model with map data...")
            ncbf_map = load_map(map_path)
            # Create UnicycleConfig with map parameters
            unicycle_config = UnicycleConfig(
                safety_radius=0.2,  # Default safety radius
                cbf_alpha=0.5,      # Default CBF alpha
                max_control_norm=2.0  # Default max control
            )
            # Create UnicycleModel from config for proper conservative loss computation
            unicycle_model = UnicycleModel(config=unicycle_config)
            logger.info(f"✅ Unicycle model loaded with {len(ncbf_map.obstacles)} obstacles")
        else:
            if map_path is None:
                logger.warning("No map file provided - conservative loss will be simplified")
            else:
                logger.warning("Unicycle model components not available - conservative loss will be simplified")
            unicycle_model = None

        # Initialize trainer
        if args.no_conservative:
            logger.info("🔧 Using standard NCBF trainer (baseline comparison)")
            trainer = NCBFTrainer(config, unicycle_model=unicycle_model)
        else:
            logger.info("🛡️ Using conservative NCBF trainer")
            trainer = ConservativeNCBFTrainer(config, unicycle_model=unicycle_model)

        # Start training
        if args.pretraining_only:
            # Pretraining-only mode
            logger.info("🔄 Running pretraining only (no main training)...")

            # Ensure pretraining is enabled
            config.enable_pretraining = True
            if config.pretrain_epochs <= 0:
                config.pretrain_epochs = 50  # Default pretraining epochs if not specified

            # Run pretraining (this will automatically save the checkpoint)
            trainer.pretrain_negative_landscape(data_path=str(data_path))

            training_history = {'train_loss': [], 'val_loss': []}  # Empty history for pretraining-only

        else:
            # Normal training modes
            logger.info("🚀 Starting NCBF training...")

            # Two-phase training if enabled
            if config.enable_pretraining and config.pretrain_epochs > 0:
                training_history = trainer.train_with_pretraining(
                    data_path=str(data_path),
                    resume_from=args.resume
                )
            else:
                # Standard training
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

        if not args.quiet and not args.pretraining_only:
            final_train_loss = training_history['train_loss'][-1]
            final_val_loss = training_history['val_loss'][-1]
            print(f"\n📈 Final Results:")
            print(f"   Training Loss: {final_train_loss:.6f}")
            print(f"   Validation Loss: {final_val_loss:.6f}")
            print(f"   Models saved to: {config.checkpoint_dir}")

            # Add conservative-specific results
            if not args.no_conservative and 'conservative_loss' in training_history:
                final_conservative_loss = training_history['conservative_loss'][-1]
                print(f"   Conservative Loss: {final_conservative_loss:.6f}")

        return 0

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


# Convenience functions for different configurations
def create_conservative_ncbf_config(**kwargs) -> NCBFConfig:
    """
    Create NCBFConfig with conservative parameters.

    Args:
        **kwargs: Conservative parameters to override

    Returns:
        NCBFConfig with conservative settings
    """
    config = create_large_ncbf_config()

    # Set conservative defaults
    config.conservative_weight = kwargs.get('conservative_weight', 1.0)
    config.temperature = kwargs.get('temperature', 0.1)
    config.num_random_controls = kwargs.get('num_random_controls', 10)
    config.enable_pretraining = kwargs.get('enable_pretraining', False)
    config.pretrain_epochs = kwargs.get('pretrain_epochs', 0)

    return config


def train_conservative_ncbf_quick(data_path: str, output_dir: str, epochs: int = 100) -> ConservativeNCBFTrainer:
    """
    Quick conservative NCBF training for testing.

    Args:
        data_path: Path to training data
        output_dir: Output directory
        epochs: Number of epochs

    Returns:
        Trained ConservativeNCBFTrainer instance
    """
    config = create_conservative_ncbf_config(
        conservative_weight=1.0,
        temperature=0.1,
        num_random_controls=5,
        max_epochs=epochs
    )
    config.checkpoint_dir = output_dir
    config.show_training_plots = False

    # Create trainer and train
    trainer = ConservativeNCBFTrainer(config)
    trainer.train(data_path)

    return trainer


def train_conservative_ncbf_full(data_path: str, output_dir: str, epochs: int = 3000) -> ConservativeNCBFTrainer:
    """
    Full conservative NCBF training with optimal parameters.

    Args:
        data_path: Path to training data
        output_dir: Output directory
        epochs: Number of epochs

    Returns:
        Trained ConservativeNCBFTrainer instance
    """
    config = create_conservative_ncbf_config(
        conservative_weight=2.0,
        temperature=0.05,
        num_random_controls=15,
        max_epochs=epochs,
        enable_pretraining=True,
        pretrain_epochs=50
    )
    config.checkpoint_dir = output_dir

    # Create trainer and train with two phases
    trainer = ConservativeNCBFTrainer(config)
    trainer.train_with_pretraining(data_path)

    return trainer


# Quick test function
def test_conservative_loss():
    """Test conservative loss computation."""
    print("🧪 Testing Conservative Loss Implementation")

    # Create minimal config
    config = create_conservative_ncbf_config(
        conservative_weight=1.0,
        temperature=0.1,
        num_random_controls=5
    )

    # Create trainer
    trainer = ConservativeNCBFTrainer(config)

    # Test with dummy data
    batch_size = 4
    safe_states = torch.randn(batch_size, 3)  # [px, py, theta]

    print(f"📊 Testing with batch size: {batch_size}")
    print(f"🎯 Temperature: {config.temperature}")
    print(f"🎲 Random controls: {config.num_random_controls}")

    # Compute conservative loss
    conservative_loss, h_proceeding = trainer.compute_conservative_loss(safe_states)

    print(f"✅ Conservative loss: {conservative_loss.item():.6f}")
    print(f"📈 h(x') shape: {h_proceeding.shape}")
    print(f"📊 h(x') range: [{h_proceeding.min().item():.4f}, {h_proceeding.max().item():.4f}]")
    print(f"🌡️  Log-sum-exp values: {trainer.temperature * torch.logsumexp(h_proceeding / config.temperature, dim=1)}")

    return trainer


if __name__ == "__main__" and len(sys.argv) == 1:
    # Quick test if no arguments provided
    print("🧪 Running conservative loss test...")
    test_conservative_loss()
    print("✅ Test completed!")