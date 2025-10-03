#!/usr/bin/env python3
"""
Simple test for improved contour plotting with intensive_training_2 weights.
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add necessary paths
sys.path.insert(0, '/home/chengrui/wk/NCBFquickDemo/work')

# Import the trainer directly
from ncbf.training.ncbf_trainer import NCBFTrainer
from ncbf.configs.ncbf_config import create_large_ncbf_config

def test_simple_contour_plotting():
    """Test the improved contour plotting functionality using the trainer."""
    print("🎨 Testing Simple Contour Plotting with Trainer")
    print("="*60)

    try:
        # Create config
        config = create_large_ncbf_config()

        # Override checkpoint directory
        config.checkpoint_dir = '/home/chengrui/wk/NCBFquickDemo/work/ncbf/weights/intensive_training_2'

        print(f"✅ Loading trainer from: {config.checkpoint_dir}")
        print(f"   Architecture: {config.hidden_dims}")
        print(f"   Activation: {config.activation}")

        # Create trainer with config
        trainer = NCBFTrainer(config)

        # Setup training components (needed for checkpoint loading)
        trainer.setup_training_components()

        # Load the best checkpoint
        checkpoint_path = '/home/chengrui/wk/NCBFquickDemo/work/ncbf/weights/intensive_training_2/best_model.pt'
        if os.path.exists(checkpoint_path):
            trainer.load_checkpoint(checkpoint_path)
            print(f"✅ Model loaded successfully with {sum(p.numel() for p in trainer.ncbf.parameters()):,} parameters")
        else:
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False

        # Load training data for visualization
        data_path = '/home/chengrui/wk/NCBFquickDemo/work/ncbf/maps/map_files/map1/training_data_new.h5'
        if os.path.exists(data_path):
            training_data = trainer.load_training_data(data_path)
            print(f"✅ Training data loaded: {training_data['num_samples']:,} samples")
        else:
            print(f"⚠️  Training data not found, proceeding without data overlay")

        # Generate the improved contour plot
        print("🎨 Generating improved contour visualization...")
        trainer.visualize_contours(resolution=100, theta_fixed=0.0)

        # Print analysis
        print(f"\n📊 Contour Analysis:")
        print(f"   Model epochs: {trainer.epoch + 1}")
        print(f"   Best validation loss: {trainer.best_val_loss:.6f}")

        return True

    except Exception as e:
        print(f"❌ Error in contour plotting: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 Testing Simple Contour Plotting")
    print("="*60)

    success = test_simple_contour_plotting()

    if success:
        print("\n✅ Simple contour plotting test completed successfully!")
    else:
        print("\n❌ Simple contour plotting test failed.")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())