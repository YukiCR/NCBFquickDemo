#!/usr/bin/env python3
"""
Test improved contour plotting with final_model weights.
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

def test_final_model_contours():
    """Test the improved contour plotting functionality using final_model."""
    print("🎨 Testing Contour Plotting with Final Model")
    print("="*60)

    try:
        # Create config
        config = create_large_ncbf_config()

        # Override checkpoint directory to final_models
        config.checkpoint_dir = '/home/chengrui/wk/NCBFquickDemo/work/ncbf/weights/final_models'

        print(f"✅ Loading trainer from: {config.checkpoint_dir}")
        print(f"   Architecture: {config.hidden_dims}")
        print(f"   Activation: {config.activation}")

        # Create trainer with config
        trainer = NCBFTrainer(config)

        # Setup training components (needed for checkpoint loading)
        trainer.setup_training_components()

        # Load the final model from intensive_training_2
        final_model_path = '/home/chengrui/wk/NCBFquickDemo/work/ncbf/weights/intensive_training_2/final_model.pt'
        if os.path.exists(final_model_path):
            # Load just the model state dict (final model doesn't have full checkpoint)
            checkpoint = torch.load(final_model_path, map_location=trainer.device)
            trainer.ncbf.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Final model loaded successfully with {sum(p.numel() for p in trainer.ncbf.parameters()):,} parameters")
        else:
            print(f"❌ Final model not found: {final_model_path}")
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
        print(f"   Model loaded from final training state")
        print(f"   Final validation loss: {checkpoint.get('training_history', {}).get('val_loss', [])[-1] if 'training_history' in checkpoint else 'N/A'}")

        return True

    except Exception as e:
        print(f"❌ Error in contour plotting: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 Testing Final Model Contour Plotting")
    print("="*60)

    success = test_final_model_contours()

    if success:
        print("\n✅ Final model contour plotting test completed successfully!")
    else:
        print("\n❌ Final model contour plotting test failed.")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())