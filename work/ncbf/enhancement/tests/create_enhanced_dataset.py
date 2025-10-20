"""
Create enhanced dataset in proper maps location for visualization.
Uses AD enhancer to generate pseudo-negative data.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from enhancer_config import ADConfig
from ad_enhancer import ADEnhancer

def create_enhanced_dataset():
    """Create enhanced dataset with AD-generated pseudo-negatives."""

    # Use existing safe-only training data
    input_path = "/home/chengrui/wk/NCBFquickDemo/work/ncbf/maps/map_files/map1/training_data_safe_only.h5"
    output_path = "/home/chengrui/wk/NCBFquickDemo/work/ncbf/maps/map_files/map1/training_data_ad_enhanced.h5"

    print(f"📂 Creating enhanced dataset from: {input_path}")
    print(f"📁 Output will be saved to: {output_path}")

    try:
        # Create AD configuration with good parameters for visualization
        config = ADConfig(
            input_dataset_path=input_path,
            ad_method='ocsvm',
            target_ratio=0.2,  # 1:1 ratio for good visualization
            kernel='rbf',
            nu=0.001,
            gamma=8.0,
            threshold_quantile=0.00005,  # Conservative threshold for clear separation
            use_full_state=False,  # Use position only for spatial clarity
            workspace_padding=0.0
        )

        print(f"⚙️ Configuration:")
        print(f"   AD method: {config.ad_method}")
        print(f"   Target ratio: {config.target_ratio}")
        print(f"   OneClassSVM nu: {config.nu}")
        print(f"   Threshold quantile: {config.threshold_quantile}")

        # Create and run enhancer
        enhancer = ADEnhancer(config)
        print(f"🚀 Created AD enhancer")

        # Test enhancement
        print(f"\n🔧 Running enhancement pipeline...")
        enhanced_path = enhancer.enhance_dataset(output_path)

        print(f"✅ Enhancement completed!")
        print(f"📁 Enhanced dataset saved to: {enhanced_path}")

        # Verify results
        print(f"\n🔍 Verifying enhanced dataset...")
        from enhancer_utils import load_hdf5_dataset

        enhanced_data = load_hdf5_dataset(enhanced_path)

        print(f"   Enhanced states shape: {enhanced_data['states'].shape}")
        print(f"   Enhanced labels shape: {enhanced_data['labels'].shape}")

        # Check balance
        num_safe = int(np.sum(enhanced_data['labels'] == 1))
        num_unsafe = int(np.sum(enhanced_data['labels'] == 0))
        print(f"   Safe samples: {num_safe}")
        print(f"   Unsafe samples: {num_unsafe}")
        print(f"   Safe ratio: {num_safe / len(enhanced_data['labels']):.2%}")

        # Verify we have the expected ratio
        original_safe = len(load_hdf5_dataset(input_path)['states'])
        expected_negative = int(original_safe * config.target_ratio)
        print(f"   Original safe samples: {original_safe}")
        print(f"   Expected negative samples: {expected_negative}")
        print(f"   Actual negative samples: {num_unsafe}")

        # Check metadata
        metadata = enhanced_data.get('metadata', {})
        print(f"   Enhancement method: {metadata.get('enhancement_method', 'unknown')}")
        print(f"   AD method used: {metadata.get('ad_method', 'unknown')}")

        print(f"✅ Enhanced dataset created successfully!")
        print(f"\n🎉 Dataset ready for visualization!")
        return True

    except Exception as e:
        print(f"❌ Creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_enhanced_dataset()
    if success:
        print("\n🎉 Enhanced dataset creation completed!")
        print("\n💡 Next step: Visualize the dataset to check negative data generation:")
        print("   python /home/chengrui/wk/NCBFquickDemo/work/ncbf/maps/visualize_training_data.py &")
        print("   --training_data map1/training_data_ad_enhanced.h5 &")
        print("   --map map1/map1.json &")
        print("   --output results/ad_enhanced_visualization.png &")
        print("   --save-only")
        sys.exit(0)
    else:
        print("\n💥 Enhanced dataset creation failed!")
        sys.exit(1)