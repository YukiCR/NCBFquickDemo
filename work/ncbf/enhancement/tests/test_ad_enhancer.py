"""
Test for ADEnhancer implementation.
Tests anomaly detection-based pseudo-negative generation.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from enhancer_config import ADConfig
from ad_enhancer import ADEnhancer  # Import from the specific module

def test_ad_enhancer():
    """Test AD enhancer with real dataset."""

    # Use existing safe-only training data
    input_path = "/home/chengrui/wk/NCBFquickDemo/work/ncbf/maps/map_files/map1/training_data_safe_only.h5"
    output_path = "/home/chengrui/wk/NCBFquickDemo/work/ncbf/enhancement/tests/test_ad_enhanced.h5"

    print(f"📂 Testing AD enhancer with: {input_path}")

    try:
        # Create AD configuration
        config = ADConfig(
            input_dataset_path=input_path,
            ad_method='ocsvm',
            target_ratio=0.3,  # 1:2 ratio of negative:safe
            kernel='rbf',
            nu=0.0,
            threshold_quantile=0.05,  # Conservative threshold
            use_full_state=False,  # Use position only
            workspace_padding=0.5
        )

        print(f"⚙️ Configuration:")
        print(f"   AD method: {config.ad_method}")
        print(f"   Target ratio: {config.target_ratio}")
        print(f"   OneClassSVM nu: {config.nu}")
        print(f"   Threshold quantile: {config.threshold_quantile}")

        # Create and test enhancer
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

        # Check data integrity
        assert num_safe == original_safe, f"Safe count mismatch: {num_safe} vs {original_safe}"
        assert abs(num_unsafe - expected_negative) <= 10, f"Negative count off by more than 10"

        # Check metadata
        metadata = enhanced_data.get('metadata', {})
        print(f"   Enhancement method: {metadata.get('enhancement_method', 'unknown')}")
        print(f"   AD method used: {metadata.get('ad_method', 'unknown')}")

        print(f"✅ All verifications passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ad_enhancer()
    if success:
        print("\n🎉 AD enhancer test passed!")
        sys.exit(0)
    else:
        print("\n💥 AD enhancer test failed!")
        sys.exit(1)