"""
Basic test for enhancer_utils.py functionality.
Tests load/save operations with real dataset.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from enhancer_utils import load_hdf5_dataset, save_hdf5_dataset, get_workspace_bounds

def test_load_and_save():
    """Test loading a real dataset and saving a copy."""

    # Use existing training data
    input_path = "/home/chengrui/wk/NCBFquickDemo/work/ncbf/maps/map_files/map1/training_data_new.h5"
    output_path = "/home/chengrui/wk/NCBFquickDemo/work/ncbf/enhancement/tests/test_output.h5"

    print(f"📂 Loading dataset from: {input_path}")

    try:
        # Load dataset
        data = load_hdf5_dataset(input_path)

        print(f"✅ Dataset loaded successfully!")
        print(f"   States shape: {data['states'].shape}")
        print(f"   Labels shape: {data['labels'].shape}")
        print(f"   Actions present: {data['actions'] is not None}")
        print(f"   Number of samples: {data['metadata']['num_samples']}")
        print(f"   Safe samples: {data['metadata']['num_safe']}")
        print(f"   Unsafe samples: {data['metadata']['num_unsafe']}")

        # Test workspace bounds calculation
        min_bound, max_bound = get_workspace_bounds(data['states'])
        print(f"📏 Workspace bounds: [{min_bound:.2f}, {max_bound:.2f}]")

        # Save dataset
        print(f"\n💾 Saving dataset to: {output_path}")

        # Add some metadata for testing
        test_metadata = {
            'enhancement_method': 'test_utils',
            'enhancement_timestamp': 'test_timestamp',
            'original_dataset': input_path
        }

        save_hdf5_dataset(output_path, data, test_metadata)

        print(f"✅ Dataset saved successfully!")

        # Verify by loading the saved file
        print(f"\n🔄 Verifying saved file...")
        loaded_data = load_hdf5_dataset(output_path)

        # Check shapes match
        assert loaded_data['states'].shape == data['states'].shape, "States shape mismatch"
        assert loaded_data['labels'].shape == data['labels'].shape, "Labels shape mismatch"

        # Check data integrity (sample a few values)
        assert np.allclose(loaded_data['states'][:10], data['states'][:10]), "States data mismatch"
        assert np.allclose(loaded_data['labels'][:10], data['labels'][:10]), "Labels data mismatch"

        print(f"✅ Data integrity verified!")
        print(f"✅ Utils test completed successfully!")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_load_and_save()
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)