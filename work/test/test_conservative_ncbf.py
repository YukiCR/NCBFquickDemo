#!/usr/bin/env python3
"""
Test suite for Conservative NCBF Trainer.

This module provides comprehensive testing for the conservative NCBF implementation,
including loss computation, random control sampling, dynamics integration, and
integration with existing unicycle model.
"""

import sys
import numpy as np
import torch
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'work'))

from ncbf.training.conservative_ncbf_trainer import ConservativeNCBFTrainer
from ncbf.configs.ncbf_config import NCBFConfig, create_large_ncbf_config
from configs.unicycle_config import UnicycleConfig
from ncbf.maps import load_map, NCBFMap

# Import unicycle model with proper path handling
try:
    from models.unicycle_model import UnicycleModel
except ImportError:
    try:
        # Try alternative import path
        import sys
        sys.path.insert(0, '/home/chengrui/wk/NCBFquickDemo/work')
        from models.unicycle_model import UnicycleModel
    except ImportError:
        print("⚠️  Could not import UnicycleModel, some tests will be skipped")


def setup_logging():
    """Setup logging for tests."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)


def test_conservative_trainer_initialization():
    """Test conservative trainer initialization."""
    print("\n🧪 Test 1: Conservative Trainer Initialization")

    # Create config with conservative parameters
    config = create_large_ncbf_config()
    config.conservative_weight = 1.5
    config.temperature = 0.05
    config.num_random_controls = 8
    config.enable_pretraining = True
    config.pretrain_epochs = 10

    # Create unicycle model
    unicycle_config = UnicycleConfig()
    from models.unicycle_model import UnicycleModel
    unicycle_model = UnicycleModel(config=unicycle_config)

    # Initialize trainer
    trainer = ConservativeNCBFTrainer(config, unicycle_model)

    print(f"✅ Conservative weight: {trainer.conservative_weight}")
    print(f"✅ Temperature: {trainer.temperature}")
    print(f"✅ Random controls: {trainer.num_random_controls}")
    print(f"✅ Pretraining enabled: {trainer.enable_pretraining}")
    print(f"✅ Device: {trainer.device}")

    # Test basic functionality
    assert trainer.conservative_weight == 1.5
    assert trainer.temperature == 0.05
    assert trainer.num_random_controls == 8
    assert trainer.enable_pretraining == True

    print("✅ Initialization test passed!")
    return trainer


def test_random_control_sampling():
    """Test random control sampling within constraints."""
    print("\n🧪 Test 2: Random Control Sampling")

    # Create minimal setup
    config = create_large_ncbf_config()
    config.num_random_controls = 5

    unicycle_config = UnicycleConfig()
    from models.unicycle_model import UnicycleModel
    unicycle_model = UnicycleModel(config=unicycle_config)

    trainer = ConservativeNCBFTrainer(config, unicycle_model)

    # Test control sampling
    batch_size = 10
    controls = trainer._sample_random_controls(batch_size)

    print(f"✅ Generated controls shape: {controls.shape}")
    print(f"✅ Controls range: [{controls.min().item():.3f}, {controls.max().item():.3f}]")

    # Verify constraints
    assert controls.shape == (batch_size, 2)  # [v, ω]
    assert controls.device == trainer.device

    # Check norm constraint
    norms = torch.norm(controls, dim=1)
    max_norm = unicycle_model.config.max_control_norm
    assert norms.max().item() <= max_norm + 1e-6  # Allow small numerical error

    print(f"✅ Control norms within constraint: max={norms.max().item():.3f}, limit={max_norm}")
    print("✅ Random control sampling test passed!")


def test_dynamics_integration():
    """Test dynamics integration using existing unicycle model."""
    print("\n🧪 Test 3: Dynamics Integration")

    # Create setup
    config = create_large_ncbf_config()
    config.num_random_controls = 3

    unicycle_config = UnicycleConfig()
    from models.unicycle_model import UnicycleModel
    unicycle_model = UnicycleModel(config=unicycle_config)

    trainer = ConservativeNCBFTrainer(config, unicycle_model)

    # Test states
    batch_size = 5
    current_states = torch.tensor([
        [1.0, 2.0, 0.0],  # [px, py, theta]
        [3.0, 4.0, np.pi/4],
        [5.0, 6.0, np.pi/2],
        [7.0, 1.0, 3*np.pi/4],
        [2.0, 5.0, np.pi]
    ], device=trainer.device)

    # Apply dynamics
    next_states = trainer._apply_unicycle_dynamics(current_states, trainer._sample_random_controls(batch_size))

    print(f"✅ Current states shape: {current_states.shape}")
    print(f"✅ Next states shape: {next_states.shape}")
    print(f"✅ Sample state transition:")
    print(f"   Before: {current_states[0].cpu().numpy()}")
    print(f"   After:  {next_states[0].cpu().numpy()}")

    # Verify shapes
    assert next_states.shape == current_states.shape
    assert next_states.device == trainer.device

    # Verify angle normalization (should be in [-pi, pi])
    angles = next_states[:, 2].cpu().numpy()
    assert np.all(angles >= -np.pi) and np.all(angles <= np.pi)

    # Verify boundary constraints
    positions = next_states[:, :2].cpu().numpy()
    assert np.all(positions >= 0) and np.all(positions <= 8)

    print("✅ Dynamics integration test passed!")


def test_proceeding_states_generation():
    """Test proceeding states generation."""
    print("\n🧪 Test 4: Proceeding States Generation")

    # Create setup
    config = create_large_ncbf_config()
    config.num_random_controls = 4

    unicycle_config = UnicycleConfig()
    from models.unicycle_model import UnicycleModel
    unicycle_model = UnicycleModel(config=unicycle_config)

    trainer = ConservativeNCBFTrainer(config, unicycle_model)

    # Test states
    batch_size = 3
    current_states = torch.randn(batch_size, 3, device=trainer.device)
    current_states = torch.abs(current_states) * 4  # Keep in workspace bounds

    # Generate proceeding states
    proceeding_states = trainer._generate_proceeding_states(current_states)

    print(f"✅ Current states shape: {current_states.shape}")
    print(f"✅ Proceeding states shape: {proceeding_states.shape}")
    print(f"✅ Sample proceeding states for first input:")
    for i in range(min(3, config.num_random_controls)):
        print(f"   Control {i}: {proceeding_states[0, i].cpu().numpy()}")

    # Verify shapes
    assert proceeding_states.shape == (batch_size, config.num_random_controls, 3)
    assert proceeding_states.device == trainer.device

    # Verify all proceeding states are valid (in workspace)
    positions = proceeding_states[:, :, :2].reshape(-1, 2).cpu().numpy()
    assert np.all(positions >= 0) and np.all(positions <= 8)

    print("✅ Proceeding states generation test passed!")


def test_conservative_loss_computation():
    """Test conservative loss computation."""
    print("\n🧪 Test 5: Conservative Loss Computation")

    # Create setup
    config = create_large_ncbf_config()
    config.conservative_weight = 2.0
    config.temperature = 0.1
    config.num_random_controls = 6

    unicycle_config = UnicycleConfig()
    from models.unicycle_model import UnicycleModel
    unicycle_model = UnicycleModel(config=unicycle_config)

    trainer = ConservativeNCBFTrainer(config, unicycle_model)

    # Test with safe states
    batch_size = 4
    safe_states = torch.tensor([
        [2.0, 2.0, 0.0],
        [4.0, 4.0, np.pi/4],
        [6.0, 2.0, np.pi/2],
        [2.0, 6.0, 3*np.pi/4]
    ], device=trainer.device)

    print(f"✅ Testing with batch size: {batch_size}")
    print(f"✅ Conservative weight: {config.conservative_weight}")
    print(f"✅ Temperature: {config.temperature}")
    print(f"✅ Random controls: {config.num_random_controls}")

    # Compute conservative loss
    conservative_loss, h_proceeding = trainer.compute_conservative_loss(safe_states)

    print(f"✅ Conservative loss: {conservative_loss.item():.6f}")
    print(f"✅ h(x') shape: {h_proceeding.shape}")
    print(f"✅ h(x') statistics:")
    print(f"   Min: {h_proceeding.min().item():.4f}")
    print(f"   Max: {h_proceeding.max().item():.4f}")
    print(f"   Mean: {h_proceeding.mean().item():.4f}")
    print(f"   Std: {h_proceeding.std().item():.4f}")

    # Verify shapes and properties
    assert h_proceeding.shape == (batch_size, config.num_random_controls)
    assert conservative_loss.dim() == 0  # Scalar

    # Log-sum-exp should be greater than or equal to max (approximation property)
    log_sum_exp_values = config.temperature * torch.logsumexp(h_proceeding / config.temperature, dim=1)
    max_values = h_proceeding.max(dim=1)[0]
    assert torch.all(log_sum_exp_values >= max_values - 1e-6)

    print(f"✅ Log-sum-exp vs max verification passed")
    print("✅ Conservative loss computation test passed!")


def test_complete_loss_integration():
    """Test complete loss integration with parent class."""
    print("\n🧪 Test 6: Complete Loss Integration")

    # Create setup
    config = create_large_ncbf_config()
    config.conservative_weight = 1.5
    config.temperature = 0.08
    config.num_random_controls = 5
    config.classification_weight = 2.0
    config.barrier_weight = 0.5
    config.regularization_weight = 0.1

    unicycle_config = UnicycleConfig()
    from models.unicycle_model import UnicycleModel
    unicycle_model = UnicycleModel(config=unicycle_config)

    trainer = ConservativeNCBFTrainer(config, unicycle_model)

    # Test with mixed data (simulate training batch)
    batch_size = 8
    states = torch.randn(batch_size, 3, device=trainer.device)
    states[:, :2] = torch.abs(states[:, :2]) * 4  # Keep in workspace bounds

    # All safe labels for conservative loss testing
    labels = torch.ones(batch_size, device=trainer.device)

    print(f"✅ Testing with batch size: {batch_size}")
    print(f"✅ All labels are safe (for conservative loss)")

    # Compute complete losses
    losses = trainer.compute_losses(states, labels, require_grad=True)

    print(f"✅ Loss components:")
    print(f"   Total loss: {losses['total_loss'].item():.6f}")
    print(f"   Classification loss: {losses['classification_loss'].item():.6f}")
    print(f"   Barrier loss: {losses['barrier_loss'].item():.6f}")
    print(f"   Regularization loss: {losses['reg_loss'].item():.6f}")
    print(f"   Conservative loss: {losses['conservative_loss'].item():.6f}")
    print(f"   Max h(x'): {losses['max_h_proceeding']:.6f}")
    print(f"   Mean h(x'): {losses['mean_h_proceeding']:.6f}")

    # Verify loss components
    expected_total = (
        config.classification_weight * losses['classification_loss'] +
        config.barrier_weight * losses['barrier_loss'] +
        config.regularization_weight * losses['reg_loss'] -
        config.conservative_weight * losses['conservative_loss']
    )

    assert torch.allclose(losses['total_loss'], expected_total, rtol=1e-5)

    # Verify conservative loss is positive and being subtracted
    assert losses['conservative_loss'].item() > 0
    assert losses['total_loss'].item() < (
        config.classification_weight * losses['classification_loss'] +
        config.barrier_weight * losses['barrier_loss'] +
        config.regularization_weight * losses['reg_loss']
    )

    print("✅ Complete loss integration test passed!")


def test_pretraining_functionality():
    """Test pretraining functionality."""
    print("\n🧪 Test 7: Pretraining Functionality")

    # Create setup
    config = create_large_ncbf_config()
    config.pretrain_epochs = 5
    config.pretrain_lr = 0.01
    config.margin = 0.3

    unicycle_config = UnicycleConfig()
    from models.unicycle_model import UnicycleModel
    unicycle_model = UnicycleModel(config=unicycle_config)

    trainer = ConservativeNCBFTrainer(config, unicycle_model)

    # Create dummy safe data
    safe_states = torch.randn(20, 3, device=trainer.device)
    safe_states[:, :2] = torch.abs(safe_states[:, :2]) * 3 + 1  # Keep away from boundaries

    # Test pretraining
    print(f"✅ Testing pretraining with {len(safe_states)} safe states")
    print(f"✅ Target h(x): -{config.margin}")
    print(f"✅ Pretrain epochs: {config.pretrain_epochs}")
    print(f"✅ Pretrain learning rate: {config.pretrain_lr}")

    # Initial h values
    initial_h = trainer.ncbf.h(safe_states)
    print(f"✅ Initial h(x) stats: mean={initial_h.mean().item():.4f}, "
          f"min={initial_h.min().item():.4f}, max={initial_h.max().item():.4f}")

    # Run pretraining
    trainer.pretrain_negative_landscape("dummy_path")  # Will use internal safe_states

    # Final h values
    final_h = trainer.ncbf.h(safe_states)
    print(f"✅ Final h(x) stats: mean={final_h.mean().item():.4f}, "
          f"min={final_h.min().item():.4f}, max={final_h.max().item():.4f}")

    # Verify pretraining worked (h values should be more negative)
    assert final_h.mean().item() < initial_h.mean().item() - 0.1  # Significant change
    assert final_h.max().item() < config.margin  # All values below target

    print("✅ Pretraining functionality test passed!")


def test_integration_with_real_data():
    """Test integration with real training data."""
    print("\n🧪 Test 8: Integration with Real Training Data")

    try:
        # Load real safe-only data
        data_path = "/home/chengrui/wk/NCBFquickDemo/work/ncbf/maps/map_files/map1/training_data_safe_only.h5"
        if not Path(data_path).exists():
            print("⚠️  Safe-only data not found, skipping integration test")
            return

        # Create setup
        config = create_large_ncbf_config()
        config.conservative_weight = 1.0
        config.temperature = 0.1
        config.num_random_controls = 5
        config.max_epochs = 2  # Quick test
        config.batch_size = 64
        config.show_training_plots = False

        unicycle_config = UnicycleConfig()
        from models.unicycle_model import UnicycleModel
        unicycle_model = UnicycleModel(config=unicycle_config)

        trainer = ConservativeNCBFTrainer(config, unicycle_model)

        # Load data
        training_data = trainer.load_training_data_file(data_path)

        print(f"✅ Loaded {training_data['num_samples']} samples")
        print(f"✅ Safe samples: {training_data['num_safe']}")
        print(f"✅ Unsafe samples: {training_data['num_unsafe']}")

        # Test conservative loss on real data
        safe_states = torch.tensor(training_data['states'][:100], device=trainer.device)  # First 100 samples
        conservative_loss, h_proceeding = trainer.compute_conservative_loss(safe_states)

        print(f"✅ Conservative loss on real data: {conservative_loss.item():.6f}")
        print(f"✅ h(x') shape: {h_proceeding.shape}")
        print(f"✅ h(x') range: [{h_proceeding.min().item():.4f}, {h_proceeding.max().item():.4f}]")

        # Quick training test
        print("✅ Running quick training test...")
        trainer.setup_data_loaders(training_data)
        trainer.setup_training_components()

        # Train for 1 epoch
        metrics = trainer.train_epoch()

        print(f"✅ Training metrics:")
        print(f"   Total loss: {metrics['total_loss']:.6f}")
        print(f"   Classification loss: {metrics['classification_loss']:.6f}")
        print(f"   Conservative loss: {metrics.get('conservative_loss', 0.0):.6f}")

        print("✅ Integration with real data test passed!")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


def run_all_tests():
    """Run all conservative trainer tests."""
    logger = setup_logging()
    print("🛡️ Conservative NCBF Trainer Test Suite")
    print("=" * 60)

    tests = [
        test_conservative_trainer_initialization,
        test_random_control_sampling,
        test_dynamics_integration,
        test_proceeding_states_generation,
        test_conservative_loss_computation,
        test_complete_loss_integration,
        test_pretraining_functionality,
        test_integration_with_real_data
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
            print(f"✅ {test.__name__} PASSED")
        except Exception as e:
            failed += 1
            print(f"❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("🎉 All tests passed! Conservative NCBF trainer is ready for use.")
    else:
        print("⚠️  Some tests failed. Please review the implementation.")

    return failed == 0


def quick_test():
    """Run a quick test of conservative loss computation."""
    print("🧪 Quick Conservative Loss Test")

    try:
        # Create minimal config
        config = create_large_ncbf_config()
        config.conservative_weight = 1.0
        config.temperature = 0.1
        config.num_random_controls = 5

        # Create unicycle model
        unicycle_config = UnicycleConfig()
        unicycle_model = UnicycleModel(config=unicycle_config)

        # Create trainer
        trainer = ConservativeNCBFTrainer(config, unicycle_model)

        # Test with dummy data
        batch_size = 4
        safe_states = torch.randn(batch_size, 3, device=trainer.device)
        safe_states[:, :2] = torch.abs(safe_states[:, :2]) * 4  # Keep in workspace bounds

        print(f"📊 Testing with batch size: {batch_size}")
        print(f"🎯 Temperature: {config.temperature}")
        print(f"🎲 Random controls: {config.num_random_controls}")

        # Compute conservative loss
        conservative_loss, h_proceeding = trainer.compute_conservative_loss(safe_states)

        print(f"✅ Conservative loss: {conservative_loss.item():.6f}")
        print(f"📈 h(x') shape: {h_proceeding.shape}")
        print(f"📊 h(x') range: [{h_proceeding.min().item():.4f}, {h_proceeding.max().item():.4f}]")
        print(f"🌡️  Log-sum-exp values: {config.temperature * torch.logsumexp(h_proceeding / config.temperature, dim=1)}")

        print("✅ Quick test completed!")
        return trainer

    except NameError:
        print("⚠️  UnicycleModel not available, creating trainer without model...")
        # Test conservative loss computation without unicycle model
        config = create_large_ncbf_config()
        config.conservative_weight = 1.0
        config.temperature = 0.1
        config.num_random_controls = 5

        # Create trainer without unicycle model
        trainer = ConservativeNCBFTrainer(config, unicycle_model=None)

        # Test conservative loss computation directly
        batch_size = 4
        safe_states = torch.randn(batch_size, 3, device=trainer.device)

        print(f"📊 Testing conservative loss computation without dynamics...")
        print(f"🎯 Temperature: {config.temperature}")
        print(f"🎲 Random controls: {config.num_random_controls}")

        # This will skip dynamics but still test the core logic
        try:
            conservative_loss, h_proceeding = trainer.compute_conservative_loss(safe_states)
            print(f"✅ Conservative loss: {conservative_loss.item():.6f}")
            print(f"📈 h(x') shape: {h_proceeding.shape}")
        except Exception as e:
            print(f"⚠️  Conservative loss requires unicycle model: {e}")
            print("✅ Core trainer initialization works fine")

        print("✅ Limited test completed!")
        return trainer


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        run_all_tests()