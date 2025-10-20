"""
Conservative Neural Control Barrier Function (NCBF) Trainer.

This module implements the conservative loss from Tabbara et al. 2025:
L_c = (λ_c/|X_safe|) * Σ[τ * ln(Σ exp(h(x')/τ))]

The conservative loss uses log-sum-exp to approximate the maximum NCBF value
over proceeding states, implementing the conservative Q-learning principle
of assuming worst-case (unsafe) for unknown states.
"""

# CRITICAL: Add work directory to path BEFORE any other imports
import sys
from pathlib import Path
work_dir = Path(__file__).parent.parent.parent  # /home/chengrui/wk/NCBFquickDemo/work
sys.path.insert(0, str(work_dir))

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import logging

# Import parent class and components
try:
    from .ncbf_trainer import NCBFTrainer
    from ..configs.ncbf_config import NCBFConfig
except ImportError:
    # Fallback for when running outside package context
    from ncbf.training.ncbf_trainer import NCBFTrainer
    from ncbf.configs.ncbf_config import NCBFConfig

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

logger = logging.getLogger(__name__)


class ConservativeNCBFTrainer(NCBFTrainer):
    """
    Conservative NCBF trainer with log-sum-exp conservative loss.

    This class extends NCBFTrainer to implement conservative learning from
    safe-only data using the principle: if we don't know a state, assume
    the worst case - the state is unsafe.

    Key features:
    - Log-sum-exp approximation of max h(x') over proceeding states
    - Random control sampling within existing unicycle constraints
    - Two-phase training support (pretraining + main training)
    - Integration with existing unicycle model dynamics
    """

    def __init__(self, config: NCBFConfig, unicycle_model=None):
        """
        Initialize conservative NCBF trainer.

        Args:
            config: NCBFConfig with conservative training parameters
            unicycle_model: UnicycleModel instance for dynamics
        """
        super().__init__(config, unicycle_model)

        # Conservative loss parameters
        self.conservative_weight = getattr(config, 'conservative_weight', 1.0)
        self.temperature = getattr(config, 'temperature', 0.1)  # τ parameter
        self.num_random_controls = getattr(config, 'num_random_controls', 10)

        # Two-phase training parameters
        self.enable_pretraining = getattr(config, 'enable_pretraining', False)
        self.pretrain_epochs = getattr(config, 'pretrain_epochs', 0)
        self.pretrain_lr = getattr(config, 'pretrain_lr', 0.001)

        logger.info(f"🛡️ Conservative NCBF Trainer initialized:")
        logger.info(f"   Conservative weight: {self.conservative_weight}")
        logger.info(f"   Temperature: {self.temperature}")
        logger.info(f"   Random controls: {self.num_random_controls}")
        logger.info(f"   Pretraining: {self.enable_pretraining} ({self.pretrain_epochs} epochs)")

    def compute_conservative_loss(self, safe_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute conservative loss: L_c = τ * ln(Σ exp(h(x')/τ)) for proceeding states.

        Args:
            safe_states: Safe states [batch_size, state_dim]

        Returns:
            conservative_loss: Scalar loss value
            h_proceeding: h(x') values [batch_size, num_random_controls]
        """
        batch_size = safe_states.shape[0]

        # Generate proceeding states using existing unicycle model dynamics
        proceeding_states = self._generate_proceeding_states(safe_states)
        # Shape: [batch_size, num_random_controls, state_dim]

        # Compute h(x') for all proceeding states
        # Reshape for batch processing: [batch_size * num_random_controls, state_dim]
        flat_proceeding = proceeding_states.reshape(-1, proceeding_states.shape[-1])
        h_proceeding = self.ncbf.h(flat_proceeding)  # [batch_size * num_random_controls]

        # Reshape back: [batch_size, num_random_controls]
        h_proceeding = h_proceeding.reshape(batch_size, self.num_random_controls)

        # Log-sum-exp: τ * ln(Σ exp(h(x')/τ))
        # This approximates max(h(x')) over all proceeding states
        log_sum_exp = self.temperature * torch.logsumexp(h_proceeding / self.temperature, dim=1)

        # Conservative loss: minimize max h(x') over proceeding states
        conservative_loss = log_sum_exp.mean()

        return conservative_loss, h_proceeding

    def _generate_proceeding_states(self, current_states: torch.Tensor) -> torch.Tensor:
        """
        Generate proceeding states using existing unicycle model dynamics.

        Args:
            current_states: Current states [batch_size, state_dim]

        Returns:
            proceeding_states: [batch_size, num_random_controls, state_dim]
        """
        batch_size = current_states.shape[0]
        proceeding_states = []

        for i in range(self.num_random_controls):
            # Sample random control using existing constraint
            controls = self._sample_random_controls(batch_size)

            # Apply dynamics using existing unicycle model methods
            next_states = self._apply_unicycle_dynamics(current_states, controls)
            proceeding_states.append(next_states)

        # Stack: [batch_size, num_random_controls, state_dim]
        return torch.stack(proceeding_states, dim=1)

    def _sample_random_controls(self, batch_size: int) -> torch.Tensor:
        """
        Sample realistic random controls using the unicycle PD controller.

        Instead of sampling random controls from maximum norm (which creates
        unrealistic aggressive controls), we:
        1. Set random target positions in the workspace
        2. Use the PD controller to compute nominal controls toward those targets
        3. This produces much more realistic, behavior-driven control inputs

        Args:
            batch_size: Number of samples to generate

        Returns:
            controls: [batch_size, 2] where each row is [v, ω] from PD controller
        """
        if self.unicycle_model is None:
            # Fallback to original method if no unicycle model available
            return self._sample_random_controls_fallback(batch_size)

        controls = []
        # TODO: use the map file to determine workspace size
        workspace_size = 8.0  # Default workspace size

        for i in range(batch_size):
            # Save current state to restore later
            original_state = self.unicycle_model.state.copy()
            original_target = self.unicycle_model.target.copy()

            try:
                # Set random target position in workspace
                random_target = np.array([
                    np.random.uniform(0, workspace_size),
                    np.random.uniform(0, workspace_size)
                ])
                self.unicycle_model.set_target(random_target)

                # Use PD controller to get nominal control toward target
                # Using pd_control_proportional for better behavior
                control = self.unicycle_model.pd_control_proportional()

                # Add to list (as numpy array)
                controls.append(control)

            finally:
                # Always restore original state and target
                self.unicycle_model.state = original_state
                self.unicycle_model.target = original_target

        # Convert list of numpy arrays to single tensor
        controls_array = np.array(controls)  # [batch_size, 2]
        return torch.tensor(controls_array, device=self.device, dtype=torch.float32)

    def _sample_random_controls_fallback(self, batch_size: int) -> torch.Tensor:
        """
        Fallback method for random control sampling when unicycle model is not available.

        Args:
            batch_size: Number of samples to generate

        Returns:
            controls: [batch_size, 2] where each row is [v, ω]
        """
        # Use smaller, more reasonable control range instead of max norm
        max_norm = 1.0  # Reduced from 2.0 for more reasonable controls

        # Sample random directions and magnitudes
        angles = torch.rand(batch_size, device=self.device) * 2 * np.pi
        magnitudes = torch.rand(batch_size, device=self.device) * max_norm

        # Convert to [v, ω] control space
        v = magnitudes * torch.cos(angles)
        ω = magnitudes * torch.sin(angles)

        return torch.stack([v, ω], dim=1)  # [batch_size, 2]

    def _apply_unicycle_dynamics(self, states: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        """
        Apply unicycle dynamics using existing model methods.

        Args:
            states: Current states [batch_size, state_dim]
            controls: Control inputs [batch_size, 2]

        Returns:
            next_states: Next states [batch_size, state_dim]
        """
        batch_size = states.shape[0]
        next_states = []
        dt = self.unicycle_model.config.dt  # Use existing time step

        for i in range(batch_size):
            # Get current state and control
            current_state = states[i].cpu().numpy()  # [px, py, theta]
            control = controls[i].cpu().numpy()  # [v, ω]

            # Apply control constraints using existing method
            constrained_control = self.unicycle_model._apply_control_constraints(control)

            # Compute state derivative using existing dynamics
            state_derivative = self.unicycle_model.dynamics(current_state, constrained_control)

            # Euler integration: x_next = x_current + x_dot * dt
            next_state = current_state + state_derivative * dt

            # Normalize angle using existing method
            next_state[2] = self.unicycle_model._normalize_angle(next_state[2])

            # Boundary checking (workspace limits)
            next_state[0] = np.clip(next_state[0], 0, 8)  # px
            next_state[1] = np.clip(next_state[1], 0, 8)  # py

            next_states.append(next_state)

        return torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)

    def compute_losses(self, states: torch.Tensor, labels: torch.Tensor, require_grad: bool = True) -> Dict[str, torch.Tensor]:
        """
        Override loss computation to include conservative term.

        Args:
            states: Input states [batch_size, state_dim]
            labels: Safety labels [batch_size] (1=safe, 0=unsafe)
            require_grad: Whether to compute gradients

        Returns:
            Dictionary with losses including conservative term
        """
        # Get original losses from parent class
        losses = super().compute_losses(states, labels, require_grad)

        # Add conservative loss for safe samples only
        safe_mask = labels == 1
        if safe_mask.any() and require_grad:
            safe_states = states[safe_mask]
            conservative_loss, h_proceeding = self.compute_conservative_loss(safe_states)
            losses['conservative_loss'] = conservative_loss

            # Add to total loss (minimize max h(x') over proceeding states)
            losses['total_loss'] += self.conservative_weight * conservative_loss

            # Store statistics for monitoring
            losses['max_h_proceeding'] = h_proceeding.max().item()
            losses['mean_h_proceeding'] = h_proceeding.mean().item()
            losses['min_h_proceeding'] = h_proceeding.min().item()
        else:
            losses['conservative_loss'] = torch.tensor(0.0, device=self.device)
            losses['max_h_proceeding'] = 0.0
            losses['mean_h_proceeding'] = 0.0
            losses['min_h_proceeding'] = 0.0

        return losses

    def pretrain_negative_landscape(self, data_path: str) -> None:
        """
        Pretrain to ensure initial negative h(x) values (optional phase 1).

        This implements the two-phase training where we first initialize
        the NCBF to output negative values, then train with conservative loss.

        Args:
            data_path: Path to training data
        """
        # Ensure training components are set up (optimizer, etc.)
        if self.optimizer is None:
            self.setup_training_components()

        # Load training data to get dataset size and map info
        training_data = self.load_training_data(data_path)
        num_samples = len(training_data['states'])  # Use same number of samples as training data

        # Get map workspace bounds for random sampling
        # TODO: use the map file to determine workspace size
        workspace_size = 8.0  # Default workspace size
        if 'map_info' in training_data and training_data['map_info']:
            workspace_size = training_data['map_info'].get('workspace_size', 8.0)

        # Use lower learning rate for pretraining
        original_lr = self.optimizer.param_groups[0]['lr']
        self.optimizer.param_groups[0]['lr'] = self.config.pretrain_lr

        print(f"📊 Pretraining with {num_samples} randomly sampled states across workspace")
        print(f"🗺️  Workspace size: {workspace_size}m × {workspace_size}m")
        print(f"🎯 Target: h(x) = -{self.config.margin} (negative for all sampled states)")
        print(f"📈 Learning rate: {self.config.pretrain_lr}")
        print(f"📦 Batch size: {self.config.batch_size}")

        for epoch in range(self.config.pretrain_epochs):
            # Sample random states uniformly across the workspace for this epoch
            # Random x, y positions within [0, workspace_size]
            random_states = np.random.uniform(0, workspace_size, size=(num_samples, 2))

            # Random theta (orientation) in [-π, π]
            random_theta = np.random.uniform(-np.pi, np.pi, size=(num_samples, 1))

            # Combine to create full state vectors [x, y, theta]
            random_states_full = np.concatenate([random_states, random_theta], axis=1)

            # Convert to tensor
            all_states_tensor = torch.tensor(random_states_full, dtype=torch.float32, device=self.device)

            # Create negative targets (all states should have h(x) = -margin)
            all_negative_targets = torch.full((num_samples,), -self.config.margin, device=self.device)

            # Mini-batch processing
            epoch_loss = 0.0
            epoch_mean_h = 0.0
            epoch_min_h = float('inf')
            epoch_max_h = float('-inf')
            num_batches = 0
            final_batch_loss = 0.0  # Keep track of last batch loss for saving

            # Process in mini-batches
            for i in range(0, num_samples, self.config.batch_size):
                # Get batch indices
                batch_end = min(i + self.config.batch_size, num_samples)
                batch_size_actual = batch_end - i

                # Get batch data
                batch_states = all_states_tensor[i:batch_end]
                batch_targets = all_negative_targets[i:batch_end]

                # Forward pass and loss computation
                h_pred = self.ncbf.h(batch_states)
                batch_loss = torch.mean((h_pred - batch_targets) ** 2)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                # Accumulate statistics
                epoch_loss += batch_loss.item() * batch_size_actual  # Weight by batch size
                epoch_mean_h += h_pred.mean().item() * batch_size_actual
                epoch_min_h = min(epoch_min_h, h_pred.min().item())
                epoch_max_h = max(epoch_max_h, h_pred.max().item())
                num_batches += 1
                final_batch_loss = batch_loss.item()  # Keep track for saving

            # Compute epoch averages
            epoch_loss /= num_samples
            epoch_mean_h /= num_samples

            if epoch % 10 == 0:
                print(f"   Epoch {epoch}: Loss = {epoch_loss:.4f}, "
                      f"Mean h(x) = {epoch_mean_h:.4f}, "
                      f"Min h(x) = {epoch_min_h:.4f}, "
                      f"Max h(x) = {epoch_max_h:.4f}, "
                      f"Batches = {num_batches}")

        # Use the final batch loss for checkpoint saving
        pretrain_loss_value = final_batch_loss

        # Restore original learning rate
        self.optimizer.param_groups[0]['lr'] = original_lr
        print("✅ Pretraining completed")

        # Save pre-trained model checkpoint using the base class method
        logger.info("💾 Saving pre-trained model checkpoint...")

        # Create metrics for the checkpoint (pretraining doesn't have validation metrics)
        pretrain_metrics = {
            'total_loss': pretrain_loss_value,
            'classification_loss': 0.0,
            'barrier_loss': 0.0,
            'conservative_loss': 0.0 if hasattr(self, 'conservative_weight') else None
        }

        # Use epoch -1 to indicate this is a pretraining checkpoint
        # This will create checkpoint_epoch_0.pt (epoch will be +1 in the saving method) and potentially best_model.pt if it's the best so far
        self._save_checkpoint(epoch=-1, metrics=pretrain_metrics)
        logger.info(f"💾 Pre-trained model checkpoint saved (final loss: {pretrain_loss_value:.4f})")

    def train_with_pretraining(self, data_path: str, resume_from: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Two-phase training: pretraining + main conservative training.

        Args:
            data_path: Path to HDF5 training data file
            resume_from: Optional path to checkpoint for resuming training

        Returns:
            Complete training history including pretraining
        """
        logger.info("🚀 Starting Two-Phase Conservative NCBF Training")
        logger.info("=" * 60)

        # Phase 1: Pretraining (optional)
        if self.enable_pretraining and self.pretrain_epochs > 0:
            logger.info("🔄 Phase 1: Pretraining negative landscape...")
            self.pretrain_negative_landscape(data_path)
            logger.info("✅ Pretraining completed")

            # Reset optimizer for main training
            self.setup_training_components()
            logger.info("🔄 Reset optimizer for main training phase")

        # Phase 2: Main conservative training
        logger.info("🚀 Phase 2: Conservative NCBF training...")
        training_history = self.train(data_path, resume_from)

        return training_history

    def _plot_training_progress(self) -> None:
        """Override to include conservative loss in plots."""
        if not self.training_history['train_loss']:
            return

        # Call parent method for basic plots
        super()._plot_training_progress()

        # Add conservative loss plot if available
        if 'conservative_loss' in self.training_history and self.training_history['conservative_loss']:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            epochs = range(1, len(self.training_history['conservative_loss']) + 1)
            ax.plot(epochs, self.training_history['conservative_loss'], 'purple',
                   linewidth=2, label='Conservative Loss')

            # Add proceeding h(x) statistics
            if 'max_h_proceeding' in self.training_history:
                ax.plot(epochs, self.training_history['max_h_proceeding'], 'red',
                       linewidth=2, label='Max h(x\')', alpha=0.7)
                ax.plot(epochs, self.training_history['mean_h_proceeding'], 'orange',
                       linewidth=2, label='Mean h(x\')', alpha=0.7)

            ax.set_title('Conservative Loss and Proceeding State Statistics')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss / h(x) Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.checkpoint_dir / 'conservative_progress.png', dpi=150, bbox_inches='tight')

            if self.config.show_training_plots:
                plt.show(block=False)
                plt.pause(0.1)
            else:
                plt.close()

    def _generate_training_report(self) -> None:
        """Override to include conservative training statistics."""
        super()._generate_training_report()

        # Append conservative training info
        report_path = self.checkpoint_dir / 'training_report.txt'

        with open(report_path, 'a') as f:
            f.write("\nConservative Training Statistics:\n")
            f.write(f"  Conservative weight: {self.conservative_weight}\n")
            f.write(f"  Temperature: {self.temperature}\n")
            f.write(f"  Random controls: {self.num_random_controls}\n")
            f.write(f"  Pretraining enabled: {self.enable_pretraining}\n")
            if self.enable_pretraining:
                f.write(f"  Pretrain epochs: {self.pretrain_epochs}\n")

            # Add final conservative statistics if available
            if 'conservative_loss' in self.training_history and self.training_history['conservative_loss']:
                f.write(f"  Final conservative loss: {self.training_history['conservative_loss'][-1]:.6f}\n")
                if 'max_h_proceeding' in self.training_history:
                    f.write(f"  Final max h(x'): {self.training_history['max_h_proceeding'][-1]:.6f}\n")
                    f.write(f"  Final mean h(x'): {self.training_history['mean_h_proceeding'][-1]:.6f}\n")

        logger.info(f"📋 Updated training report with conservative statistics")


# Convenience function for quick conservative training
def create_conservative_ncbf_trainer(config: NCBFConfig, unicycle_model=None) -> ConservativeNCBFTrainer:
    """
    Convenience function to create conservative NCBF trainer.

    Args:
        config: NCBFConfig with conservative parameters
        unicycle_model: Optional unicycle model instance

    Returns:
        ConservativeNCBFTrainer instance
    """
    return ConservativeNCBFTrainer(config, unicycle_model)