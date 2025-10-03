"""
Neural Control Barrier Function (NCBF) Training Infrastructure.

This module provides a complete training pipeline for NCBF neural networks,
including data loading, loss computation, training loops, visualization,
and model evaluation with contour plots.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Import NCBF components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ncbf.models.ncbf import NCBF
from ncbf.configs.ncbf_config import NCBFConfig
from safe_control.cbf_function import CBFFunction
from ncbf.maps.map_manager import NCBFMap, load_map

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NCBFTrainer:
    """
    Complete training infrastructure for Neural Control Barrier Functions.

    Manages:
    - Data loading and preprocessing from HDF5 files
    - Loss function computation (classification + barrier + regularization)
    - Training loop with GPU acceleration and mixed precision
    - Real-time monitoring and visualization
    - Model checkpointing and validation
    - Contour visualization and evaluation
    """

    def __init__(self, config: NCBFConfig, unicycle_model=None):
        """
        Initialize trainer with configuration and optional unicycle model.

        Args:
            config: NCBFConfig instance with all training parameters
            unicycle_model: UnicycleModel instance for dynamics (needed for barrier loss)
        """
        # Validate configuration
        if not hasattr(config, 'input_dim') or not hasattr(config, 'hidden_dims'):
            raise TypeError(f"config must be NCBFConfig-like object with required attributes")

        self.config = config
        self.unicycle_model = unicycle_model

        # Setup device (GPU/CPU)
        self.device = self._setup_device()
        logger.info(f"Using device: {self.device}")

        # Initialize NCBF model
        self.ncbf = NCBF(config).to(self.device)
        logger.info(f"NCBF model initialized: {self.ncbf}")

        # Training components (will be initialized during setup)
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # For mixed precision training

        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'classification_loss': [], 'barrier_loss': [], 'reg_loss': [],
            'learning_rate': [], 'safe_ratio': [], 'unsafe_ratio': []
        }

        # Data storage for visualization
        self.training_states = None
        self.training_labels = None
        self.obstacle_data = None

        # Create checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration for reproducibility
        config_path = self.checkpoint_dir / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)

    def _setup_device(self) -> torch.device:
        """Setup compute device (GPU/CPU) based on configuration."""
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                logger.info(f"CUDA available, using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device('cpu')
                logger.info("CUDA not available, using CPU")
        else:
            device = torch.device(self.config.device)
            logger.info(f"Using specified device: {device}")

        return device

    def load_training_data(self, data_path: str) -> Dict[str, Any]:
        """
        Load training data from HDF5 file and create PyTorch DataLoaders.

        Data structure from map_manager: {'states': [N, 3], 'labels': [N, 1], ...}

        Args:
            data_path: Path to HDF5 training data file

        Returns:
            Dictionary containing loaded training data
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Training data file not found: {data_path}")

        logger.info(f"Loading training data from: {data_path}")

        # Load from HDF5 using existing map_manager functionality
        # First, try to load the actual map to get real obstacle data
        map_dir = data_path.parent
        map_file = map_dir / "map1.json"  # Assume map file is in same directory

        if map_file.exists():
            # Load actual map with real obstacles
            actual_map = load_map(map_file)
            temp_map = actual_map
            # Store actual obstacle data for visualization
            self.obstacle_data = actual_map.obstacles
            logger.info(f"   Loaded actual map with {len(actual_map.obstacles)} obstacles")
        else:
            # Fallback to temporary map with dummy obstacle
            temp_map = NCBFMap(obstacles=[np.array([1.0, 1.0, 0.1])], workspace_size=8.0)
            logger.warning(f"   Map file not found: {map_file}, using dummy obstacle")

        training_data = temp_map.load_training_data(data_path)

        logger.info(f"✅ Training data loaded successfully!")
        logger.info(f"   Total samples: {training_data['num_samples']:,}")
        logger.info(f"   Safe samples: {training_data['num_safe']:,} ({100*training_data['num_safe']/training_data['num_samples']:.1f}%)")
        logger.info(f"   Unsafe samples: {training_data['num_unsafe']:,} ({100*training_data['num_unsafe']/training_data['num_samples']:.1f}%)")

        # Store data for visualization
        self.training_states = training_data['states']
        self.training_labels = training_data['labels'].squeeze()
        self.obstacle_data = training_data.get('map_info', {}).get('obstacles', [])

        return training_data

    def setup_data_loaders(self, training_data: Dict[str, Any]) -> None:
        """
        Create PyTorch DataLoaders from loaded training data.

        Args:
            training_data: Dictionary containing states and labels
        """
        logger.info("Setting up data loaders...")

        # Convert to PyTorch tensors
        states = torch.tensor(training_data['states'], dtype=torch.float32)
        labels = torch.tensor(training_data['labels'], dtype=torch.float32).squeeze()

        # Create dataset
        dataset = TensorDataset(states, labels)

        # Create train/validation split
        train_size = int(len(dataset) * (1 - self.config.validation_split))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )

        logger.info(f"Training set: {len(train_dataset):,} samples")
        logger.info(f"Validation set: {len(val_dataset):,} samples")

        # Create DataLoaders with GPU optimization
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True  # Important for BatchNorm stability
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

        logger.info("✅ Data loaders created successfully")

    def setup_training_components(self) -> None:
        """Initialize optimizer, scheduler, and mixed precision components."""
        logger.info("Setting up training components...")

        # Optimizer
        if self.config.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.ncbf.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.ncbf.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        elif self.config.optimizer.lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.ncbf.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        # Learning rate scheduler
        if self.config.scheduler.lower() == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma
            )
        elif self.config.scheduler.lower() == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_epochs
            )
        elif self.config.scheduler.lower() == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=10,
                factor=0.5
            )
        else:
            self.scheduler = None

        # Mixed precision training
        if self.config.mixed_precision and self.device.type == 'cuda':
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled")
        else:
            self.scaler = None

        logger.info(f"✅ Optimizer: {self.config.optimizer}")
        logger.info(f"✅ Scheduler: {self.config.scheduler}")
        logger.info(f"✅ Mixed precision: {self.scaler is not None}")

    def compute_losses(self, states: torch.Tensor, labels: torch.Tensor,
                      require_grad: bool = True) -> Dict[str, torch.Tensor]:
        """
        Compute the complete NCBF loss according to CLAUDE.md specifications.

        L_total = λ₁ * L_classification + λ₂ * L_barrier + λ₃ * L_reg

        Args:
            states: Input states [batch_size, state_dim]
            labels: Safety labels [batch_size] (1=safe, 0=unsafe)
            require_grad: Whether to compute gradients for barrier loss

        Returns:
            Dictionary with individual losses and total loss
        """
        # Classification Loss (Primary)
        # L_classification = max(0, -h(x_safe) + margin) + max(0, h(x_unsafe) + margin)
        h_values = self.ncbf.h(states)  # [batch_size]

        # Separate safe and unsafe samples
        safe_mask = labels == 1
        unsafe_mask = labels == 0

        # Classification loss components
        if safe_mask.any():
            h_safe = h_values[safe_mask]
            safe_loss = torch.relu(-h_safe + self.config.margin).mean()
        else:
            safe_loss = torch.tensor(0.0, device=self.device)

        if unsafe_mask.any():
            h_unsafe = h_values[unsafe_mask]
            unsafe_loss = torch.relu(h_unsafe + self.config.margin).mean()
        else:
            unsafe_loss = torch.tensor(0.0, device=self.device)

        classification_loss = safe_loss + unsafe_loss

        # Barrier Derivative Loss (Safety Enforcement)
        # L_barrier = max(0, -max_u{dh/dt} - α*h(x))
        # max_u{dh/dt} = ∇h(x)·f(x) + u_max*||∇h(x)·g(x)||

        if require_grad and self.unicycle_model is not None:
            grad_h = self.ncbf.grad_h(states)  # [batch_size, state_dim]

            # Compute f(x) and g(x) from unicycle dynamics
            # We need to implement dynamics computation based on unicycle model
            # For now, we'll use a simplified approach
            barrier_loss = self._compute_barrier_loss(states, h_values, grad_h)
        else:
            barrier_loss = torch.tensor(0.0, device=self.device)

        # SDF Regularization Loss
        # L_reg = ||∇h(x)||² with target norm ≈ 1 for SDF behavior
        if require_grad:
            grad_h = self.ncbf.grad_h(states)
            grad_norm = torch.norm(grad_h, dim=1)
            # Encourage gradient norm ≈ 1 (SDF-like behavior)
            reg_loss = torch.mean((grad_norm - 1.0) ** 2)
        else:
            reg_loss = torch.tensor(0.0, device=self.device)

        # Total loss
        total_loss = (
            self.config.classification_weight * classification_loss +
            self.config.barrier_weight * barrier_loss +
            self.config.regularization_weight * reg_loss
        )

        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'barrier_loss': barrier_loss,
            'reg_loss': reg_loss
        }

    def _compute_barrier_loss(self, states: torch.Tensor, h_values: torch.Tensor,
                             grad_h: torch.Tensor) -> torch.Tensor:
        """
        Compute barrier derivative loss using unicycle dynamics with transformation matrix.

        For unicycle: max_u{dh/dt} = ∇h(x)·f(x) + u_max*||∇h(x)·M_virtual||
        where M_virtual is the transformation matrix for virtual control space.
        L_barrier = max(0, -max_u{dh/dt} - α*h(x))
        """
        if self.unicycle_model is None:
            return torch.tensor(0.0, device=self.device)

        batch_size = states.shape[0]
        barrier_terms = []

        # Get control norm constraint from unicycle model config
        u_max = self.unicycle_model.config.max_control_norm  # This is max_control_norm from config

        for i in range(batch_size):
            # Get single state and gradient
            x = states[i].cpu().numpy()  # [px, py, theta]
            grad_h_i = grad_h[i].cpu().numpy()  # [dH/dpx, dH/dpy, dH/dtheta]
            h_i = h_values[i].item()

            # Compute f(x) - drift dynamics (unicycle has no drift)
            f_x = self.unicycle_model.f(x)  # Should be zeros for unicycle

            # Use transformation matrix instead of g(x) for virtual control space
            M_virtual = self.unicycle_model.get_transformation_matrix(x)  # Shape (3, 2)

            # Compute ∇h(x)·f(x) - this will be 0 for unicycle since f(x) = 0
            grad_h_dot_f = np.dot(grad_h_i, f_x)

            # Compute ∇h(x)·M_virtual - gradient projected onto virtual control directions
            grad_h_dot_M = np.dot(grad_h_i, M_virtual)  # Shape (2,) for virtual controls

            # Compute u_max*||∇h(x)·M_virtual|| - maximum control effort in virtual space
            # This is the key term: max_u{dh/dt} = ∇h(x)·f(x) + u_max*||∇h(x)·M_virtual||
            max_control_effort = u_max * np.linalg.norm(grad_h_dot_M)

            # Total max dh/dt
            max_dh_dt = grad_h_dot_f + max_control_effort

            # Barrier condition: max_u{dh/dt} >= -alpha*h(x)
            # Loss: max(0, -max_u{dh/dt} - alpha*h(x))
            # Use config.margin as alpha (CBF parameter)
            alpha = self.config.margin
            barrier_violation = np.maximum(0, -max_dh_dt - alpha * h_i)
            barrier_terms.append(barrier_violation)

        # Return mean barrier loss across batch
        barrier_loss = np.mean(barrier_terms)
        return torch.tensor(barrier_loss, device=self.device, dtype=torch.float32)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch and return metrics."""
        self.ncbf.train()
        total_metrics = {'total_loss': 0.0, 'classification_loss': 0.0,
                        'barrier_loss': 0.0, 'reg_loss': 0.0}
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Training Epoch {self.epoch + 1}")

        for batch_idx, (states, labels) in enumerate(pbar):
            states, labels = states.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision training
            if self.scaler is not None:
                with autocast():
                    losses = self.compute_losses(states, labels)

                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses = self.compute_losses(states, labels)
                losses['total_loss'].backward()
                self.optimizer.step()

            # Update metrics
            for key in total_metrics:
                total_metrics[key] += losses[key].item()
            num_batches += 1

            # Update progress bar
            if batch_idx % self.config.log_frequency == 0:
                pbar.set_postfix({
                    'Loss': f"{losses['total_loss'].item():.4f}",
                    'Cls': f"{losses['classification_loss'].item():.4f}",
                    'Bar': f"{losses['barrier_loss'].item():.4f}",
                    'Reg': f"{losses['reg_loss'].item():.4f}"
                })

        # Average metrics over epoch
        for key in total_metrics:
            total_metrics[key] /= num_batches

        return total_metrics

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch and return metrics."""
        self.ncbf.eval()
        total_metrics = {'total_loss': 0.0, 'classification_loss': 0.0,
                        'barrier_loss': 0.0, 'reg_loss': 0.0}
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validation Epoch {self.epoch + 1}")

            for states, labels in pbar:
                states, labels = states.to(self.device), labels.to(self.device)

                # Compute losses without gradients
                losses = self.compute_losses(states, labels, require_grad=False)

                # Update metrics
                for key in total_metrics:
                    total_metrics[key] += losses[key].item()
                num_batches += 1

        # Average metrics over epoch
        for key in total_metrics:
            total_metrics[key] /= num_batches

        return total_metrics

    def train(self, data_path: str, resume_from: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Main training loop with comprehensive monitoring.

        Args:
            data_path: Path to HDF5 training data file
            resume_from: Optional path to checkpoint for resuming training

        Returns:
            Training history dictionary
        """
        logger.info("🚀 Starting NCBF Training")
        logger.info("=" * 60)

        start_time = time.time()

        # Setup training
        training_data = self.load_training_data(data_path)
        self.setup_data_loaders(training_data)
        self.setup_training_components()

        if resume_from:
            self.load_checkpoint(resume_from)

        logger.info(f"Training for {self.config.max_epochs} epochs")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Device: {self.device}")

        # Training loop
        for epoch in range(self.epoch, self.config.max_epochs):
            self.epoch = epoch
            epoch_start = time.time()

            # Training phase
            train_metrics = self.train_epoch()

            # Validation phase
            val_metrics = self.validate_epoch()

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_loss'])
                else:
                    self.scheduler.step()

            # Update training history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.training_history['train_loss'].append(train_metrics['total_loss'])
            self.training_history['val_loss'].append(val_metrics['total_loss'])
            self.training_history['classification_loss'].append(train_metrics['classification_loss'])
            self.training_history['barrier_loss'].append(train_metrics['barrier_loss'])
            self.training_history['reg_loss'].append(train_metrics['reg_loss'])
            self.training_history['learning_rate'].append(current_lr)

            # Compute safety statistics on validation set
            safe_ratio, unsafe_ratio = self._compute_safety_statistics()
            self.training_history['safe_ratio'].append(safe_ratio)
            self.training_history['unsafe_ratio'].append(unsafe_ratio)

            # Logging
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch + 1}/{self.config.max_epochs} "
                       f"({epoch_time:.1f}s) - "
                       f"Train Loss: {train_metrics['total_loss']:.4f}, "
                       f"Val Loss: {val_metrics['total_loss']:.4f}, "
                       f"LR: {current_lr:.2e}")

            # Visualization
            if (epoch + 1) % self.config.plot_frequency == 0:
                self._plot_training_progress()

            # Checkpoint saving
            if self._should_save_checkpoint(epoch, val_metrics):
                self._save_checkpoint(epoch, val_metrics)

            # Early stopping
            if self._check_early_stopping(val_metrics):
                logger.info(f"🛑 Early stopping triggered at epoch {epoch + 1}")
                break

        # Final evaluation and saving
        total_time = time.time() - start_time
        logger.info(f"🎉 Training completed in {total_time:.1f} seconds")

        self._save_final_model()
        self._generate_training_report()

        if self.config.show_training_plots:
            plt.show()

        return self.training_history

    def _should_save_checkpoint(self, epoch: int, val_metrics: Dict[str, float]) -> bool:
        """Determine if checkpoint should be saved."""
        # Save based on frequency or best validation loss
        if (epoch + 1) % self.config.save_frequency == 0:
            return True

        if self.config.save_best_only and val_metrics['total_loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['total_loss']
            return True

        return False

    def _check_early_stopping(self, val_metrics: Dict[str, float]) -> bool:
        """Check if early stopping should be triggered."""
        if len(self.training_history['val_loss']) < self.config.early_stopping_patience:
            return False

        # Check if validation loss has improved in the last N epochs
        recent_losses = self.training_history['val_loss'][-self.config.early_stopping_patience:]
        best_recent = min(recent_losses[:-1])  # Exclude current epoch

        return val_metrics['total_loss'] > best_recent

    def _compute_safety_statistics(self) -> Tuple[float, float]:
        """Compute safety statistics on validation set."""
        self.ncbf.eval()
        all_predictions = []

        with torch.no_grad():
            for states, _ in self.val_loader:
                states = states.to(self.device)
                h_values = self.ncbf.h(states)
                predictions = (h_values > 0).float()  # 1 if safe, 0 if unsafe
                all_predictions.append(predictions.cpu())

        all_predictions = torch.cat(all_predictions)
        safe_ratio = (all_predictions == 1).float().mean().item()
        unsafe_ratio = (all_predictions == 0).float().mean().item()

        return safe_ratio, unsafe_ratio

    def _plot_training_progress(self) -> None:
        """Generate training progress plots."""
        if not self.training_history['train_loss']:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.training_history['train_loss']) + 1)

        # 1. Total loss curves
        axes[0, 0].plot(epochs, self.training_history['train_loss'], 'b-', label='Training', linewidth=2)
        axes[0, 0].plot(epochs, self.training_history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')

        # 2. Individual loss components
        axes[0, 1].plot(epochs, self.training_history['classification_loss'], 'g-', label='Classification', linewidth=2)
        axes[0, 1].plot(epochs, self.training_history['barrier_loss'], 'orange', label='Barrier', linewidth=2)
        axes[0, 1].plot(epochs, self.training_history['reg_loss'], 'purple', label='Regularization', linewidth=2)
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')

        # 3. Learning rate
        axes[1, 0].plot(epochs, self.training_history['learning_rate'], 'brown', linewidth=2)
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Safety statistics
        if self.training_history['safe_ratio']:
            axes[1, 1].plot(epochs, self.training_history['safe_ratio'], 'b-', label='Safe %', linewidth=2)
            axes[1, 1].plot(epochs, self.training_history['unsafe_ratio'], 'r-', label='Unsafe %', linewidth=2)
            axes[1, 1].set_title('Safety Distribution on Validation Set')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Percentage')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'training_progress.png', dpi=150, bbox_inches='tight')

        if self.config.show_training_plots:
            plt.show(block=False)
            plt.pause(0.1)
        else:
            plt.close()

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.ncbf.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config.__dict__,
            'metrics': metrics
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt'
        torch.save(checkpoint, checkpoint_path)

        # Also save as best model if it's the best so far
        if metrics['total_loss'] <= self.best_val_loss:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"💾 Best model saved to: {best_path}")

        logger.info(f"💾 Checkpoint saved to: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.ncbf.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Load training state
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']

        logger.info(f"✅ Checkpoint loaded from epoch {self.epoch + 1}")

    def _save_final_model(self) -> None:
        """Save final trained model."""
        final_path = self.checkpoint_dir / 'final_model.pt'

        # Save just the model and config (not training state)
        torch.save({
            'model_state_dict': self.ncbf.state_dict(),
            'config': self.config.__dict__,
            'training_history': self.training_history
        }, final_path)

        logger.info(f"🎯 Final model saved to: {final_path}")

    def _generate_training_report(self) -> None:
        """Generate comprehensive training report."""
        report_path = self.checkpoint_dir / 'training_report.txt'

        with open(report_path, 'w') as f:
            f.write("NCBF Training Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Training Duration: {len(self.training_history['train_loss'])} epochs\n")
            f.write(f"Final Training Loss: {self.training_history['train_loss'][-1]:.6f}\n")
            f.write(f"Final Validation Loss: {self.training_history['val_loss'][-1]:.6f}\n")
            f.write(f"Best Validation Loss: {self.best_val_loss:.6f}\n\n")

            f.write("Configuration:\n")
            for key, value in self.config.__dict__.items():
                f.write(f"  {key}: {value}\n")

            f.write("\nFinal Model Statistics:\n")
            model_info = self.ncbf.get_model_info()
            for key, value in model_info.items():
                f.write(f"  {key}: {value}\n")

        logger.info(f"📋 Training report saved to: {report_path}")

    def visualize_contours(self, resolution: int = None, theta_fixed: float = None) -> None:
        """
        Generate improved contour visualization of the learned h(x) function.

        Creates:
        - 2D contour plot of h(x) over workspace with equal axes
        - Enhanced safety boundary visualization with obstacles
        - Gradient field visualization
        - Training data overlay
        """
        if resolution is None:
            resolution = self.config.contour_resolution
        if theta_fixed is None:
            theta_fixed = self.config.evaluation_theta

        logger.info("Generating improved contour visualization...")

        # Create evaluation grid
        x_range = np.linspace(0, 8, resolution)
        y_range = np.linspace(0, 8, resolution)
        X, Y = np.meshgrid(x_range, y_range)

        # Evaluate h(x) on grid (fix theta for 2D visualization)
        grid_states = np.column_stack([
            X.ravel(), Y.ravel(),
            np.full(X.size, theta_fixed)
        ])

        # Convert to tensor and move to correct device
        grid_states_tensor = torch.tensor(grid_states, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            h_values = self.ncbf.h(grid_states_tensor).cpu().numpy().reshape(X.shape)

        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Contour plot with zero line - IMPROVED
        contour = axes[0, 0].contour(X, Y, h_values, levels=20, cmap='RdBu_r', alpha=0.8)
        axes[0, 0].clabel(contour, inline=True, fontsize=8)
        zero_contour = axes[0, 0].contour(X, Y, h_values, levels=[0], colors='black', linewidths=3)
        axes[0, 0].set_title('h(x) Contours with Safety Boundary (h=0)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('X Position (m)', fontsize=10)
        axes[0, 0].set_ylabel('Y Position (m)', fontsize=10)
        axes[0, 0].set_aspect('equal')  # Ensure equal aspect ratio
        axes[0, 0].grid(True, alpha=0.3)

        # 2. True obstacles + safety boundary + 0-level set - COMPLETE VISUALIZATION
        ax = axes[0, 1]

        # Load the actual map and get safety radius from config
        try:
            # Load map to get obstacles
            map_path = '/home/chengrui/wk/NCBFquickDemo/work/ncbf/maps/map_files/map1/map1.json'
            obstacle_map = load_map(map_path)
            obstacles = obstacle_map.obstacles

            # Get safety radius from unicycle config
            from configs.unicycle_config import UnicycleConfig
            unicycle_config = UnicycleConfig()
            safety_radius = unicycle_config.safety_radius

            print(f"📍 Loaded {len(obstacles)} obstacles from map")
            print(f"📏 Safety radius: {safety_radius}m")

            # Draw each obstacle with its safety boundary
            for i, obs in enumerate(obstacles):
                obs_x, obs_y, obs_radius = obs
                print(f"   Obstacle {i+1}: center=({obs_x:.2f}, {obs_y:.2f}), radius={obs_radius:.2f}")

                # 1. Draw TRUE SAFETY BOUNDARY (obstacle + safety_radius) - ORANGE DASHED
                safety_boundary_radius = obs_radius + safety_radius
                safety_circle = plt.Circle((obs_x, obs_y), safety_boundary_radius,
                                         facecolor='orange', alpha=0.3, fill=True,
                                         edgecolor='orange', linewidth=2, linestyle='--')
                ax.add_patch(safety_circle)

                # 2. Draw TRUE OBSTACLE (red solid) on top
                obstacle_circle = plt.Circle((obs_x, obs_y), obs_radius,
                                           facecolor='red', alpha=0.8, fill=True,
                                           edgecolor='darkred', linewidth=2)
                ax.add_patch(obstacle_circle)

                # 3. Mark center with black dot
                ax.plot(obs_x, obs_y, 'ko', markersize=4, alpha=0.9)

            # 4. Add 0-LEVEL SET from learned NCBF (extract as lines to avoid covering)
            try:
                # Extract 0-level contour paths
                zero_contour = ax.contour(X, Y, h_values, levels=[0])
                if zero_contour.collections:
                    paths = zero_contour.collections[0].get_paths()

                    # Plot each path segment as individual lines
                    for path in paths:
                        vertices = path.vertices
                        x_vals = vertices[:, 0]
                        y_vals = vertices[:, 1]
                        # Plot as blue line (learned boundary)
                        ax.plot(x_vals, y_vals, 'b-', linewidth=4, alpha=0.9)

                # Add label for learned boundary
                if paths:
                    mid_path = paths[len(paths)//2]
                    mid_vertex = mid_path.vertices[len(mid_path.vertices)//2]
                    ax.text(mid_vertex[0], mid_vertex[1], 'h=0', fontsize=10,
                           ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2',
                           facecolor='white', alpha=0.8))
            except:
                pass  # Skip if no 0-level contour found

        except Exception as e:
            print(f"⚠️  Could not load map: {e}")
            # Fallback - draw placeholder if map loading fails
            ax.text(4, 4, 'No Map Data', fontsize=14, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

        # Professional formatting
        ax.set_title('True Obstacles + Safety Boundary + Learned h=0', fontsize=12, fontweight='bold')
        ax.set_xlabel('X Position (m)', fontsize=10)
        ax.set_ylabel('Y Position (m)', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)

        # Comprehensive legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='orange', alpha=0.3, label='True Safety Boundary'),
            Patch(facecolor='red', alpha=0.8, label='True Obstacles'),
            plt.Line2D([0], [0], color='blue', linewidth=4, label='Learned h=0'),
            plt.Line2D([0], [0], marker='o', color='black', markersize=4, label='Center', linestyle='None')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

        # 3. Gradient magnitude - IMPROVED
        grad_x, grad_y = np.gradient(h_values, x_range[1] - x_range[0], y_range[1] - y_range[0])
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        im = axes[1, 0].imshow(grad_magnitude, extent=[0, 8, 0, 8], origin='lower', cmap='viridis', aspect='equal')
        axes[1, 0].set_title('Gradient Magnitude ||∇h(x)||', fontsize=12, fontweight='bold')
        cbar = plt.colorbar(im, ax=axes[1, 0], shrink=0.8)
        cbar.set_label('Gradient Magnitude', fontsize=9)
        axes[1, 0].set_xlabel('X Position (m)', fontsize=10)
        axes[1, 0].set_ylabel('Y Position (m)', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Training data overlay - IMPROVED
        if self.training_states is not None and self.training_labels is not None:
            safe_mask = self.training_labels == 1
            unsafe_mask = self.training_labels == 0

            # Sample training data for visualization (to avoid overcrowding)
            sample_size = min(1000, len(self.training_states))
            indices = np.random.choice(len(self.training_states), sample_size, replace=False)

            sample_states = self.training_states[indices]
            sample_labels = self.training_labels[indices]

            safe_sample_mask = sample_labels == 1
            unsafe_sample_mask = sample_labels == 0

            axes[1, 1].scatter(sample_states[safe_sample_mask, 0],
                              sample_states[safe_sample_mask, 1],
                              c='blue', s=3, alpha=0.7, label=f'Safe ({np.sum(safe_sample_mask)})',
                              edgecolors='none')
            axes[1, 1].scatter(sample_states[unsafe_sample_mask, 0],
                              sample_states[unsafe_sample_mask, 1],
                              c='red', s=3, alpha=0.7, label=f'Unsafe ({np.sum(unsafe_sample_mask)})',
                              edgecolors='none')

            axes[1, 1].set_title('Training Data Distribution', fontsize=12, fontweight='bold')
            axes[1, 1].legend(fontsize=9, loc='upper right')
            axes[1, 1].set_xlabel('X Position (m)', fontsize=10)
            axes[1, 1].set_ylabel('Y Position (m)', fontsize=10)
            axes[1, 1].set_aspect('equal')  # Ensure equal aspect ratio
            axes[1, 1].grid(True, alpha=0.3)

        # Global improvements
        for ax in axes.flat:
            ax.set_xlim(0, 8)
            ax.set_ylim(0, 8)
            ax.tick_params(labelsize=9)

        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'contour_evaluation.png', dpi=300, bbox_inches='tight')

        if self.config.show_training_plots:
            plt.show()
        else:
            plt.close()

        logger.info(f"📊 Contour visualization saved to: {self.checkpoint_dir / 'contour_evaluation.png'}")

    def __str__(self) -> str:
        """String representation of the trainer."""
        return (f"NCBFTrainer(epoch={self.epoch}, device={self.device}, "
                f"best_val_loss={self.best_val_loss:.4f})")

    def __repr__(self) -> str:
        """Detailed representation of the trainer."""
        return self.__str__()