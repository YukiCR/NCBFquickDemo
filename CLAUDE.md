# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Neural Control Barrier Functions (NCBF) implementation for safe unicycle robot navigation. The project combines classical control theory with neural networks to learn safety certificates for robotic systems.

## Essential Commands

### Environment Setup
```bash
# Activate conda environment (required for all operations)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate latentSafety

# Install dependencies (if needed)
pip install -r requirements.txt
```

### Training Neural CBF
```bash
cd /home/chengrui/wk/NCBFquickDemo/work/ncbf/training
python train_ncbf.py --data /home/chengrui/wk/NCBFquickDemo/work/ncbf/maps/map_files/map1/training_data_large.h5 \
                     --config large --epochs 3000 \
                     --output_dir /home/chengrui/wk/NCBFquickDemo/work/ncbf/weights/my_training \
                     --visualize
```

### Running Control Simulation
```bash
cd /home/chengrui/wk/NCBFquickDemo/work/sim
python sim_ncbf.py --initial-x 0.5 --initial-y 0.5 --goal-x 7 --goal-y 7 --paper-ready \
                   --output results/control_simulation.png --no-display
```

### Testing and Validation
```bash
# Run validation tests (no formal test framework - individual scripts)
cd /home/chengrui/wk/NCBFquickDemo/work/test

# Basic functionality test
python simple_ncbf_test.py

# Comprehensive NCBF vs handwritten CBF comparison
python test_ncbf_visualization.py

# Training performance analysis
python evaluate_intensive_training.py

# Individual component tests
python test_obstacles_only.py
python test_final_model_contours.py
```

### Visualizing Training Data
```bash
cd /home/chengrui/wk/NCBFquickDemo/work/ncbf/maps
python visualize_training_data.py --training_data map1/training_data_new.h5 \
                                   --map map1/map1.json \
                                   --output results/training_analysis.png --save-only --quiet
```

### Visualizing Trained Model
```bash
cd /home/chengrui/wk/NCBFquickDemo/work/ncbf/training
python ncbf_visualization_tool.py --checkpoint /home/chengrui/wk/NCBFquickDemo/work/ncbf/weights/train_3000ep_largenn/best_model.pt \
                                  --config /home/chengrui/wk/NCBFquickDemo/work/ncbf/weights/train_3000ep_largenn/training_config.json \
                                  --map /home/chengrui/wk/NCBFquickDemo/work/ncbf/maps/map_files/map1/map1.json \
                                  --safety-radius 0.2 --resolution 100 \
                                  --output results/ncbf_model_visualization.png
```

### Map Generation and Data Preparation
```bash
# Generate new training maps
cd /home/chengrui/wk/NCBFquickDemo/work/ncbf/maps
python map_generator.py --num-obstacles 8 --workspace-size 8.0 --output my_map.json

# Generate training data for a map
python generate_training_data.py --map map1/map1.json --num-samples 10000 --output my_training_data.h5
```

## Code Architecture

### Core Inheritance Hierarchy
```
ControlAffineSystem (ABC)
├── UnicycleModel (concrete implementation with PD control)

CBFFunction (ABC)
├── NCBF (neural network implementation + nn.Module)
├── CBFsingleobs (handwritten single obstacle CBF)
└── CBFmultipleobs (handwritten multiple obstacle CBF)

CBFFilter (safety enforcement using analytical QP solution)
```

### Key Components

**System Layer** (`/work/models/`):
- `control_affine_system.py`: Abstract base for control-affine dynamics `ẋ = f(x) + g(x)u`
- `unicycle_model.py`: 3-state unicycle [px, py, θ] with PD controllers and visualization

**CBF Layer** (`/work/safe_control/`, `/work/ncbf/`):
- `cbf_function.py`: Abstract CBF interface with `h(x)` and `grad_h(x)` methods
- `ncbf/models/ncbf.py`: Neural network CBF with configurable architecture
- `cbf_filter.py`: Safety filter using Lie derivatives and analytical QP

**Training Infrastructure** (`/work/ncbf/training/`):
- `train_ncbf.py`: Main training script with configuration support
- `ncbf_trainer.py`: Training loop with loss functions and validation
- `ncbf_visualization_tool.py`: Model evaluation and contour plotting

**Configuration** (`/work/configs/`, `/work/ncbf/configs/`):
- `unicycle_config.py`: Robot parameters, safety settings, controller gains
- `ncbf_config.py`: Neural network architecture and training hyperparameters

### Critical Design Patterns

**Safety Filtering Algorithm**:
1. Compute Lie derivatives: `L_f h = ∇h·f(x)`, `L_g h = ∇h·g(x)`
2. Apply constraint: `L_f h + L_g h·u ≥ -αh(x)`
3. Solve QP analytically: minimize `||u - u_nom||²` subject to CBF constraint
4. Handle underactuation via virtual dynamics matrix `M`

**Key Methods**:
- `h(x)`: Barrier function value (≥0 indicates safe states)
- `grad_h(x)`: Gradient for Lie derivative computation
- `compute_safe_control()`: Main safety filtering with analytical solution
- `pd_control_proportional()`: Primary controller using transformation matrix

### Testing Infrastructure

**No formal testing framework** - tests are individual validation scripts in `/work/test/`:
- `simple_ncbf_test.py`: Basic model loading and contour visualization
- `test_ncbf_visualization.py`: Comprehensive neural vs handwritten CBF comparison
- `evaluate_intensive_training.py`: Training configuration performance analysis
- `test_obstacles_only.py`: Obstacle visualization verification
- `test_final_model_contours.py`: Final model contour plotting tests

**Run tests individually**: `python test_script.py` (no test runner available)

### Important Implementation Details

**State Representation**: Unicycle uses 3D state [px, py, θ] with 2D control [v, ω]
**Safety Margin**: 0.2m safety radius around obstacles (matches robot radius)
**Training Data**: 76.3% safe, 23.7% unsafe samples with obstacle-focused sampling
**Network Architecture**: Configurable MLP, default [256, 256, 128, 64, 32] → 1 output
**Control Constraints**: Norm constraint `sqrt(v² + ω²) ≤ 2.0` for explicit optimal control
**Time Step**: 0.05s for simulation integration

### Configuration Files

Primary configuration managed through dataclasses:
- `UnicycleConfig`: Robot parameters, safety settings, visualization options
- `NCBFConfig`: Network architecture, training parameters, loss weights

Pre-trained models available in `/work/ncbf/weights/train_3000ep_largenn/`
Training data and maps in `/work/ncbf/maps/map_files/map1/`

## Development Workflow

### Adding New Features
1. Follow existing inheritance patterns (extend ControlAffineSystem or CBFFunction)
2. Use configuration dataclasses for new parameters
3. Implement required abstract methods
4. Add validation tests in `/work/test/`

### Debugging Safety Issues
1. Check `h(x)` values around obstacles using visualization tools
2. Verify Lie derivative computations with finite differences
3. Test analytical QP solution against numerical solvers
4. Examine virtual dynamics matrix `M` for underactuated systems

### Performance Optimization
1. Use analytical QP solution for single constraints (faster than numerical)
2. Enable mixed precision training in NCBFConfig
3. Adjust contour resolution for visualization speed vs accuracy trade-off
4. Batch multiple safety evaluations when possible

## Theoretical notations
### NCBF Definitions and Notation
Neural Control Barrier Function h(x):
+ Learned neural network function h(x) that serves as safety certificate
+ Safety condition: h(x) ≥ 0 for safe regions, h(x) < 0 for unsafe states
+ Time derivative condition: dh/dt ≥ -αh where α > 0 is a positive parameter
+ This ensures forward invariance of safe set (if h(x(0)) ≥ 0, then h(x(t)) ≥ 0 ∀t ≥ 0)
+ CBF-QP:   The safety filter solves:
  ```
  minimize    ||u - u_nom||²
  subject to  L_f h + L_g h·u ≥ -αh(x)
  ```
    Where:
  - L_f h = ∇h·f(x) (Lie derivative along drift dynamics)
  - L_g h = ∇h·g(x) (Lie derivative along control directions)
  - α > 0 is the barrier parameter (typically 0.3-0.5)
  - h(x) is the barrier function value
  - u is the control input
+ 

### Neural CBF Loss Function Design

**Total Loss:**
```
L_total = λ₁ * L_classification + λ₂ * L_barrier + λ₃ * L_reg
```

**1. Classification Loss (Primary):**
```
L_classification = max(0, -h(x_safe) + margin) + max(0, h(x_unsafe) + margin)
```
- h(x_safe) should be ≥ margin (e.g., margin = 0.3)
- h(x_unsafe) should be ≤ -margin
- Uses hinge loss for clear safe/unsafe separation

**2. Barrier Derivative Loss (Safety Enforcement):**
```
L_barrier = max(0, -max_u{dh/dt} - α*h(x))
```
- **Key insight**: max_u{dh/dt} = ∇h(x)·f(x) + u_max*||∇h(x)·g(x)||
- Ensures existence of safe control (not all controls are safe)
- α > 0 is barrier parameter (e.g., α = 0.3)

**3. SDF Regularization Loss (Optional):**
```
L_reg = ||∇h(x)||²  # Modified to make ||∇h(x)|| ≈ 1 for SDF-like behavior
```

### Virtual Transformation Approach for Underactuated Systems

**Problem Solved**: Unicycle systems are underactuated (control: [v, ω]) but CBF constraints need to be applied as if the system were fully actuated.

**Key Insight**: By using transformation matrix M, we control a virtual point ahead of the unicycle center, making the system behave as if it were fully actuated in constraint space.

**Mathematical Foundation**:
- **Underactuated Control**: u = [v, ω] (linear velocity, angular velocity)
- **Virtual Control**: [vx, vy, vω] = M * [v, ω] where M transforms to virtual velocities
- **Transformation Matrix**: M = [[cos(θ), -D*sin(θ)], [sin(θ), D*cos(θ)], [0, 1]]
- **Virtual Point**: Point D meters ahead of robot center, where D is a design parameter

**Why This Works**:
1. **Constraint Application**: CBF constraints L_g h·u ≥ b are computed in virtual space
2. **Physical Interpretation**: We're controlling a point ahead of the robot, giving us more "leverage" for constraint satisfaction
3. **Mathematical Soundness**: Follows established MATLAB carModel.m implementation pattern
4. **Constraint Satisfaction**: Virtual point provides additional degrees of freedom for satisfying safety constraints

**Implementation Details**:
- `get_transformation_matrix(x)` method in UnicycleModel returns proper 3×2 matrix
- CBF filter uses virtual dynamics for Lie derivative computation: L_g_virtual h = ∇h·M
- Analytical QP solution handles weighted norm MᵀM for proper constraint formulation
- Control constraints still applied through unicycle's `update_state` method (norm truncation)

## Future work
1. train NCBF with safe data only : learn NCBF with safe demonstrations
2. learn latent space CBF : use a encoder to encode high-dimensional input (like vision input), than train NCBF that uses the latent vector as input
3. learn task-conditioned NCBF : safety should be context dependent, use natural language decription as condition to judge whether a state is safe