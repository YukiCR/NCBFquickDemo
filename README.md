# Neural Control Barrier Functions (NCBF) for Unicycle Navigation

A comprehensive implementation of Neural Control Barrier Functions for safe unicycle robot navigation, featuring learned safety certificates, visualization tools, and pre-generated figures.

## 🎯 Project Overview

This project implements a complete Neural Control Barrier Function (NCBF) system for safe robotic navigation, extending the original neural_clbf framework with:

- **Learned Safety Certificates**: Neural networks trained to distinguish safe from unsafe states
- **Real-time Safety Filtering**: Live constraint enforcement during robot navigation


## 📁 File Structure

```
/home/chengrui/wk/NCBFquickDemo/
├── work/                           # Main implementation directory
│   ├── models/                     # Robot models and control systems
│   │   ├── control_affine_system.py # Abstract base class for control-affine systems
│   │   ├── unicycle_model.py       # Unicycle robot with PD control and visualization
│   │   └── __pycache__/            # Compiled Python files
│   ├── configs/                    # Configuration management
│   │   ├── unicycle_config.py      # Unicycle-specific parameters and settings
│   │   └── __pycache__/            # Compiled Python files
│   ├── safe_control/               # Safety control and CBF implementation
│   │   ├── cbf_function.py         # Abstract base class for CBF functions
│   │   ├── cbf_filter.py           # CBF safety filter with Lie derivatives
│   │   ├── handwritten_cbf.py      # Handwritten CBF implementations
│   │   └── __pycache__/            # Compiled Python files
│   ├── ncbf/                       # Neural CBF implementation
│   │   ├── models/                 # Neural network models
│   │   │   └── ncbf.py            # Neural CBF implementation
│   │   ├── configs/                # NCBF configuration
│   │   │   └── ncbf_config.py     # network architecture settings and hyperparameters
│   │   ├── maps/                   # Map generation and management
│   │   │   ├── map_manager.py     # Map loading and utilities
│   │   │   ├── map_generator.py   # Random map generation
│   │   │   ├── map_generation.py  # Map data structures
│   │   │   ├── generate_training_data.py # Training data generation
│   │   │   ├── visualize_training_data.py # Training data visualization
│   │   │   └── map_files/         # Pre-generated map files
│   │   ├── training/               # NCBF training infrastructure
│   │   │   ├── train_ncbf.py      # Main training script
│   │   │   ├── ncbf_trainer.py    # Training loop implementation
│   │   │   ├── ncbf_visualization_tool.py # Model visualization
│   │   │   └── results/           # Training outputs and visualizations
│   │   └── weights/                # Trained model weights
│   ├── sim/                        # Control simulation
│   │   ├── sim_ncbf.py            # Main simulation script
│   │   └── results/               # Simulation outputs
│   └── test/                       # Testing and validation
├── pre/                           # Pre-generated results and figures
├── reference/                     # Reference materials
│   ├── mcode/                     # MATLAB reference implementations
│   └── papers/                    # Academic papers and documentation
└── CLAUDE.md                      # Project guidance for AI assistants
```

## 🔗 Class Relationships

### **Base Classes and Inheritance Hierarchy**

```
ControlAffineSystem (ABC)
├── UnicycleModel
└── [Other robot models]

CBFFunction (ABC)
├── CBFsingleobs (Handwritten CBF)
├── CBFmultipleobs (Handwritten CBF)
└── NCBF (Neural CBF + nn.Module)

NCBFConfig
└── NCBF (uses for architecture)

UnicycleConfig
└── UnicycleModel (uses for parameters)
```

### **Key Class Methods**

**ControlAffineSystem** (Abstract Base):
- `f(x)`: Drift dynamics - returns zero for unicycle (no drift when u=0)
- `g(x)`: Control input matrix, for unicycle there's a special treatment
- `state_dim()`: Returns 3 for unicycle [px, py, θ]
- `control_dim()`: Returns 2 for unicycle [v, ω]

**UnicycleModel** (Concrete Implementation):
- `pd_control_proportional()`: PD control using transformation matrix, performs better
- `pd_control_basic()`: Simple PD control for position tracking
- `get_transformation_matrix()`: Converts underactuated to virtually fully-actuated
- `update_state()`: Integrates dynamics with control constraints

**CBFFunction** (Abstract Base):
- `h(x)`: Barrier function value (h ≥ 0 for safe states)
- `grad_h(x)`: Gradient ∇h(x) for Lie derivative computation
- `is_safe(x)`: Boolean safety check
- `get_safety_level(x)`: Numerical safety assessment

**NCBF** (Neural Implementation):
- Forward pass: MLP architecture with configurable hidden layers
- `grad_h(x)`: Automatic differentiation for gradient computation
- Model saving/loading functionality
- Support for both numpy arrays and PyTorch tensors

**CBFFilter** (Safety Enforcement):
- `compute_safe_control()`: Analytical QP solution for single constraints
- Lie derivative computation: L_f h = ∇h·f(x), L_g h = ∇h·g(x)
- Real-time safety filtering

## 🛠️ Command-Line Tools

### **1. Map Generation and Management**

```bash
# Generate random maps with obstacles
source ~/miniconda3/etc/profile.d/conda.sh && conda activate latentSafety
cd /home/chengrui/wk/NCBFquickDemo/work/ncbf/maps
python map_generator.py --num-obstacles 8 --workspace-size 8.0 --output my_map.json

# Generate training data for a specific map
python generate_training_data.py --map map1/map1.json --num-samples 10000 --output my_training_data.h5
```

### **2. NCBF Training**

```bash
# Train Neural CBF with large network (our trained model)
cd /home/chengrui/wk/NCBFquickDemo/work/ncbf/training
python train_ncbf.py --data /home/chengrui/wk/NCBFquickDemo/work/ncbf/maps/map_files/map1/training_data_large.h5 \
                     --config large --epochs 3000 \
                     --output_dir /home/chengrui/wk/NCBFquickDemo/work/ncbf/weights/my_training \
                     --visualize

# Train with custom parameters
python train_ncbf.py --data map_files/map1/training_data.h5 \
                     --config medium --epochs 500 \
                     --output_dir weights/custom_training \
                     --batch-size 256 --lr 1e-3
```

### **3. NCBF Model Visualization**

```bash
# Visualize trained NCBF model (three-panel equal sizing)
python ncbf_visualization_tool.py --checkpoint /home/chengrui/wk/NCBFquickDemo/work/ncbf/weights/train_3000ep_largenn/best_model.pt \
                                  --config /home/chengrui/wk/NCBFquickDemo/work/ncbf/weights/train_3000ep_largenn/training_config.json \
                                  --map /home/chengrui/wk/NCBFquickDemo/work/ncbf/maps/map_files/map1/map1.json \
                                  --safety-radius 0.2 --resolution 100 \
                                  --output results/ncbf_model_visualization.png

# Quick visualization with lower resolution
python ncbf_visualization_tool.py --checkpoint weights/best_model.pt \
                                  --config weights/training_config.json \
                                  --resolution 50 --theta 0.0
```

### **4. Control Simulation**

```bash
# Run NCBF control simulation (paper-ready two-panel)
cd /home/chengrui/wk/NCBFquickDemo/work/sim
python sim_ncbf.py --initial-x 0.5 --initial-y 0.5 --goal-x 7 --goal-y 7 --paper-ready \
                   --output results/control_simulation.png --no-display

# Custom simulation with different parameters
python sim_ncbf.py --initial-x 1.0 --initial-y 1.0 --initial-theta 0.5 \
                   --goal-x 6.0 --goal-y 6.0 --max-sim-time 20.0 \
                   --output results/custom_simulation.png
```

### **5. Training Data Analysis**

```bash
# Visualize training data distribution (three-panel compact)
cd /home/chengrui/wk/NCBFquickDemo/work/ncbf/maps
python visualize_training_data.py --training_data map1/training_data_new.h5 \
                                   --map map1/map1.json \
                                   --output results/training_analysis.png --save-only --quiet

# Analyze training data with statistics
python visualize_training_data.py --training_data map_files/map1/training_data.h5 \
                                   --map map_files/map1/map1.json
```

## 📊 Example Results

### **Training Results**:
- **Model Size**: 111,553 parameters (large network)
- **Training Accuracy**: 90.1% safe classification on test data
- **Architecture**: [256, 256, 128, 64, 32] → 1 (configurable)

### **Control Performance**:
- **Safety Rate**: 90.1% safe classification on navigation scenarios
- **Goal Success**: Successful navigation with safety constraint enforcement
- **Real-time**: Analytical QP solution for single constraints

### **Data Statistics**:
- **Training Data**: 10,000 samples (76.3% safe, 23.7% unsafe)
- **Map Coverage**: 8×8m workspace with 8 obstacles
- **Safety Margin**: 0.2m safety radius around obstacles due to the radius of unicycle

## 🎯 Quick Start Examples

### **Basic Navigation**:
```bash
# Navigate from (0.5, 0.5) to (7, 7) with safety filtering
python sim_ncbf.py --initial-x 0.5 --initial-y 0.5 --goal-x 7 --goal-y 7
```

### **Training New Model**:
```bash
# Train on custom map with medium configuration
python train_ncbf.py --data my_map/training_data.h5 --config medium --epochs 1000
```

### **Analyze Results**:
```bash
# Visualize trained model performance
python ncbf_visualization_tool.py --checkpoint my_model.pt --map my_map.json --resolution 100
```

## 🔧 Configuration

### **Key Configuration Files**:
- `/work/configs/unicycle_config.py`: Robot parameters, safety settings
- `/work/ncbf/configs/ncbf_config.py`: Neural network architecture
- Map files in `/work/ncbf/maps/map_files/`: Pre-generated training environments

## 📚 Additional Resources

- **Reference Materials**: `/reference/` contains MATLAB implementations and academic papers
- **Test Suite**: `/test/` includes comprehensive validation scripts
- **Pre-trained Models**: `/work/ncbf/weights/` contains our trained 111,553 parameter model
- **Example Maps**: `/work/ncbf/maps/map_files/` has pre-generated training environments

---

**Note**: 
+ Some of the code/files in this project were generated with AI assistance, and there may be mistakes. Always validate the mathematical properties and safety guarantees for your specific application before deployment.
+ there may be some path problem, please fix it yourself 