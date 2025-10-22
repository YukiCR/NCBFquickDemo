# Neural Control Barrier Functions with Only Safe Data - Project Plan

## Overall Goal
Learn Neural Control Barrier Functions (NCBF) using only safe demonstration data, without access to unsafe state samples or the ability to sample unsafe states in the state space.

## Project Status Overview
**Current Focus**: Conservative learning approach from Tabbara et al. 2025 - implementing log-sum-exp conservative loss with two-phase training

## Three General Implementation Steps

### Step 1: Original Learning Pipeline with Only Safe Data  COMPLETED
**Status**: Implemented and tested - results were poor due to hinge loss limitations with only positive data

**Implementation Details**:
- [x] Created training data filter (`@work/ncbf/maps/map_filter.py`) to extract safe-only datasets
- [x] Filtered out dataset with only positive (safe) data from existing training data
- [x] Tested standard NCBF training with safe-only data
- [x] **Result**: Poor performance because hinge loss cannot work effectively without negative examples

### Step 2: Modify Learning Procedure (Conservative Learning) = IN PROGRESS
**Status**: Conservative learning implementation started - this is our current focus

**Implementation Details**:
- [x] **Conservative Loss Implementation**: Added new trainer class (`@work/ncbf/training/conservative_ncbf_trainer.py`)
- [x] **CLI Interface**: Created training script (`@work/ncbf/training/train_conservative_ncbf.py`)
- [x] Test the conservative learning pipline:
  - [x] use full data (both safe and unsafe) to train NCBF, to see if the code runs well as work/ncbf/training/train_ncbf.py and work/ncbf/training/ncbf_trainer.py
  - [x] **Result**: By setting the conservative term to 0, the added training pipline behaves like the original code. The conservative loss does add some pessisism in learning procedure, but cannot work without negative data
- [x] **Two-Phase Training**: Implemented pretraining procedure to first train state space to be unsafe
  - [x] basic code implemenatation
    - [x] tested only pre-training, after pre-training, we can have a negative landscape 
  - [x] test the pre-training, viasualize the NCBF after pretraining to see if the NCBF is negative after the pre-training phase.  
    - [x] fix the training method: we should randomly use state in the state space, and not states only in the data set
- [x] **Testing and Refinement**: Need to test conservative learning approach
  - [x] **Result**: With only safe data, the conservative loss above doesn't work well. The hinge loss is still necessary for learning the safe boundary.
- [x] **Validation**: Compare conservative learning results with baseline
  - [x] **Result**:  With unsafe data, the conservative loss works. It seems to add some pessimism for NCBF learning. But by using only safe data, both conservative loss and pre-training does not work. **We still need negative data**

**Conservative Learning Theory** (from Tabbara et al. 2025):
- Uses log-sum-exp to approximate maximum NCBF value over proceeding states
- Implements conservative Q-learning principle: assume worst-case (unsafe) for unknown states
- Loss function: $\mathcal{L}_c = \frac{\lambda_c}{| \mathcal{X}_{\mathrm{safe}} |} \sum_{x\in \mathcal{X}_{\mathrm{safe}}} \left[ \tau \ln \left( \sum \exp ( \frac{h(x^{\prime} )}{\tau}) \right) \right]$

### Step 3: Advanced Pipeline with Pseudo-Negative Data
#### Technical Plan

**Core Problem**: Safe-only datasets cannot train effective NCBF models because hinge loss requires both positive (safe) and negative (unsafe) examples to learn safety boundaries.

**Solution Approach**: Generate pseudo-negative data using three distribution-based methods that leverage different aspects of the safe demonstration data.

**Method 1: iDBF (Control Distribution)**
**Core Idea**: Use control distribution information from safe demonstrations. Sample out-of-distribution (OOD) controls using stochastic behavior cloning, then forward propagate safe states with these OOD controls to generate unsafe next states.

**General Steps**:
1. Learn the control distribution p(action|state) from safe (state, action) pairs using stochastic behavior cloning
2. Sample OOD controls from the complement of learned control distribution
3. Forward simulate safe states with OOD controls using system dynamics
4. Label resulting next states as pseudo-negative examples

**Key Insight**: States reachable through OOD controls provide natural unsafe examples that respect system dynamics.

**Method 2: Anomaly Detection (State Distribution)**
**Core Idea**: Use state distribution information from safe demonstrations. Train an in-distribution classifier on safe states, then sample random states and label out-of-distribution states as unsafe.

**General Steps**:
1. Learn the state distribution p(state) from safe demonstration data using anomaly detection
2. Build classifier to distinguish in-distribution (safe-like) vs out-of-distribution states
3. Sample candidate states across the workspace
4. Label OOD states according to classifier as pseudo-negative examples

**Key Insight**: States that are statistically different from safe demonstrations are likely to be unsafe.

**Method 3: Direct Complement (Distance Information)**
**Core Idea**: Use geometric distance information from safe states. Sample random states and label those far from any safe state as unsafe, then apply unsafe set inflation to fill gaps.

**General Steps**:
1. Sample random candidate states across the workspace
2. For each candidate, compute distance to nearest safe state
3. Label states outside R-disk of any safe state as initially unsafe (conservative estimate)
4. Apply unsafe set inflation: for each initially unsafe state, label all states within its R-disk as unsafe

**Key Insight**: Unsafe regions are geometrically separated from safe regions, and unsafe sets can be inflated from conservative initial estimates.

**Unified Enhancement Framework**
All three methods follow the same enhancement pipeline: load safe-only dataset → fit enhancer to safe data → generate pseudo-negative states → create balanced dataset → save enhanced dataset for training.

#### Code structure
1. Base class: the ABC of dataset enhacners, the base class that defines the basic methods to realize
2. Config class: a config class that manages the parameters of the dataset enhancer
3. Sub classes: `ComplementEnhancer`, `iDBFEnhancer` and `ADEnhancer` will inherit the base class to do the enhancing procedure.

#### Implementation details
<!-- this is where you should note the implementation details -->
- [x] implement base class design and utils ( see work/ncbf/enhancement/enhancer_base.py and enhancer_utils.py for implementation)
- [ ] implement AD method
  - [x] implement `ADEnhancer` class (see work/ncbf/enhancement/ad_enhancer.py)
  - [x] do basic test and tuning
  - [x] develop general command line tool for all enhancement methods and test it for AD.
    - [x] test the command line tool for AD and saved the commands in command.md 
  - [ ] improve for better performance and general use.
    - [ ] Down sampling our data via spatial uniformization (filter out the artificial density transition of safe-only dataset).
      - motivation: our data generation has a obstacle concentration strategy: in map_manager.py code we deliberately creates concentrated sampling around obstacles. And the OCSVM may interprets density drops as anomaly boundaries
      - method:   1. Divide workspace into spatial cells (grid or Voronoi) 2. Count safe states per cell 3. Sample equal number from each cell (or cap maximum per cell) 4. Result: Uniform spatial density regardless of original sampling
- [ ] implement complement method
- [ ] implement iDBF method

**Status**: Future work - advanced approaches for generating pseudo-negative data

**Planned Implementation**:
- [ ] **In-Distribution Classifier**: Implement classifier to judge if state is in-distribution
- [ ] **OOD Sampling**: Sample out-of-distribution data to become negative training examples
- [ ] **iDBF Method**: Reference Castaneda et al. approach using safe trajectory data + OOD actions
- [ ] **Pseudo-Labeling**: Generate unsafe data from safe states with adversarial actions

## Current Work Items

### Immediate Tasks (Next Steps)
1. **Test Conservative Learning**: Run conservative NCBF training with safe-only data
2. **Parameter Optimization**: Tune conservative loss parameters for best performance
3. **Two-Phase Training Validation**: Validate pretraining + main training approach
4. **Results Analysis**: Compare conservative learning with baseline methods

### Technical Implementation Details

**Conservative NCBF Trainer Features**:
- Log-sum-exp conservative loss with temperature parameter �
- Random control sampling within unicycle constraints
- Two-phase training: pretraining (negative landscape) + main training
- Integration with existing unicycle model dynamics
- Proceeding state generation using forward simulation

**Key Parameters to Tune**:
- Conservative weight (�_c): Balance between classification and conservative loss
- Temperature (�): Controls sharpness of log-sum-exp approximation
- Number of random controls: Samples for proceeding state generation
- Pretraining epochs: Duration of negative landscape initialization

## Testing and Validation Plan

### Conservative Learning Tests
1. **Basic Functionality**: Test conservative loss computation
2. **Training Convergence**: Validate training stability and convergence
3. **Safety Performance**: Test learned CBF on navigation tasks
4. **Comparison Studies**: Compare with standard NCBF and handwritten CBF

### Evaluation Metrics
- Safety violation rate in simulation
- Task completion success rate
- Conservative loss convergence
- Final h(x) value distributions

## Future Work (Beyond Current Scope)

### Advanced Approaches
1. **Latent Space CBF**: Use encoder for high-dimensional inputs (vision)
2. **Task-Conditioned NCBF**: Context-dependent safety with natural language
3. **Multi-Modal Learning**: Combine multiple data sources and modalities

### Research Directions
1. **Theoretical Analysis**: Convergence guarantees for conservative learning
2. **Scalability**: Extension to higher-dimensional state spaces
3. **Real-World Deployment**: Transfer from simulation to physical robots

## File Structure for Current Work
```
work/ncbf/training/
   conservative_ncbf_trainer.py    # Conservative learning implementation
   train_conservative_ncbf.py      # CLI for conservative training
   ncbf_trainer.py                 # Base trainer (parent class)
   results/                        # Training outputs

work/ncbf/maps/
   map_filter.py                   # Safe data filtering utilities
   map_files/map1/
       training_data_safe_only.h5  # Filtered safe-only dataset
       training_data_large.h5      # Original dataset
```

## Quick Start Commands

### Conservative Training
```bash
cd /home/chengrui/wk/NCBFquickDemo/work/ncbf/training
# Basic conservative training
python train_conservative_ncbf.py --data map1/training_data_safe_only.h5 --config large

# Two-phase training with pretraining
python train_conservative_ncbf.py --data map1/training_data_safe_only.h5 \
                                  --enable-pretraining --pretrain-epochs 50 \
                                  --config large --visualize
```

### Testing
```bash
cd /home/chengrui/wk/NCBFquickDemo/work/test
# Test conservative learning
python test_conservative_ncbf.py
```

## Notes and Observations
- Conservative learning shows promise for safe-only data scenarios
- Two-phase training helps initialize proper negative landscape
- Log-sum-exp provides smooth approximation to max operator
- Integration with existing unicycle dynamics maintains consistency
- Need careful parameter tuning for optimal performance