# Training Setup Guide for Simplified Warehouse

## Quick Start

This guide walks you through training the PPO model on the simplified warehouse layout with 2 aisles.

---

## Step 1: Build Updated Packages

First, rebuild the packages to include the simplified warehouse spawner:

```bash
cd ~/MSML_642_FinalProject
colcon build --packages-select warehouse_sim rl_nav --symlink-install
source install/setup.bash
```

---

## Step 2: Verify Workspace Coordinates

The training script uses these workspace limits (in `train_ppo.py`):
- **X_MIN, X_MAX**: -7.0 to 2.0
- **Y_MIN, Y_MAX**: -3.0 to 3.0
- **DOCK_A**: (-6.5, -2.0)
- **DOCK_B**: (-6.5, 0.0)
- **DOCK_C**: (-6.5, 2.0)

The simplified warehouse layout:
- **Aisle 0**: x = -3.0 (with shelves at y = -2.0, 0.5, 3.0)
- **Aisle 1**: x = 1.0 (with shelves at y = -2.0, 0.5, 3.0)
- **Pallets**: (-3.0, -4.0) and (1.0, -4.0)

**Note**: The pallets at y = -4.0 are slightly outside the workspace bounds (Y_MIN = -3.0), but this is fine as they're just obstacles. The robot will train within the workspace limits.

---

## Step 3: Launch the Warehouse World

**Terminal 1**: Launch Gazebo with the simplified warehouse:

```bash
cd ~/MSML_642_FinalProject
source install/setup.bash
./launch_warehouse.sh
```

This will:
1. Start Gazebo server with `warehouse_empty.world`
2. Start Gazebo client (GUI)
3. Spawn the simplified warehouse layout (2 aisles, 6 shelves, 12 boxes, 2 pallets)

**Wait for**: "Warehouse environment ready!" message

---

## Step 4: Train Stage 1 (Local Docking)

**Terminal 2**: Train Stage 1 - Basic docking near DOCK_A:

```bash
cd ~/MSML_642_FinalProject
source install/setup.bash

ros2 run rl_nav train_ppo \
    --curriculum-stage 1 \
    --timesteps 20000 \
    --seed 0
```

**What it does**:
- Robot spawns near DOCK_A (-6.5, -2.0)
- Goal is always DOCK_A
- Learns basic docking movements
- Model saved as: `ppo_runs/ppo_stage1.zip`

**Expected time**: ~10-20 minutes (depending on hardware)

**Success criteria**: Robot should consistently reach DOCK_A from nearby positions

---

## Step 5: Train Stage 2 (Aisle Navigation)

**Terminal 2** (same or new): Train Stage 2 - Longer navigation to DOCK_A:

```bash
cd ~/MSML_642_FinalProject
source install/setup.bash

ros2 run rl_nav train_ppo \
    --curriculum-stage 2 \
    --timesteps 80000 \
    --seed 0
```

**What it does**:
- Robot spawns randomly in workspace (X: -7.0 to 2.0, Y: -3.0 to 3.0)
- Goal is always DOCK_A
- Learns to navigate around shelves and obstacles
- Model saved as: `ppo_runs/ppo_stage2.zip`

**Expected time**: ~30-60 minutes

**Success criteria**: Robot should navigate from any workspace position to DOCK_A

---

## Step 6: Train Stage 3 (Virtual Sorting)

**Terminal 2**: Train Stage 3 - Virtual sorting between docks:

```bash
cd ~/MSML_642_FinalProject
source install/setup.bash

ros2 run rl_nav train_ppo \
    --curriculum-stage 3 \
    --timesteps 100000 \
    --seed 0
```

**What it does**:
- Robot spawns randomly in workspace
- Randomly assigned task class: A, B, or C
- Goal = corresponding dock (A→DOCK_A, B→DOCK_B, C→DOCK_C)
- Observation includes task_class one-hot encoding
- Model saved as: `ppo_runs/ppo_stage3_sorting.zip`

**Expected time**: ~60-90 minutes

**Success criteria**: Robot should navigate to the correct dock based on task class

---

## Step 7: Evaluate Trained Models

**Terminal 2**: Evaluate a trained model:

```bash
cd ~/MSML_642_FinalProject
source install/setup.bash

# Evaluate Stage 2
ros2 run rl_nav eval_policy \
    --model-path ppo_runs/ppo_stage2.zip \
    --curriculum-stage 2 \
    --episodes 10

# Evaluate Stage 3
ros2 run rl_nav eval_policy \
    --model-path ppo_runs/ppo_stage3_sorting.zip \
    --curriculum-stage 3 \
    --episodes 10
```

**Output**: Success rate, collisions, average reward, etc.

---

## Training with Multiple Seeds (Optional)

For better robustness, train multiple seeds in parallel:

**Terminal 2**:
```bash
ros2 run rl_nav train_ppo --curriculum-stage 3 --timesteps 100000 --seed 0
```

**Terminal 3**:
```bash
ros2 run rl_nav train_ppo --curriculum-stage 3 --timesteps 100000 --seed 1
```

**Terminal 4**:
```bash
ros2 run rl_nav train_ppo --curriculum-stage 3 --timesteps 100000 --seed 2
```

Then compare results and use the best model.

---

## Monitoring Training

### TensorBoard

While training, monitor progress with TensorBoard:

```bash
# Terminal 3
cd ~/MSML_642_FinalProject
tensorboard --logdir ppo_runs
```

Open browser to: `http://localhost:6006`

### Episode Metrics

Metrics are logged to: `ppo_runs/episode_metrics.csv`

Columns:
- `episode`: Episode number
- `stage`: Curriculum stage (1, 2, or 3)
- `steps`: Steps in episode
- `final_dist`: Final distance to goal (meters)
- `success`: 1 if successful, 0 otherwise
- `return`: Cumulative episode reward

---

## Troubleshooting

### Issue: Robot spawns outside workspace
**Solution**: Check that workspace limits in `train_ppo.py` match your world layout

### Issue: Training is too slow
**Solution**: 
- Reduce `--timesteps` for testing
- Use `--seed` for reproducibility
- Check Gazebo real-time factor (should be ~1.0)

### Issue: Robot doesn't learn
**Solution**:
- Verify reward function is working (check episode_metrics.csv)
- Ensure workspace coordinates are correct
- Try Stage 1 first to verify basic setup

### Issue: Collisions too frequent
**Solution**: 
- Verify shelf positions don't block navigation paths
- Check workspace limits allow clear paths to docks
- Consider adjusting shelf spacing in `warehouse_spawner.py`

---

## Complete Training Sequence

For a full training run:

```bash
# 1. Build packages
colcon build --packages-select warehouse_sim rl_nav --symlink-install
source install/setup.bash

# 2. Terminal 1: Launch world
./launch_warehouse.sh

# 3. Terminal 2: Train Stage 1
ros2 run rl_nav train_ppo --curriculum-stage 1 --timesteps 20000 --seed 0

# 4. Terminal 2: Train Stage 2
ros2 run rl_nav train_ppo --curriculum-stage 2 --timesteps 80000 --seed 0

# 5. Terminal 2: Train Stage 3
ros2 run rl_nav train_ppo --curriculum-stage 3 --timesteps 100000 --seed 0

# 6. Terminal 2: Evaluate
ros2 run rl_nav eval_policy --model-path ppo_runs/ppo_stage3_sorting.zip --curriculum-stage 3 --episodes 10
```

---

## Expected Results

After training, you should have:
- `ppo_runs/ppo_stage1.zip` - Basic docking policy
- `ppo_runs/ppo_stage2.zip` - Aisle navigation policy
- `ppo_runs/ppo_stage3_sorting.zip` - Virtual sorting policy
- `ppo_runs/episode_metrics.csv` - Training metrics
- `ppo_runs/` - TensorBoard logs

---

## Next Steps

After training:
1. Use `ppo_controller_node.py` to run inference with trained models
2. Use `sorting_node.py` for full sorting task execution
3. Analyze `episode_metrics.csv` for performance insights
4. Visualize training curves in TensorBoard

