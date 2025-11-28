# Two-Phase Training Guide: Stage 1 → Stage 2

## Overview

This guide explains how to train Stage 1 in an empty world, save the model, then continue training Stage 2 in the warehouse with objects using the saved checkpoint.

## Phase 1: Stage 1 Training (Empty World)

### Step 1: Launch Empty World
```bash
# Terminal 1
./launch_warehouse_empty.sh
```

### Step 2: Train Stage 1
```bash
# Terminal 2
cd ~/MSML_642_FinalProject
source install/setup.bash

# Train Stage 1 (will save model when done)
ros2 run rl_nav train_ppo \
    --timesteps 10000 \
    --logdir ~/ppo_runs \
    --save-name tb3_ppo_stage1
```

**What happens:**
- Trains in empty world
- Stage 1 goals: 0.4-0.8m
- Saves model to: `~/ppo_runs/tb3_ppo_stage1.zip`
- Should reach 75%+ success rate before advancing

**Monitor progress:**
```bash
# Check success rate
tail -20 ~/ppo_runs/episode_metrics.csv

# Watch TensorBoard
tensorboard --logdir ~/ppo_runs
```

### Step 3: Verify Stage 1 Model
```bash
# Check that model was saved
ls -lh ~/ppo_runs/tb3_ppo_stage1.zip

# Should see file with reasonable size (few MB)
```

## Phase 2: Stage 2 Training (Warehouse with Objects)

### Step 1: Switch to Warehouse with Objects
```bash
# Terminal 1: Stop empty world (Ctrl+C), then:
./launch_warehouse.sh
```

### Step 2: Resume Training from Stage 1 Checkpoint
```bash
# Terminal 2: Continue training
ros2 run rl_nav train_ppo \
    --timesteps 20000 \
    --logdir ~/ppo_runs \
    --resume ~/ppo_runs/tb3_ppo_stage1.zip \
    --save-name tb3_ppo_stage2
```

**What happens:**
- Loads `tb3_ppo_stage1.zip` checkpoint
- Continues training in warehouse with objects
- Stage 2 goals: 0.8-1.5m in clear aisles
- Saves final model to: `~/ppo_runs/tb3_ppo_stage2.zip`
- Should maintain 60-70% success rate

**The model will:**
- Keep all learned Stage 1 behaviors
- Adapt to obstacle avoidance in Stage 2
- Continue curriculum progression to Stage 3

## Complete Training Script

Here's a complete script for two-phase training:

```bash
#!/bin/bash
# two_phase_train.sh

set -e

cd ~/MSML_642_FinalProject
source install/setup.bash

LOGDIR=~/ppo_runs
STAGE1_MODEL="${LOGDIR}/tb3_ppo_stage1.zip"
STAGE2_MODEL="${LOGDIR}/tb3_ppo_stage2.zip"

echo "=========================================="
echo "Phase 1: Stage 1 Training (Empty World)"
echo "=========================================="
echo "1. Launch empty world in another terminal:"
echo "   ./launch_warehouse_empty.sh"
echo ""
echo "2. Press Enter when empty world is running..."
read

echo "Training Stage 1..."
ros2 run rl_nav train_ppo \
    --timesteps 10000 \
    --logdir "$LOGDIR" \
    --save-name tb3_ppo_stage1

if [ ! -f "$STAGE1_MODEL" ]; then
    echo "ERROR: Stage 1 model not saved!"
    exit 1
fi

echo ""
echo "Stage 1 training complete!"
echo "Model saved to: $STAGE1_MODEL"
echo ""
echo "=========================================="
echo "Phase 2: Stage 2 Training (With Objects)"
echo "=========================================="
echo "1. Stop empty world (Ctrl+C)"
echo "2. Launch warehouse with objects:"
echo "   ./launch_warehouse.sh"
echo ""
echo "3. Press Enter when warehouse is running..."
read

echo "Resuming training from Stage 1 checkpoint..."
ros2 run rl_nav train_ppo \
    --timesteps 20000 \
    --logdir "$LOGDIR" \
    --resume "$STAGE1_MODEL" \
    --save-name tb3_ppo_stage2

if [ ! -f "$STAGE2_MODEL" ]; then
    echo "ERROR: Stage 2 model not saved!"
    exit 1
fi

echo ""
echo "Training complete!"
echo "Final model: $STAGE2_MODEL"
echo "Stage 1 checkpoint: $STAGE1_MODEL"
```

## Alternative: Single Continuous Training

If you prefer to train all stages in one run:

```bash
# Start with warehouse WITH objects
./launch_warehouse.sh

# Train all stages (Stage 1 will train with objects present)
ros2 run rl_nav train_ppo --timesteps 30000
```

**Pros:**
- Simpler (one command)
- No checkpoint management
- Stage 1 still works (goals in clear areas)

**Cons:**
- Stage 1 trains with objects present (slightly harder)
- Can't easily separate Stage 1 vs Stage 2 performance

## Using Trained Models

### Evaluate Stage 1 Model
```bash
# Load and test Stage 1 model
ros2 run rl_nav evaluate_ppo --model ~/ppo_runs/tb3_ppo_stage1.zip
```

### Use Stage 2 Model for Inference
```bash
# Launch warehouse
./launch_warehouse.sh

# Run PPO controller with Stage 2 model
ros2 run rl_nav ppo_controller_node
# (Update model path in ppo_controller_node.py if needed)
```

## Troubleshooting

### Checkpoint Not Found
```bash
# Check if model exists
ls -lh ~/ppo_runs/tb3_ppo_stage1.zip

# Use absolute path if needed
ros2 run rl_nav train_ppo \
    --resume /home/nsumesh/ppo_runs/tb3_ppo_stage1.zip
```

### Model Size Issues
- Stage 1 model should be ~2-5 MB
- If too small (< 1 MB), training may have failed
- Check TensorBoard logs for training progress

### Stage 2 Performance Drops
- This is normal initially (new environment)
- Should recover within 10-20 episodes
- If success rate stays < 50%, Stage 1 may need more training

## Best Practices

1. **Save checkpoints frequently**: Consider saving after each stage
2. **Monitor TensorBoard**: Watch for learning curves
3. **Check episode metrics**: Verify success rates before advancing
4. **Keep Stage 1 model**: Useful for comparison/ablation studies
5. **Document training**: Note which model was trained with which setup

