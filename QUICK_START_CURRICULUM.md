# Quick Start: 3-Stage Curriculum Training

This guide shows you how to use the new 3-stage curriculum system with virtual pickup and sorting.

## Overview

**Stage 1**: Empty aisle → Basic movement training  
**Stage 2**: Moderate clutter → Obstacle avoidance training  
**Stage 3**: Full warehouse → Complex navigation training

## Step 1: Build the Package

```bash
cd ~/MSML_642_FinalProject
colcon build --packages-select rl_nav warehouse_sim
source install/setup.bash
```

## Step 2: Train Stage 1 (Empty Aisle)

**Terminal 1**: Launch Level 1 world
```bash
./launch_level1.sh
```

**Terminal 2**: Train PPO
```bash
ros2 run rl_nav train_ppo --level 1 --timesteps 10000
```

This trains the robot in an empty aisle with simple forward goals.

## Step 3: Train Stage 2 (Moderate Clutter)

**Terminal 1**: Launch Level 2 world
```bash
./launch_level2.sh
```

**Terminal 2**: Resume training for Stage 2
```bash
ros2 run rl_nav train_ppo \
    --level 2 \
    --resume ppo_runs/tb3_ppo.zip \
    --start-stage 2 \
    --timesteps 15000
```

This introduces 1-2 obstacles for obstacle avoidance training.

## Step 4: Train Stage 3 (Full Warehouse)

**Terminal 1**: Launch Level 3 world
```bash
./launch_level3.sh
```

**Terminal 2**: Resume training for Stage 3
```bash
ros2 run rl_nav train_ppo \
    --level 3 \
    --resume ppo_runs/tb3_ppo.zip \
    --start-stage 3 \
    --timesteps 20000
```

This uses the full warehouse with shelves, pallets, and clutter.

## Virtual Pickup and Sorting

The virtual pickup system allows the robot to:
- Detect items using camera/OpenCV (ArUco markers, color blobs)
- Pick up items when within 0.5m (virtually removes from world)
- Drop items at sorting bins when reaching goal

### Using Virtual Pickup Node

**Terminal 1**: Launch warehouse world
```bash
./launch_level3.sh
```

**Terminal 2**: Run virtual pickup node
```bash
ros2 run rl_nav virtual_pickup_node
```

**Terminal 3**: Run sorting node (uses trained PPO)
```bash
ros2 run rl_nav sorting_node
```

## Automatic Curriculum Progression

If you want automatic progression (no manual stage switching):

```bash
# Start with Level 3 world (has all objects)
./launch_level3.sh

# Train with automatic curriculum
ros2 run rl_nav train_ppo --timesteps 30000
```

The training will automatically progress:
- Stage 1 → Stage 2 when: 25+ episodes, 75%+ success rate, 3+ streak
- Stage 2 → Stage 3 when: 30+ episodes, 70%+ success rate, 3+ streak

## Arguments Reference

- `--level 1/2/3`: Select curriculum level/world (1=empty, 2=moderate, 3=full)
- `--start-stage 1/2/3`: Set starting curriculum stage
- `--resume PATH`: Resume from checkpoint
- `--timesteps N`: Total training timesteps
- `--logdir DIR`: Directory for logs/models (default: ppo_runs)

## World Files

- `warehouse_stage1.world`: Empty aisle (no obstacles)
- `warehouse_stage2.world`: Moderate clutter (2 obstacles)
- `warehouse_stage3.world`: Full warehouse (shelves + objects)

## Notes

- **Level vs Stage**: `--level` selects the world file, `--start-stage` sets the curriculum stage
- **Virtual Pickup**: Works without physical manipulation - academically valid approach
- **Camera Detection**: Uses ArUco markers or color blob detection (falls back to distance-based if camera unavailable)

