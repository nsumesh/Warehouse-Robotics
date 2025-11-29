# Sorting Node with PPO Navigation

## Overview

The `sorting_node` is an **integrated node** that combines:
1. **Task Management**: Manages sorting tasks with 3 bins (light, heavy, fragile)
2. **PPO Navigation**: Uses your trained PPO policy to navigate to pickup locations and drop-off bins
3. **Goal Detection**: Detects when goals are reached (distance-based, not timeouts)

## How It Works

### 3 Bins for Sorting

1. **Light Items Bin**: `(-4.0, -4.0)`
2. **Heavy Items Bin**: `(0.0, -4.0)`
3. **Fragile Items Bin**: `(4.0, -4.0)`

### Pickup Locations

- **Light items**: `(-2.0, 0.0)`
- **Heavy items**: `(0.0, 0.0)`
- **Fragile items**: `(2.0, 0.0)`

### Workflow (FSM)

```
IDLE → GO_PICKUP → GO_DROPOFF → IDLE
  ↑                                ↓
  └────────────────────────────────┘
```

1. **IDLE**: Select next task from queue
2. **GO_PICKUP**: Navigate to pickup location using PPO
3. **GO_DROPOFF**: Navigate to corresponding bin using PPO
4. **IDLE**: Task complete, start next task

## Usage

### Option 1: Launch File (Recommended)
```bash
# Terminal 1: Launch warehouse
./launch_warehouse.sh

# Terminal 2: Launch sorting node
ros2 launch rl_nav sorting_demo.launch.py
```

### Option 2: Direct Node
```bash
# Terminal 1: Launch warehouse
./launch_warehouse.sh

# Terminal 2: Run sorting node
ros2 run rl_nav sorting_node
```

## What Happens

1. **Node loads trained PPO model** from `ppo_runs/tb3_ppo.zip`
2. **Creates task queue** with 5 random tasks (light/heavy/fragile)
3. **For each task**:
   - Navigates to pickup location using PPO
   - Detects arrival (within 0.4m)
   - Navigates to corresponding drop-off bin using PPO
   - Detects arrival and completes task
4. **Logs progress** for each phase

## Configuration

### Adjust Bin Locations

Edit `sorting_node.py`:
```python
self.drop_bins = {
    "light": (-4.0, -4.0),    # Change these coordinates
    "heavy": (0.0, -4.0),
    "fragile": (4.0, -4.0),
}
```

### Adjust Pickup Locations

```python
self.pickup_locations = {
    "light": (-2.0, 0.0),     # Change these coordinates
    "heavy": (0.0, 0.0),
    "fragile": (2.0, 0.0),
}
```

### Adjust Goal Detection Threshold

```python
self.goal_reached_threshold = 0.4  # meters (same as training success threshold)
```

### Change Number of Tasks

```python
self._build_initial_tasks(num_tasks=10)  # More tasks
```

## Example Output

```
[INFO] SortingNode initialized with PPO navigation
[INFO] Task queue: ['heavy', 'light', 'fragile', 'heavy', 'light']
[INFO] [Task] HEAVY: Navigating to pickup @ (0.0, 0.0)
[INFO] [Task] HEAVY: Reached pickup. Navigating to heavy bin @ (0.0, -4.0)
[INFO] [Task] ✓ Completed sorting HEAVY item to heavy bin
[INFO] [Task] LIGHT: Navigating to pickup @ (-2.0, 0.0)
...
[INFO] All sorting tasks completed!
```

## Features

✅ **Uses trained PPO policy** - Same model from curriculum training  
✅ **Real goal detection** - Distance-based (0.4m threshold), not timeouts  
✅ **3-bin sorting** - Light, heavy, fragile items  
✅ **FSM workflow** - Proper state machine for task management  
✅ **Timeout protection** - 60s max per phase (safety)  
✅ **Integrated navigation** - No separate PPO controller needed  

## Troubleshooting

### Model Not Found
```
ERROR: PPO model not found at: ppo_runs/tb3_ppo.zip
```
**Solution**: Train a model first or specify correct path in code

### Robot Not Moving
- Check if warehouse is running
- Check if robot is spawned
- Check `/odom` and `/scan` topics are publishing

### Goals Not Reached
- Increase `goal_reached_threshold` (e.g., 0.5m)
- Check if pickup/bin locations are correct for your world
- Verify PPO model was trained successfully

### Tasks Not Completing
- Check logs for timeout messages
- Increase `max_task_time` if needed
- Verify goal locations are reachable

## Next Steps

1. **Customize bin locations** to match your warehouse layout
2. **Add item detection** (simulate pickup/dropoff)
3. **Add metrics tracking** (time per task, success rate)
4. **Integrate with camera** for actual item detection

