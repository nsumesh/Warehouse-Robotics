# Curriculum Stage Fix - Stage 2 & 3 Behavior

## Problem Identified

Looking at your training metrics:
- **Stage 1**: Many successes (final_dist ~0.38-0.40m), but also timeouts
- **Stage 2**: Many failures (final_dist > 0.4m), lots of timeouts, low success rate

The issue was that **Stage 2 was still too similar to Stage 1**:
- Goals were still forward-biased (like Stage 1)
- Area overlapped with Stage 1
- Not enough variation to prepare for Stage 3

## Fixes Applied

### 1. Stage 2: More Variation in Goal Direction

**Before:**
```python
yaw = random.uniform(-0.3, 0.3)  # Small orientation
goal_angle = random.uniform(-0.4, 0.4)  # Mostly forward
```

**After:**
```python
yaw = random.uniform(-math.pi, math.pi)  # Any orientation (360°)
goal_angle = random.uniform(0, 2 * math.pi)  # ANY direction (360°)
```

**Key Change**: Goals can now be in **any direction** relative to start, not just forward. This is a major difference from Stage 1.

### 2. Stage 2: Goal Relative to Position (Not Yaw)

**Before:**
```python
gx = sx + goal_dist * math.cos(yaw + goal_angle)  # Relative to robot heading
```

**After:**
```python
gx = sx + goal_dist * math.cos(goal_angle)  # Absolute direction
```

**Key Change**: Goals are now in **absolute world coordinates**, not relative to robot heading. This forces the robot to learn to turn and navigate in any direction.

### 3. Stage 3: Already Correct

Stage 3 was already using full 360° directions, so it's fine. Just verified it's working correctly.

### 4. Better Logging

Added logging to show:
- Stage transitions with area and goal distance info
- Current stage in episode logs
- Debug logs for start/goal positions

## Expected Behavior Now

### Stage 1 (Episodes 1-25+)
- **Area**: Small (1.5m x 1.5m around x=-2.0)
- **Goals**: 0.4-0.8m, **always forward** (relative to robot heading)
- **Orientation**: Small variation (±0.15 rad)
- **Behavior**: Simple forward movement

### Stage 2 (After Stage 1 succeeds)
- **Area**: Larger (5m x 6m, x=-2.5 to 2.5, y=-3.0 to 3.0)
- **Goals**: 0.8-1.5m, **ANY direction** (360°)
- **Orientation**: Any orientation (360°)
- **Behavior**: Must learn to turn and navigate in all directions

### Stage 3 (After Stage 2 succeeds)
- **Area**: Full warehouse (10m x 15m)
- **Goals**: 3.0-5.0m, **ANY direction** (360°)
- **Behavior**: Long-distance navigation with obstacles

## Key Differences Between Stages

| Feature | Stage 1 | Stage 2 | Stage 3 |
|---------|---------|---------|---------|
| Goal Distance | 0.4-0.8m | 0.8-1.5m | 3.0-5.0m |
| Goal Direction | Forward only | **Any direction** | Any direction |
| Area Size | 1.5m x 1.5m | 5m x 6m | 10m x 15m |
| Orientation | ±0.15 rad | **360°** | 360° |
| Complexity | Very easy | Medium | Hard |

## Testing

After rebuilding, you should see:

1. **Stage 1**: Quick successes (5-30 steps), final_dist ~0.38-0.40m
2. **Stage 2 transition**: Log message showing area and goal ranges
3. **Stage 2**: More varied start/goal positions, goals in different directions
4. **Stage 2 metrics**: May initially drop (new behavior), then recover
5. **Stage 3**: Long-distance goals, full warehouse navigation

## Rebuild and Retrain

```bash
cd ~/MSML_642_FinalProject
rm -rf build install log
colcon build --packages-select rl_nav --symlink-install
source install/setup.bash

# Retrain (or continue from checkpoint)
ros2 run rl_nav train_ppo --timesteps 20000
```

## Monitoring Stage Transitions

Watch for log messages like:
```
[Curriculum] Advanced to STAGE 2 (success_rate=0.75, streak=3, episodes=25)
[Curriculum] Stage 2: Goals 0.8-1.5m, Area (-2.5 to 2.5, -3.0 to 3.0)
```

This confirms the stage transition and shows the new parameters.

