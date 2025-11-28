# Stage 2 Performance Collapse - Fix Applied

## Problem Identified

After successful Stage 1 training (90% success rate), the robot advanced to Stage 2 but performance **completely collapsed**:
- Success rate dropped from 90% → 0-5%
- All episodes timing out (300 steps)
- Final distances of 1-5m (goals are 1.5-2.5m away)
- Negative episode rewards

## Root Cause

**The jump from Stage 1 to Stage 2 was too large:**
- Stage 1: 0.4-0.8m goals, mostly forward, small area
- Stage 2: 1.5-2.5m goals, **random direction** (0-360°), larger area

This is a **3x increase in distance** AND a **complete change in task difficulty** (forward-only → any direction).

## Fixes Applied

### 1. Gradual Goal Distance Increase
**Before:**
```python
STAGE2_GOAL_MIN = 1.5  # 3x larger than Stage 1 max
STAGE2_GOAL_MAX = 2.5
```

**After:**
```python
STAGE2_GOAL_MIN = 0.8   # Only 2x Stage 1 max (gradual)
STAGE2_GOAL_MAX = 1.5   # Much closer to Stage 1
```

### 2. Keep Forward-Biased Goals
**Before:**
```python
yaw = random.uniform(0, 2 * math.pi)  # Any direction
goal_angle = random.uniform(0, 2 * math.pi)  # Random
```

**After:**
```python
yaw = random.uniform(-0.3, 0.3)  # Small orientation
goal_angle = random.uniform(-0.4, 0.4)  # Mostly forward
```

### 3. Stricter Curriculum Progression
**Before:**
- Stage 1 threshold: 65% (too lenient)
- Min episodes: 20

**After:**
- Stage 1 threshold: **75%** (ensure mastery before advancing)
- Min episodes: **25** (more practice)
- Stage 2 min episodes: **30** (more time to learn)

## Expected Improvements

1. **Smoother transition**: 0.8-1.5m goals are only 2x Stage 1, not 3x
2. **Forward bias**: Robot can use Stage 1 skills (forward movement)
3. **More practice**: 25+ episodes in Stage 1 ensures solid foundation
4. **Gradual difficulty**: Stage 2 is now "Stage 1 but slightly longer"

## Training Strategy

**For empty warehouse (Stage 1 & 2):**
- Stage 1: 0.4-0.8m goals → should reach 75%+ success
- Stage 2: 0.8-1.5m goals → should maintain 70%+ success
- Stage 3: Only advance when ready (with objects)

**For warehouse with objects:**
- Use `launch_warehouse.sh` for Stage 3 training
- Stage 3 goals: 3.0-5.0m (navigate around obstacles)

## Next Training Run

```bash
# Rebuild
cd ~/MSML_642_FinalProject
rm -rf build install log
colcon build --packages-select rl_nav --symlink-install
source install/setup.bash

# Train - should now maintain success in Stage 2
ros2 run rl_nav train_ppo --timesteps 20000
```

## Monitoring

Watch for:
- **Stage 1**: Should reach 75%+ success before advancing (was 90% before, now threshold is 75%)
- **Stage 2**: Should maintain 60-70% success (not collapse to 0%)
- **Episode length**: Should stay reasonable (50-150 steps, not all 300)
- **Final distances**: Should be < 0.6m for successes, < 1.0m for "close"

## If Stage 2 Still Struggles

If success rate still drops below 50% in Stage 2:
1. **Further reduce Stage 2 goals**: 0.6-1.2m
2. **Add intermediate stage**: Stage 1.5 with 0.6-1.0m goals
3. **Increase Stage 1 threshold**: Require 80%+ before advancing
4. **Keep Stage 2 forward-only**: Don't allow random directions yet

