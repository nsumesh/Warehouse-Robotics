# Stage 3 Two-Phase Training Guide

## Overview

Stage 3 training now implements **full two-phase virtual sorting**:
- **Phase 1 (GO_PICKUP)**: Robot navigates to PICKUP location `(-4.0, 0.0)` where items are spawned
- **Phase 2 (GO_DROPOFF)**: After reaching pickup, robot navigates to the correct dock (A, B, or C) based on task class

## What Changed

1. **Two-Phase Training**: Each episode has two phases (pickup → dropoff)
2. **Item Spawning**: Items spawn at pickup location at episode start
3. **Phase Transitions**: Automatic transition from pickup to dropoff when pickup is reached
4. **Phase-Aware Observations**: Task class encoding changes based on phase
5. **Reward Structure**: Separate rewards for reaching pickup (5.0) and correct dock (10.0)

## Training Command

### Basic Training (Single Run)

```bash
# Make sure Gazebo is running first
./launch_warehouse.sh

# Train Stage 3 with two-phase sorting
ros2 run rl_nav train_ppo --timesteps 300000 --curriculum-stage 3 --seed 0
```

### Recommended Training Parameters

**For 300,000 timesteps (as you mentioned):**
```bash
ros2 run rl_nav train_ppo --timesteps 300000 --curriculum-stage 3 --seed 0
```

**For better robustness (multiple seeds):**
```bash
# Terminal 1
ros2 run rl_nav train_ppo --timesteps 300000 --curriculum-stage 3 --seed 0

# Terminal 2
ros2 run rl_nav train_ppo --timesteps 300000 --curriculum-stage 3 --seed 1

# Terminal 3
ros2 run rl_nav train_ppo --timesteps 300000 --curriculum-stage 3 --seed 2
```

## Training Expectations

### Episode Structure
- Each episode: Start → PICKUP (with items) → Dock (A/B/C)
- Max steps per episode: **600 steps** (increased from 500 for two phases)
- Success requires: Reaching pickup AND reaching correct dock

### What to Monitor

1. **Episode Logs**: Look for phase transitions
   ```
   [EP X] stage=3 task=A steps=Y dist=Z success=1 return=R
   ```

2. **Success Rate**: Should improve over time
   - Early training: 5-15% success (learning both phases)
   - Mid training: 20-40% success
   - Late training: 50-70%+ success

3. **Average Reward**: Should increase over time
   - Early: Negative (timeouts, wrong docks)
   - Mid: Slightly positive (reaching pickup)
   - Late: Positive (completing full task)

4. **Episode Length**: Should decrease as policy improves
   - Early: Often hitting 600 step timeout
   - Late: Completing in 200-400 steps

## Training Time Estimates

**For 300,000 timesteps:**
- **Approximate episodes**: ~500-1000 episodes (depending on episode length)
- **Training time**: ~8-15 hours (depends on hardware and Gazebo performance)
- **FPS**: Expect 2-6 FPS (frames per second)

## Key Differences from Previous Training

| Aspect | Old Stage 3 | New Stage 3 |
|--------|-------------|-------------|
| Goal | Direct to dock | PICKUP → Dock (two phases) |
| Items | None | Spawned at pickup |
| Max Steps | 500 | 600 |
| Observation | Always task class | Phase-aware (dummy during pickup) |
| Success | Reach dock | Reach pickup + correct dock |

## Troubleshooting

### If training is very slow:
- Check Gazebo performance (reduce physics update rate if needed)
- Ensure no other heavy processes running
- Consider reducing `--timesteps` for initial testing

### If success rate stays very low (<10%):
- May need more timesteps (try 500,000)
- Check that items are spawning correctly
- Verify robot can reach pickup location

### If robot keeps going in circles:
- This should be fixed with proper two-phase training
- If still happening, check that phase transitions are working
- Verify observation encoding is correct

## After Training

Once training completes, the model will be saved as:
```
ppo_runs/ppo_stage3_sorting.zip
```

This model can be used directly with `sorting_node.py` - no modifications needed!

## Next Steps

1. **Start Training**: Run the command above
2. **Monitor Progress**: Watch episode logs and success rates
3. **Evaluate**: After training, test with `sorting_node.py`
4. **Iterate**: If needed, adjust timesteps or retrain with different seeds

