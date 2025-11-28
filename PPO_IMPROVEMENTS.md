# PPO Training Improvements for Phase 4

## Problem Identified
PPO rewards were not improving due to:
1. **Too large warehouse** - Goals were too far (up to 7m in Stage 3)
2. **Sparse rewards** - Only +20/-20 for success/collision
3. **Poor reward shaping** - Progress reward too small relative to penalties
4. **No oscillation detection** - Robot could spin in place
5. **No near-miss penalty** - As required by proposal
6. **Suboptimal PPO hyperparameters** - Too large network, too long rollouts

## Key Assumptions Made

### 1. Simplified Stage 1 (Clear Aisle)
- **Assumption**: Stage 1 uses a small 1.5m x 1.5m area in a clear aisle
- **Location**: x=-2.0 (clear aisle between shelves at x=-4 and x=0)
- **Goals**: Very short (0.5-1.0m) to ensure easy learning
- **Options**: Use `launch_warehouse_empty.sh` for truly empty world, or use clear aisle with `launch_warehouse.sh`
- **Rationale**: Robot must learn basic forward movement before handling obstacles

### 2. Reduced Warehouse Size & Clear Area Selection
- **Stage 1**: 1.5m x 1.5m area at x=-2.0 (clear aisle, avoids shelves)
- **Stage 2**: 5m x 6m area using clear aisles (x=-2.5 to 2.5, avoids shelf rows)
- **Stage 3**: 10m x 15m area including all clear spaces and navigating around shelves
- **Rationale**: Uses clear aisles between shelves (x=-4, 0, 4) to avoid obstacles in early stages

### 3. Progressive Goal Distances
- **Stage 1**: 0.5-1.0m (very easy)
- **Stage 2**: 1.5-2.5m (medium)
- **Stage 3**: 3.0-5.0m (hard, but not too hard)
- **Rationale**: Gradual increase prevents overwhelming the agent

## Improvements Implemented

### 1. Redesigned Reward Function

**Before:**
```python
r = 2.0 * (prev_dist - dist)  # progress
    - 0.01                     # time penalty
    - 0.3 * heading_error      # heading
    - 20.0 if collision
    + 20.0 if success
```

**After:**
```python
r = 5.0 * (prev_dist - dist)      # 2.5x higher progress reward
    - 0.005                       # smaller time penalty
    - 0.2 * heading_error         # reduced heading penalty
    - 0.5 * oscillation_penalty  # NEW: penalize back-and-forth
    - 1.0 * near_miss_penalty     # NEW: penalize getting too close
    - 30.0 if collision           # larger collision penalty
    + 50.0-70.0 if success       # larger, distance-scaled success bonus
```

**Key Changes:**
- **Higher progress reward** (2.0 → 5.0): Makes progress more valuable
- **Oscillation detection**: Tracks last 5 positions, penalizes movement without progress
- **Near-miss penalty**: Penalizes getting within 0.5m of obstacles (as per proposal)
- **Distance-scaled success**: Closer = bigger bonus (up to 70.0)
- **Reward clipping**: Clips to [-50, 100] to prevent extreme values

### 2. Improved PPO Hyperparameters

**Before:**
```python
n_steps=512
batch_size=128
net_arch=[128, 128]
gamma=0.995
gae_lambda=0.98
```

**After:**
```python
n_steps=256              # shorter rollouts = faster updates
batch_size=64            # smaller batches = more frequent updates
n_epochs=10              # more epochs per update
net_arch=[64, 64]        # smaller network = faster training
gamma=0.99               # standard discount
gae_lambda=0.95          # standard GAE
ent_coef=0.01            # entropy for exploration
max_grad_norm=0.5        # gradient clipping
activation_fn=Tanh       # bounded outputs
```

**Rationale:**
- Smaller network trains faster and generalizes better initially
- Shorter rollouts allow more frequent policy updates
- Standard hyperparameters are more stable

### 3. Slower, More Controlled Actions

**Before:**
```python
(0.15, 0.8)   # forward + left (fast turn)
(0.00, 0.8)   # rotate left (fast)
```

**After:**
```python
(0.12, 0.6)   # forward + left (slower turn)
(0.00, 0.6)   # rotate left (slower)
```

**Rationale:** Slower actions = more control = less oscillation

### 4. Better Curriculum Progression

**Before:**
- Stage 1: 10 episodes, 75% success rate
- Stage 2: 12 episodes, 80% success rate

**After:**
- Stage 1: 15 episodes, 70% success rate (more lenient)
- Stage 2: 20 episodes, 75% success rate
- Increased max_steps: 200 → 300 (for longer goals)

**Rationale:** More episodes in Stage 1 ensures solid foundation before advancing

### 5. Improved Success Detection

**Before:**
- Success threshold: 0.4m
- Fixed success bonus: +20.0

**After:**
- Success threshold: 0.3m (tighter)
- Distance-scaled bonus: 50.0 + 20.0 * (closeness)
- Timeout penalty: -5.0 (not as harsh as collision)

## Expected Improvements

1. **Faster Learning**: Smaller areas + shorter goals = quicker success
2. **More Stable Rewards**: Better shaping + clipping = smoother learning curve
3. **Less Oscillation**: Oscillation penalty prevents spinning in place
4. **Better Exploration**: Entropy coefficient encourages exploration
5. **Progressive Difficulty**: Curriculum ensures agent masters basics first

## Training Recommendations

1. **Start with Stage 1 only**: Train for 5000-10000 timesteps in Stage 1
2. **Monitor TensorBoard**: Watch for:
   - Increasing episode reward
   - Decreasing collision rate
   - Increasing success rate
3. **Check episode_metrics.csv**: Verify success rate > 70% before Stage 2
4. **If stuck**: Reduce Stage 1 area further or increase goal distance range

## Testing the Changes

```bash
# OPTION 1: Empty world for Stage 1 (recommended for initial training)
./launch_warehouse_empty.sh  # No objects spawned
ros2 run rl_nav train_ppo --timesteps 10000 --logdir ~/ppo_runs

# OPTION 2: Full warehouse with objects (for Stage 2+)
./launch_warehouse.sh  # Spawns shelves and boxes
ros2 run rl_nav train_ppo --timesteps 20000 --logdir ~/ppo_runs

# 3. Monitor TensorBoard
tensorboard --logdir ~/ppo_runs

# 4. Check metrics
cat ~/ppo_runs/episode_metrics.csv | tail -20
```

## Next Steps

If rewards still don't improve:
1. **Further reduce Stage 1**: Use 1m x 1m area, 0.3-0.7m goals
2. **Increase progress reward**: Try 8.0 or 10.0 instead of 5.0
3. **Reduce action space**: Remove rotation actions, only forward+turn
4. **Add reward normalization**: Normalize rewards by episode length

