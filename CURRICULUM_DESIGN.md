# Curriculum Design: Empty → Cluttered → Full Warehouse

## Recommended Training Progression

### Stage 1: Empty World (Basic Movement)
**World**: `launch_warehouse_empty.sh` (no objects)
**Goals**: 0.4-0.8m, forward-biased
**Purpose**: Learn basic forward movement, turning, goal-reaching
**Success Threshold**: 75%+ before advancing

**Why empty?**
- Robot needs to learn basic motor skills first
- No distractions from obstacles
- Fast learning of fundamental behaviors

### Stage 2: Warehouse with Objects (Obstacle Avoidance)
**World**: `launch_warehouse.sh` (with shelves/boxes)
**Goals**: 0.8-1.5m, forward-biased, in clear aisles
**Purpose**: Learn to navigate around obstacles while maintaining goal-seeking
**Success Threshold**: 70%+ before advancing

**Why with objects?**
- Introduces obstacle avoidance gradually
- Short goals (0.8-1.5m) prevent overwhelming
- Clear aisles provide safe navigation paths
- Prepares for Stage 3 complexity

**Key Design:**
- Goals stay in clear aisles (x=-2.5 to 2.5, avoids shelf rows)
- Short distances (only 2x Stage 1, not 3x)
- Forward-biased (can use Stage 1 skills)
- Objects present but goals avoid them

### Stage 3: Full Warehouse (Complex Navigation)
**World**: `launch_warehouse.sh` (with all objects)
**Goals**: 3.0-5.0m, any direction, full area
**Purpose**: Full warehouse navigation with obstacles
**Success Threshold**: 70%+ (final stage)

**Why full complexity?**
- Long-distance navigation
- Complex obstacle avoidance
- Realistic warehouse scenarios

## Training Workflow

### Phase 1: Stage 1 Training (Empty World)
```bash
# Terminal 1: Launch empty world
./launch_warehouse_empty.sh

# Terminal 2: Train Stage 1
ros2 run rl_nav train_ppo --timesteps 10000
# Monitor: Should reach 75%+ success, then advance to Stage 2
```

### Phase 2: Stage 2 Training (With Objects)
```bash
# IMPORTANT: Switch to warehouse with objects!
# Terminal 1: Stop empty world, launch with objects
./launch_warehouse.sh

# Terminal 2: Continue training (will use Stage 2)
ros2 run rl_nav train_ppo --timesteps 15000
# Monitor: Should maintain 60-70% success in Stage 2
```

### Phase 3: Stage 3 Training (Full Warehouse)
```bash
# Continue with warehouse.sh (already running)
# Terminal 2: Continue training
ros2 run rl_nav train_ppo --timesteps 20000
# Monitor: Should learn full warehouse navigation
```

## Alternative: Single Training Run

If you want to train all stages in one run:

```bash
# Start with warehouse WITH objects (for Stage 2+)
./launch_warehouse.sh

# Train all stages
ros2 run rl_nav train_ppo --timesteps 30000
```

**Note**: Stage 1 will train in empty world if you use `launch_warehouse_empty.sh`, but Stage 2+ need objects. For single run, start with `launch_warehouse.sh` - Stage 1 will still work (just with objects present, but goals are in clear areas).

## Why This Design?

1. **Gradual Complexity**: Empty → Objects → Full navigation
2. **Skill Building**: Each stage builds on previous skills
3. **Obstacle Introduction**: Stage 2 introduces obstacles while keeping goals manageable
4. **Realistic Training**: Stage 2+ use actual warehouse environment

## Current Implementation

The code currently supports this, but you need to:
- Use `launch_warehouse_empty.sh` for Stage 1 (optional, but recommended)
- Use `launch_warehouse.sh` for Stage 2+ (required for obstacle learning)

The coordinates are already set up to use clear aisles in Stage 2, so even with objects present, the robot will navigate in safe areas.

