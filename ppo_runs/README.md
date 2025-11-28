# PPO Training Runs

This directory contains all training outputs, models, and logs.

## Directory Structure

```
ppo_runs/
├── episode_metrics.csv          # Training metrics (episode, stage, success, etc.)
├── tb3_ppo_stage1.zip          # Stage 1 trained model
├── tb3_ppo_stage2.zip          # Stage 2 trained model  
├── tb3_ppo.zip                 # Final trained model
├── PPO_1/                      # TensorBoard logs (run 1)
├── PPO_2/                      # TensorBoard logs (run 2)
└── ...
```

## Usage

### Training
```bash
# From repo root
ros2 run rl_nav train_ppo --timesteps 10000 --logdir ppo_runs
```

### Loading Models
```bash
# Models are saved relative to repo root
ros2 run rl_nav train_ppo --resume ppo_runs/tb3_ppo_stage1.zip
```

## Git Tracking

- `episode_metrics.csv` is tracked in git (useful for analysis)
- Model files (`.zip`) are typically NOT tracked (large files)
- TensorBoard logs are NOT tracked

To track specific models, uncomment them in `.gitignore` or use Git LFS.

