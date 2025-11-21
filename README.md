# PPO Navigation (TurtleBot3 + Gazebo + ROS 2 Humble)

This repo trains a simple **Proximal Policy Optimization (PPO)** policy to drive a **TurtleBot3 Waffle Pi** toward a goal in a Gazebo warehouse world. The policy learns from a compact observation (down‑sampled LiDAR + goal vector) and a small discrete action set (forward / turn). The package name is `rl_nav` and the main entrypoint is `train_ppo`.

---

## 1) Prerequisites

* **ROS 2 Humble** + **Gazebo Classic**
* TurtleBot3 (Humble) stacks installed/built:

  * `turtlebot3_description`, `turtlebot3_gazebo`
  * `gazebo_ros` (factory plugin) / `gazebo_ros_pkgs`
* This project checked out at: `~/MSML_642_FinalProject`
* Python deps (user‑level ok):

```bash
python3 -m pip install --user "numpy<2" gymnasium==0.29.1 stable-baselines3==2.3.0 torch opencv-python matplotlib
```

> Why these pins? They avoid known Gym/NumPy incompatibilities and match Stable‑Baselines3 expectations.

<<<<<<< Updated upstream
---

## 2) Build the package

```bash
cd ~/MSML_642_FinalProject
colcon build --packages-select rl_nav --symlink-install

# Source overlays (order matters)
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
source ~/gazebo_ros_pkgs_ws/install/setup.bash
source ~/MSML_642_FinalProject/install/setup.bash

export TURTLEBOT3_MODEL=waffle_pi
```

---

## 3) Launch simulation

Launch the warehouse world (adjust the absolute path if needed):

```bash
ros2 launch gazebo_ros gazebo.launch.py \
  world:=/home/nsumesh/MSML_642_FinalProject/gazebo_worlds/small_warehouse.world
```

Spawn the TB3 once per Gazebo session:

```bash
ros2 run gazebo_ros spawn_entity.py \
  -file $(ros2 pkg prefix turtlebot3_gazebo)/share/turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf \
  -entity tb3 -x 0 -y 0 -z 0.1
```

Sanity‑check topics:

```bash
ros2 topic list | grep -E "/scan|/odom|/cmd_vel"
```

---

## 4) Train the PPO policy

Start training (choose timesteps and optional log dir):

```bash
ros2 run rl_nav train_ppo --timesteps 30000 --logdir ~/ppo_runs
```

Artifacts:

* **Model**: `~/ppo_runs/tb3_ppo.zip`
* **TensorBoard logs**: under `~/ppo_runs/` (per run/time)

Visualize training:

```bash
tensorboard --logdir ~/ppo_runs
```

> **Note on stopping**: PPO may run a few extra seconds after the exact `--timesteps` to finish the current rollout chunk (default `n_steps=512`). This is expected. Press `Ctrl+C` to interrupt earlier if needed.

---

## 5) What the agent sees & does (quick intuition)

* **Observation (27 floats):** 24 downsampled laser ranges (clipped & normalized) + goal vector `(dx, dy)` in the robot frame + current yaw.
* **Actions (5 discrete):**

  1. forward + turn left `(v=0.12, w=+0.8)`
  2. straight forward `(v=0.12, w=0.0)`
  3. forward + turn right `(v=0.12, w=-0.8)`
  4. in‑place left `(v=0.0, w=+0.8)`
  5. in‑place right `(v=0.0, w=-0.8)`
* **Rewards:** `+10` on goal, `-5` on collision, `-0.01` per step, small shaping bonus for getting closer to the goal.

---

## 6) (Optional) Quick policy evaluation

If Gazebo is running and `tb3` is spawned, you can run a quick ad‑hoc evaluation without an extra node:

```bash
python3 - <<'PY'
import time, numpy as np
import rclpy
from rl_nav.train_ppo import Tb3Env, GymTb3
from stable_baselines3 import PPO

MODEL = "~/ppo_runs/tb3_ppo.zip"  # change if needed

rclpy.init()
node = Tb3Env()
env = GymTb3(node)
model = PPO.load(MODEL.replace('~','/home/nsumesh'), env=env)

obs = env.reset()
for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    if done:
        obs = env.reset()

rclpy.shutdown()
PY
```

---

## 7) Common issues & fixes

* **`/spawn_entity` unavailable**: Start Gazebo first (`gazebo.launch.py`) and then run the spawn command; ensure `gazebo_ros` factory plugin is present (it is in the default launch).
* **“Entity already exists”**: Use a new `-entity` name (e.g., `tb3_2`) or remove the old one from Gazebo GUI.
* **Gym/NumPy errors**: Pin to `numpy<2` and use `gymnasium` (as in the pip line above). Reinstall `opencv-python` if `_ARRAY_API` errors appear.
* **Stops after timesteps but lingers**: That’s PPO completing a rollout; reduce batch sizes in code (`n_steps`) if you need tighter stop behavior.

---

## 8) File/Path quick reference

* World: `/home/nsumesh/MSML_642_FinalProject/gazebo_worlds/small_warehouse.world`
* Package: `~/MSML_642_FinalProject/src/rl_nav`
* Entry point: `ros2 run rl_nav train_ppo`
* Saved model & logs: `~/ppo_runs/`

---

## 9) Rebuilding after edits

```bash
cd ~/MSML_642_FinalProject
colcon build --packages-select rl_nav --symlink-install
source ~/MSML_642_FinalProject/install/setup.bash
```

That’s it—launch Gazebo, spawn the robot, run `train_ppo`, and watch the policy learn.
=======
# Launch full simulation (with GUI)
bash scripts/run_all.sh

## Compiling and running custom world with objects
# build project
cd ~/ros2_ws
colcon build --packages-select warehouse_sim
source install/setup.bash

# launch warehouse world
ros2 launch warehouse_sim warehouse_with_objects.launch.py



>>>>>>> Stashed changes
