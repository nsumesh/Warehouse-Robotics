#!/usr/bin/env python3
"""
PPO training for simplified warehouse navigation with virtual sorting.

This implements a 3-stage curriculum:
- Stage 1: Short local docking near DOCK_A
- Stage 2: Longer aisle navigation to DOCK_A
- Stage 3: Virtual sorting between DOCK_A/DOCK_B/DOCK_C

Usage:
  ros2 run rl_nav train_ppo --timesteps 20000 --curriculum-stage 1

Parallel seeds (for robustness):
  Run multiple seeds in separate terminals:
  Terminal 1: ros2 run rl_nav train_ppo --timesteps 80000 --curriculum-stage 3 --seed 0
  Terminal 2: ros2 run rl_nav train_ppo --timesteps 80000 --curriculum-stage 3 --seed 1
  Terminal 3: ros2 run rl_nav train_ppo --timesteps 80000 --curriculum-stage 3 --seed 2
"""
import os
import math
import time
import random
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SetEntityState, SpawnEntity
from gazebo_msgs.msg import EntityState
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import torch

# Workspace limits - limited aisle workspace
X_MIN, X_MAX = -7.0, 2.0
Y_MIN, Y_MAX = -3.0, 3.0

# Docking bays - virtual sorting categories
# Each dock represents a different sorting category
DOCK_A = (-6.5, -2.0)  # Sorting category A
DOCK_B = (-6.5, 0.0)   # Sorting category B
DOCK_C = (-6.5, 2.0)   # Sorting category C

# Optional pickup zone (for future use)
PICKUP = (-4.0, 0.0)


def angle_diff(a: float, b: float) -> float:
    """Smallest signed difference between two angles in [-pi, pi]."""
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def spawn_tb3(node: Node) -> bool:
    """Spawn TurtleBot3 in Gazebo."""
    model_path = os.path.expanduser(
        "~/turtlebot3_ws/install/turtlebot3_gazebo/share/"
        "turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf"
    )
    if not os.path.exists(model_path):
        node.get_logger().error(f"TB3 model not found: {model_path}")
        return False

    with open(model_path, "r") as f:
        sdf_xml = f.read()

    spawn_cli = node.create_client(SpawnEntity, "/spawn_entity")
    if not spawn_cli.wait_for_service(timeout_sec=10.0):
        node.get_logger().error("Service /spawn_entity not available")
        return False

    req = SpawnEntity.Request()
    req.name = "tb3"
    req.xml = sdf_xml
    req.robot_namespace = ""
    req.reference_frame = "world"
    # Spawn in workspace center
    req.initial_pose.position.x = (X_MIN + X_MAX) / 2.0
    req.initial_pose.position.y = (Y_MIN + Y_MAX) / 2.0
    req.initial_pose.position.z = 0.01
    yaw = 0.0
    req.initial_pose.orientation.z = math.sin(yaw / 2.0)
    req.initial_pose.orientation.w = math.cos(yaw / 2.0)

    future = spawn_cli.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is None:
        node.get_logger().error("Failed to spawn TB3")
        return False
    node.get_logger().info("TB3 spawned successfully")
    return True


class Tb3Env(Node):
    """ROS2 environment for PPO training. Observation: 24 LiDAR bins + [dx, dy, yaw] + [task_class one-hot] = 30 dims."""

    def __init__(self, curriculum_stage=1):
        super().__init__("tb3_env")
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self._on_scan, 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.reset_cli = self.create_client(SetEntityState, "/set_entity_state")

        self.max_range = 3.5
        self.num_scan_bins = 24
        self.scan = None
        self.pose = np.zeros(3, dtype=np.float32)
        self.goal = np.array([0.0, 6.0], dtype=np.float32)
        self.collision = False
        self.min_obstacle_dist = self.max_range

        self.prev_dist = None
        self.episode_idx = 0
        self.episode_steps = 0
        self.cumulative_reward = 0.0
        self.step_time = 0.15
        self.max_steps = 300

        self.actions = [
            (0.12, 0.6), (0.15, 0.0), (0.12, -0.6), (0.00, 0.6), (0.00, -0.6)
        ]
        self.recent_positions = deque(maxlen=10)
        self.curriculum_stage = curriculum_stage
        self.task_class = None  # For Stage 3: 'A', 'B', or 'C'
        self.metrics_path = None

    def _on_scan(self, msg: LaserScan):
        """Process LiDAR scan into bins."""
        ranges = msg.ranges
        n = len(ranges)
        if n == 0:
            self.scan = np.ones(self.num_scan_bins, dtype=np.float32)
            self.collision = False
            return

        step = max(1, n // self.num_scan_bins)
        bins = []
        min_dist = self.max_range
        collision = False

        for i in range(0, n, step):
            v = ranges[i]
            if math.isnan(v) or v <= 0.0:
                v = self.max_range
            if v < min_dist:
                min_dist = v
            if v < 0.18:
                collision = True
            bins.append(min(v, self.max_range) / self.max_range)
            if len(bins) == self.num_scan_bins:
                break

        if len(bins) < self.num_scan_bins:
            bins.extend([1.0] * (self.num_scan_bins - len(bins)))

        self.scan = np.array(bins, dtype=np.float32)
        self.collision = collision
        self.min_obstacle_dist = float(min_dist)

    def _on_odom(self, msg: Odometry):
        """Update robot pose [x, y, yaw]."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose[:] = (x, y, yaw)

    def sample_start_and_goal(self):
        """Sample start pose and goal based on curriculum stage."""
        if self.curriculum_stage == 1:
            # Stage 1: Short local docking near DOCK_A
            # Start close to DOCK_A, goal is DOCK_A
            dock_x, dock_y = DOCK_A
            sx = dock_x + random.uniform(-0.5, 0.5)
            sy = dock_y + random.uniform(-0.5, 0.5)
            # Ensure within workspace
            sx = max(X_MIN, min(X_MAX, sx))
            sy = max(Y_MIN, min(Y_MAX, sy))
            yaw = random.uniform(-math.pi, math.pi)
            gx, gy = DOCK_A
            self.task_class = None
            return (sx, sy, yaw), (gx, gy)
        
        elif self.curriculum_stage == 2:
            # Stage 2: Longer aisle navigation to DOCK_A
            # Start sampled in workspace, goal is DOCK_A
            sx = random.uniform(X_MIN, X_MAX)
            sy = random.uniform(Y_MIN, Y_MAX)
            yaw = random.uniform(-math.pi, math.pi)
            gx, gy = DOCK_A
            self.task_class = None
            return (sx, sy, yaw), (gx, gy)
        
        else:  # Stage 3
            # Stage 3: Virtual sorting between DOCK_A/DOCK_B/DOCK_C
            # Start sampled in workspace, randomly choose task class
            sx = random.uniform(X_MIN, X_MAX)
            sy = random.uniform(Y_MIN, Y_MAX)
            yaw = random.uniform(-math.pi, math.pi)
            # Randomly choose task class A, B, or C
            self.task_class = random.choice(['A', 'B', 'C'])
            if self.task_class == 'A':
                gx, gy = DOCK_A
            elif self.task_class == 'B':
                gx, gy = DOCK_B
            else:  # 'C'
                gx, gy = DOCK_C
            return (sx, sy, yaw), (gx, gy)

    def _obs(self):
        """Build 30-dim observation: 24 scan bins + [dx, dy, yaw] + [task_class one-hot]."""
        if self.scan is None:
            scan = np.ones(self.num_scan_bins, dtype=np.float32)
        else:
            scan = self.scan
        dx, dy = (self.goal - self.pose[:2])
        tail = np.array([dx, dy, self.pose[2]], dtype=np.float32)
        
        # Task class one-hot encoding (3 dims)
        # For Stage 3: [1,0,0] for A, [0,1,0] for B, [0,0,1] for C
        # For Stage 1/2: use dummy [1,0,0]
        if self.curriculum_stage == 3 and self.task_class:
            if self.task_class == 'A':
                task_onehot = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            elif self.task_class == 'B':
                task_onehot = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            else:  # 'C'
                task_onehot = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            # Stage 1/2: dummy one-hot
            task_onehot = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        return np.concatenate([scan, tail, task_onehot], axis=0)

    def reset(self):
        """Reset environment for new episode."""
        if self.episode_steps > 0 and self.prev_dist is not None:
            final_dist = float(np.linalg.norm(self.goal - self.pose[:2]))
            success = int(final_dist < 0.4 and not self.collision)

            # Log metrics
            if self.metrics_path is None:
                self.metrics_path = os.path.join("ppo_runs", "episode_metrics.csv")
            os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
            with open(self.metrics_path, "a") as f:
                f.write(f"{self.episode_idx},{self.curriculum_stage},{self.episode_steps},"
                        f"{final_dist:.3f},{success},{self.cumulative_reward:.3f}\n")

            task_info = f"task={self.task_class}" if self.task_class else ""
            self.get_logger().info(f"[EP {self.episode_idx}] stage={self.curriculum_stage} {task_info} "
                                  f"steps={self.episode_steps} dist={final_dist:.2f}m "
                                  f"success={success} return={self.cumulative_reward:.2f}")

        # Reset counters
        self.episode_idx += 1
        self.episode_steps = 0
        self.cumulative_reward = 0.0
        self.collision = False
        self.min_obstacle_dist = self.max_range
        self.recent_positions.clear()

        # Sample new start/goal using curriculum stage
        (sx, sy, syaw), (gx, gy) = self.sample_start_and_goal()
        self.goal[:] = (gx, gy)

        # Teleport robot
        if self.reset_cli.wait_for_service(timeout_sec=5.0):
            req = SetEntityState.Request()
            req.state = EntityState()
            req.state.name = "tb3"
            req.state.pose.position.x = float(sx)
            req.state.pose.position.y = float(sy)
            req.state.pose.position.z = 0.01
            cy, syq = math.cos(syaw / 2.0), math.sin(syaw / 2.0)
            req.state.pose.orientation.z = syq
            req.state.pose.orientation.w = cy
            self.reset_cli.call_async(req)

        # Wait for sensors
        t0 = time.time()
        self.scan = None
        while (self.scan is None) and (time.time() - t0 < 2.0):
            rclpy.spin_once(self, timeout_sec=0.05)

        self.prev_dist = float(np.linalg.norm(self.goal - self.pose[:2]))
        return self._obs()

    def step(self, a_idx: int):
        """Execute action and compute simplified reward."""
        v, w = self.actions[int(a_idx)]
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_pub.publish(twist)

        end_time = self.get_clock().now().nanoseconds + int(self.step_time * 1e9)
        while self.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self, timeout_sec=0.01)

        self.episode_steps += 1
        obs = self._obs()
        dx, dy = (self.goal - self.pose[:2])
        dist = float(math.hypot(dx, dy))

        # Reward constants
        k_progress = 1.0
        k_time = 0.01
        k_collision = 5.0
        k_success = 10.0
        SUCCESS_RADIUS = 0.4

        # Progress reward
        if self.prev_dist is None:
            delta = 0.0
        else:
            delta = self.prev_dist - dist
        self.prev_dist = dist

        r = k_progress * delta
        r -= k_time  # Time penalty every step

        done = False
        success = False

        # Collision penalty
        if self.collision:
            r -= k_collision
            done = True

        # Success bonus
        if dist < SUCCESS_RADIUS and not self.collision:
            # For Stage 3: check if reached correct dock
            if self.curriculum_stage == 3 and self.task_class:
                # Check which dock is closest
                dist_to_a = math.hypot(self.pose[0] - DOCK_A[0], self.pose[1] - DOCK_A[1])
                dist_to_b = math.hypot(self.pose[0] - DOCK_B[0], self.pose[1] - DOCK_B[1])
                dist_to_c = math.hypot(self.pose[0] - DOCK_C[0], self.pose[1] - DOCK_C[1])
                
                # Find closest dock
                closest_dock = min([('A', dist_to_a), ('B', dist_to_b), ('C', dist_to_c)], key=lambda x: x[1])
                
                if closest_dock[0] == self.task_class and closest_dock[1] < SUCCESS_RADIUS:
                    # Correct dock
                    r += k_success
                    success = True
                    done = True
                elif closest_dock[1] < SUCCESS_RADIUS:
                    # Wrong dock - small penalty
                    r -= 2.0
                    done = True
            else:
                # Stage 1/2: any success is good
                r += k_success
                success = True
                done = True

        # Timeout
        if self.episode_steps >= self.max_steps:
            done = True

        self.cumulative_reward += r

        info = {
            "distance_to_goal": dist,
            "curriculum_stage": self.curriculum_stage,
            "episode_steps": self.episode_steps,
            "success": success,
            "task_class": self.task_class,
        }
        return obs, r, done, info


class GymTb3(gym.Env):
    """Gymnasium wrapper for Stable-Baselines3."""
    metadata = {"render_modes": []}

    def __init__(self, node: Tb3Env):
        super().__init__()
        self.node = node
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.node.actions))

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        obs = self.node.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.node.step(int(action))
        terminated = done
        truncated = (self.node.episode_steps >= self.node.max_steps and not terminated)
        return obs, reward, terminated, truncated, info


def main():
    """
    Train PPO policy for simplified warehouse navigation with virtual sorting.
    
    Example commands:
    # Stage 1: Short local docking
    ros2 run rl_nav train_ppo --timesteps 20000 --curriculum-stage 1
    
    # Stage 2: Longer aisle navigation
    ros2 run rl_nav train_ppo --timesteps 80000 --curriculum-stage 2
    
    # Stage 3: Virtual sorting between docks
    ros2 run rl_nav train_ppo --timesteps 100000 --curriculum-stage 3
    
    # With seed for reproducibility
    ros2 run rl_nav train_ppo --timesteps 80000 --curriculum-stage 3 --seed 0
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO policy for TurtleBot3 navigation with virtual sorting")
    parser.add_argument("--timesteps", type=int, default=20000, help="Total timesteps to train")
    parser.add_argument("--logdir", type=str, default="ppo_runs",
                        help="Directory for logs and models (default: ppo_runs)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--curriculum-stage", type=int, default=1, choices=[1, 2, 3],
                        help="Curriculum stage: 1=local docking, 2=aisle nav, 3=virtual sorting")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    rclpy.init()
    if not os.path.isabs(args.logdir):
        args.logdir = os.path.abspath(args.logdir)
    os.makedirs(args.logdir, exist_ok=True)

    # Set seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Create environment with curriculum stage
    node = Tb3Env(curriculum_stage=args.curriculum_stage)
    node.metrics_path = os.path.join(args.logdir, "episode_metrics.csv")
    if not os.path.exists(node.metrics_path):
        with open(node.metrics_path, "w") as f:
            f.write("episode,stage,steps,final_dist,success,return\n")

    node.get_logger().info(f"Training with curriculum stage {args.curriculum_stage}")

    if not spawn_tb3(node):
        node.get_logger().error("TB3 spawn failed")
        rclpy.shutdown()
        return

    env = GymTb3(node)

    if args.resume:
        if not os.path.isabs(args.resume):
            resume_path = os.path.join(args.logdir, args.resume)
            if not os.path.exists(resume_path):
                resume_path = os.path.abspath(args.resume)
        else:
            resume_path = os.path.expanduser(args.resume)

        if os.path.exists(resume_path):
            node.get_logger().info(f"Loading checkpoint: {resume_path}")
            model = PPO.load(resume_path, env=env)
        else:
            node.get_logger().warn(f"Checkpoint not found: {resume_path}. Starting fresh.")
            args.resume = None

    if not args.resume:
        model = PPO(
            "MlpPolicy", env, verbose=1, tensorboard_log=args.logdir,
            n_steps=256, batch_size=64, n_epochs=10, learning_rate=3e-4,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
            vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[64, 64], activation_fn=torch.nn.Tanh),
            seed=args.seed,
        )
        node.get_logger().info("Starting fresh training")

    model.learn(total_timesteps=args.timesteps)
    
    # Save model with stage-specific name
    if args.curriculum_stage == 1:
        model_name = "ppo_stage1.zip"
    elif args.curriculum_stage == 2:
        model_name = "ppo_stage2.zip"
    else:  # Stage 3
        model_name = "ppo_stage3_sorting.zip"
    
    out = os.path.join(args.logdir, model_name)
    model.save(out)
    node.get_logger().info(f"Saved policy to {out}")
    print(f"Saved policy to {out}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
