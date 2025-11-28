#!/usr/bin/env python3
"""
train_ppo.py

PPO training for TurtleBot3 Waffle Pi in the warehouse world.

Workflow
--------
1) In one terminal, start the world (NO robot):

   ./launch_warehouse.sh

2) In another terminal, run PPO training:

   ros2 run rl_nav train_ppo -- --timesteps 20000

This script:
  - Spawns the TB3 once via /spawn_entity.
  - Teleports TB3 at the start of every episode via /set_entity_state.
  - Trains a PPO policy with a simple, shaped reward and curriculum.

Reward (per step)
-----------------
r =  2.0 * (prev_dist - dist)      # progress toward goal
     - 0.01                        # small time penalty
     - 0.3 * heading_error        # penalize not facing goal
     - 20.0   on collision        # big penalty
     + 20.0   on success          # big bonus

Curriculum
----------
Stage 1: very easy straight corridor (short run)
Stage 2: medium runs in central area
Stage 3: longer runs in a larger region

Stage progression:
  Stage 1 -> 2  when:
      len(recent_episodes) >= MIN_EPISODES_STAGE1
      AND success_rate >= 0.75
      AND success_streak >= 3

  Stage 2 -> 3  when:
      len(recent_episodes) >= MIN_EPISODES_STAGE2
      AND success_rate >= 0.80
      AND success_streak >= 3

NOTE: If Stage 1 start/goal are not in the exact bottom aisle,
      use /odom while teleoping the robot to read the correct
      (x, y) and update STAGE1_SX/STAGE1_SY below.
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


# -------------------------------------------------------------------
# CONFIG: coordinates & curriculum
# -------------------------------------------------------------------

# Approximate bottom-aisle spawn (update with /odom if needed)
STAGE1_SX = 2.04      # center of corridor in x
STAGE1_SY = -7.3     # bottom aisle y  (adjust if needed)

# Stage 1 goal distance straight ahead (in meters)
STAGE1_GOAL_MIN = 1.2
STAGE1_GOAL_MAX = 1.6

# Stage 2 band (medium difficulty corridor)
STAGE2_SX_MIN = -0.7
STAGE2_SX_MAX = 0.7
STAGE2_SY_MIN = -6.8
STAGE2_SY_MAX = -5.6

# Stage 3 region (wider, further goals)
STAGE3_SX_MIN = -1.0
STAGE3_SX_MAX = 1.0
STAGE3_SY_MIN = -9.5
STAGE3_SY_MAX = -7.0

# Curriculum thresholds
MIN_EPISODES_STAGE1 = 10
MIN_EPISODES_STAGE2 = 12
STAGE1_SUCCESS_THRESHOLD = 0.75
STAGE2_SUCCESS_THRESHOLD = 0.80


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def angle_diff(a: float, b: float) -> float:
    """
    Smallest signed difference between two angles (rad) in [-pi, pi].
    """
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def spawn_tb3(node: Node) -> bool:
    """
    Spawn the TurtleBot3 Waffle Pi using the /spawn_entity service.

    Assumes the TB3 Gazebo model is installed in your turtlebot3_ws.
    Adjust the path if your workspace layout is different.
    """
    logger = node.get_logger()
    logger.info("Waiting for /spawn_entity service...")

    model_path = os.path.expanduser(
        "~/turtlebot3_ws/install/turtlebot3_gazebo/share/"
        "turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf"
    )

    if not os.path.exists(model_path):
        logger.error(f"TB3 model not found at: {model_path}")
        return False

    with open(model_path, "r") as f:
        sdf_xml = f.read()

    spawn_cli = node.create_client(SpawnEntity, "/spawn_entity")
    if not spawn_cli.wait_for_service(timeout_sec=10.0):
        logger.error("Service /spawn_entity not available.")
        return False

    req = SpawnEntity.Request()
    req.name = "tb3"
    req.xml = sdf_xml
    req.robot_namespace = ""
    req.reference_frame = "world"

    # Initial pose = what you measured from Gazebo (/odom)
    req.initial_pose.position.x = 2.04
    req.initial_pose.position.y = -7.30
    req.initial_pose.position.z = 0.01  # small offset above ground

    # Orientation from your z, w (yaw ≈ 0)
    yaw = 0.0
    req.initial_pose.orientation.x = 0.0
    req.initial_pose.orientation.y = 0.0
    req.initial_pose.orientation.z = math.sin(yaw / 2.0)
    req.initial_pose.orientation.w = math.cos(yaw / 2.0)

    future = spawn_cli.call_async(req)
    rclpy.spin_until_future_complete(node, future)

    if future.result() is None:
        logger.error("Failed to spawn TB3 via /spawn_entity.")
        return False

    logger.info("TB3 spawned successfully (train_ppo).")
    return True


# -------------------------------------------------------------
# TB3 Environment Node
# -------------------------------------------------------------

class Tb3Env(Node):
    """
    ROS2-based environment for PPO.

    Observation:
        - 24 binned laser ranges (normalized to [0, 1])
        - dx, dy to goal
        - robot yaw
      => 27-dimensional float32 vector

    Actions (discrete, 0..4):
        0: forward + left turn
        1: forward straight
        2: forward + right turn
        3: rotate left in place
        4: rotate right in place
    """

    def __init__(self):
        super().__init__("tb3_env")

        # --- ROS I/O ---
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self._on_scan, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self._on_odom, 10
        )
        self.reset_cli = self.create_client(SetEntityState, "/set_entity_state")

        # --- Core state ---
        self.max_range = 3.5
        self.num_scan_bins = 24

        self.scan = None  # np.array shape (24,)
        self.pose = np.zeros(3, dtype=np.float32)  # x, y, yaw
        self.goal = np.array([0.0, 6.0], dtype=np.float32)
        self.collision = False
        self.min_obstacle_dist = self.max_range

        self.prev_dist = None
        self.episode_idx = 0
        self.episode_steps = 0
        self.cumulative_reward = 0.0

        self.step_time = 0.15    # seconds per RL step
        self.max_steps = 200     # max steps per episode

        # Discrete actions (v, w)
        self.actions = [
            (0.15,  0.8),   # forward + left
            (0.15,  0.0),   # forward
            (0.15, -0.8),   # forward + right
            (0.00,  0.8),   # rotate left
            (0.00, -0.8),   # rotate right
        ]

        # --- Curriculum state ---
        self.stage = 1
        self.recent_successes = deque(maxlen=20)
        self.success_streak = 0

        # --- Metrics logging ---
        self.metrics_path = os.path.expanduser(
            "~/MSML_642_FinalProject/ppo_runs/episode_metrics.csv"
        )
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w") as f:
                f.write(
                    "episode,stage,steps,final_dist,success,return\n"
                )

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------

    def _on_scan(self, msg: LaserScan):
        """
        Downsample LaserScan into num_scan_bins bins.
        Also set collision and track min obstacle distance.
        """
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
        """
        Update robot pose = [x, y, yaw].
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.pose[:] = (x, y, yaw)

    # ---------------------------------------------------------
    # Curriculum: start/goal sampling
    # ---------------------------------------------------------
    def _sample_stage1(self):
        """
        Stage 1: VERY EASY straight corridor near the measured pose.
        Start near (2.04, -7.30), facing +X, goal 1.2–1.6 m ahead in +X.
        """
        base_x = 2.04
        base_y = -7.30

        # Small noise around start pose
        sx = base_x + random.uniform(-0.1, 0.1)
        sy = base_y + random.uniform(-0.05, 0.05)
        yaw = 0.0 + random.uniform(-0.05, 0.05)  # almost straight in +X

        # Goal: straight ahead in +X
        gx = sx + random.uniform(1.2, 1.6)
        gy = sy + random.uniform(-0.05, 0.05)

        return (sx, sy, yaw), (gx, gy)

    def _sample_stage2(self):
        """
        Stage 2: medium runs in central aisle.
        """
        sx = random.uniform(STAGE2_SX_MIN, STAGE2_SX_MAX)
        sy = random.uniform(STAGE2_SY_MIN, STAGE2_SY_MAX)
        yaw = math.pi / 2.0 + random.uniform(-0.25, 0.25)

        gx = random.uniform(STAGE2_SX_MIN, STAGE2_SX_MAX)
        gy = sy + random.uniform(2.0, 3.0)

        return (sx, sy, yaw), (gx, gy)

    def _sample_stage3(self):
        """
        Stage 3: longer runs, broader area.
        """
        sx = random.uniform(STAGE3_SX_MIN, STAGE3_SX_MAX)
        sy = random.uniform(STAGE3_SY_MIN, STAGE3_SY_MAX)
        yaw = math.pi / 2.0 + random.uniform(-0.4, 0.4)

        gx = random.uniform(STAGE3_SX_MIN, STAGE3_SX_MAX)
        gy = random.uniform(3.0, 7.0)

        return (sx, sy, yaw), (gx, gy)

    def _choose_start_goal(self):
        if self.stage == 1:
            return self._sample_stage1()
        elif self.stage == 2:
            return self._sample_stage2()
        else:
            return self._sample_stage3()

    # ---------------------------------------------------------
    # Observation
    # ---------------------------------------------------------

    def _obs(self):
        """
        27-dim observation: 24 scan bins + [dx, dy, yaw].
        """
        if self.scan is None:
            scan = np.ones(self.num_scan_bins, dtype=np.float32)
        else:
            scan = self.scan

        dx, dy = (self.goal - self.pose[:2])
        tail = np.array([dx, dy, self.pose[2]], dtype=np.float32)
        return np.concatenate([scan, tail], axis=0)

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------

    def reset(self):
        """
        Reset environment for a new episode:
          - Log previous episode (if any)
          - Update curriculum
          - Sample new start/goal
          - Teleport TB3 via /set_entity_state
          - Wait for sensors and return initial observation
        """
        # Log previous episode & update curriculum
        if self.episode_steps > 0 and self.prev_dist is not None:
            final_dist = float(np.linalg.norm(self.goal - self.pose[:2]))
            success = int(final_dist < 0.4 and not self.collision)

            # Update success history
            self.recent_successes.append(success)
            if success:
                self.success_streak += 1
            else:
                self.success_streak = 0

            sr = sum(self.recent_successes) / len(self.recent_successes)

            prev_stage = self.stage

            # Stage 1 -> 2
            if (
                self.stage == 1
                and len(self.recent_successes) >= MIN_EPISODES_STAGE1
                and sr >= STAGE1_SUCCESS_THRESHOLD
                and self.success_streak >= 3
            ):
                self.stage = 2
                self.get_logger().info(
                    f"[Curriculum] Advanced to STAGE 2 "
                    f"(success_rate={sr:.2f}, streak={self.success_streak})"
                )

            # Stage 2 -> 3
            elif (
                self.stage == 2
                and len(self.recent_successes) >= MIN_EPISODES_STAGE2
                and sr >= STAGE2_SUCCESS_THRESHOLD
                and self.success_streak >= 3
            ):
                self.stage = 3
                self.get_logger().info(
                    f"[Curriculum] Advanced to STAGE 3 "
                    f"(success_rate={sr:.2f}, streak={self.success_streak})"
                )

            # Append metrics (use previous stage for logging)
            with open(self.metrics_path, "a") as f:
                f.write(
                    f"{self.episode_idx},"
                    f"{prev_stage},"
                    f"{self.episode_steps},"
                    f"{final_dist:.3f},"
                    f"{success},"
                    f"{self.cumulative_reward:.3f}\n"
                )

            self.get_logger().info(
                f"[EP {self.episode_idx}] "
                f"stage={prev_stage} steps={self.episode_steps} "
                f"final_dist={final_dist:.2f} success={success} "
                f"return={self.cumulative_reward:.2f} "
                f"(sr={sr:.2f}, streak={self.success_streak})"
            )

        # Reset counters for new episode
        self.episode_idx += 1
        self.episode_steps = 0
        self.cumulative_reward = 0.0
        self.collision = False
        self.min_obstacle_dist = self.max_range

        # Sample curriculum start/goal
        (sx, sy, syaw), (gx, gy) = self._choose_start_goal()
        self.goal[:] = (gx, gy)

        # Teleport TB3
        if not self.reset_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/set_entity_state service not available.")
        else:
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

        # Wait briefly for sensors to update
        t0 = time.time()
        self.scan = None
        while (self.scan is None) and (time.time() - t0 < 2.0):
            rclpy.spin_once(self, timeout_sec=0.05)

        self.prev_dist = float(np.linalg.norm(self.goal - self.pose[:2]))
        return self._obs()

    # ---------------------------------------------------------
    # Step with SIMPLE reward
    # ---------------------------------------------------------

    def step(self, a_idx: int):
        """
        Apply discrete action, step the sim, compute SIMPLE reward.
        SIMPLE reward:
            r =  2.0 * (prev_dist - dist)      # progress
                 - 0.01                        # time penalty
                 - 0.3 * heading_error         # heading shaping
                 - 20.0 if collision
                 + 20.0 if success
        """
        # Apply action
        v, w = self.actions[int(a_idx)]
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_pub.publish(twist)

        # Let simulation run for step_time seconds
        end_time = self.get_clock().now().nanoseconds + int(self.step_time * 1e9)
        while self.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self, timeout_sec=0.01)

        self.episode_steps += 1

        # Observation and geometry
        obs = self._obs()
        dx, dy = (self.goal - self.pose[:2])
        dist = float(math.hypot(dx, dy))

        # Progress (positive if getting closer)
        if self.prev_dist is None:
            delta = 0.0
        else:
            delta = self.prev_dist - dist
        self.prev_dist = dist

        # Base reward: progress + small time penalty
        r = 2.0 * delta
        r -= 0.01

        # Heading shaping: encourage facing the goal direction
        goal_heading = math.atan2(dy, dx)
        heading_error = abs(angle_diff(goal_heading, self.pose[2]))
        r -= 0.3 * heading_error

        done = False
        success = False

        # Collision penalty
        if self.collision:
            r -= 20.0
            done = True

        # Success bonus
        if (dist < 0.4) and (not self.collision):
            r += 20.0
            success = True
            done = True

        # Max steps → end episode
        if self.episode_steps >= self.max_steps:
            done = True

        self.cumulative_reward += r

        info = {
            "distance_to_goal": dist,
            "stage": self.stage,
            "episode_steps": self.episode_steps,
            "success": success,
        }
        return obs, r, done, info


# -------------------------------------------------------------
# Gymnasium Wrapper for Stable-Baselines3
# -------------------------------------------------------------

class GymTb3(gym.Env):
    """
    Gymnasium wrapper around Tb3Env for Stable-Baselines3 PPO.
    """
    metadata = {"render_modes": []}

    def __init__(self, node: Tb3Env):
        super().__init__()
        self.node = node
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32
        )
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
        truncated = (
            self.node.episode_steps >= self.node.max_steps and not terminated
        )
        return obs, reward, terminated, truncated, info


# -------------------------------------------------------------
# Main: PPO Training
# -------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument(
        "--logdir",
        type=str,
        default=os.path.expanduser("~/MSML_642_FinalProject/ppo_runs"),
    )
    args = parser.parse_args()

    rclpy.init()
    node = Tb3Env()

    # Spawn TB3 once (world must already be running)
    if not spawn_tb3(node):
        node.get_logger().error("TB3 spawn failed. Exiting.")
        rclpy.shutdown()
        return

    env = GymTb3(node)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=args.logdir,
        n_steps=512,                 # longer rollout for better gradient estimates
        batch_size=128,
        learning_rate=3e-4,
        gamma=0.995,                 # slightly longer horizon
        gae_lambda=0.98,
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    model.learn(total_timesteps=args.timesteps)

    os.makedirs(args.logdir, exist_ok=True)
    out = os.path.join(args.logdir, "tb3_ppo.zip")
    model.save(out)
    print("Saved policy to", out)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
