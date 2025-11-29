#!/usr/bin/env python3
"""
train_ppo.py

PPO training for TurtleBot3 Waffle Pi in the warehouse world.

Workflow
--------
1) In one terminal, start the world (NO robot):

   # For Stage 1 (optional - easier learning):
   ./launch_warehouse_empty.sh
   
   # For Stage 2+ (required - introduces obstacles):
   ./launch_warehouse.sh

2) In another terminal, run PPO training:

   ros2 run rl_nav train_ppo --timesteps 20000
   
Note: Stage 1 can train in empty world, but Stage 2+ should use
      warehouse with objects to learn obstacle avoidance.

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
import torch


# -------------------------------------------------------------------
# CONFIG: coordinates & curriculum
# -------------------------------------------------------------------
# ASSUMPTIONS FOR STABLE LEARNING:
# 1. Stage 1: Very small area, very short goals (0.4-0.8m) - EMPTY WORLD (no objects)
# 2. Stage 2: Slightly larger area, short goals (0.8-1.5m) - WAREHOUSE WITH OBJECTS (learn obstacle avoidance)
# 3. Stage 3: Full warehouse area, longer goals (3-5m) - WAREHOUSE WITH OBJECTS (full navigation)
#
# IMPORTANT: Stage 2 should use launch_warehouse.sh (with objects) to learn obstacle avoidance
#            while keeping goals short and in clear aisles to avoid overwhelming the agent.

# Stage 1: VERY EASY - small area, short straight goals
# NOTE: Shelves are at x=-4, 0, 4. Use clear aisle between shelves.
# Aisle between x=-4 and x=0 is at x=-2, or between x=0 and x=4 is at x=2
STAGE1_SX = -2.0      # clear aisle between shelves (or use 2.0)
STAGE1_SY = 0.0       # start in middle of aisle (shelves at y=-6 to y=6)
STAGE1_AREA_SIZE = 1.5  # 1.5m x 1.5m area for Stage 1 (smaller to avoid shelves)

# Stage 1 goal distance - VERY SHORT for easy learning
STAGE1_GOAL_MIN = 0.4   # 0.4m minimum (very easy, almost trivial)
STAGE1_GOAL_MAX = 0.8   # 0.8m maximum (still easy)

# Stage 2: Medium difficulty - GRADUAL increase from Stage 1
# Use clear aisles between shelves, but keep goals closer initially
# Shelves at x=-4, 0, 4. Clear aisles at x=-2, 2, and also x=-6, 6
STAGE2_SX_MIN = -2.5  # clear aisle area
STAGE2_SX_MAX = 2.5   # can use both aisles
STAGE2_SY_MIN = -3.0  # avoid shelf rows (shelves at y=-6, -3.6, -1.2, 1.2, 3.6, 6.0)
STAGE2_SY_MAX = 3.0
STAGE2_GOAL_MIN = 0.8   # Start closer to Stage 1 (was 1.5)
STAGE2_GOAL_MAX = 1.5   # Gradual increase (was 2.5)

# Stage 3: Full warehouse - use all clear areas and navigate around shelves
# Shelves span from y=-6 to y=6, with spacing 2.4
# Clear areas: between shelves and at edges
STAGE3_SX_MIN = -5.0  # wider area including edges
STAGE3_SX_MAX = 5.0
STAGE3_SY_MIN = -8.0  # include pallet area at y=-9
STAGE3_SY_MAX = 7.0   # above top shelves
STAGE3_GOAL_MIN = 3.0
STAGE3_GOAL_MAX = 5.0

# Curriculum thresholds - more lenient for Stage 1, stricter for Stage 2
MIN_EPISODES_STAGE1 = 25  # More episodes to ensure solid learning
MIN_EPISODES_STAGE2 = 30  # More episodes in Stage 2 before advancing
STAGE1_SUCCESS_THRESHOLD = 0.75  # Higher threshold - ensure mastery (was 0.65)
STAGE2_SUCCESS_THRESHOLD = 0.70  # Keep same for Stage 2


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

    # Initial pose = start in clear aisle for Stage 1 (between shelves)
    # Shelves are at x=-4, 0, 4, so use x=-2 or x=2 for clear aisle
    req.initial_pose.position.x = -2.0
    req.initial_pose.position.y = 0.0
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
        self.max_steps = 300     # increased for longer goals

        # Discrete actions (v, w) - slightly slower for more control
        self.actions = [
            (0.12,  0.6),   # forward + left (slower turn)
            (0.15,  0.0),   # forward
            (0.12, -0.6),   # forward + right (slower turn)
            (0.00,  0.6),   # rotate left (slower)
            (0.00, -0.6),   # rotate right (slower)
        ]
        
        # For oscillation detection
        self.recent_positions = deque(maxlen=10)  # track last 10 positions
        self.recent_actions = deque(maxlen=5)     # track last 5 actions

        # --- Curriculum state ---
        self.stage = 1  # Can be overridden by main() when resuming
        self.recent_successes = deque(maxlen=20)
        self.success_streak = 0

        # --- Metrics logging ---
        # Use relative path in repo (will be set by main() with actual logdir)
        self.metrics_path = None  # Will be set in main()

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
        Stage 1: VERY EASY - small area, very short goals.
        ASSUMPTION: Empty corridor, no obstacles.
        """
        # Start in small area around STAGE1_SX, STAGE1_SY (clear aisle)
        sx = STAGE1_SX + random.uniform(-STAGE1_AREA_SIZE/2, STAGE1_AREA_SIZE/2)
        sy = STAGE1_SY + random.uniform(-STAGE1_AREA_SIZE/2, STAGE1_AREA_SIZE/2)
        yaw = random.uniform(-0.15, 0.15)  # even smaller orientation variation

        # Goal: very close, ALWAYS forward (straight ahead)
        goal_dist = random.uniform(STAGE1_GOAL_MIN, STAGE1_GOAL_MAX)
        goal_angle = random.uniform(-0.15, 0.15)  # very small angle variation (almost straight)
        
        gx = sx + goal_dist * math.cos(yaw + goal_angle)
        gy = sy + goal_dist * math.sin(yaw + goal_angle)

        return (sx, sy, yaw), (gx, gy)

    def _sample_stage2(self):
        """
        Stage 2: medium runs - larger area, medium goals, MORE VARIATION in direction.
        This is distinct from Stage 1: goals can be in any direction relative to start.
        """
        # Start anywhere in Stage 2 area
        sx = random.uniform(STAGE2_SX_MIN, STAGE2_SX_MAX)
        sy = random.uniform(STAGE2_SY_MIN, STAGE2_SY_MAX)
        yaw = random.uniform(-math.pi, math.pi)  # Any orientation (full 360)

        # Goal at medium distance - ANY direction (not just forward)
        goal_dist = random.uniform(STAGE2_GOAL_MIN, STAGE2_GOAL_MAX)
        goal_angle = random.uniform(0, 2 * math.pi)  # Random direction (full 360)
        
        # Goal relative to start position (not relative to yaw)
        gx = sx + goal_dist * math.cos(goal_angle)
        gy = sy + goal_dist * math.sin(goal_angle)
        
        # Keep goal within Stage 2 bounds
        gx = max(STAGE2_SX_MIN, min(STAGE2_SX_MAX, gx))
        gy = max(STAGE2_SY_MIN, min(STAGE2_SY_MAX, gy))

        return (sx, sy, yaw), (gx, gy)

    def _sample_stage3(self):
        """
        Stage 3: longer runs in full warehouse area with obstacles.
        Goals can be anywhere, requiring navigation around shelves.
        """
        # Start anywhere in Stage 3 area (full warehouse)
        sx = random.uniform(STAGE3_SX_MIN, STAGE3_SX_MAX)
        sy = random.uniform(STAGE3_SY_MIN, STAGE3_SY_MAX)
        yaw = random.uniform(0, 2 * math.pi)  # Any orientation

        # Goal at longer distance - any direction
        goal_dist = random.uniform(STAGE3_GOAL_MIN, STAGE3_GOAL_MAX)
        goal_angle = random.uniform(0, 2 * math.pi)  # Random direction
        
        # Goal relative to start position
        gx = sx + goal_dist * math.cos(goal_angle)
        gy = sy + goal_dist * math.sin(goal_angle)
        
        # Keep goal within Stage 3 bounds
        gx = max(STAGE3_SX_MIN, min(STAGE3_SX_MAX, gx))
        gy = max(STAGE3_SY_MIN, min(STAGE3_SY_MAX, gy))

        return (sx, sy, yaw), (gx, gy)

    def _choose_start_goal(self):
        """Choose start/goal based on current curriculum stage"""
        if self.stage == 1:
            start_pose, goal_pose = self._sample_stage1()
            self.get_logger().debug(f"Stage 1: start=({start_pose[0]:.2f}, {start_pose[1]:.2f}), goal=({goal_pose[0]:.2f}, {goal_pose[1]:.2f})")
            return start_pose, goal_pose
        elif self.stage == 2:
            start_pose, goal_pose = self._sample_stage2()
            self.get_logger().debug(f"Stage 2: start=({start_pose[0]:.2f}, {start_pose[1]:.2f}), goal=({goal_pose[0]:.2f}, {goal_pose[1]:.2f})")
            return start_pose, goal_pose
        else:  # Stage 3
            start_pose, goal_pose = self._sample_stage3()
            self.get_logger().debug(f"Stage 3: start=({start_pose[0]:.2f}, {start_pose[1]:.2f}), goal=({goal_pose[0]:.2f}, {goal_pose[1]:.2f})")
            return start_pose, goal_pose

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
            # Use same success threshold as in step() function (0.4m)
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
                    f"(success_rate={sr:.2f}, streak={self.success_streak}, episodes={len(self.recent_successes)})"
                )
                self.get_logger().info(
                    f"[Curriculum] Stage 2: Goals {STAGE2_GOAL_MIN}-{STAGE2_GOAL_MAX}m, "
                    f"Area ({STAGE2_SX_MIN:.1f} to {STAGE2_SX_MAX:.1f}, {STAGE2_SY_MIN:.1f} to {STAGE2_SY_MAX:.1f})"
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
                    f"(success_rate={sr:.2f}, streak={self.success_streak}, episodes={len(self.recent_successes)})"
                )
                self.get_logger().info(
                    f"[Curriculum] Stage 3: Goals {STAGE3_GOAL_MIN}-{STAGE3_GOAL_MAX}m, "
                    f"Area ({STAGE3_SX_MIN:.1f} to {STAGE3_SX_MAX:.1f}, {STAGE3_SY_MIN:.1f} to {STAGE3_SY_MAX:.1f})"
                )

            # Append metrics (use previous stage for logging)
            if self.metrics_path is None:
                # Fallback if not set (shouldn't happen, but safety check)
                self.metrics_path = os.path.join("ppo_runs", "episode_metrics.csv")
            os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
            with open(self.metrics_path, "a") as f:
                f.write(
                    f"{self.episode_idx},"
                    f"{prev_stage},"
                    f"{self.episode_steps},"
                    f"{final_dist:.3f},"
                    f"{success},"
                    f"{self.cumulative_reward:.3f}\n"
                )

            # Log episode with stage info
            stage_str = f"stage={prev_stage}"
            if prev_stage != self.stage:
                stage_str += f"→{self.stage}"  # Show stage transition
            
            self.get_logger().info(
                f"[EP {self.episode_idx}] "
                f"{stage_str} steps={self.episode_steps} "
                f"final_dist={final_dist:.2f}m success={success} "
                f"return={self.cumulative_reward:.2f} "
                f"(sr={sr:.2f}, streak={self.success_streak})"
            )

        # Reset counters for new episode
        self.episode_idx += 1
        self.episode_steps = 0
        self.cumulative_reward = 0.0
        self.collision = False
        self.min_obstacle_dist = self.max_range
        self.recent_positions.clear()
        self.recent_actions.clear()

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
        Apply discrete action, step the sim, compute IMPROVED reward.
        
        IMPROVED reward function:
            r =  5.0 * (prev_dist - dist)      # higher progress reward
                 - 0.005                       # smaller time penalty
                 - 0.2 * heading_error         # reduced heading penalty
                 - 0.5 * oscillation_penalty   # NEW: penalize oscillation
                 - 1.0 * near_miss_penalty     # NEW: penalize getting too close
                 - 30.0 if collision           # larger collision penalty
                 + 50.0 if success             # larger success bonus
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

        # Track position for oscillation detection
        self.recent_positions.append((self.pose[0], self.pose[1]))
        self.recent_actions.append(int(a_idx))

        # Progress (positive if getting closer)
        if self.prev_dist is None:
            delta = 0.0
        else:
            delta = self.prev_dist - dist
        self.prev_dist = dist

        # Base reward: higher progress reward
        r = 8.0 * delta  # Increased from 5.0 to 8.0 for stronger signal
        r -= 0.003  # Even smaller time penalty

        # Heading shaping: encourage facing the goal direction
        goal_heading = math.atan2(dy, dx)
        heading_error = abs(angle_diff(goal_heading, self.pose[2]))
        # Reduce heading penalty - let robot learn to turn while moving
        r -= 0.1 * heading_error  # Further reduced from 0.2

        # NEW: Oscillation detection - penalize going back and forth
        oscillation_penalty = 0.0
        if len(self.recent_positions) >= 5:
            # Check if robot is moving back and forth
            positions = list(self.recent_positions)
            total_movement = 0.0
            for i in range(1, len(positions)):
                dx_pos = positions[i][0] - positions[i-1][0]
                dy_pos = positions[i][1] - positions[i-1][1]
                total_movement += math.hypot(dx_pos, dy_pos)
            
            # If robot moved a lot but didn't get closer, it's oscillating
            if total_movement > 0.5 and delta < 0.05:
                oscillation_penalty = 1.0
        
        r -= 0.5 * oscillation_penalty

        # NEW: Near-miss penalty - penalize getting too close to obstacles
        near_miss_penalty = 0.0
        if self.min_obstacle_dist < 0.5:  # within 0.5m of obstacle
            near_miss_penalty = (0.5 - self.min_obstacle_dist) / 0.5  # 0 to 1
        r -= 1.0 * near_miss_penalty

        done = False
        success = False

        # Collision penalty - larger
        if self.collision:
            r -= 30.0  # Increased from 20.0
            done = True

        # Success bonus - larger and distance-based
        success_threshold = 0.4  # Slightly more lenient (was 0.3)
        if (dist < success_threshold) and (not self.collision):
            # Bonus scales with how close we got
            distance_bonus = (success_threshold - dist) / success_threshold
            r += 50.0 + 20.0 * distance_bonus  # Base 50 + up to 20 more
            success = True
            done = True
        # Also give partial credit for getting close (but not perfect)
        elif dist < 0.6 and not self.collision:
            # Partial success reward - encourage getting closer
            r += 10.0 * (0.6 - dist) / 0.6  # Up to 10.0 reward for getting within 0.6m

        # Max steps → end episode
        if self.episode_steps >= self.max_steps:
            done = True
            # Penalty for timeout - but less harsh if we got close
            if dist > 0.6:
                r -= 10.0  # Penalty for timing out far from goal
            elif dist > success_threshold:
                r -= 2.0   # Small penalty if we got close but timed out

        # Clip reward to prevent extreme values
        r = max(-50.0, min(100.0, r))

        self.cumulative_reward += r

        info = {
            "distance_to_goal": dist,
            "stage": self.stage,
            "episode_steps": self.episode_steps,
            "success": success,
            "oscillation": oscillation_penalty,
            "near_miss": near_miss_penalty,
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

    parser = argparse.ArgumentParser(
        description="Train PPO policy for TurtleBot3 navigation with curriculum learning"
    )
    parser.add_argument("--timesteps", type=int, default=20000,
                        help="Total timesteps to train")
    parser.add_argument(
        "--logdir",
        type=str,
        default="ppo_runs",
        help="Directory for logs and saved models (relative to repo root, default: ppo_runs)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from (e.g., ppo_runs/tb3_ppo_stage1.zip)"
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default="tb3_ppo",
        help="Base name for saved model (default: tb3_ppo)"
    )
    parser.add_argument(
        "--start-stage",
        type=int,
        default=None,
        help="Curriculum stage to start at (1, 2, or 3). If not specified and resuming, will try to infer from checkpoint name or start at Stage 2"
    )
    args = parser.parse_args()

    rclpy.init()
    
    # Resolve logdir to absolute path (relative to current working directory)
    if not os.path.isabs(args.logdir):
        # Use current working directory (should be repo root when running from there)
        args.logdir = os.path.abspath(args.logdir)
    
    os.makedirs(args.logdir, exist_ok=True)
    
    node = Tb3Env()
    
    # Set metrics path in node
    node.metrics_path = os.path.join(args.logdir, "episode_metrics.csv")
    if not os.path.exists(node.metrics_path):
        with open(node.metrics_path, "w") as f:
            f.write("episode,stage,steps,final_dist,success,return\n")
    
    # Set starting stage if specified or if resuming
    if args.start_stage is not None:
        if args.start_stage < 1 or args.start_stage > 3:
            node.get_logger().error(f"Invalid start-stage: {args.start_stage}. Must be 1, 2, or 3.")
            rclpy.shutdown()
            return
        node.stage = args.start_stage
        node.get_logger().info(f"Starting at curriculum Stage {args.start_stage}")
    elif args.resume:
        # If resuming but no stage specified, default to Stage 2
        # (assuming Stage 1 was completed in previous training)
        node.stage = 2
        node.get_logger().info(
            "Resuming training: Starting at Stage 2 (assuming Stage 1 was completed). "
            "Use --start-stage to override."
        )

    # Spawn TB3 once (world must already be running)
    if not spawn_tb3(node):
        node.get_logger().error("TB3 spawn failed. Exiting.")
        rclpy.shutdown()
        return

    env = GymTb3(node)

    # Load existing model or create new one
    if args.resume:
        # Resolve resume path (can be relative or absolute)
        if not os.path.isabs(args.resume):
            # Try relative to logdir first, then current working directory
            resume_path = os.path.join(args.logdir, args.resume)
            if not os.path.exists(resume_path):
                resume_path = os.path.abspath(args.resume)
        else:
            resume_path = os.path.expanduser(args.resume)
        
        if os.path.exists(resume_path):
            node.get_logger().info(f"Loading checkpoint from: {resume_path}")
            model = PPO.load(resume_path, env=env)
            node.get_logger().info("Resuming training from checkpoint")
        else:
            node.get_logger().warn(f"Checkpoint not found: {resume_path}. Starting fresh training.")
            args.resume = None  # Clear resume flag
    else:
        if args.resume:
            node.get_logger().warn(
                f"Checkpoint not found: {args.resume}. Starting fresh training."
            )
        
        # Improved PPO hyperparameters for stable learning
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=args.logdir,
            n_steps=256,                 # shorter rollout for faster updates (was 512)
            batch_size=64,              # smaller batch for more frequent updates (was 128)
            n_epochs=10,                 # more epochs per update for better learning
            learning_rate=3e-4,
            gamma=0.99,                  # standard discount (was 0.995)
            gae_lambda=0.95,             # standard GAE lambda (was 0.98)
            clip_range=0.2,              # PPO clip range
            ent_coef=0.01,               # entropy coefficient for exploration
            vf_coef=0.5,                 # value function coefficient
            max_grad_norm=0.5,           # gradient clipping
            policy_kwargs=dict(
                net_arch=[64, 64],       # smaller network for faster training (was [128, 128])
                activation_fn=torch.nn.Tanh,  # Tanh activation for bounded outputs
            ),
        )
        node.get_logger().info("Starting fresh training")

    model.learn(total_timesteps=args.timesteps)

    # Save model
    out = os.path.join(args.logdir, f"{args.save_name}.zip")
    model.save(out)
    node.get_logger().info(f"Saved policy to {out}")
    print(f"Saved policy to {out}")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
