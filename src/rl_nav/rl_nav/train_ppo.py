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
from gazebo_msgs.srv import SetEntityState, SpawnEntity, DeleteEntity
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Point, Quaternion
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import torch

# Import constants and utilities
from rl_nav.constants import (X_MIN, X_MAX, Y_MIN, Y_MAX, DOCK_A, DOCK_B, DOCK_C, PICKUP,SUCCESS_RADIUS, ACTIONS, MAX_RANGE, NUM_SCAN_BINS)
from rl_nav.gazebo_utils import spawn_tb3, spawn_entity, delete_entity
from rl_nav.item_utils import generate_item_sdf, get_item_color
from rl_nav.navigation_utils import process_scan_to_bins
from rl_nav.observation_utils import build_observation
from rl_nav.reward_utils import (K_PROGRESS, K_TIME, K_COLLISION, K_SUCCESS, K_PICKUP_SUCCESS,K_CLOSE_ZONE, K_WRONG_DOCK, calculate_progress_reward,check_close_zone_bonus, check_dock_success)


class Tb3Env(Node):

    def __init__(self, curriculum_stage=1):
        super().__init__("tb3_env")
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self._on_scan, 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.reset_cli = self.create_client(SetEntityState, "/set_entity_state")

        self.max_range = MAX_RANGE
        self.num_scan_bins = NUM_SCAN_BINS
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
        # Stage-specific max steps
        if curriculum_stage == 1:
            self.max_steps = 200  # Short for local docking
        elif curriculum_stage == 2:
            self.max_steps = 400  # Longer for aisle navigation
        else:  # Stage 3
            self.max_steps = 600  # Increased for two-phase sorting (pickup + dropoff)

        self.in_close_zone = False  # Track if in close zone for one-time bonus

        self.actions = ACTIONS
        self.recent_positions = deque(maxlen=10)
        self.curriculum_stage = curriculum_stage
        self.task_class = None  # For Stage 3: 'A', 'B', or 'C'
        self.metrics_path = None
        
        # For Stage 3: two-phase training (GO_PICKUP → GO_DROPOFF)
        self.phase = None  # "GO_PICKUP" or "GO_DROPOFF" for Stage 3
        self.pickup_goal = None  # Store pickup location
        self.dropoff_goal = None  # Store dock location
        self.pickup_reached = False  # Track if pickup was reached
        self.pickup_reached_time = None  # Track when pickup was reached (for stability)
        
        # Item management for Stage 3
        self.spawn_client = None
        self.delete_client = None
        self.current_item_id = None  # Item ID for current task
        self.item_counter = {'A': 0, 'B': 0, 'C': 0}  # Counter for unique IDs
        if curriculum_stage == 3:
            self.spawn_client = self.create_client(SpawnEntity, "/spawn_entity")
            self.delete_client = self.create_client(DeleteEntity, "/delete_entity")

    def _on_scan(self, msg: LaserScan):
        """Process LiDAR scan into bins."""
        self.scan, self.collision, self.min_obstacle_dist = process_scan_to_bins(
            msg, self.num_scan_bins, self.max_range
        )

    def _on_odom(self, msg: Odometry):
        """Update robot pose [x, y, yaw]."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose[:] = (x, y, yaw)


    def _spawn_item_at_pickup(self, task_class):
        """Spawn a single item at pickup point for the given task class."""
        if not self.spawn_client or not self.spawn_client.service_is_ready():
            return None
        
        # Generate unique item ID
        self.item_counter[task_class] += 1
        item_id = f"item_{task_class}_{self.item_counter[task_class]}"
        
        pickup_x, pickup_y = PICKUP
        
        # Generate SDF for this item
        color_rgba = get_item_color(task_class)
        item_sdf = generate_item_sdf(item_id, color_rgba)
        
        # Spawn at pickup location (single item, centered)
        item_z = 0.15  # Half of box height (0.3m box)
        if spawn_entity(self, self.spawn_client, item_id, item_sdf, pickup_x, pickup_y, item_z):
            self.get_logger().info(f"Spawned {item_id} at pickup ({pickup_x:.2f}, {pickup_y:.2f})")
            return item_id
        return None

    def _cleanup_item(self, item_id):
        """Delete an item from Gazebo."""
        return delete_entity(self, self.delete_client, item_id)

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
            # Stage 3: Two-phase virtual sorting (PICKUP → DOCK)
            # Phase 1: Navigate to PICKUP (with items)
            # Phase 2: Navigate to correct dock after pickup
            sx = random.uniform(X_MIN, X_MAX)
            sy = random.uniform(Y_MIN, Y_MAX)
            yaw = random.uniform(-math.pi, math.pi)
            # Randomly choose task class A, B, or C
            self.task_class = random.choice(['A', 'B', 'C'])
            # Determine dock location based on task class
            if self.task_class == 'A':
                dock_x, dock_y = DOCK_A
            elif self.task_class == 'B':
                dock_x, dock_y = DOCK_B
            else:  # 'C'
                dock_x, dock_y = DOCK_C
            
            # Store both goals for two-phase training
            self.pickup_goal = PICKUP
            self.dropoff_goal = (dock_x, dock_y)
            self.phase = "GO_PICKUP"
            self.pickup_reached = False
            self.pickup_reached_time = None
            
            # Initial goal is PICKUP (phase 1)
            gx, gy = PICKUP
            return (sx, sy, yaw), (gx, gy)

    def _obs(self):
        """Build 30-dim observation: 24 scan bins + [dx, dy, yaw] + [task_class one-hot]."""
        if self.scan is None:
            scan = np.ones(self.num_scan_bins, dtype=np.float32)
        else:
            scan = self.scan
        
        # Use observation utility for consistent encoding
        task_class_for_obs = self.task_class if (self.curriculum_stage == 3 and self.phase == "GO_DROPOFF") else None
        return build_observation(scan, self.pose, self.goal, task_class_for_obs, self.phase)

    def reset(self):
        """Reset environment for new episode."""
        if self.episode_steps > 0 and self.prev_dist is not None:
            final_dist = float(np.linalg.norm(self.goal - self.pose[:2]))
            success = int(final_dist < SUCCESS_RADIUS and not self.collision)

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

        # Clean up items from previous episode (Stage 3 only)
        if self.curriculum_stage == 3 and self.current_item_id:
            self._cleanup_item(self.current_item_id)
            self.current_item_id = None
        
        # Reset counters
        self.episode_idx += 1
        self.episode_steps = 0
        self.cumulative_reward = 0.0
        self.collision = False
        self.min_obstacle_dist = self.max_range
        self.recent_positions.clear()
        
        # Reset phase tracking for Stage 3
        if self.curriculum_stage == 3:
            self.pickup_reached = False
            self.pickup_reached_time = None

        # Sample new start/goal using curriculum stage
        (sx, sy, syaw), (gx, gy) = self.sample_start_and_goal()
        self.goal[:] = (gx, gy)
        
        # For Stage 3: spawn item at pickup
        if self.curriculum_stage == 3 and self.task_class:
            # Wait a bit for services to be ready
            if self.spawn_client:
                self.spawn_client.wait_for_service(timeout_sec=2.0)
            self.current_item_id = self._spawn_item_at_pickup(self.task_class)

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
        self.in_close_zone = False  # Reset close zone flag
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

        # Progress reward
        r = calculate_progress_reward(self.prev_dist, dist, K_PROGRESS)
        self.prev_dist = dist
        r -= K_TIME  # Time penalty every step

        # Intermediate bonus for getting close to goal (one-time when entering)
        bonus, self.in_close_zone = check_close_zone_bonus(dist, self.in_close_zone, K_CLOSE_ZONE)
        r += bonus

        done = False
        success = False

        # Collision penalty
        if self.collision:
            r -= K_COLLISION
            done = True

        # For Stage 3: Handle two-phase training (PICKUP → DOCK)
        if self.curriculum_stage == 3 and self.task_class and not done:
            if self.phase == "GO_PICKUP":
                # Phase 1: Check if reached PICKUP
                dist_to_pickup = math.hypot(self.pose[0] - PICKUP[0], self.pose[1] - PICKUP[1])
                if dist_to_pickup < SUCCESS_RADIUS and not self.collision:
                    # Check stability (been close for at least 1 second)
                    now = time.time()
                    if self.pickup_reached_time is None:
                        self.pickup_reached_time = now
                    elif now - self.pickup_reached_time >= 1.0:  # Stable for 1 second
                        # Reached pickup: transition to phase 2
                        r += K_PICKUP_SUCCESS  # Reward for reaching pickup
                        self.phase = "GO_DROPOFF"
                        self.pickup_reached = True
                        # Switch goal to dock
                        self.goal[:] = self.dropoff_goal
                        # Delete item (virtual pickup)
                        if self.current_item_id:
                            self._cleanup_item(self.current_item_id)
                        # Reset distance tracking for new phase
                        self.prev_dist = float(np.linalg.norm(self.goal - self.pose[:2]))
                        self.in_close_zone = False
                        self.pickup_reached_time = None
                    # Don't set done=True - continue to phase 2
                else:
                    # Reset timer if not close anymore
                    if dist_to_pickup >= SUCCESS_RADIUS:
                        self.pickup_reached_time = None
            
            elif self.phase == "GO_DROPOFF":
                # Phase 2: Check if reached correct dock
                if dist < SUCCESS_RADIUS and not self.collision:
                    dock_success, wrong_dock, _ = check_dock_success(self.pose, self.task_class, SUCCESS_RADIUS)
                    if dock_success:
                        # Correct dock - full success!
                        r += K_SUCCESS
                        success = True
                        done = True
                    elif wrong_dock:
                        # Wrong dock - penalty
                        r -= K_WRONG_DOCK
                        done = True
        
        # For Stage 1/2: standard success check
        elif dist < SUCCESS_RADIUS and not self.collision and not done:
            r += K_SUCCESS
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
    
    # Stage 3: Two-phase virtual sorting (PICKUP → DOCK) with items
    ros2 run rl_nav train_ppo --timesteps 300000 --curriculum-stage 3
    
    # With seed for reproducibility (recommended: 300k timesteps)
    ros2 run rl_nav train_ppo --timesteps 300000 --curriculum-stage 3 --seed 0
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
