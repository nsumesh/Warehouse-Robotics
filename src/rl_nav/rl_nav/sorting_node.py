#!/usr/bin/env python3
"""
Integrated Sorting Node with PPO Navigation

This node:
1. Manages sorting tasks (3 bins: light, heavy, fragile)
2. Uses trained PPO policy to navigate to pickup locations and drop-off bins
3. Detects goal arrival based on distance (not timeouts)
4. Implements FSM: IDLE → GO_PICKUP → GO_DROPOFF → IDLE
"""

import os
import time
import math
import random
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from stable_baselines3 import PPO


class SortingNode(Node):
    """
    Integrated sorting node that uses trained PPO policy for navigation.
    
    Workflow:
    1. Select task from queue (light/heavy/fragile item)
    2. Navigate to pickup location using PPO
    3. Navigate to corresponding drop-off bin using PPO
    4. Repeat for next task
    """

    def __init__(self):
        super().__init__("sorting_node")

        # ---------------------
        #  Load PPO model
        # ---------------------
        model_relative = os.path.abspath("ppo_runs/tb3_ppo.zip")
        model_home = os.path.expanduser("~/MSML_642_FinalProject/ppo_runs/tb3_ppo.zip")
        
        if os.path.exists(model_relative):
            MODEL_PATH = model_relative
        elif os.path.exists(model_home):
            MODEL_PATH = model_home
        else:
            MODEL_PATH = model_relative
        
        if not os.path.exists(MODEL_PATH):
            self.get_logger().error(
                f"PPO model not found at: {MODEL_PATH}\n"
                "Train a model first: ros2 run rl_nav train_ppo"
            )
            raise RuntimeError("PPO model missing.")
        
        self.model = PPO.load(MODEL_PATH)
        self.get_logger().info(f"Loaded PPO model: {MODEL_PATH}")

        # ---------------------
        #  ROS I/O
        # ---------------------
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self._scan_cb, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self._odom_cb, 10
        )

        # ---------------------
        #  Sorting Configuration
        # ---------------------
        # 3 bins for sorting: light, heavy, fragile
        # Pickup locations (where items are located)
        self.pickup_locations = {
            "light": (-2.0, 0.0),
            "heavy": (0.0, 0.0),
            "fragile": (2.0, 0.0),
        }

        # Drop-off bins (where items should be sorted)
        self.drop_bins = {
            "light": (-4.0, -4.0),    # Bin 1: Light items
            "heavy": (0.0, -4.0),     # Bin 2: Heavy items
            "fragile": (4.0, -4.0),   # Bin 3: Fragile items
        }

        # ---------------------
        #  State
        # ---------------------
        self.scan = None
        self.pose = np.zeros(3, dtype=np.float32)  # x, y, yaw
        self.current_goal = None  # (x, y) tuple
        self.goal_reached_threshold = 0.4  # meters (same as training)

        # Task management
        self.task_queue = []
        self.current_task = None  # "light", "heavy", or "fragile"
        self.phase = "IDLE"  # "GO_PICKUP", "GO_DROPOFF", "IDLE"
        self.task_start_time = None
        self.max_task_time = 60.0  # seconds (timeout for safety)

        # PPO actions (must match train_ppo.py)
        self.actions = [
            (0.12,  0.6),   # forward + left
            (0.15,  0.0),   # forward
            (0.12, -0.6),   # forward + right
            (0.00,  0.6),   # rotate left
            (0.00, -0.6),   # rotate right
        ]

        # ---------------------
        #  Initialize
        # ---------------------
        self._build_initial_tasks(num_tasks=5)
        self.control_timer = self.create_timer(0.15, self.control_step)
        self.task_timer = self.create_timer(0.5, self.task_manager)

        self.get_logger().info("SortingNode initialized with PPO navigation")
        self.get_logger().info(f"Task queue: {self.task_queue}")

    # =======================================================
    #   CALLBACKS
    # =======================================================
    def _scan_cb(self, msg):
        """Process LiDAR scan (same as PPO controller)"""
        rng = np.array(msg.ranges, dtype=np.float32)
        N = len(rng)
        BINS = 24
        step = max(1, N // BINS)

        bins = []
        for i in range(0, N, step):
            v = rng[i]
            if math.isnan(v) or v <= 0:
                v = 3.5
            bins.append(min(v, 3.5) / 3.5)
            if len(bins) == BINS:
                break

        while len(bins) < BINS:
            bins.append(1.0)

        self.scan = np.array(bins, dtype=np.float32)

    def _odom_cb(self, msg):
        """Update robot pose"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        self.pose[:] = (x, y, yaw)

    # =======================================================
    #   OBSERVATION (for PPO)
    # =======================================================
    def _obs(self):
        """Build observation vector for PPO (same format as training)"""
        if self.scan is None or self.current_goal is None:
            return None
        
        dx, dy = (np.array(self.current_goal) - self.pose[:2])
        tail = np.array([dx, dy, self.pose[2]], dtype=np.float32)
        return np.concatenate([self.scan, tail], axis=0)

    # =======================================================
    #   TASK MANAGEMENT
    # =======================================================
    def _build_initial_tasks(self, num_tasks=5):
        """Build initial task queue"""
        item_types = list(self.pickup_locations.keys())
        for _ in range(num_tasks):
            item_type = random.choice(item_types)
            self.task_queue.append(item_type)

    def _distance_to_goal(self):
        """Calculate distance to current goal"""
        if self.current_goal is None:
            return float('inf')
        dx, dy = (np.array(self.current_goal) - self.pose[:2])
        return float(math.hypot(dx, dy))

    def _goal_reached(self):
        """Check if current goal has been reached"""
        dist = self._distance_to_goal()
        return dist < self.goal_reached_threshold

    def task_manager(self):
        """Manage task queue and FSM transitions"""
        now = time.time()

        # Check for timeout
        if self.task_start_time is not None:
            elapsed = now - self.task_start_time
            if elapsed > self.max_task_time:
                self.get_logger().warn(
                    f"Task timeout after {elapsed:.1f}s. Moving to next phase/task."
                )
                self._advance_phase()

        # FSM: Handle phase transitions
        if self.phase == "IDLE":
            if self.task_queue:
                # Start new task
                self.current_task = self.task_queue.pop(0)
                self.phase = "GO_PICKUP"
                self.task_start_time = now
                
                px, py = self.pickup_locations[self.current_task]
                self.current_goal = (px, py)
                
                self.get_logger().info(
                    f"[Task] {self.current_task.upper()}: Navigating to pickup @ ({px:.1f}, {py:.1f})"
                )

        elif self.phase == "GO_PICKUP":
            if self._goal_reached():
                # Reached pickup location, go to drop-off bin
                self.phase = "GO_DROPOFF"
                self.task_start_time = now
                
                dx, dy = self.drop_bins[self.current_task]
                self.current_goal = (dx, dy)
                
                self.get_logger().info(
                    f"[Task] {self.current_task.upper()}: Reached pickup. "
                    f"Navigating to {self.current_task} bin @ ({dx:.1f}, {dy:.1f})"
                )

        elif self.phase == "GO_DROPOFF":
            if self._goal_reached():
                # Completed task
                self.get_logger().info(
                    f"[Task] ✓ Completed sorting {self.current_task.upper()} item "
                    f"to {self.current_task} bin"
                )
                
                # Reset for next task
                self.current_task = None
                self.phase = "IDLE"
                self.current_goal = None
                self.task_start_time = None
                
                if not self.task_queue:
                    self.get_logger().info("All sorting tasks completed!")

    def _advance_phase(self):
        """Force advance to next phase (for timeout handling)"""
        if self.phase == "GO_PICKUP":
            # Skip to drop-off
            self.phase = "GO_DROPOFF"
            dx, dy = self.drop_bins[self.current_task]
            self.current_goal = (dx, dy)
            self.task_start_time = time.time()
        elif self.phase == "GO_DROPOFF":
            # Skip to next task
            self.current_task = None
            self.phase = "IDLE"
            self.current_goal = None
            self.task_start_time = None

    # =======================================================
    #   PPO CONTROL LOOP
    # =======================================================
    def control_step(self):
        """PPO control loop - runs at 0.15s intervals"""
        if self.current_goal is None:
            # No active goal, stop robot
            msg = Twist()
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.cmd_pub.publish(msg)
            return

        # Get observation
        obs = self._obs()
        if obs is None:
            return

        # Get action from PPO policy
        action, _ = self.model.predict(obs, deterministic=True)

        # Apply action
        v, w = self.actions[int(action)]
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

        # Log distance periodically
        dist = self._distance_to_goal()
        if int(time.time() * 2) % 10 == 0:  # Log every ~5 seconds
            self.get_logger().debug(
                f"Phase: {self.phase}, Distance to goal: {dist:.2f}m"
            )


def main(args=None):
    rclpy.init(args=args)
    node = SortingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
