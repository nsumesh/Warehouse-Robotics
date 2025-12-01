#!/usr/bin/env python3
"""
Sorting Node with PPO Navigation

Manages sorting tasks (3 bins: light, heavy, fragile) using trained PPO policy.
FSM: IDLE → GO_PICKUP → GO_DROPOFF → IDLE
"""
import os
import time
import math
import random
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from stable_baselines3 import PPO


class SortingNode(Node):
    """Sorting node using trained PPO for navigation."""

    def __init__(self):
        super().__init__("sorting_node")

        # Load PPO model
        model_relative = os.path.abspath("ppo_runs/tb3_ppo.zip")
        model_home = os.path.expanduser("~/MSML_642_FinalProject/ppo_runs/tb3_ppo.zip")
        model_path = model_relative if os.path.exists(model_relative) else model_home
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"PPO model not found: {model_path}\nTrain first: ros2 run rl_nav train_ppo")
            raise RuntimeError("PPO model missing")
        
        self.model = PPO.load(model_path)
        self.get_logger().info(f"Loaded PPO model: {model_path}")

        # ROS I/O
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self._scan_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self._odom_cb, 10)

        # Configuration
        self.pickup_locations = {"light": (-2.0, 0.0), "heavy": (0.0, 0.0), "fragile": (2.0, 0.0)}
        self.drop_bins = {"light": (-4.0, -4.0), "heavy": (0.0, -4.0), "fragile": (4.0, -4.0)}
        self.actions = [(0.12, 0.6), (0.15, 0.0), (0.12, -0.6), (0.00, 0.6), (0.00, -0.6)]

        # State
        self.scan = None
        self.pose = np.zeros(3, dtype=np.float32)
        self.current_goal = None
        self.goal_reached_threshold = 0.4
        self.task_queue = []
        self.current_task = None
        self.phase = "IDLE"
        self.task_start_time = None
        self.max_task_time = 60.0

        # Initialize
        self._build_initial_tasks(5)
        self.control_timer = self.create_timer(0.15, self.control_step)
        self.task_timer = self.create_timer(0.5, self.task_manager)
        self.get_logger().info(f"SortingNode initialized. Task queue: {self.task_queue}")

    def _scan_cb(self, msg):
        """Process LiDAR scan into 24 bins."""
        rng = np.array(msg.ranges, dtype=np.float32)
        n, bins = len(rng), 24
        step = max(1, n // bins)
        scan_bins = []
        for i in range(0, n, step):
            v = rng[i]
            if math.isnan(v) or v <= 0:
                v = 3.5
            scan_bins.append(min(v, 3.5) / 3.5)
            if len(scan_bins) == bins:
                break
        while len(scan_bins) < bins:
            scan_bins.append(1.0)
        self.scan = np.array(scan_bins, dtype=np.float32)

    def _odom_cb(self, msg):
        """Update robot pose."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose[:] = (x, y, yaw)

    def _obs(self):
        """Build observation for PPO."""
        if self.scan is None or self.current_goal is None:
            return None
        dx, dy = (np.array(self.current_goal) - self.pose[:2])
        tail = np.array([dx, dy, self.pose[2]], dtype=np.float32)
        return np.concatenate([self.scan, tail], axis=0)

    def _build_initial_tasks(self, num_tasks=5):
        """Build initial task queue."""
        item_types = list(self.pickup_locations.keys())
        self.task_queue = [random.choice(item_types) for _ in range(num_tasks)]

    def _distance_to_goal(self):
        """Calculate distance to goal."""
        if self.current_goal is None:
            return float('inf')
        dx, dy = (np.array(self.current_goal) - self.pose[:2])
        return float(math.hypot(dx, dy))

    def _goal_reached(self):
        """Check if goal reached."""
        return self._distance_to_goal() < self.goal_reached_threshold

    def task_manager(self):
        """Manage task queue and FSM transitions."""
        now = time.time()

        # Timeout check
        if self.task_start_time and (now - self.task_start_time) > self.max_task_time:
            self.get_logger().warn("Task timeout, advancing phase")
                self._advance_phase()

        # FSM
        if self.phase == "IDLE" and self.task_queue:
                self.current_task = self.task_queue.pop(0)
                self.phase = "GO_PICKUP"
                self.task_start_time = now
                px, py = self.pickup_locations[self.current_task]
                self.current_goal = (px, py)
            self.get_logger().info(f"[Task] {self.current_task.upper()}: Going to pickup @ ({px:.1f}, {py:.1f})")

        elif self.phase == "GO_PICKUP" and self._goal_reached():
                self.phase = "GO_DROPOFF"
                self.task_start_time = now
                dx, dy = self.drop_bins[self.current_task]
                self.current_goal = (dx, dy)
            self.get_logger().info(f"[Task] {self.current_task.upper()}: Reached pickup. Going to bin @ ({dx:.1f}, {dy:.1f})")

        elif self.phase == "GO_DROPOFF" and self._goal_reached():
            self.get_logger().info(f"[Task] Completed sorting {self.current_task.upper()} item")
                self.current_task = None
                self.phase = "IDLE"
                self.current_goal = None
                self.task_start_time = None
                if not self.task_queue:
                self.get_logger().info("All tasks completed!")

    def _advance_phase(self):
        """Advance to next phase on timeout."""
        if self.phase == "GO_PICKUP":
            self.phase = "GO_DROPOFF"
            dx, dy = self.drop_bins[self.current_task]
            self.current_goal = (dx, dy)
            self.task_start_time = time.time()
        elif self.phase == "GO_DROPOFF":
            self.current_task = None
            self.phase = "IDLE"
            self.current_goal = None
            self.task_start_time = None

    def control_step(self):
        """PPO control loop."""
        if self.current_goal is None:
            msg = Twist()
            self.cmd_pub.publish(msg)
            return

        obs = self._obs()
        if obs is None:
            return

        action, _ = self.model.predict(obs, deterministic=True)
        v, w = self.actions[int(action)]
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)


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
