#!/usr/bin/env python3
"""
PPO Controller Node

Runs inference using trained PPO policy and spawns TurtleBot3.
"""
import os
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SpawnEntity
from stable_baselines3 import PPO


def spawn_tb3(node):
    """Spawn TurtleBot3 in Gazebo."""
    robot_path = os.path.expanduser(
        "~/turtlebot3_ws/install/turtlebot3_gazebo/share/"
        "turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf"
    )
    if not os.path.exists(robot_path):
        node.get_logger().error(f"TB3 SDF not found: {robot_path}")
        return False

    sdf_xml = open(robot_path, "r").read()
    cli = node.create_client(SpawnEntity, "/spawn_entity")
    cli.wait_for_service(timeout_sec=5.0)

    req = SpawnEntity.Request()
    req.name = "tb3"
    req.xml = sdf_xml
    req.robot_namespace = ""
    req.reference_frame = "world"
    req.initial_pose.position.x = 0.0
    req.initial_pose.position.y = -8.0
    req.initial_pose.position.z = 0.02

    future = cli.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=4.0)
    if future.result():
        node.get_logger().info("TB3 spawned successfully")
        return True
    return False


class PPOController(Node):
    """PPO controller for navigation."""

    def __init__(self):
        super().__init__("ppo_controller")
        self.get_logger().info("Spawning TB3...")
        spawn_tb3(self)

        # Load PPO model
        model_relative = os.path.abspath("ppo_runs/tb3_ppo.zip")
        model_home = os.path.expanduser("~/MSML_642_FinalProject/ppo_runs/tb3_ppo.zip")
        model_path = model_relative if os.path.exists(model_relative) else model_home

        if not os.path.exists(model_path):
            self.get_logger().error(f"PPO model not found: {model_path}\nTrain first.")
            raise RuntimeError("PPO model missing")

        self.model = PPO.load(model_path)
        self.get_logger().info(f"Loaded PPO model: {model_path}")

        # ROS I/O
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self._scan_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.goal_sub = self.create_subscription(PoseStamped, "/ppo_goal", self._goal_cb, 10)

        self.scan = None
        self.pose = np.zeros(3, dtype=np.float32)
        self.goal = np.array([0.0, 2.0], dtype=np.float32)
        self.actions = [(0.12, 0.6), (0.15, 0.0), (0.12, -0.6), (0.00, 0.6), (0.00, -0.6)]

        self.timer = self.create_timer(0.15, self.control_step)

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

    def _goal_cb(self, msg):
        """Update goal position."""
        self.goal[0] = msg.pose.position.x
        self.goal[1] = msg.pose.position.y
        self.get_logger().info(f"Updated goal → ({self.goal[0]:.2f}, {self.goal[1]:.2f})")

    def obs(self):
        """Build observation for PPO."""
        if self.scan is None:
            return None
        dx, dy = (self.goal - self.pose[:2])
        tail = np.array([dx, dy, self.pose[2]], dtype=np.float32)
        return np.concatenate([self.scan, tail], axis=0)

    def control_step(self):
        """PPO control loop."""
        obs = self.obs()
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
    node = PPOController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
