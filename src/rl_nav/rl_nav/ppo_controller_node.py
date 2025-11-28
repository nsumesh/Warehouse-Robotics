#!/usr/bin/env python3
"""
ppo_controller_node.py

Runs inference using a trained PPO policy AND spawns the TurtleBot3
if it is not already present in the world.

This ensures:
- launch_warehouse.sh can load ONLY the warehouse
- this controller handles:
    1) TB3 spawn
    2) PPO inference
"""

import os
import numpy as np
import math
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SpawnEntity
from stable_baselines3 import PPO


# ============================================================
#   SPAWN TB3 (same logic as train_ppo.py)
# ============================================================
def spawn_tb3(node):
    """
    Spawns TB3 once. If TB3 already exists, Gazebo will respond with an error,
    but this node will continue safely.
    """

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

    # Initial spawn position — this does NOT matter;
    # PPO controller will continuously overwrite cmd_vel.
    req.initial_pose.position.x = 0.0
    req.initial_pose.position.y = -8.0
    req.initial_pose.position.z = 0.02

    future = cli.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=4.0)

    if future.result() is not None:
        node.get_logger().info("TB3 spawned successfully (ppo_controller).")
        return True
    else:
        node.get_logger().warn("TB3 already exists or failed to spawn.")
        return False


# ============================================================
#   PPO CONTROLLER NODE
# ============================================================
class PPOController(Node):
    def __init__(self):
        super().__init__("ppo_controller")

        # ---------------------
        #  Spawn the robot
        # ---------------------
        self.get_logger().info("Attempting to spawn TB3...")
        spawn_tb3(self)

        # ---------------------
        #  Load PPO model
        # ---------------------
        # Try to find model in repo ppo_runs directory (relative to current working dir)
        model_relative = os.path.abspath("ppo_runs/tb3_ppo.zip")
        model_home = os.path.expanduser("~/MSML_642_FinalProject/ppo_runs/tb3_ppo.zip")
        
        if os.path.exists(model_relative):
            MODEL_PATH = model_relative
        elif os.path.exists(model_home):
            MODEL_PATH = model_home
        else:
            MODEL_PATH = model_relative  # Use repo path as default
        if not os.path.exists(MODEL_PATH):
            self.get_logger().error(
                f"PPO model not found.\nExpected at: {MODEL_PATH}\n"
                f"Train a model first."
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
        self.goal_sub = self.create_subscription(
            PoseStamped, "/ppo_goal", self._goal_cb, 10
        )

        # state
        self.scan = None
        self.pose = np.zeros(3, dtype=np.float32)
        self.goal = np.array([0.0, 2.0], dtype=np.float32)

        # control loop
        self.timer = self.create_timer(0.15, self.control_step)

    # =======================================================
    #   CALLBACKS
    # =======================================================
    def _scan_cb(self, msg):
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
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2*(q.w*q.z + q.x*q.y),
            1 - 2*(q.y*q.y + q.z*q.z)
        )
        self.pose[:] = (x, y, yaw)

    def _goal_cb(self, msg):
        self.goal[0] = msg.pose.position.x
        self.goal[1] = msg.pose.position.y
        self.get_logger().info(
            f"Updated goal → ({self.goal[0]:.2f}, {self.goal[1]:.2f})"
        )

    # =======================================================
    #   OBSERVATION
    # =======================================================
    def obs(self):
        if self.scan is None:
            return None
        dx, dy = (self.goal - self.pose[:2])
        tail = np.array([dx, dy, self.pose[2]], dtype=np.float32)
        return np.concatenate([self.scan, tail], axis=0)

    # =======================================================
    #   CONTROL LOOP
    # =======================================================
    def control_step(self):
        obs = self.obs()
        if obs is None:
            return

        action, _ = self.model.predict(obs, deterministic=True)

        # MATCH EXACT ACTIONS IN TRAIN_PPO
        ACTIONS = [
            (0.15, +0.8),
            (0.15,  0.0),
            (0.15, -0.8),
            (0.00, +0.8),
            (0.00, -0.8),
        ]

        v, w = ACTIONS[int(action)]

        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)


# ============================================================
#   MAIN
# ============================================================
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
