#!/usr/bin/env python3
from typing import List

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from stable_baselines3 import PPO

LIDAR_SAMPLES = 24
OBSERVATION_DIM = 27
DEFAULT_MODEL_PATH = "ppo_warehouse_policy.zip"


class PPOController(Node):
    def __init__(self):
        super().__init__('ppo_controller')
        self.declare_parameter('model_path', DEFAULT_MODEL_PATH)
        model_path = self.get_parameter('model_path').get_parameter_value().string_value

        self.get_logger().info(f"Loading PPO policy from: {model_path}")
        self.model = PPO.load(model_path)

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self._on_scan, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.actions: List[tuple[float, float]] = [
            (0.12, 0.8),
            (0.12, 0.0),
            (0.12, -0.8),
            (0.00, 0.8),
            (0.00, -0.8),
        ]

        self.max_range = 3.5

    def _on_scan(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        if ranges.size == 0:
            self.get_logger().warn('Received empty laser scan; skipping action.')
            return

        indices = np.linspace(0, ranges.size - 1, LIDAR_SAMPLES).astype(int)
        sampled = np.clip(ranges[indices], 0.0, msg.range_max if msg.range_max > 0 else self.max_range)
        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        obs[:LIDAR_SAMPLES] = sampled / self.max_range

        action_idx, _ = self.model.predict(obs, deterministic=True)
        self._publish_action(int(action_idx))

    def _publish_action(self, action_idx: int):
        if action_idx < 0 or action_idx >= len(self.actions):
            self.get_logger().warn(f"Action index {action_idx} out of bounds; ignoring.")
            return

        v, w = self.actions[action_idx]
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_pub.publish(twist)


def main():
    rclpy.init()
    node = PPOController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
