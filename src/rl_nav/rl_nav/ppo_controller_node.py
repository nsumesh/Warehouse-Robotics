import os
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from stable_baselines3 import PPO


class PPOController(Node):
    def __init__(self):
        super().__init__("ppo_controller")

        # Load trained model
        MODEL_PATH = os.path.expanduser(
            "~/MSML_642_FinalProject/ppo_runs/tb3_ppo.zip"
        )
        if not os.path.exists(MODEL_PATH):
            self.get_logger().warn(
                f"Model path {MODEL_PATH} does not exist. "
                "Update PPOController MODEL_PATH or train a model first."
            )
        self.model = PPO.load(MODEL_PATH)
        self.get_logger().info(f"Loaded PPO model: {MODEL_PATH}")

        # ROS I/O
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

        # State
        self.scan = None
        self.pose = np.zeros(3, dtype=np.float32)
        self.goal = np.array([3.0, 0.0], dtype=np.float32)

        self.timer = self.create_timer(0.15, self.control_step)

    # --------------------------------------------------------
    def _scan_cb(self, msg: LaserScan):
        rng = np.array(msg.ranges, dtype=np.float32)
        inds = np.linspace(0, len(rng) - 1, 24).astype(int)
        r = np.clip(rng[inds], 0.0, 3.5)
        self.scan = r / 3.5

    # --------------------------------------------------------
    def _odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = np.arctan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y * q.y + q.z * q.z),
        )
        self.pose[:] = (x, y, yaw)

    # --------------------------------------------------------
    def _goal_cb(self, msg: PoseStamped):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.goal = np.array([x, y], dtype=np.float32)
        self.get_logger().info(f"Updated PPO goal to ({x:.2f}, {y:.2f})")

    # --------------------------------------------------------
    def obs(self):
        if self.scan is None:
            return None
        dx, dy = (self.goal - self.pose[:2])
        tail = np.array([dx, dy, self.pose[2]], dtype=np.float32)
        return np.concatenate([self.scan, tail], axis=0)

    # --------------------------------------------------------
    def control_step(self):
        obs = self.obs()
        if obs is None:
            return

        action, _ = self.model.predict(obs, deterministic=True)

        # Same action mapping as train_ppo.py
        ACTIONS = [
            (0.12, 0.8),
            (0.12, 0.0),
            (0.12, -0.8),
            (0.0, 0.8),
            (0.0, -0.8),
        ]
        v, w = ACTIONS[int(action)]

        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z *= 0.7 
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
