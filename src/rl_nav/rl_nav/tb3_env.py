import rclpy, math, numpy as np, time
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from .reset_utils import ResetClient, random_start

OBS_SAMPLES = 60
MAX_RANGE = 3.5
GOAL = np.array([-4.0, 6.0], dtype=np.float32)

class TB3Env(Node):
    def __init__(self):
        super().__init__('tb3_env')
        self.scan = None
        self.odom = None
        self.cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.reset_cli = ResetClient()

        self.timer = self.create_timer(0.1, self.step_loop)
        self._reset_episode()

    def _scan_cb(self, msg): self.scan = np.array(msg.ranges, dtype=np.float32)
    def _odom_cb(self, msg): self.odom = msg

    def _obs(self):
        if self.scan is None or self.odom is None: return None
        idx = np.linspace(0, len(self.scan)-1, OBS_SAMPLES).astype(int)
        obs_scan = np.clip(self.scan[idx], 0.0, MAX_RANGE)
        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        vec = GOAL - np.array([x, y], dtype=np.float32)
        dist = float(np.linalg.norm(vec))
        bearing = math.atan2(vec[1], vec[0])
        return np.concatenate([obs_scan / MAX_RANGE, [dist, bearing]]).astype(np.float32)

    def _collision(self):
        return False if self.scan is None else np.nanmin(self.scan) < 0.18

    async def _teleport_random(self):
        x, y, yaw = random_start()
        await self.reset_cli.set_pose(x, y, yaw)

    def _reset_episode(self):
        self.prev_dist = None
        self.episode_start = time.time()
        rclpy.get_global_executor().create_task(self._teleport_random())

    def _policy(self, obs):
        t = Twist()
        min_range = np.min(obs[:-2]) * MAX_RANGE
        dist = obs[-2]; bearing = obs[-1]
        t.linear.x = 0.15 if min_range > 0.4 else 0.0
        t.angular.z = float(np.clip(2.0 * bearing, -1.0, 1.0))
        return t

    def step_loop(self):
        obs = self._obs()
        if obs is None: return
        if self._collision() or (time.time() - self.episode_start) > 30:
            self.get_logger().info("Episode reset")
            self._reset_episode(); return
        self.cmd.publish(self._policy(obs))

def main():
    rclpy.init()
    node = TB3Env()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
