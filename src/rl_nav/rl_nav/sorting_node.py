import time
import random

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class SortingNode(Node):
    """
    Very simple 'sorting' logic:

    - Defines 3 item types: light, heavy, fragile
    - Each has a pickup location and a drop-off bin location
    - Publishes sequential /ppo_goal targets for PPOController
    - Uses fixed timeouts instead of true success detection
    """

    def __init__(self):
        super().__init__("sorting_node")

        self.goal_pub = self.create_publisher(PoseStamped, "/ppo_goal", 10)

        # Positions must be tuned based on your lane world.
        # These are just reasonable starting guesses.
        self.pickup_locations = {
            "light": (-2.0, 0.0),
            "heavy": (0.0, 0.0),
            "fragile": (2.0, 0.0),
        }

        self.drop_bins = {
            "light": (-4.0, -4.0),
            "heavy": (0.0, -4.0),
            "fragile": (4.0, -4.0),
        }

        self.task_queue = []
        self._build_initial_tasks(num_tasks=5)

        self.current_task = None
        self.phase = "IDLE"  # "GO_PICKUP", "GO_DROPOFF"
        self.last_phase_time = time.time()

        self.timer = self.create_timer(0.5, self.loop)
        self.get_logger().info("SortingNode started. Tasks queued.")

    def _build_initial_tasks(self, num_tasks=5):
        item_types = list(self.pickup_locations.keys())
        for _ in range(num_tasks):
            item_type = random.choice(item_types)
            self.task_queue.append(item_type)
        self.get_logger().info(f"Initial task queue: {self.task_queue}")

    def _publish_goal(self, x, y):
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        # yaw is irrelevant to PPO obs construction
        self.goal_pub.publish(msg)

    def loop(self):
        now = time.time()

        if self.current_task is None:
            if not self.task_queue:
                self.get_logger().info("All sorting tasks completed.")
                return

            # Start new sorting task
            self.current_task = self.task_queue.pop(0)
            self.phase = "GO_PICKUP"
            self.last_phase_time = now

            px, py = self.pickup_locations[self.current_task]
            self.get_logger().info(
                f"[Task] {self.current_task}: going to pickup @ ({px:.1f}, {py:.1f})"
            )
            self._publish_goal(px, py)
            return

        # Very rough timing-based 'success' logic:
        if self.phase == "GO_PICKUP":
            if now - self.last_phase_time > 20.0:
                dx, dy = self.drop_bins[self.current_task]
                self.phase = "GO_DROPOFF"
                self.last_phase_time = now
                self.get_logger().info(
                    f"[Task] {self.current_task}: going to drop-off @ ({dx:.1f}, {dy:.1f})"
                )
                self._publish_goal(dx, dy)

        elif self.phase == "GO_DROPOFF":
            if now - self.last_phase_time > 20.0:
                self.get_logger().info(
                    f"[Task] Completed sorting for item type: {self.current_task}"
                )
                self.current_task = None
                self.phase = "IDLE"
                self.last_phase_time = now


def main(args=None):
    rclpy.init(args=args)
    node = SortingNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
