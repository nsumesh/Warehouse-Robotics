import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Twist
import random, math

def random_start():
    x = random.uniform(-10.0, -6.0)
    y = random.uniform(-10.0, -6.0)
    yaw = random.uniform(-math.pi, math.pi)
    return x, y, yaw

class ResetClient(Node):
    def __init__(self):
        super().__init__('reset_client')
        self.cli = self.create_client(SetEntityState, '/set_entity_state')

    async def set_pose(self, x, y, yaw):
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('wait /set_entity_state...')
        s = EntityState()
        s.name = 'tb3'
        s.pose.position.x = x; s.pose.position.y = y; s.pose.position.z = 0.01
        s.pose.orientation.z = math.sin(yaw/2.0)
        s.pose.orientation.w = math.cos(yaw/2.0)
        s.twist = Twist()
        req = SetEntityState.Request(); req.state = s
        await self.cli.call_async(req)
