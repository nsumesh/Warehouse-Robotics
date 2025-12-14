import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class ColorDockingNode(Node):
	def __init__(self):
		super().__init__('color_docking_node')
		self.bridge = CvBridge()
		self.sub = self.create_subscription(
			Image,
			'/camera/image_raw',
			self.callback,
			10
		)
		self.cmd_pub = self.create_publisher(Twist, '/cmd/vel', 10)
		
		self.forward_speed = 0.12
		self.turn_gain = 0.0025
		self.get_logger().info('Color docking node started')
	
	def callback(self, msg: Image):
		try:
			frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
		except CvBridgeError as e:
			self.get_logger().error(f'cv_bridge error: {e}')
			return
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		lower = np.array([100, 150, 50])
		upper = np.array([140, 255, 255])
		mask = cv2.inRange(hsv, lower, upper)
		moments = cv2.moments(mask)
		
		twist = Twist()
		
		error_x = cx - (frame.shape[1] // 2)
		
		if abs(error_x) > 30:
			twist.angular.z = -0.003 * error_x
			twist.linear.x = 0.05
		
		else:
			twist.angular.z = 0.0
			twist.linear.x = 0.15
		
			self.get_logger().info(
				f"Publishing cmd_vel: lin.x{twist.linear.x:.3f}, ang.z={twist.angular.z:.3f}"
		)
		self.cmd_pub.publish(twist)
		
def main(args=None):
	rclpy.init(args=args)
	node = ColorDockingNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()

if __name__ == '__main__':
	main()
