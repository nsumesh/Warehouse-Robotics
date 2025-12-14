import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
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
		
		self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
		self.get_logger().info("Color docking node started.")
	
	def callback(self, msg):
		frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		
		lower = np.array([100, 150, 50])
		upper = np.array([140, 255, 255])
		
		mask = cv2.inRange(hsv, lower, upper)
		moments = cv2.moments(mask)
		
		twist = Twist()
		
		if moments["m00"] > 800000:
			cx = int(moments["m10"] / moments["m00"])
			
			error = cx - (frame.shape[1] // 2)
			twist.angular.z = -float(error) / 300
			
			twist.linear.x = 0.1
			
			self.get_logger().info(f"Blue marker detected.")
		else:
			twist.linear.x = 0.0
			twist.angular.z = 0.0
		
		self.cmd_pub.publish(twist)
	
def main(args=None):
	rclpy.init(args=args)
	node = ColorDockingNode()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == "__main__":
	main()
