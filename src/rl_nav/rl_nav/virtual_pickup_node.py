#!/usr/bin/env python3
"""
Virtual Pickup and Sorting Node

Detects items using camera/OpenCV, picks up when within 0.5m, drops at sorting bins.
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose, Point, Quaternion

try:
    import cv2
    from cv_bridge import CvBridge
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False


class VirtualPickupNode(Node):
    """Virtual pickup: detect items, pick up when close, drop at bins."""
    
    def __init__(self):
        super().__init__("virtual_pickup_node")
        self.odom_sub = self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.camera_sub = self.create_subscription(Image, "/camera/image_raw", self._camera_cb, 10)
        
        self.delete_client = self.create_client(DeleteEntity, "/delete_entity")
        self.spawn_client = self.create_client(SpawnEntity, "/spawn_entity")
        
        self.get_logger().info("Waiting for Gazebo services...")
        self.delete_client.wait_for_service(timeout_sec=5.0)
        self.spawn_client.wait_for_service(timeout_sec=5.0)

        self.pose = np.zeros(3, dtype=np.float32)
        self.camera_image = None
        self.bridge = CvBridge() if CV_AVAILABLE else None
        if not CV_AVAILABLE:
            self.get_logger().warn("OpenCV not available, using distance-based detection")
        
        self.picked_item = None
        self.picked_item_type = None
        self.items_in_world = {
            "item_light_1": {"pose": (-2.0, 0.0, 0.2), "type": "light"},
            "item_heavy_1": {"pose": (0.0, 0.0, 0.2), "type": "heavy"},
            "item_fragile_1": {"pose": (2.0, 0.0, 0.2), "type": "fragile"},
        }

        self.pickup_distance_threshold = 0.5
        self.sorting_bins = {
            "light": (-4.0, -4.0, 0.1),
            "heavy": (0.0, -4.0, 0.1),
            "fragile": (4.0, -4.0, 0.1),
        }
        self.current_goal = None
        self.goal_reached_threshold = 0.4

        self.timer = self.create_timer(0.1, self.update_loop)
        self.get_logger().info("VirtualPickupNode initialized")
    
    def _odom_cb(self, msg):
        """Update robot pose."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose[:] = (x, y, yaw)
    
    def _camera_cb(self, msg):
        """Process camera image."""
        if self.bridge:
            try:
                self.camera_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception:
                pass
    
    def _detect_items_camera(self):
        """Detect items using ArUco markers or color blobs."""
        if not self.camera_image or not CV_AVAILABLE or not self.bridge:
            return []
        
        detected_items = []
        
        # ArUco marker detection
        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            corners, ids, _ = cv2.aruco.detectMarkers(self.camera_image, aruco_dict)
            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    marker_corners = corners[i][0]
                    marker_size = np.linalg.norm(marker_corners[0] - marker_corners[1])
                    item_type = ["light", "heavy", "fragile"][marker_id % 3]
                    detected_items.append({"id": f"item_{item_type}_{marker_id}", "type": item_type,
                                          "distance": marker_size})
        except Exception:
            pass
        
        # Color blob detection (fallback)
        if not detected_items:
            try:
                hsv = cv2.cvtColor(self.camera_image, cv2.COLOR_BGR2HSV)
                color_ranges = {
                    "light": ([20, 100, 100], [30, 255, 255]),
                    "heavy": ([0, 100, 100], [10, 255, 255]),
                    "fragile": ([100, 100, 100], [130, 255, 255]),
                }
                for item_type, (lower, upper) in color_ranges.items():
                    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if cv2.contourArea(contour) > 100:
                            detected_items.append({"id": f"item_{item_type}_{len(detected_items)}",
                                                   "type": item_type, "distance": 1000.0 / cv2.contourArea(contour)})
            except Exception:
                pass
        
        return detected_items
    
    def _distance_to_item(self, item_pose):
        """Calculate distance to item."""
        dx = item_pose[0] - self.pose[0]
        dy = item_pose[1] - self.pose[1]
        return math.hypot(dx, dy)
    
    def _pickup_item(self, item_id, item_type):
        """Delete item from world (virtual pickup)."""
        if self.picked_item is not None:
            return False
        
        req = DeleteEntity.Request()
        req.name = item_id
        future = self.delete_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        
        if future.result() and future.result().success:
            self.picked_item = item_id
            self.picked_item_type = item_type
            if item_id in self.items_in_world:
                del self.items_in_world[item_id]
            self.get_logger().info(f"Picked up {item_type} item: {item_id}")
            return True
            return False
    
    def _drop_item(self):
        """Spawn item at sorting bin."""
        if self.picked_item is None:
            return False
        
        item_type = self.picked_item_type
        if item_type not in self.sorting_bins:
            return False
        
        bin_x, bin_y, bin_z = self.sorting_bins[item_type]
        item_sdf = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{self.picked_item}_sorted">
    <static>false</static>
    <pose>{bin_x} {bin_y} {bin_z} 0 0 0</pose>
    <link name="link">
      <collision name="collision">
        <geometry><box><size>0.2 0.2 0.2</size></box></geometry>
      </collision>
      <visual name="visual">
        <geometry><box><size>0.2 0.2 0.2</size></box></geometry>
      </visual>
    </link>
  </model>
</sdf>"""
        
        req = SpawnEntity.Request()
        req.name = f"{self.picked_item}_sorted"
        req.xml = item_sdf
        req.robot_namespace = ""
        req.reference_frame = "world"
        req.initial_pose = Pose()
        req.initial_pose.position = Point(x=float(bin_x), y=float(bin_y), z=float(bin_z))
        req.initial_pose.orientation = Quaternion(w=1.0)
        
        future = self.spawn_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        
        if future.result():
            self.get_logger().info(f"Dropped {item_type} item at bin ({bin_x:.1f}, {bin_y:.1f})")
            self.picked_item = None
            self.picked_item_type = None
            return True
            return False
    
    def _goal_reached(self):
        """Check if goal reached."""
        if self.current_goal is None:
            return False
        dx = self.current_goal[0] - self.pose[0]
        dy = self.current_goal[1] - self.pose[1]
        return math.hypot(dx, dy) < self.goal_reached_threshold
    
    def update_loop(self):
        """Main loop: check pickup and drop-off."""
        if self.picked_item is None:
            detected_items = self._detect_items_camera()
            if not detected_items:
                for item_id, item_data in list(self.items_in_world.items()):
                    if self._distance_to_item(item_data["pose"]) < self.pickup_distance_threshold:
                        self._pickup_item(item_id, item_data["type"])
                        break
            else:
                for item in detected_items:
                    if item["distance"] < self.pickup_distance_threshold:
                        for item_id, item_data in list(self.items_in_world.items()):
                            if item_data["type"] == item["type"]:
                                if self._distance_to_item(item_data["pose"]) < self.pickup_distance_threshold:
                                    self._pickup_item(item_id, item_data["type"])
                                    break
        
        if self.picked_item is not None and self._goal_reached():
            self._drop_item()
    
    def set_goal(self, x, y):
        """Set navigation goal."""
        self.current_goal = (x, y)
    
    def get_picked_item(self):
        """Get currently picked item."""
        return self.picked_item, self.picked_item_type


def main(args=None):
    rclpy.init(args=args)
    node = VirtualPickupNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
