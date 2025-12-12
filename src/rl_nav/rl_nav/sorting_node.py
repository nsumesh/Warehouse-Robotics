
import os
import sys
import time
import math
import random
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from stable_baselines3 import PPO

from rl_nav.constants import dockA, dockB, dockC, pickup, success_region, robot_actions,docking_time
from rl_nav.gazebo_functions import robot_initilization, entity_spawned, delete_entity, reset_robot_position, docking_blue_box, delete_blue_box
from rl_nav.box_functions import generate_item, get_item_color
from rl_nav.navigation_functions import euclidean_distance, goal_reached, check_collision, process_scan_to_bins
from rl_nav.docking_functions import process_camera_image, docking_complete, docking_control
from rl_nav.observation_functions import observation
from rl_nav.fsm import FSM


class SortingNode(Node):
    def __init__(self):
        super().__init__("sorting_node")
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        model_path = os.path.join(project_root, "ppo_runs", "ppo_stage3_sorting.zip")
        if not os.path.exists(model_path):
            model_path = os.path.join(os.getcwd(), "ppo_runs", "ppo_stage3_sorting.zip")
        if not os.path.exists(model_path):
            self.get_logger().error("Model file not found")
            raise FileNotFoundError("PPO model not found")
        self.fsm = FSM(self)        
        self.task_timer = self.create_timer(0.5, self.fsm.move)
        self.model = PPO.load(model_path)
        self.get_logger().info(f"Loaded Stage 3 PPO model: {model_path}")
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_for_bins, 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.update_pose, 10)
        self.camera_sub = self.create_subscription(Image, '/camera/image_raw', self.process_camera, 10)
        self.bridge = CvBridge()
        self.delete_client = self.create_client(DeleteEntity, "/delete_entity")
        self.spawn_client = self.create_client(SpawnEntity, "/spawn_entity")
        self.pickup_location = pickup
        self.drop_docks = {'A': dockA,'B': dockB,'C': dockC}
        self.actions = robot_actions
        self.scan = None
        self.pose = np.zeros(3, dtype=np.float32)
        self.current_goal = None
        self.task = None 
        self.goal_region = success_region 
        self.goal_reached_time = None  
        self.task_queue = []
        self.current_task = None
        self.phase = "idle" 
        self.task_start_time = None
        self.max_task_time = 480.0 
        self.last_log_time = None  
        self.last_stuck_dist = None  
        self.collision_check_enabled = False  
        self.start_time = None 
        self.last_collision_time = None  
        self.items_at_pickup = {}  
        self.active_items = {}  
        self.item_counter = {'A': 0, 'B': 0, 'C': 0} 
        self.current_item_id = None 
        self.items_spawned_for_current_task = False  
        self.item_dropped_for_current_task = False  
        self.dropped_items = {}  
        self.blue_marker_detected = False
        self.blue_marker_area = 0
        self.blue_marker_centered = False
        self.blue_marker_error_x = 0  
        self.docking_complete = False
        self.docking_stable_time = None
        self.docking_stable_duration = 3.0
        self.max_docking_time = docking_time
        self.docking_start_time = None
        self.tasks(5)
        self.control_timer = self.create_timer(0.15, self.step)
        self.get_logger().info("Sorting is up and running. Task queue:" +str(self.task_queue))

    def scan_for_bins(self, msg):
        self.scan, _, _ = process_scan_to_bins(msg, 24, 3.5)

    def update_pose(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (orientation.w * orientation.z + orientation.x * orientation.y), 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z))
        self.pose[:] = (position.x, position.y, yaw)

    def process_camera(self, msg):
        if self.phase != "docking":
            if self.blue_marker_detected or self.docking_complete:
                self.blue_marker_detected = False
                self.blue_marker_area = 0
                self.blue_marker_centered = False
                self.blue_marker_error_x = 0
                self.docking_complete = False
                self.docking_stable_time = None
            return
        
        result = process_camera_image(self.bridge, msg, self.phase)
        if result is None or 'error' in result:
            if result and 'error' in result:
                self.get_logger().warn("Camera error")
            self.blue_marker_detected = False
            self.blue_marker_area = 0
            self.blue_marker_centered = False
            self.blue_marker_error_x = 0
            self.docking_stable_time = None
            return
    
        self.blue_marker_detected = result['detected']
        self.blue_marker_area = result['area']
        self.blue_marker_error_x = result['error_x']
        self.blue_marker_centered = result['centered']
        
        if self.blue_marker_detected:
            complete, new_time = docking_complete(self.blue_marker_area, self.blue_marker_centered,self.docking_stable_time, self.docking_stable_duration)
            self.docking_complete = complete
            self.docking_stable_time = new_time
            if complete:
                self.get_logger().info("Docking complete")
        else:
            self.docking_stable_time = None
            self.docking_complete = False

    def build_observation(self):
        if self.phase == "dropoff":
            task = self.task
        else: 
            task = None
        return observation(self.scan, self.pose, self.current_goal, task, self.phase)

    def tasks(self, num_tasks):
        categories = ['A', 'B', 'C']
        self.task_queue = ['A','B','C']
        #self.task_queue = random.choice(categories, size=num_tasks, replace=False)
        for task in categories:
            self.items_at_pickup[task] = []
        for task in self.task_queue:
            self.item_counter[task] = self.item_counter.get(task, 0) + 1
            item_id = "item_" + task +"_"+str(self.item_counter[task])
            self.items_at_pickup[task].append(item_id)
            self.active_items[item_id] = {'task': task, 'spawned':False, 'picked':False}

    def goal_distance(self):
        return euclidean_distance(self.pose, self.current_goal)
    
    def goal_reached_check(self):
        reached, new_time = goal_reached(self.pose, self.current_goal, self.goal_region, self.goal_reached_time, 3.0)
        self.goal_reached_time = new_time
        return reached

    def check_collision(self):
        return check_collision(self.scan, 3.5, 0.10) 

    def spawn_items_for_task(self):
        if not self.spawn_client.service_is_ready():
            self.get_logger().warn("Spawn service not ready")
            return False
        if self.current_task is None:
            return False
        task = self.current_task
        if task not in self.items_at_pickup or len(self.items_at_pickup[task]) == 0:
            self.get_logger().warn("No items available")
            return False
        item_id = self.items_at_pickup[task][0]  
        if item_id in self.active_items and self.active_items[item_id]['spawned']:
            return True
        pickup_x, pickup_y = self.pickup_location        
        color = get_item_color(task)
        item = generate_item(item_id, color)
        if entity_spawned(self, self.spawn_client, item_id, item, pickup_x, pickup_y, 0.15):
            self.active_items[item_id]['spawned'] = True
            return True
        self.get_logger().warn("Failed to spawn item")
        return False

    def reset_robot_position(self):
        success = reset_robot_position(self)
        return success

    def virtual_pickup(self, item_name, task):
        if item_name in self.active_items:
            self.active_items[item_name]['picked'] = True
        
        if delete_entity(self, self.delete_client, item_name):
            self.get_logger().info("Picked up item")
            return True
        else:
            self.get_logger().info("Picked up item")
            return True 

    def virtual_dropoff(self, task, item_id):
        if item_id is None:
            item_id = "sorted_" + str(task) + "_"+str(int(time.time()))
        if not self.spawn_client.service_is_ready():
            self.get_logger().warn("Simulating dropoff")
            return True
        dock_x, dock_y = self.drop_docks[task]
        color = get_item_color(task)
        item = generate_item(item_id, color)
        if entity_spawned(self, self.spawn_client, item_id, item, dock_x, dock_y, 0.15):
            message = "Dropped " + str(item_id) + "item at Dock " + str(task) 
            self.get_logger().info(message)
            self.dropped_items[item_id] = {'dropoff_time': time.time(),'task': task}
            return True
        else:
            self.get_logger().warn("Simulating dropoff")
            return True

    def docking_box(self, x, y):
        return docking_blue_box(self, self.spawn_client, self.task, x, y)

    def delete_dock_box(self):
        return delete_blue_box(self, self.delete_client, self.task)

    def cleanup_dropped_items(self):
        curr_time = time.time()
        item_k = list(self.dropped_items.keys())
        for item_id in item_k:
            info = self.dropped_items.get(item_id)
            if info is None:
                continue
            d_time = info.get("dropoff_time", 0)
            diff = curr_time-d_time
            if diff>=2.5:
                delete = delete_entity(self, self.delete_client, item_id)
                if delete:
                    self.get_logger().debug("Cleaned up item")
                if item_id in self.dropped_items:
                    del self.dropped_items[item_id]
 
    def reset_task_state(self):
        self.current_task = None
        self.task = None
        self.current_item_id = None
        self.phase = "idle"
        self.current_goal = None
        self.task_start_time = None
        self.docking_complete = False
        self.blue_marker_detected = False
        self.blue_marker_area = 0
        self.blue_marker_centered = False
        self.blue_marker_error_x = 0
        self.docking_stable_time = None
        self.item_dropped_for_current_task = False 
        
        if not self.task_queue:
            self.get_logger().info("All tasks completed!")

    def prepare_docking(self):
        docking_target_x, docking_target_y = self.drop_docks[self.task]
        robot_pose_x, robot_pose_y = self.pose[0], self.pose[1]
        distance_x, distance_y = docking_target_x - robot_pose_x, docking_target_y - robot_pose_y
        dist = math.hypot(distance_x, distance_y)
        desired_offset = 2.5
        if dist > 0.1:
            scale = desired_offset / dist
            target_x = robot_pose_x - distance_x * scale
            target_y = robot_pose_y - distance_y * scale
        else:
            target_x, target_y = robot_pose_x - 2.5, robot_pose_y
        self.docking_box(target_x, target_y)

    def advance_phase(self):
        if self.phase == "pickup":
            self.phase = "dropoff"
            self.task = self.current_task
            self.current_goal = self.drop_docks[self.task]
            self.goal_reached_time = None  
            self.task_start_time = time.time()
        elif self.phase == "dropoff":
            self.current_task = None
            self.task = None
            self.phase = "idle"
            self.current_goal = None
            self.task_start_time = None
        elif self.phase == "docking":
            self.delete_dock_box()
            self.reset_task_state()

    def step(self):
        if self.phase == "docking":
            msg = Twist()
            linear, angular = docking_control(self.blue_marker_detected, self.blue_marker_error_x)
            msg.linear.x = linear
            msg.angular.z = angular
            self.cmd_pub.publish(msg)
            return
        if self.current_goal is None:
            msg = Twist()
            self.cmd_pub.publish(msg)
            return
        observation = self.build_observation()
        if observation is None:
            if self.scan is None:
                self.get_logger().warn("Waiting for LiDAR scan...")
            return
        action, _ = self.model.predict(observation, deterministic=True)
        v, w = self.actions[int(action)]
        if self.last_log_time is not None:
            if time.time() - self.last_log_time > 5.0:
                distance = self.goal_distance()
                self.last_log_time = time.time()
        else:
            self.last_log_time = time.time() 
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def cleanup_items(self):
        if not self.delete_client.service_is_ready():
            return
        for item_id in list(self.active_items.keys()):
            if self.active_items[item_id]['spawned']:
                try:
                    if delete_entity(self, self.delete_client, item_id):
                        self.get_logger().info(f"Cleaned up {item_id}")
                except Exception as e:
                    self.get_logger().warn(f"Exception cleaning up {item_id}: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SortingNode()
    if robot_initilization(node, spawn_x=0.0, spawn_y=0.0)==False:
        node.get_logger().error("TB3 spawn failed - make sure Gazebo is running")
        node.get_logger().error("Launch Gazebo first: ./launch_warehouse.sh")
        node.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass
        return
    node.get_logger().info("Sensors initializing")
    import time
    time.sleep(2.0)
    
    node.get_logger().info("Starting tasks")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down sorting")
    finally:
        try:
            node.cleanup_items()
            node.destroy_node()
            rclpy.shutdown()
        except:
            pass  
if __name__ == "__main__":
    main()