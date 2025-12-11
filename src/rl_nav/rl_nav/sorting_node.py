#!/usr/bin/env python3
"""
Sorting Node with Stage 3 PPO Navigation

Manages virtual sorting tasks using trained Stage 3 PPO policy.
Task classes: A, B, C (mapped to DOCK_A, DOCK_B, DOCK_C)
FSM: IDLE → GO_PICKUP → GO_DROPOFF → IDLE (repeat for each task)
     After all tasks: GO_DROPOFF → GO_DOCKING → IDLE (final docking only)
"""
import os
import sys
import signal
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

from rl_nav.constants import dockA, dockB, dockC, pickup, success_region, sucess_close_bonus, warehouse_x_limit_max, warehouse_x_limit_min, warehouse_y_limit_max, warehouse_y_limit_min, robot_actions, docking_fsm_distance, docking_time
from rl_nav.gazebo_functions import robot_initilization, entity_spawned, delete_entity, reset_robot_position, docking_blue_box, delete_blue_box
from rl_nav.box_functions import generate_item, get_item_color
from rl_nav.navigation_functions import euclidean_distance, goal_reached, check_collision, process_scan_to_bins
from rl_nav.rl_nav.docking_functions import process_camera_image, docking_complete, docking_control
from rl_nav.observation_functions import observation


class SortingNode(Node):
    def __init__(self):
        super().__init__("sorting_node")
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ppo_runs", "ppo_stage3_sorting.zip")        
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
        self.last_stuck_check = None  
        self.last_stuck_dist = None  
        self.collision_check_enabled = False  
        self.start_time = None  #
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
        self.blue_marker_error_x = 0  # For control calculation
        self.docking_complete = False
        self.docking_stable_time = None
        self.docking_stable_duration = 3.0
        self.max_docking_time = docking_time
        self.docking_start_time = None
        self.docking_transition_distance = docking_fsm_distance
        self.build_initial_tasks(5)
        self.control_timer = self.create_timer(0.15, self.control_step)
        self.task_timer = self.create_timer(0.5, self.task_manager)
        self.get_logger().info(f"SortingNode initialized. Task queue: {self.task_queue}")

    def scan_for_bins(self, msg):
        self.scan, _, _ = process_scan_to_bins(msg, 24, 3.5)

    def update_pose(self, msg):
        position = msg.pose.position
        orientation = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (orientation.w * orientation.z + orientation.x * orientation.y), 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z))
        self.pose[:] = (position.x, position.y, yaw)

    def process_camera(self, msg):
        if self.phase != "docking":
            if self.blue_marker_detected or self.docking_complete:
                self.blue_marker_detected = False
                self.blue_marker_area = 0
                self.blue_marker_centered = False
                self.blue_marker_error = 0
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
        self.blue_marker_error_x = result['error']
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
            task_class = self.task_class
        else: 
            task_class = None
        return observation(self.scan, self.pose, self.current_goal, task_class, self.phase)

    def tasks(self, num_tasks):
        categories = ['A', 'B', 'C']
        self.task_queue = ['B','C']
        for task in categories:
            self.items_at_pickup[task] = []
        for task in self.task_queue:
            self.item_counter[task] = self.item_counter.get(task, 0) + 1
            item_id = "item_" + task +"_"+str(self.item_counter[task])
            self.items_at_pickup.append(item_id)
            self.active_items[item_id] = {'task': task, 'spawned':False, 'picked':False}

    def goal_distance(self):
        return euclidean_distance(self.pose, self.current_goal)

    def goal_reached(self):
        is_reached, new_time = goal_reached(self.pose, self.current_goal, self.goal_reached_threshold,self.goal_region, 3.0)
        self._goal_reached_time = new_time
        return is_reached

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
        if success:
            self.last_stuck_check = None
            self.last_stuck_dist = None
        return success

    def virtual_pickup(self, item_name, task):
        if item_name in self.active_items:
            self.active_items[item_name]['picked'] = True
        
        if delete_entity(self, self.delete_client, item_name):
            self.get_logger().info("Picked up item")
            return True
        else:
            self.get_logger.info("Picked up item")
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
            self.dropped_items[item_id] = {'dropoff_time': time.time(),'task_class': task}
            return True
        else:
            self.get_logger().warn("Simulating dropoff")
            return True

    def docking_box(self, x, y):
        return docking_blue_box(self, self.spawn_client, self.task, x, y)

    def delete_dock_box(self):
        return delete_blue_box(self, self.delete_client, self.task_class)

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
                    self.get_logger.debug("Cleaned up item")
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
        self.blue_marker_error = 0
        self.docking_stable_time = None
        self.item_dropped_for_current_task = False 
        
        if not self.task_queue:
            self.get_logger().info("All tasks completed!")

    # def task_manager(self):
    #     """Manage task queue and FSM transitions."""
    #     now = time.time()

    #     # Cleanup dropped items periodically
    #     self._cleanup_dropped_items()

    #     # Check for collision or stuck (only during active navigation)
    #     if self.phase in ["GO_PICKUP", "GO_DROPOFF", "GO_DOCKING"]:
    #         # Enable collision checking after 2 seconds (let robot settle)
    #         if self._start_time is None:
    #             self._start_time = time.time()
    #         if time.time() - self._start_time >= 2.0:
    #             self._collision_check_enabled = True
            
    #         # Only check collisions after delay
    #         if self._collision_check_enabled and self._check_collision():
    #             # Debounce: only reset if collision persists for 3 seconds
    #             now = time.time()
    #             if self._last_collision_time is None:
    #                 self._last_collision_time = now
    #             elif now - self._last_collision_time >= 3.0:
    #                 self.get_logger().warn("Persistent collision detected! Resetting robot position...")
    #                 self._reset_robot_position()  # Non-blocking
    #                 self.task_start_time = time.time()
    #                 self._last_collision_time = None
    #                 return
    #         else:
    #             # No collision - reset debounce timer
    #             self._last_collision_time = None
            
    #         # if self._check_stuck():
    #         #     self.get_logger().warn("Robot stuck! Resetting position...")
    #         #     if self._reset_robot_position():
    #         #         self.task_start_time = time.time()
    #         #         return
    #         #     else:
    #         #         self.get_logger().warn("Reset failed, skipping task")
    #         #         self._advance_phase()
    #         #         return

    #     # Timeout check
    #     if self.task_start_time and (now - self.task_start_time) > self.max_task_time:
    #         self.get_logger().warn("Task timeout, advancing phase")
    #         self._advance_phase()

    #     # FSM
    #     if self.phase == "IDLE" and self.task_queue:
    #         # Start new task: go to pickup
    #         self.current_task = self.task_queue.pop(0)
    #         self.phase = "GO_PICKUP"
    #         self.task_start_time = now
    #         self.current_goal = self.pickup_location
    #         self._goal_reached_time = None  # Reset goal reached timer
    #         self.task_class = None  # No task class yet (not picked up)
    #         self._items_spawned_for_current_task = False  # Reset spawn flag for new task
    #         self._item_dropped_for_current_task = False # Reset dropoff flag for new task
    #         self.get_logger().info(f"[Task] {self.current_task}: Going to PICKUP @ ({self.pickup_location[0]:.1f}, {self.pickup_location[1]:.1f})")

    #     elif self.phase == "GO_PICKUP":
    #         # Check if robot is close enough to spawn items (within 1.5m)
    #         dist_to_pickup = self._distance_to_goal()
    #         if dist_to_pickup < 1.5 and not self._items_spawned_for_current_task:
    #             # Spawn items now (only once per task)
    #             if self._spawn_items_for_current_task():
    #                 self._items_spawned_for_current_task = True
            
    #         # Check if goal reached
    #         if self._goal_reached():
    #             # Reached pickup: assign task class and go to dock
    #             self.phase = "GO_DROPOFF"
    #             self.task_start_time = now
    #             self.task_class = self.current_task  # Set task class for PPO observation
    #             self.current_goal = self.drop_docks[self.task_class]
    #             self._goal_reached_time = None  # Reset goal reached timer
                
    #             # Get the item ID for this task
    #             if self.items_at_pickup.get(self.current_task) and len(self.items_at_pickup[self.current_task]) > 0:
    #                 item_id = self.items_at_pickup[self.current_task].pop(0)  # Get first item of this class
    #             else:
    #                 # Fallback if no items available
    #                 self.item_counter[self.current_task] += 1
    #                 item_id = f"item_{self.current_task}_{self.item_counter[self.current_task]}"
    #                 self.get_logger().warn(f"No items available for {self.current_task}, using fallback ID: {item_id}")
                
    #             # Store item_id for dropoff
    #             self.current_item_id = item_id
                
    #             # Virtual pickup: delete item from pickup zone
    #             self._virtual_pickup(item_id, self.task_class)
                
    #             self.get_logger().info(f"[Task] {self.current_task}: Picked up {item_id}. Going to DOCK_{self.current_task} @ ({self.current_goal[0]:.1f}, {self.current_goal[1]:.1f})")

    #     elif self.phase == "GO_DROPOFF":
    #         dist_to_dock = self._distance_to_goal()
            
    #         # Drop item when goal reached (0.7m) - consistent with training SUCCESS_RADIUS
    #         if self._goal_reached() and not self._item_dropped_for_current_task:
    #             # Drop off the item first
    #             item_id = getattr(self, 'current_item_id', None)
    #             self._virtual_dropoff(self.task_class, item_id)
    #             self._item_dropped_for_current_task = True
    #             self.get_logger().info(f"Dropped off {self.current_task} item at DOCK_{self.task_class} (0.7m threshold)")
            
    #         # Check if this is the last task - transition to docking when close enough (1.5m)
    #         # BUT only after item has been dropped (at 0.7m)
    #         if not self.task_queue and self._item_dropped_for_current_task and dist_to_dock < self.docking_transition_distance:
    #             # All tasks done - now perform final docking
    #             self.phase = "GO_DOCKING"
    #             self.docking_start_time = now
    #             self.docking_complete = False
    #             self.blue_marker_detected = False
    #             self.blue_marker_area = 0
    #             self.blue_marker_centered = False
    #             self.blue_marker_error_x = 0
    #             self.docking_stable_time = None
                
    #             # Spawn blue box 2-3m away from robot's current position (away from dock)
    #             # This ensures the box never spawns under the robot
    #             dock_x, dock_y = self.drop_docks[self.task_class]
    #             robot_x, robot_y = self.pose[0], self.pose[1]
                
    #             # Calculate vector from robot to dock
    #             dx = dock_x - robot_x
    #             dy = dock_y - robot_y
    #             dist_to_dock = math.hypot(dx, dy)
                
    #             # Target should be 2.5m away from robot, in direction AWAY from dock
    #             # This makes robot back up to dock position
    #             desired_offset = 2.5  # meters from robot
                
    #             if dist_to_dock > 0.1:
    #                 # Normalize direction vector and place target 2.5m away from robot
    #                 # in direction away from dock (negative direction)
    #                 norm = dist_to_dock
    #                 target_x = robot_x - (dx / norm) * desired_offset
    #                 target_y = robot_y - (dy / norm) * desired_offset
    #             else:
    #                 # Fallback: if robot is exactly at dock, place target 2.5m behind robot
    #                 target_x = robot_x - 2.5
    #                 target_y = robot_y

    #             if self._spawn_blue_box_at_dock(target_x, target_y):
    #                 self.get_logger().info(
    #                     f"[Final] All tasks complete! Docking target offset from DOCK_{self.task_class} "
    #                     f"at ({target_x:.2f}, {target_y:.2f})"
    #                 )
    #             else:
    #                 self.get_logger().warn("Failed to spawn blue box - completing without final docking")
    #                 self._reset_task_state()
    #         elif self._item_dropped_for_current_task and self.task_queue:
    #             # Item dropped and more tasks remaining - move to next task
    #             self.get_logger().info(f"[Task] {self.current_task} complete. {len(self.task_queue)} task(s) remaining.")
    #             self._reset_task_state()

    #     elif self.phase == "GO_DOCKING":
    #         # Check docking completion
    #         if self.docking_complete:
    #             # Final docking successful!
    #             self.get_logger().info(f"Completed final docking at DOCK_{self.task_class}")
                
    #             # Clean up blue box
    #             self._delete_blue_box()
                
    #             # Reset for next task (will be IDLE since queue is empty)
    #             self._reset_task_state()
                
    #         elif self.docking_start_time and (now - self.docking_start_time) > self.max_docking_time:
    #             # Timeout - final docking took too long
    #             self.get_logger().warn("Final docking timeout - completing anyway")
    #             self._delete_blue_box()
    #             self._reset_task_state()

    def _advance_phase(self):
        """Advance to next phase on timeout."""
        if self.phase == "GO_PICKUP":
            # Skip pickup, go straight to dropoff
            self.phase = "GO_DROPOFF"
            self.task_class = self.current_task
            self.current_goal = self.drop_docks[self.task_class]
            self._goal_reached_time = None  # Reset goal reached timer
            self.task_start_time = time.time()
        elif self.phase == "GO_DROPOFF":
            # Skip dropoff, mark task as failed
            self.current_task = None
            self.task_class = None
            self.phase = "IDLE"
            self.current_goal = None
            self.task_start_time = None
        elif self.phase == "GO_DOCKING":
            # Final docking timeout: item already dropped, just clean up and reset
            self._delete_blue_box()
            self._reset_task_state()

    def control_step(self):
        """PPO control loop or color-based docking control."""
        # DOCKING PHASE: Use color-based control
        if self.phase == "GO_DOCKING":
            msg = Twist()
            linear_x, angular_z = calculate_docking_control(
                self.blue_marker_detected, self.blue_marker_error_x
            )
            msg.linear.x = linear_x
            msg.angular.z = angular_z
            self.cmd_pub.publish(msg)
            return
        
        # PPO CONTROL for other phases
        if self.current_goal is None:
            msg = Twist()
            self.cmd_pub.publish(msg)
            return

        obs = self._obs()
        if obs is None:
            # Log when observation is None
            if self.scan is None:
                self.get_logger().warn("Waiting for LiDAR scan...")
            return

        # Use Stage 3 model (expects 30-dim obs with task_class one-hot)
        action, _ = self.model.predict(obs, deterministic=True)
        v, w = self.actions[int(action)]
        
        # Log periodically to debug
        if self._last_log_time is not None:
            if time.time() - self._last_log_time > 5.0:
                dist = self._distance_to_goal()
                self.get_logger().info(f"[Control] phase={self.phase} goal={self.current_goal} "
                                     f"dist={dist:.2f}m task_class={self.task_class}")
                self._last_log_time = time.time()
        else:
            self._last_log_time = time.time()
        
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def cleanup_items(self):
        """Clean up all spawned items on shutdown."""
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
    
    # Handle SIGINT gracefully
    def signal_handler(sig, frame):
        node.get_logger().info("Received interrupt signal, shutting down...")
        try:
            node.destroy_node()
            rclpy.shutdown()
        except:
            pass
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Spawn robot in Gazebo before starting
    if not spawn_tb3(node):
        node.get_logger().error("TB3 spawn failed - make sure Gazebo is running")
        node.get_logger().error("Launch Gazebo first: ./launch_warehouse.sh")
        node.destroy_node()
        # Don't shutdown here - let finally block handle it
        try:
            rclpy.shutdown()
        except:
            pass
        return
    
    node.get_logger().info("Robot spawned successfully. Starting sorting tasks...")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down sorting node...")
    finally:
        try:
            node.cleanup_items()
            node.destroy_node()
            rclpy.shutdown()
        except:
            pass  # Ignore errors if already shut down


if __name__ == "__main__":
    main()
