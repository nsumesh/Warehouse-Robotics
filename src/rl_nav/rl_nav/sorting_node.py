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

# Import constants and utilities
from rl_nav.constants import (
    DOCK_A, DOCK_B, DOCK_C, PICKUP, SUCCESS_RADIUS, X_MIN, X_MAX, Y_MIN, Y_MAX,
    ACTIONS, DOCKING_TRANSITION_DISTANCE, MAX_DOCKING_TIME
)
from rl_nav.gazebo_utils import (
    spawn_tb3, spawn_entity, delete_entity, reset_robot_position,
    spawn_blue_box_at_dock, delete_blue_box
)
from rl_nav.item_utils import generate_item_sdf, get_item_color
from rl_nav.navigation_utils import (
    distance_to_goal, goal_reached, check_collision, process_scan_to_bins
)
from rl_nav.rl_nav.docking_functions import (
    process_camera_image, is_docking_complete, calculate_docking_control
)
from rl_nav.observation_utils import build_observation


class SortingNode(Node):
    """Sorting node using Stage 3 trained PPO for virtual sorting."""

    def __init__(self):
        super().__init__("sorting_node")

        # Load Stage 3 PPO model
        model_paths = [
            os.path.abspath("ppo_runs/ppo_stage3_sorting.zip"),
            os.path.expanduser("~/MSML_642_FinalProject/ppo_runs/ppo_stage3_sorting.zip"),
            os.path.join(os.path.dirname(__file__), "../../../ppo_runs/ppo_stage3_sorting.zip"),
        ]
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            self.get_logger().error("Stage 3 PPO model not found. Train first:")
            self.get_logger().error("  ros2 run rl_nav train_ppo --curriculum-stage 3 --timesteps 100000")
            raise RuntimeError("PPO model missing")
        
        self.model = PPO.load(model_path)
        self.get_logger().info(f"Loaded Stage 3 PPO model: {model_path}")

        # ROS I/O
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self._scan_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.camera_sub = self.create_subscription(Image, '/camera/image_raw', self._camera_cb, 10)
        self.bridge = CvBridge()
        
        # Gazebo services for virtual pickup/dropoff
        self.delete_client = self.create_client(DeleteEntity, "/delete_entity")
        self.spawn_client = self.create_client(SpawnEntity, "/spawn_entity")
        if self.delete_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info("Gazebo services available for virtual sorting")
        else:
            self.get_logger().warn("Gazebo services not available - virtual sorting will be simulated")

        # Configuration - match Stage 3 training setup
        self.pickup_location = PICKUP
        self.drop_docks = {
            'A': DOCK_A,
            'B': DOCK_B,
            'C': DOCK_C,
        }
        self.actions = ACTIONS

        # State
        self.scan = None
        self.pose = np.zeros(3, dtype=np.float32)
        self.current_goal = None
        self.task_class = None  # 'A', 'B', or 'C'
        self.goal_reached_threshold = 0.7  # Closer to training threshold (0.6m)
        self._goal_reached_time = None  # Track when goal was first reached (for stability check)
        self.task_queue = []
        self.current_task = None
        self.phase = "IDLE"  # IDLE → GO_PICKUP → GO_DROPOFF → GO_DOCKING → IDLE
        self.task_start_time = None
        self.max_task_time = 480.0  # Increased timeout for better navigation
        self._last_log_time = None  # For debug logging
        self._last_stuck_check = None  # For stuck detection
        self._last_stuck_dist = None  # For stuck detection
        self._collision_check_enabled = False  # Delay collision checking
        self._start_time = None  # Track when node started
        self._last_collision_time = None  # Debounce collision detection

        # Object tracking: maps task_class -> list of item IDs
        self.items_at_pickup = {}  # Maps task_class -> list of item IDs
        self.active_items = {}  # Maps item_id -> {'task_class': 'A', 'spawned': True, 'picked': False}
        self.item_counter = {'A': 0, 'B': 0, 'C': 0}  # Counter for unique IDs
        self.current_item_id = None  # Store current item ID for dropoff
        self._items_spawned_for_current_task = False  # Track if items spawned for current task
        self._item_dropped_for_current_task = False  # Track if item dropped for current task
        self.dropped_items = {}  # Maps item_id -> {'dropoff_time': timestamp, 'task_class': 'A'}

        # Docking state variables
        self.blue_marker_detected = False
        self.blue_marker_area = 0
        self.blue_marker_centered = False
        self.blue_marker_error_x = 0  # For control calculation
        self.docking_complete = False
        self.docking_stable_time = None
        self.docking_stable_duration = 3.0
        self.max_docking_time = MAX_DOCKING_TIME
        self.docking_start_time = None
        self.docking_transition_distance = DOCKING_TRANSITION_DISTANCE

        # Initialize
        self._build_initial_tasks(5)
        # Don't spawn items immediately - wait until robot is close to pickup
        self.control_timer = self.create_timer(0.15, self.control_step)
        self.task_timer = self.create_timer(0.5, self.task_manager)
        self.get_logger().info(f"SortingNode initialized. Task queue: {self.task_queue}")

    def _scan_cb(self, msg):
        """Process LiDAR scan into 24 bins (match training)."""
        self.scan, _, _ = process_scan_to_bins(msg, 24, 3.5)

    def _odom_cb(self, msg):
        """Update robot pose."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose[:] = (x, y, yaw)

    def _camera_cb(self, msg):
        """Process camera image for blue marker detection during GO_DOCKING."""
        # Safety: Reset docking state if not in GO_DOCKING phase
        if self.phase != "GO_DOCKING":
            # Clear any stale docking state to prevent issues
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
                self.get_logger().warn(f"Camera processing error: {result['error']}")
            # Reset state on error to prevent stale data
            self.blue_marker_detected = False
            self.blue_marker_area = 0
            self.blue_marker_centered = False
            self.blue_marker_error_x = 0
            self.docking_stable_time = None
            return
        
        # Update state from detection result
        self.blue_marker_detected = result['detected']
        self.blue_marker_area = result['area']
        self.blue_marker_error_x = result['error_x']
        self.blue_marker_centered = result['centered']
        
        # Check docking completion
        if self.blue_marker_detected:
            complete, new_stable_time = is_docking_complete(
                self.blue_marker_area, self.blue_marker_centered,
                self.docking_stable_time, self.docking_stable_duration
            )
            self.docking_complete = complete
            self.docking_stable_time = new_stable_time
            if complete:
                self.get_logger().info("Docking complete: marker large, centered, and stable")
        else:
            # No marker detected - reset stability timer and completion flag
            self.docking_stable_time = None
            self.docking_complete = False

    def _obs(self):
        """Build 30-dim observation: 24 scan + [dx, dy, yaw] + task_class one-hot."""
        task_class_for_obs = self.task_class if self.phase == "GO_DROPOFF" else None
        return build_observation(self.scan, self.pose, self.current_goal, task_class_for_obs, self.phase)

    def _build_initial_tasks(self, num_tasks=5):
        """Build task queue and assign item IDs."""
        categories = ['A', 'B', 'C']
        self.task_queue = ['B','C']
        
        # Initialize items_at_pickup dictionary
        for task_class in categories:
            self.items_at_pickup[task_class] = []
        
        # Assign IDs based on task queue
        for task_class in self.task_queue:
            self.item_counter[task_class] += 1
            item_id = f"item_{task_class}_{self.item_counter[task_class]}"
            self.items_at_pickup[task_class].append(item_id)
            self.active_items[item_id] = {
                'task_class': task_class,
                'spawned': False,
                'picked': False
            }

    def _distance_to_goal(self):
        """Calculate distance to goal."""
        return distance_to_goal(self.pose, self.current_goal)

    def _goal_reached(self):
        """Check if goal reached (with stability check)."""
        is_reached, new_time = goal_reached(
            self.pose, self.current_goal, self.goal_reached_threshold,
            self._goal_reached_time, 3.0
        )
        self._goal_reached_time = new_time
        return is_reached

    def _check_collision(self):
        """Check if robot is too close to obstacles."""
        return check_collision(self.scan, 3.5, 0.10)  # 10cm threshold

    # def _check_stuck(self):
    #     """Check if robot is stuck (distance not improving)."""
    #     current_dist = self._distance_to_goal()
    #     is_stuck, new_check_time, new_dist = check_stuck(
    #         current_dist, self._last_stuck_check, self._last_stuck_dist
    #     )
    #     self._last_stuck_check = new_check_time
    #     self._last_stuck_dist = new_dist
    #     if is_stuck:
    #         self.get_logger().warn(f"Robot appears stuck: dist={current_dist:.2f}m (no improvement)")
    #     return is_stuck


    def _spawn_items_for_current_task(self):
        """Spawn items for the current task class at pickup point when robot is close."""
        if not self.spawn_client.service_is_ready():
            self.get_logger().warn("Spawn service not ready - items won't be spawned")
            return False
        
        if self.current_task is None:
            return False
        
        # Get items for current task class
        task_class = self.current_task
        if task_class not in self.items_at_pickup or len(self.items_at_pickup[task_class]) == 0:
            self.get_logger().warn(f"No items available for task class {task_class}")
            return False
        
        # Only spawn the FIRST item for this task (the one that will be picked up)
        item_id = self.items_at_pickup[task_class][0]  # Get first item
        
        # Skip if already spawned
        if item_id in self.active_items and self.active_items[item_id]['spawned']:
            return True
        
        self.get_logger().info(f"Spawning item for task {task_class} at pickup point")
        pickup_x, pickup_y = self.pickup_location
        
        # Spawn single item at pickup point
        item_x = pickup_x
        item_y = pickup_y
        item_z = 0.15  # Half of box height (0.3m box)
        
        # Generate SDF for this item
        color_rgba = get_item_color(task_class)
        item_sdf = generate_item_sdf(item_id, color_rgba)
        
        # Spawn item
        if spawn_entity(self, self.spawn_client, item_id, item_sdf, item_x, item_y, item_z):
            self.active_items[item_id]['spawned'] = True
            self.get_logger().info(f"Spawned {item_id} at pickup ({item_x:.2f}, {item_y:.2f})")
            return True
        else:
            self.get_logger().warn(f"Failed to spawn {item_id}")
            return False

    def _reset_robot_position(self):
        """Reset robot to a safe position when stuck."""
        success = reset_robot_position(self)
        if success:
            # Reset stuck detection after successful reset
            self._last_stuck_check = None
            self._last_stuck_dist = None
        return success

    def _virtual_pickup(self, item_name, task_class):
        """Virtual pickup: delete item from pickup zone."""
        # Mark item as picked in tracking
        if item_name in self.active_items:
            self.active_items[item_name]['picked'] = True
        
        if delete_entity(self, self.delete_client, item_name):
            self.get_logger().info(f"Picked up {task_class} item: {item_name}")
            return True
        else:
            self.get_logger().warn(f"Failed to delete {item_name} - simulating pickup")
            return True  # Continue anyway

    def _virtual_dropoff(self, task_class, item_id=None):
        """Virtual dropoff: spawn item at dock."""
        if item_id is None:
            item_id = f"sorted_{task_class}_{int(time.time())}"
        
        if not self.spawn_client.service_is_ready():
            self.get_logger().warn("Spawn service not ready - simulating dropoff")
            return True
        
        dock_x, dock_y = self.drop_docks[task_class]
        
        # Generate SDF with same properties as pickup item
        color_rgba = get_item_color(task_class)
        item_sdf = generate_item_sdf(item_id, color_rgba)
        
        # Spawn at dock location
        if spawn_entity(self, self.spawn_client, item_id, item_sdf, dock_x, dock_y, 0.15):
            self.get_logger().info(f"Dropped {task_class} item ({item_id}) at DOCK_{task_class} ({dock_x:.1f}, {dock_y:.1f})")
            # Track dropped item for delayed deletion
            self.dropped_items[item_id] = {
                'dropoff_time': time.time(),
                'task_class': task_class
            }
            return True
        else:
            self.get_logger().warn(f"Failed to spawn at dock - simulating dropoff")
            return True  # Continue anyway

    def _spawn_blue_box_at_dock(self, x, y):
        """Spawn blue box marker at dock location for visual docking."""
        return spawn_blue_box_at_dock(self, self.spawn_client, self.task_class, x, y)

    def _delete_blue_box(self):
        """Delete blue box marker after docking."""
        return delete_blue_box(self, self.delete_client, self.task_class)

    def _cleanup_dropped_items(self):
        """Delete items from dropoff points after 2-3 seconds."""
        now = time.time()
        items_to_delete = []
        
        for item_id, item_info in self.dropped_items.items():
            elapsed = now - item_info['dropoff_time']
            if elapsed >= 2.5:  # Delete after 2.5 seconds
                items_to_delete.append(item_id)
        
        for item_id in items_to_delete:
            if delete_entity(self, self.delete_client, item_id):
                self.get_logger().debug(f"Cleaned up dropped item: {item_id}")
            del self.dropped_items[item_id]

    def _reset_task_state(self):
        """Reset state after task completion."""
        self.current_task = None
        self.task_class = None
        self.current_item_id = None
        self.phase = "IDLE"
        self.current_goal = None
        self.task_start_time = None
        self.docking_complete = False
        self.blue_marker_detected = False
        self.blue_marker_area = 0
        self.blue_marker_centered = False
        self.blue_marker_error_x = 0
        self.docking_stable_time = None
        self._item_dropped_for_current_task = False # Reset the flag
        
        if not self.task_queue:
            self.get_logger().info("All tasks completed!")

    def task_manager(self):
        """Manage task queue and FSM transitions."""
        now = time.time()

        # Cleanup dropped items periodically
        self._cleanup_dropped_items()

        # Check for collision or stuck (only during active navigation)
        if self.phase in ["GO_PICKUP", "GO_DROPOFF", "GO_DOCKING"]:
            # Enable collision checking after 2 seconds (let robot settle)
            if self._start_time is None:
                self._start_time = time.time()
            if time.time() - self._start_time >= 2.0:
                self._collision_check_enabled = True
            
            # Only check collisions after delay
            if self._collision_check_enabled and self._check_collision():
                # Debounce: only reset if collision persists for 3 seconds
                now = time.time()
                if self._last_collision_time is None:
                    self._last_collision_time = now
                elif now - self._last_collision_time >= 3.0:
                    self.get_logger().warn("Persistent collision detected! Resetting robot position...")
                    self._reset_robot_position()  # Non-blocking
                    self.task_start_time = time.time()
                    self._last_collision_time = None
                    return
            else:
                # No collision - reset debounce timer
                self._last_collision_time = None
            
            # if self._check_stuck():
            #     self.get_logger().warn("Robot stuck! Resetting position...")
            #     if self._reset_robot_position():
            #         self.task_start_time = time.time()
            #         return
            #     else:
            #         self.get_logger().warn("Reset failed, skipping task")
            #         self._advance_phase()
            #         return

        # Timeout check
        if self.task_start_time and (now - self.task_start_time) > self.max_task_time:
            self.get_logger().warn("Task timeout, advancing phase")
            self._advance_phase()

        # FSM
        if self.phase == "IDLE" and self.task_queue:
            # Start new task: go to pickup
            self.current_task = self.task_queue.pop(0)
            self.phase = "GO_PICKUP"
            self.task_start_time = now
            self.current_goal = self.pickup_location
            self._goal_reached_time = None  # Reset goal reached timer
            self.task_class = None  # No task class yet (not picked up)
            self._items_spawned_for_current_task = False  # Reset spawn flag for new task
            self._item_dropped_for_current_task = False # Reset dropoff flag for new task
            self.get_logger().info(f"[Task] {self.current_task}: Going to PICKUP @ ({self.pickup_location[0]:.1f}, {self.pickup_location[1]:.1f})")

        elif self.phase == "GO_PICKUP":
            # Check if robot is close enough to spawn items (within 1.5m)
            dist_to_pickup = self._distance_to_goal()
            if dist_to_pickup < 1.5 and not self._items_spawned_for_current_task:
                # Spawn items now (only once per task)
                if self._spawn_items_for_current_task():
                    self._items_spawned_for_current_task = True
            
            # Check if goal reached
            if self._goal_reached():
                # Reached pickup: assign task class and go to dock
                self.phase = "GO_DROPOFF"
                self.task_start_time = now
                self.task_class = self.current_task  # Set task class for PPO observation
                self.current_goal = self.drop_docks[self.task_class]
                self._goal_reached_time = None  # Reset goal reached timer
                
                # Get the item ID for this task
                if self.items_at_pickup.get(self.current_task) and len(self.items_at_pickup[self.current_task]) > 0:
                    item_id = self.items_at_pickup[self.current_task].pop(0)  # Get first item of this class
                else:
                    # Fallback if no items available
                    self.item_counter[self.current_task] += 1
                    item_id = f"item_{self.current_task}_{self.item_counter[self.current_task]}"
                    self.get_logger().warn(f"No items available for {self.current_task}, using fallback ID: {item_id}")
                
                # Store item_id for dropoff
                self.current_item_id = item_id
                
                # Virtual pickup: delete item from pickup zone
                self._virtual_pickup(item_id, self.task_class)
                
                self.get_logger().info(f"[Task] {self.current_task}: Picked up {item_id}. Going to DOCK_{self.current_task} @ ({self.current_goal[0]:.1f}, {self.current_goal[1]:.1f})")

        elif self.phase == "GO_DROPOFF":
            dist_to_dock = self._distance_to_goal()
            
            # Drop item when goal reached (0.7m) - consistent with training SUCCESS_RADIUS
            if self._goal_reached() and not self._item_dropped_for_current_task:
                # Drop off the item first
                item_id = getattr(self, 'current_item_id', None)
                self._virtual_dropoff(self.task_class, item_id)
                self._item_dropped_for_current_task = True
                self.get_logger().info(f"Dropped off {self.current_task} item at DOCK_{self.task_class} (0.7m threshold)")
            
            # Check if this is the last task - transition to docking when close enough (1.5m)
            # BUT only after item has been dropped (at 0.7m)
            if not self.task_queue and self._item_dropped_for_current_task and dist_to_dock < self.docking_transition_distance:
                # All tasks done - now perform final docking
                self.phase = "GO_DOCKING"
                self.docking_start_time = now
                self.docking_complete = False
                self.blue_marker_detected = False
                self.blue_marker_area = 0
                self.blue_marker_centered = False
                self.blue_marker_error_x = 0
                self.docking_stable_time = None
                
                # Spawn blue box 2-3m away from robot's current position (away from dock)
                # This ensures the box never spawns under the robot
                dock_x, dock_y = self.drop_docks[self.task_class]
                robot_x, robot_y = self.pose[0], self.pose[1]
                
                # Calculate vector from robot to dock
                dx = dock_x - robot_x
                dy = dock_y - robot_y
                dist_to_dock = math.hypot(dx, dy)
                
                # Target should be 2.5m away from robot, in direction AWAY from dock
                # This makes robot back up to dock position
                desired_offset = 2.5  # meters from robot
                
                if dist_to_dock > 0.1:
                    # Normalize direction vector and place target 2.5m away from robot
                    # in direction away from dock (negative direction)
                    norm = dist_to_dock
                    target_x = robot_x - (dx / norm) * desired_offset
                    target_y = robot_y - (dy / norm) * desired_offset
                else:
                    # Fallback: if robot is exactly at dock, place target 2.5m behind robot
                    target_x = robot_x - 2.5
                    target_y = robot_y

                if self._spawn_blue_box_at_dock(target_x, target_y):
                    self.get_logger().info(
                        f"[Final] All tasks complete! Docking target offset from DOCK_{self.task_class} "
                        f"at ({target_x:.2f}, {target_y:.2f})"
                    )
                else:
                    self.get_logger().warn("Failed to spawn blue box - completing without final docking")
                    self._reset_task_state()
            elif self._item_dropped_for_current_task and self.task_queue:
                # Item dropped and more tasks remaining - move to next task
                self.get_logger().info(f"[Task] {self.current_task} complete. {len(self.task_queue)} task(s) remaining.")
                self._reset_task_state()

        elif self.phase == "GO_DOCKING":
            # Check docking completion
            if self.docking_complete:
                # Final docking successful!
                self.get_logger().info(f"Completed final docking at DOCK_{self.task_class}")
                
                # Clean up blue box
                self._delete_blue_box()
                
                # Reset for next task (will be IDLE since queue is empty)
                self._reset_task_state()
                
            elif self.docking_start_time and (now - self.docking_start_time) > self.max_docking_time:
                # Timeout - final docking took too long
                self.get_logger().warn("Final docking timeout - completing anyway")
                self._delete_blue_box()
                self._reset_task_state()

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
