#!/usr/bin/env python3
"""
Sorting Node with Stage 3 PPO Navigation

Manages virtual sorting tasks using trained Stage 3 PPO policy.
Task classes: A, B, C (mapped to DOCK_A, DOCK_B, DOCK_C)
FSM: IDLE → GO_PICKUP → GO_DROPOFF → IDLE
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
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from stable_baselines3 import PPO

# Import constants and spawn function from train_ppo.py
from rl_nav.train_ppo import DOCK_A, DOCK_B, DOCK_C, PICKUP, SUCCESS_RADIUS, spawn_tb3, X_MIN, X_MAX, Y_MIN, Y_MAX


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
        self.actions = [(0.12, 0.6), (0.15, 0.0), (0.12, -0.6), (0.00, 0.6), (0.00, -0.6)]

        # State
        self.scan = None
        self.pose = np.zeros(3, dtype=np.float32)
        self.current_goal = None
        self.task_class = None  # 'A', 'B', or 'C'
        self.goal_reached_threshold = 0.7  # Closer to training threshold (0.6m)
        self._goal_reached_time = None  # Track when goal was first reached (for stability check)
        self.task_queue = []
        self.current_task = None
        self.phase = "IDLE"  # IDLE → GO_PICKUP → GO_DROPOFF → IDLE
        self.task_start_time = None
        self.max_task_time = 180.0  # Increased timeout for better navigation
        self._last_log_time = None  # For debug logging
        self._last_stuck_check = None  # For stuck detection
        self._last_stuck_dist = None  # For stuck detection
        self._collision_check_enabled = False  # Delay collision checking
        self._start_time = None  # Track when node started

        # Object tracking: maps task_class -> list of item IDs
        self.items_at_pickup = {}  # Maps task_class -> list of item IDs
        self.active_items = {}  # Maps item_id -> {'task_class': 'A', 'spawned': True, 'picked': False}
        self.item_counter = {'A': 0, 'B': 0, 'C': 0}  # Counter for unique IDs
        self.current_item_id = None  # Store current item ID for dropoff

        # Initialize
        self._build_initial_tasks(5)
        # Spawn items at pickup point
        self._spawn_initial_items()
        self.control_timer = self.create_timer(0.15, self.control_step)
        self.task_timer = self.create_timer(0.5, self.task_manager)
        self.get_logger().info(f"SortingNode initialized. Task queue: {self.task_queue}")

    def _scan_cb(self, msg):
        """Process LiDAR scan into 24 bins (match training)."""
        rng = np.array(msg.ranges, dtype=np.float32)
        n, bins = len(rng), 24
        step = max(1, n // bins)
        scan_bins = []
        for i in range(0, n, step):
            v = rng[i]
            if math.isnan(v) or v <= 0:
                v = 3.5
            scan_bins.append(min(v, 3.5) / 3.5)
            if len(scan_bins) == bins:
                break
        while len(scan_bins) < bins:
            scan_bins.append(1.0)
        self.scan = np.array(scan_bins, dtype=np.float32)

    def _odom_cb(self, msg):
        """Update robot pose."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose[:] = (x, y, yaw)

    def _obs(self):
        """Build 30-dim observation: 24 scan + [dx, dy, yaw] + task_class one-hot."""
        if self.scan is None or self.current_goal is None:
            return None
        
        # LiDAR bins (24 dims)
        scan = self.scan
        
        # Goal relative pose (3 dims)
        dx, dy = (np.array(self.current_goal) - self.pose[:2])
        tail = np.array([dx, dy, self.pose[2]], dtype=np.float32)
        
        # Task class one-hot (3 dims)
        # During GO_PICKUP: use dummy [1,0,0] (model should still work for basic navigation)
        # During GO_DROPOFF: use actual task class for Stage 3 model
        if self.phase == "GO_DROPOFF" and self.task_class:
            if self.task_class == 'A':
                task_onehot = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            elif self.task_class == 'B':
                task_onehot = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            elif self.task_class == 'C':
                task_onehot = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                task_onehot = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            # GO_PICKUP or no task class: use dummy encoding
            task_onehot = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        return np.concatenate([scan, tail, task_onehot], axis=0)

    def _build_initial_tasks(self, num_tasks=5):
        """Build task queue and assign item IDs."""
        categories = ['A', 'B', 'C']
        self.task_queue = [random.choice(categories) for _ in range(num_tasks)]
        
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
        if self.current_goal is None:
            return float('inf')
        dx, dy = (np.array(self.current_goal) - self.pose[:2])
        return float(math.hypot(dx, dy))

    def _goal_reached(self):
        """Check if goal reached (with stability check)."""
        dist = self._distance_to_goal()
        if dist < self.goal_reached_threshold:
            # Goal is close - check if it's been close for at least 2 seconds
            now = time.time()
            if self._goal_reached_time is None:
                self._goal_reached_time = now
            elif now - self._goal_reached_time >= 3.0:  # Stable for 3 seconds
                return True
        else:
            # Reset timer if we're not close anymore
            self._goal_reached_time = None
        return False

    def _check_collision(self):
        """Check if robot is too close to obstacles."""
        if self.scan is None:
            return False
        
        # Ensure scan is valid (not all NaN or invalid)
        valid_scan = self.scan[~np.isnan(self.scan)]
        if len(valid_scan) == 0:
            return False
        
        # Check if any LiDAR reading is very close (collision threshold)
        min_dist = np.min(valid_scan) * 3.5  # Convert normalized to meters
        return min_dist < 0.15  # Reduced from 0.2 to 0.15m (15cm threshold)

    def _check_stuck(self):
        """Check if robot is stuck (distance not improving)."""
        # Don't check for stuck if very close to goal (robot might be fine-tuning position)
        current_dist = self._distance_to_goal()
        if current_dist < 1.0:
            return False
        
        if self._last_stuck_check is None:
            self._last_stuck_check = time.time()
            self._last_stuck_dist = current_dist
            return False
        
        # Check every 15 seconds (more lenient)
        if time.time() - self._last_stuck_check > 15.0:
            # If robot is close to goal (< 1.5m), be more lenient with stuck detection
            if current_dist < 1.5:
                # When close, require less improvement (0.1m instead of 0.2m)
                improvement_threshold = 0.1
            else:
                # When far, require more improvement
                improvement_threshold = 0.2
            
            # If distance hasn't improved by threshold in 15 seconds, consider stuck
            if abs(current_dist - self._last_stuck_dist) < improvement_threshold:
                self.get_logger().warn(f"Robot appears stuck: dist={current_dist:.2f}m (no improvement)")
                return True
            self._last_stuck_check = time.time()
            self._last_stuck_dist = current_dist
        return False

    def _generate_item_sdf(self, item_name, color_rgba):
        """Generate SDF for a sortable item box."""
        size = [0.2, 0.2, 0.3]  # 20cm x 20cm x 30cm box
        mass = 0.5
        
        # Calculate inertia (for box: I = m/12 * (h^2 + d^2))
        ixx = (mass / 12.0) * (size[1]**2 + size[2]**2)
        iyy = (mass / 12.0) * (size[0]**2 + size[2]**2)
        izz = (mass / 12.0) * (size[0]**2 + size[1]**2)
        
        sdf = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{item_name}">
    <static>false</static>
    <pose>0 0 0 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{ixx}</ixx>
          <iyy>{iyy}</iyy>
          <izz>{izz}</izz>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry><box><size>{size[0]} {size[1]} {size[2]}</size></box></geometry>
      </collision>
      <visual name="visual">
        <geometry><box><size>{size[0]} {size[1]} {size[2]}</size></box></geometry>
        <material>
          <ambient>{color_rgba[0]} {color_rgba[1]} {color_rgba[2]} {color_rgba[3]}</ambient>
          <diffuse>{color_rgba[0]} {color_rgba[1]} {color_rgba[2]} {color_rgba[3]}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
        return sdf

    def _spawn_item(self, item_name, item_sdf, x, y, z):
        """Spawn a single item in Gazebo."""
        from geometry_msgs.msg import Pose, Point, Quaternion
        
        req = SpawnEntity.Request()
        req.name = item_name
        req.xml = item_sdf
        req.robot_namespace = ""
        req.reference_frame = "world"
        
        # Set pose - required for items to spawn at correct location
        req.initial_pose = Pose()
        req.initial_pose.position = Point(x=float(x), y=float(y), z=float(z))
        req.initial_pose.orientation = Quaternion(w=1.0)  # No rotation
        
        future = self.spawn_client.call_async(req)
        try:
            rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
            if future.result() and future.result().success:
                return True
        except Exception as e:
            self.get_logger().warn(f"Exception spawning {item_name}: {e}")
        return False

    def _spawn_initial_items(self):
        """Spawn all items at pickup point based on task queue."""
        if not self.spawn_client.service_is_ready():
            self.get_logger().warn("Spawn service not ready - items won't be spawned")
            return
        
        self.get_logger().info(f"Attempting to spawn items: {self.items_at_pickup}")
        pickup_x, pickup_y = self.pickup_location
        
        # Color mapping for visual distinction
        colors = {
            'A': [0.8, 0.2, 0.2, 1.0],  # Red
            'B': [0.2, 0.8, 0.2, 1.0],  # Green
            'C': [0.2, 0.2, 0.8, 1.0],  # Blue
        }
        
        # Spawn items in a grid pattern at pickup point
        spacing = 0.3  # 30cm spacing between items
        items_per_row = 3
        
        for task_class, item_list in self.items_at_pickup.items():
            for idx, item_id in enumerate(item_list):
                row = idx // items_per_row
                col = idx % items_per_row
                
                # Calculate position offset from pickup center
                offset_x = (col - items_per_row/2 + 0.5) * spacing
                offset_y = row * spacing
                
                item_x = pickup_x + offset_x
                item_y = pickup_y + offset_y
                item_z = 0.15  # Half of box height (0.3m box)
                
                # Generate SDF for this item
                item_sdf = self._generate_item_sdf(item_id, colors[task_class])
                
                # Spawn item
                if self._spawn_item(item_id, item_sdf, item_x, item_y, item_z):
                    self.active_items[item_id]['spawned'] = True
                    self.get_logger().info(f"Spawned {item_id} at pickup ({item_x:.2f}, {item_y:.2f})")
                else:
                    self.get_logger().warn(f"Failed to spawn {item_id}")

    def _reset_robot_position(self):
        """Reset robot to a safe position when stuck."""
        from gazebo_msgs.srv import SetEntityState
        from gazebo_msgs.msg import EntityState
        
        reset_cli = self.create_client(SetEntityState, "/set_entity_state")
        if not reset_cli.wait_for_service(timeout_sec=1.0):  # Quick check
            self.get_logger().warn("Reset service not available - skipping reset")
            return False
        
        # Reset to workspace center
        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = "tb3"
        req.state.pose.position.x = (X_MIN + X_MAX) / 2.0
        req.state.pose.position.y = (Y_MIN + Y_MAX) / 2.0
        req.state.pose.position.z = 0.1
        req.state.pose.orientation.w = 1.0
        
        future = reset_cli.call_async(req)
        
        # Wait with shorter timeout and don't block
        try:
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)  # Shorter timeout
        except Exception as e:
            self.get_logger().warn(f"Exception waiting for reset service: {e}")
            return False
        
        # Check if future is done
        if not future.done():
            self.get_logger().warn("Reset service call timed out - service may be slow")
            return False
        
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().warn(f"Exception getting reset result: {e}")
            return False
            
        if result is None:
            self.get_logger().warn("Reset service call returned None")
            return False
        
        if result.success:
            self.get_logger().info("Robot position reset to workspace center")
            # Reset stuck detection after successful reset
            self._last_stuck_check = None
            self._last_stuck_dist = None
            return True
        else:
            self.get_logger().warn("Reset failed: service returned success=False")
            return False

    def _virtual_pickup(self, item_name, task_class):
        """Virtual pickup: delete item from pickup zone."""
        # Mark item as picked in tracking
        if item_name in self.active_items:
            self.active_items[item_name]['picked'] = True
        
        if not self.delete_client.service_is_ready():
            self.get_logger().warn("Gazebo service not ready - simulating pickup")
            return True
        
        req = DeleteEntity.Request()
        req.name = item_name
        future = self.delete_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        
        if future.result() and future.result().success:
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
        
        # Use same color as pickup for consistency
        colors = {
            'A': [0.8, 0.2, 0.2, 1.0],  # Red
            'B': [0.2, 0.8, 0.2, 1.0],  # Green
            'C': [0.2, 0.2, 0.8, 1.0],  # Blue
        }
        
        # Generate SDF with same properties as pickup item
        item_sdf = self._generate_item_sdf(item_id, colors[task_class])
        
        # Spawn at dock location
        if self._spawn_item(item_id, item_sdf, dock_x, dock_y, 0.15):
            self.get_logger().info(f"Dropped {task_class} item ({item_id}) at DOCK_{task_class} ({dock_x:.1f}, {dock_y:.1f})")
            return True
        else:
            self.get_logger().warn(f"Failed to spawn at dock - simulating dropoff")
            return True  # Continue anyway

    def task_manager(self):
        """Manage task queue and FSM transitions."""
        now = time.time()

        # Check for collision or stuck (only during active navigation)
        if self.phase in ["GO_PICKUP", "GO_DROPOFF"]:
            # Enable collision checking after 2 seconds (let robot settle)
            if self._start_time is None:
                self._start_time = time.time()
            if time.time() - self._start_time >= 2.0:
                self._collision_check_enabled = True
            
            # Only check collisions after delay
            if self._collision_check_enabled and self._check_collision():
                self.get_logger().warn("Collision detected! Resetting robot position...")
                if self._reset_robot_position():
                    # Reset task start time to give it another chance
                    self.task_start_time = time.time()
                    return
                else:
                    # If reset fails, skip to next phase
                    self.get_logger().warn("Reset failed, advancing phase")
                    self._advance_phase()
                    return
            
            if self._check_stuck():
                self.get_logger().warn("Robot stuck! Resetting position...")
                if self._reset_robot_position():
                    self.task_start_time = time.time()
                    return
                else:
                    self.get_logger().warn("Reset failed, skipping task")
                    self._advance_phase()
                    return

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
            self.get_logger().info(f"[Task] {self.current_task}: Going to PICKUP @ ({self.pickup_location[0]:.1f}, {self.pickup_location[1]:.1f})")

        elif self.phase == "GO_PICKUP" and self._goal_reached():
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

        elif self.phase == "GO_DROPOFF" and self._goal_reached():
            # Reached dock: drop off item
            item_id = getattr(self, 'current_item_id', None)
            self._virtual_dropoff(self.task_class, item_id)
            self.get_logger().info(f"[Task] ✓ Completed sorting {self.current_task} item")
            
            # Reset for next task
            self.current_task = None
            self.task_class = None
            self.current_item_id = None
            self.phase = "IDLE"
            self.current_goal = None
            self.task_start_time = None
            
            if not self.task_queue:
                self.get_logger().info("All tasks completed!")

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

    def control_step(self):
        """PPO control loop."""
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
                req = DeleteEntity.Request()
                req.name = item_id
                future = self.delete_client.call_async(req)
                try:
                    rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
                    if future.result() and future.result().success:
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
