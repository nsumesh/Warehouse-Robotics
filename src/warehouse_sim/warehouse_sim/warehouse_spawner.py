#!/usr/bin/env python3
"""
Warehouse World Generator with Pickable Objects - ROS 2 Version
Creates an empty warehouse and spawns various pickable boxes at different heights
"""

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose, Point, Quaternion
import random
import math
import time


class WarehouseObjectSpawner(Node):
    def __init__(self):
        super().__init__('warehouse_object_spawner')
        
        # Create service clients
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')
        
        # Wait for services to be available
        self.get_logger().info('Waiting for Gazebo services...')
        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Spawn service not available, waiting...')
        
        while not self.delete_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Delete service not available, waiting...')
        
        self.get_logger().info("Warehouse Object Spawner initialized")
    
    def generate_box_sdf(self, size, mass, color_rgba, friction=1.0):
        """
        Generate SDF for a pickable box
        
        Args:
            size: [x, y, z] dimensions in meters
            mass: mass in kg
            color_rgba: [r, g, b, a] color values (0-1)
            friction: friction coefficient
        """
        # Calculate inertia (for box: I = m/12 * (h^2 + d^2))
        ixx = (mass / 12.0) * (size[1]**2 + size[2]**2)
        iyy = (mass / 12.0) * (size[0]**2 + size[2]**2)
        izz = (mass / 12.0) * (size[0]**2 + size[1]**2)
        
        sdf = f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='pickable_box'>
    <pose>0 0 0 0 0 0</pose>
    <link name='link'>
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
      <collision name='collision'>
        <geometry>
          <box>
            <size>{size[0]} {size[1]} {size[2]}</size>
          </box>
        </geometry>
        <surface>
          <contact>
            <ode>
              <kp>1000000.0</kp>
              <kd>100.0</kd>
              <max_vel>0.01</max_vel>
              <min_depth>0.001</min_depth>
            </ode>
          </contact>
          <friction>
            <ode>
              <mu>{friction}</mu>
              <mu2>{friction}</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
      <visual name='visual'>
        <geometry>
          <box>
            <size>{size[0]} {size[1]} {size[2]}</size>
          </box>
        </geometry>
        <material>
          <ambient>{color_rgba[0]} {color_rgba[1]} {color_rgba[2]} {color_rgba[3]}</ambient>
          <diffuse>{color_rgba[0]} {color_rgba[1]} {color_rgba[2]} {color_rgba[3]}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
        return sdf
    
    def generate_shelf_sdf(self, width, depth, height, shelf_height):
        """Generate SDF for a simple shelf structure"""
        sdf = f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='shelf'>
    <static>true</static>
    <link name='base'>
      <!-- Left vertical support -->
      <collision name='left_support_collision'>
        <pose>0 0 {height/2} 0 0 0</pose>
        <geometry>
          <box>
            <size>0.05 {depth} {height}</size>
          </box>
        </geometry>
      </collision>
      <visual name='left_support_visual'>
        <pose>0 0 {height/2} 0 0 0</pose>
        <geometry>
          <box>
            <size>0.05 {depth} {height}</size>
          </box>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1</ambient>
          <diffuse>0.5 0.5 0.5 1</diffuse>
        </material>
      </visual>
      
      <!-- Right vertical support -->
      <collision name='right_support_collision'>
        <pose>{width} 0 {height/2} 0 0 0</pose>
        <geometry>
          <box>
            <size>0.05 {depth} {height}</size>
          </box>
        </geometry>
      </collision>
      <visual name='right_support_visual'>
        <pose>{width} 0 {height/2} 0 0 0</pose>
        <geometry>
          <box>
            <size>0.05 {depth} {height}</size>
          </box>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1</ambient>
          <diffuse>0.5 0.5 0.5 1</diffuse>
        </material>
      </visual>
      
      <!-- Shelf surface -->
      <collision name='shelf_collision'>
        <pose>{width/2} 0 {shelf_height} 0 0 0</pose>
        <geometry>
          <box>
            <size>{width} {depth} 0.02</size>
          </box>
        </geometry>
      </collision>
      <visual name='shelf_visual'>
        <pose>{width/2} 0 {shelf_height} 0 0 0</pose>
        <geometry>
          <box>
            <size>{width} {depth} 0.02</size>
          </box>
        </geometry>
        <material>
          <ambient>0.7 0.5 0.3 1</ambient>
          <diffuse>0.7 0.5 0.3 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
        return sdf
    
    def spawn_object(self, name, sdf_string, x, y, z, roll=0, pitch=0, yaw=0):
        """Spawn an object in Gazebo"""
        request = SpawnEntity.Request()
        request.name = name
        request.xml = sdf_string
        request.robot_namespace = ""
        request.reference_frame = "world"
        
        # Set pose
        request.initial_pose = Pose()
        request.initial_pose.position = Point(x=float(x), y=float(y), z=float(z))
        
        # Convert RPY to quaternion
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        request.initial_pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        
        # Call service
        future = self.spawn_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        
        if future.result() is not None:
            self.get_logger().info(f"Spawned {name} at ({x:.2f}, {y:.2f}, {z:.2f})")
            return True
        else:
            self.get_logger().error(f"Failed to spawn {name}")
            return False
    
    def clear_all_objects(self):
        """Delete all spawned objects"""
        objects_to_delete = []
        
        # Collect all object names
        for i in range(100):
            objects_to_delete.extend([
                f"ground_box_{i}",
                f"low_shelf_box_{i}_0",
                f"low_shelf_box_{i}_1",
                f"medium_shelf_box_{i}_0",
                f"medium_shelf_box_{i}_1",
                f"medium_shelf_box_{i}_2",
                f"high_shelf_box_{i}_0",
                f"high_shelf_box_{i}_1",
                f"shelf_low_{i}",
                f"shelf_medium_{i}",
                f"shelf_high_{i}",
                f"pallet_{i}",
                f"pallet_box_{i}_0",
                f"pallet_box_{i}_1",
                f"pallet_box_{i}_2",
                f"pallet_box_{i}_3",
                f"scattered_box_{i}"
            ])
        
        for obj_name in objects_to_delete:
            request = DeleteEntity.Request()
            request.name = obj_name
            try:
                self.delete_client.call_async(request)
            except:
                pass
    
    def spawn_warehouse_environment(self):
        """Spawn complete warehouse with shelves and pickable boxes"""
        
        self.get_logger().info("Creating warehouse environment...")
        
        # Box type definitions
        box_types = {
            'small': {'size': [0.1, 0.1, 0.1], 'mass': 0.3, 'color': [0.8, 0.3, 0.2, 1.0]},
            'medium': {'size': [0.15, 0.15, 0.15], 'mass': 0.5, 'color': [0.3, 0.6, 0.8, 1.0]},
            'large': {'size': [0.2, 0.2, 0.2], 'mass': 0.8, 'color': [0.2, 0.7, 0.3, 1.0]},
            'rectangular': {'size': [0.25, 0.15, 0.1], 'mass': 0.6, 'color': [0.9, 0.7, 0.2, 1.0]}
        }
        
        # 1. GROUND LEVEL BOXES
        self.get_logger().info("Spawning ground level boxes...")
        ground_positions = [
            (2, 2), (2, -2), (-2, 2), (-2, -2),
            (4, 0), (-4, 0), (0, 4), (0, -4),
            (3, 3), (-3, -3)
        ]
        
        for idx, (x, y) in enumerate(ground_positions):
            box_type = random.choice(list(box_types.keys()))
            box_config = box_types[box_type]
            z = box_config['size'][2] / 2
            
            sdf = self.generate_box_sdf(
                box_config['size'],
                box_config['mass'],
                box_config['color']
            )
            self.spawn_object(f"ground_box_{idx}", sdf, x, y, z)
            time.sleep(0.1)
        
        # 2. LOW SHELVES with boxes
        self.get_logger().info("Spawning low shelves with boxes...")
        low_shelf_configs = [
            {'x': 6, 'y': 2, 'yaw': 0},
            {'x': 6, 'y': -2, 'yaw': 0},
            {'x': -6, 'y': 2, 'yaw': 0},
            {'x': -6, 'y': -2, 'yaw': 0}
        ]
        
        for idx, config in enumerate(low_shelf_configs):
            shelf_sdf = self.generate_shelf_sdf(
                width=1.0, depth=0.4, height=0.8, shelf_height=0.5
            )
            self.spawn_object(
                f"shelf_low_{idx}",
                shelf_sdf,
                config['x'], config['y'], 0,
                yaw=config['yaw']
            )
            time.sleep(0.1)
            
            for box_idx in range(2):
                box_type = random.choice(['small', 'medium'])
                box_config = box_types[box_type]
                
                box_x = config['x'] + 0.25 + (box_idx * 0.3)
                box_y = config['y']
                box_z = 0.5 + box_config['size'][2] / 2 + 0.01
                
                sdf = self.generate_box_sdf(
                    box_config['size'],
                    box_config['mass'],
                    box_config['color']
                )
                self.spawn_object(
                    f"low_shelf_box_{idx}_{box_idx}",
                    sdf, box_x, box_y, box_z
                )
                time.sleep(0.1)
        
        # 3. MEDIUM SHELVES with boxes
        self.get_logger().info("Spawning medium shelves with boxes...")
        medium_shelf_configs = [
            {'x': 8, 'y': 0, 'yaw': 0},
            {'x': -8, 'y': 0, 'yaw': 0},
            {'x': 0, 'y': 6, 'yaw': 1.57},
            {'x': 0, 'y': -6, 'yaw': 1.57}
        ]
        
        for idx, config in enumerate(medium_shelf_configs):
            shelf_sdf = self.generate_shelf_sdf(
                width=1.2, depth=0.4, height=1.5, shelf_height=1.0
            )
            self.spawn_object(
                f"shelf_medium_{idx}",
                shelf_sdf,
                config['x'], config['y'], 0,
                yaw=config['yaw']
            )
            time.sleep(0.1)
            
            for box_idx in range(3):
                box_type = random.choice(['small', 'medium', 'rectangular'])
                box_config = box_types[box_type]
                
                box_x = config['x'] + 0.2 + (box_idx * 0.25) if config['yaw'] == 0 else config['x']
                box_y = config['y'] if config['yaw'] == 0 else config['y'] + 0.2 + (box_idx * 0.25)
                box_z = 1.0 + box_config['size'][2] / 2 + 0.01
                
                sdf = self.generate_box_sdf(
                    box_config['size'],
                    box_config['mass'],
                    box_config['color']
                )
                self.spawn_object(
                    f"medium_shelf_box_{idx}_{box_idx}",
                    sdf, box_x, box_y, box_z
                )
                time.sleep(0.1)
        
        # 4. HIGH SHELVES with boxes
        self.get_logger().info("Spawning high shelves with boxes...")
        high_shelf_configs = [
            {'x': 10, 'y': 3, 'yaw': 0},
            {'x': -10, 'y': -3, 'yaw': 0}
        ]
        
        for idx, config in enumerate(high_shelf_configs):
            shelf_sdf = self.generate_shelf_sdf(
                width=1.5, depth=0.5, height=2.0, shelf_height=1.5
            )
            self.spawn_object(
                f"shelf_high_{idx}",
                shelf_sdf,
                config['x'], config['y'], 0,
                yaw=config['yaw']
            )
            time.sleep(0.1)
            
            for box_idx in range(2):
                box_type = 'small'
                box_config = box_types[box_type]
                
                box_x = config['x'] + 0.3 + (box_idx * 0.4)
                box_y = config['y']
                box_z = 1.5 + box_config['size'][2] / 2 + 0.01
                
                sdf = self.generate_box_sdf(
                    box_config['size'],
                    box_config['mass'],
                    box_config['color']
                )
                self.spawn_object(
                    f"high_shelf_box_{idx}_{box_idx}",
                    sdf, box_x, box_y, box_z
                )
                time.sleep(0.1)
        
        # 5. PALLET BOXES
        self.get_logger().info("Spawning pallet boxes...")
        pallet_positions = [
            (5, 5), (5, -5), (-5, 5), (-5, -5)
        ]
        
        for idx, (x, y) in enumerate(pallet_positions):
            pallet_sdf = f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='pallet'>
    <static>true</static>
    <link name='link'>
      <collision name='collision'>
        <geometry>
          <box><size>1.0 1.0 0.15</size></box>
        </geometry>
      </collision>
      <visual name='visual'>
        <geometry>
          <box><size>1.0 1.0 0.15</size></box>
        </geometry>
        <material>
          <ambient>0.6 0.4 0.2 1</ambient>
          <diffuse>0.6 0.4 0.2 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
            self.spawn_object(f"pallet_{idx}", pallet_sdf, x, y, 0.075)
            time.sleep(0.1)
            
            for box_idx in range(4):
                box_type = random.choice(['medium', 'large'])
                box_config = box_types[box_type]
                
                offset_x = -0.25 + (box_idx % 2) * 0.5
                offset_y = -0.25 + (box_idx // 2) * 0.5
                
                box_x = x + offset_x
                box_y = y + offset_y
                box_z = 0.15 + box_config['size'][2] / 2 + 0.01
                
                sdf = self.generate_box_sdf(
                    box_config['size'],
                    box_config['mass'],
                    box_config['color']
                )
                self.spawn_object(
                    f"pallet_box_{idx}_{box_idx}",
                    sdf, box_x, box_y, box_z
                )
                time.sleep(0.1)
        
        # 6. SCATTERED BOXES
        self.get_logger().info("Spawning scattered boxes...")
        for idx in range(15):
            x = random.uniform(-9, 9)
            y = random.uniform(-9, 9)
            
            if abs(x) < 1 and abs(y) < 1:
                continue
            
            box_type = random.choice(list(box_types.keys()))
            box_config = box_types[box_type]
            z = box_config['size'][2] / 2
            yaw = random.uniform(0, 2 * math.pi)
            
            sdf = self.generate_box_sdf(
                box_config['size'],
                box_config['mass'],
                box_config['color']
            )
            self.spawn_object(
                f"scattered_box_{idx}",
                sdf, x, y, z, yaw=yaw
            )
            time.sleep(0.1)
        
        self.get_logger().info("Warehouse environment created successfully!")
        self.get_logger().info("Object categories:")
        self.get_logger().info("  - Ground boxes: Easy to reach")
        self.get_logger().info("  - Low shelf boxes (0.5m): Moderate difficulty")
        self.get_logger().info("  - Medium shelf boxes (1.0m): Challenging")
        self.get_logger().info("  - High shelf boxes (1.5m): Very challenging")
        self.get_logger().info("  - Pallet boxes (0.15m): Slightly elevated")
        self.get_logger().info("  - Scattered boxes: Various positions")
    # def spawn_clustered_warehouse(self):
    #   """
    #   Create a realistic warehouse using 3 natural clusters:
    #   - Cluster A: North dense storage
    #   - Cluster B: Center racks (symmetrical)
    #   - Cluster C: South mixed shelves + pallets
    #   """
    #   self.get_logger().info("Spawning clustered warehouse layout...")

    #   # ============= CLUSTER A (North storage zone) =============
    #   clusterA_origin = (0, 8)   # centered, y=8
    #   clusterA_rows = 2
    #   clusterA_cols = 4
    #   shelf_width = 1.2
    #   spacing_x = 2.0
    #   spacing_y = 1.5

    #   for r in range(clusterA_rows):
    #       for c in range(clusterA_cols):
    #           x = clusterA_origin[0] + (c - clusterA_cols/2) * spacing_x
    #           y = clusterA_origin[1] + r * spacing_y

    #           shelf_sdf = self.generate_shelf_sdf(
    #               width=shelf_width, depth=0.45, height=1.7, shelf_height=1.0
    #           )

    #           name = f"clusterA_shelf_{r}_{c}"
    #           self.spawn_object(name, shelf_sdf, x, y, 0)

    #           # Add boxes on this shelf
    #           for b in range(3):
    #               box_type = random.choice(['small', 'medium', 'rectangular'])
    #               config = random.choice([
    #                   ([0.12,0.12,0.12],0.4,[0.9,0.3,0.3,1]),
    #                   ([0.15,0.15,0.15],0.5,[0.3,0.7,0.9,1]),
    #                   ([0.25,0.15,0.10],0.6,[0.9,0.7,0.2,1])
    #               ])
    #               box_x = x - 0.3 + b * 0.3
    #               box_y = y
    #               box_z = 1.0 + config[0][2]/2 + 0.01

    #               sdf = self.generate_box_sdf(config[0], config[1], config[2])
    #               self.spawn_object(f"clusterA_box_{r}_{c}_{b}", sdf, box_x, box_y, box_z)



    #   # ============= CLUSTER B (Center island racks) =============
    #   clusterB_positions = [
    #       (-4, 0), (0, 0), (4, 0),
    #       (-4, -2), (0, -2), (4, -2)
    #   ]

    #   for idx, (x, y) in enumerate(clusterB_positions):
    #       shelf_sdf = self.generate_shelf_sdf(
    #           width=1.0, depth=0.40, height=1.3, shelf_height=0.9
    #       )
    #       self.spawn_object(f"clusterB_shelf_{idx}", shelf_sdf, x, y, 0)

    #       # Boxes on shelves
    #       for b in range(2):
    #           size = [0.15, 0.15, 0.12]
    #           mass = 0.4
    #           color = [0.6, 0.6, 0.9, 1]
    #           z = 0.9 + size[2]/2 + 0.01
    #           bx = x - 0.2 + b*0.3
    #           by = y

    #           sdf = self.generate_box_sdf(size, mass, color)
    #           self.spawn_object(f"clusterB_box_{idx}_{b}", sdf, bx, by, z)



    #   # ============= CLUSTER C (South zone: mixed storage + pallets) =============
    #   clusterC_shelves = [
    #       (-5, -7), (-3, -7), (-1, -7), (1, -7), (3, -7)
    #   ]

    #   for idx, (x, y) in enumerate(clusterC_shelves):
    #       shelf_sdf = self.generate_shelf_sdf(
    #           width=1.4, depth=0.45, height=1.6, shelf_height=1.1
    #       )
    #       self.spawn_object(f"clusterC_shelf_{idx}", shelf_sdf, x, y, 0)

    #   # Pallet zone
    #   pallet_spots = [
    #       (-4, -9), (-2, -9), (2, -9), (4, -9)
    #   ]

    #   for idx, (x, y) in enumerate(pallet_spots):
    #       pallet_sdf = """
    #   <sdf version='1.6'>
    #   <model name='pallet'>
    #   <static>true</static>
    #   <link name='link'>
    #   <visual name='v'><geometry><box><size>1 1 0.15</size></box></geometry></visual>
    #   <collision name='c'><geometry><box><size>1 1 0.15</size></box></geometry></collision>
    #   </link>
    #   </model>
    #   </sdf>
    #   """
    #       self.spawn_object(f"clusterC_pallet_{idx}", pallet_sdf, x, y, 0.075)

    #       # Boxes on pallet
    #       for b in range(3):
    #           size = [0.2, 0.2, 0.18]
    #           mass = 0.7
    #           color = [0.8, 0.5, 0.2, 1]
    #           bx = x - 0.25 + (b * 0.25)
    #           by = y
    #           bz = 0.15 + size[2]/2 + 0.01

    #           sdf = self.generate_box_sdf(size, mass, color)
    #           self.spawn_object(f"clusterC_pallet_box_{idx}_{b}", sdf, bx, by, bz)



    #   self.get_logger().info("Clustered warehouse created successfully!")
    def spawn_warehouse_lanes(self):
        """Spawn simplified warehouse layout: two aisles with 3 shelves each, minimal obstacles."""
        self.get_logger().info("Spawning simplified warehouse layout...")

        # SIMPLIFIED: Two aisles, fewer shelves for limited workspace
        lane_x_positions = [-3.0, 1.0]   # TWO aisles with more spacing (4.0m apart)
        num_shelves_per_lane = 3    # 3 shelves per aisle
        start_y = -2.0              # Start within workspace bounds
        spacing_y = 2.5              # 2.5m spacing between shelves (more room)

        shelf_width = 1.4
        shelf_depth = 0.45
        shelf_height = 1.6
        shelf_board_z = 1.1   # height of the shelf surface

        # Simple box config for shelf boxes
        box_size = [0.18, 0.18, 0.16]
        box_mass = 0.6
        box_color = [0.9, 0.6, 0.2, 1.0]

        # ----- Two aisles with 3 shelves each -----
        for lane_idx, x in enumerate(lane_x_positions):
            for i in range(num_shelves_per_lane):
                y = start_y + i * spacing_y  # y = -2.0, 0.0, 2.0

                shelf_sdf = self.generate_shelf_sdf(
                    width=shelf_width,
                    depth=shelf_depth,
                    height=shelf_height,
                    shelf_height=shelf_board_z,
                )

                shelf_name = f"lane{lane_idx}_shelf_{i}"
                self.spawn_object(shelf_name, shelf_sdf, x, y, 0.0, yaw=0.0)
                time.sleep(0.1)

                # 2 boxes per shelf (was 3) for simpler layout
                num_boxes = 2
                for b in range(num_boxes):
                    bx = x - shelf_width / 2 + 0.3 + b * 0.35
                    by = y
                    bz = shelf_board_z + box_size[2] / 2 + 0.01

                    sdf = self.generate_box_sdf(box_size, box_mass, box_color)
                    box_name = f"lane{lane_idx}_shelfbox_{i}_{b}"
                    self.spawn_object(box_name, sdf, bx, by, bz)
                    time.sleep(0.1)

        # ----- Two pallets at one end (one per aisle) -----
        pallet_y = -4.0  # Closer to workspace (was -9.0)
        pallet_x_positions = [-3.0, 1.0]  # Two pallets (one per aisle, matching aisle positions)

        for idx, x in enumerate(pallet_x_positions):
            pallet_sdf = """<?xml version='1.0'?>
        <sdf version='1.6'>
        <model name='pallet'>
        <static>true</static>
        <link name='link'>
        <collision name='collision'>
        <geometry><box><size>1.0 1.0 0.15</size></box></geometry>
        </collision>
        <visual name='visual'>
        <geometry><box><size>1.0 1.0 0.15</size></box></geometry>
        <material>
          <ambient>0.7 0.7 0.7 1</ambient>
          <diffuse>0.7 0.7 0.7 1</diffuse>
        </material>
        </visual>
        </link>
        </model>
        </sdf>
        """
            self.spawn_object(f"lane_pallet_{idx}", pallet_sdf, x, pallet_y, 0.075)
            time.sleep(0.1)

            # 1 box on pallet (was 2) for minimal clutter
            size = [0.25, 0.20, 0.18]
            mass = 0.8
            color = [0.8, 0.5, 0.2, 1]
            bx = x
            by = pallet_y
            bz = 0.15 + size[2] / 2 + 0.01

            sdf = self.generate_box_sdf(size, mass, color)
            self.spawn_object(f"lane_pallet_box_{idx}_0", sdf, bx, by, bz)
            time.sleep(0.1)

        self.get_logger().info("Simplified warehouse created successfully!")
        self.get_logger().info("Layout: 2 aisles, 6 shelves, 12 boxes, 2 pallets (22 objects total)")



def main(args=None):
    rclpy.init(args=args)
    
    spawner = WarehouseObjectSpawner()
    # Optional: Clear existing objects
    spawner.get_logger().info("Clearing existing objects...")
    spawner.clear_all_objects()
    time.sleep(1)
    
    # Spawn warehouse environment
    spawner.spawn_warehouse_lanes()
    
    spawner.get_logger().info("Warehouse setup complete!")
    
    # Keep node alive
    try:
        rclpy.spin(spawner)
    except KeyboardInterrupt:
        pass
    
    spawner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
