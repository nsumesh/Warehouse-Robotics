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
        
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')
        
        self.get_logger().info('Waiting for Gazebo services...')
        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Spawn service not available, waiting...')
        
        while not self.delete_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Delete service not available, waiting...')
        
        self.get_logger().info("Warehouse Object Spawner initialized")
    
    def generate_box_sdf(self, size, mass, color_rgba, friction=1.0):
        """Generate SDF for box"""
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
        
        request.initial_pose = Pose()
        request.initial_pose.position = Point(x=float(x), y=float(y), z=float(z))
        
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
                f"scattered_box_{i}",
                # Add lanes objects to delete list
                f"lane_pallet_{i}",
                f"lane_pallet_box_{i}_0",
                f"lane{0}_shelf_{i}", f"lane{1}_shelf_{i}",
                f"lane{0}_shelfbox_{i}_0", f"lane{0}_shelfbox_{i}_1",
                f"lane{1}_shelfbox_{i}_0", f"lane{1}_shelfbox_{i}_1"
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
        
        box_types = {
            'small': {'size': [0.1, 0.1, 0.1], 'mass': 0.3, 'color': [0.8, 0.3, 0.2, 1.0]},
            'medium': {'size': [0.15, 0.15, 0.15], 'mass': 0.5, 'color': [0.3, 0.6, 0.8, 1.0]},
            'large': {'size': [0.2, 0.2, 0.2], 'mass': 0.8, 'color': [0.2, 0.7, 0.3, 1.0]},
            'rectangular': {'size': [0.25, 0.15, 0.1], 'mass': 0.6, 'color': [0.9, 0.7, 0.2, 1.0]}
        }
        
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

    def spawn_warehouse_lanes(self):
        self.get_logger().info("Spawning warehouse layout...")
        lane_x_positions = [-3.0, 1.0]   # TWO aisles with more spacing (4.0m apart)
        num_shelves_per_lane = 3    # 3 shelves per aisle
        start_y = -2.0              # Start within workspace bounds
        spacing_y = 2.5             # 2.5m spacing between shelves (more room)

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

                num_boxes = 2
                for b in range(num_boxes):
                    bx = x - shelf_width / 2 + 0.3 + b * 0.35
                    by = y
                    bz = shelf_board_z + box_size[2] / 2 + 0.01

                    sdf = self.generate_box_sdf(box_size, box_mass, box_color)
                    box_name = f"lane{lane_idx}_shelfbox_{i}_{b}"
                    self.spawn_object(box_name, sdf, bx, by, bz)
                    time.sleep(0.1)

        pallet_y = -4.0  
        pallet_x_positions = [-3.0, 1.0]  

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
    spawner.get_logger().info("Clearing existing objects")
    spawner.clear_all_objects()
    time.sleep(1)
    spawner.spawn_warehouse_lanes()
    spawner.get_logger().info("Warehouse setup complete")
    try:
        rclpy.spin(spawner)
    except KeyboardInterrupt:
        pass
    
    spawner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()