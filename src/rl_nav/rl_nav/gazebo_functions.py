'''
gazebo_functions.py : This set up the functions for robot spawning, entity management, position reset and the docking marker placement
'''

import os
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, SetEntityState
from gazebo_msgs.msg import EntityState
from rl_nav.constants import warehouse_x_limit_min, warehouse_x_limit_max, warehouse_y_limit_min, warehouse_y_limit_max


def robot_initilization(node, spawn_x=None, spawn_y=None):
    robot_path = os.path.expanduser("~/turtlebot3_ws/install/turtlebot3_gazebo/share/" "turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf")
    if not os.path.exists(robot_path):
        node.get_logger().error("WafflePi cannot be found")
        return False
    with open(robot_path, "r") as f:
        xml = f.read()
    service = node.create_client(SpawnEntity, "/spawn_entity")
    if not service.wait_for_service(timeout_sec=5.0):
        node.get_logger().error("Spawn entity service not available")
        return False
    request = SpawnEntity.Request()
    request.name = "tb3"
    request.xml = xml
    request.robot_namespace = ""
    request.reference_frame = "world"
    if spawn_x is None:
        request.initial_pose.position.x = (warehouse_x_limit_max + warehouse_x_limit_min) / 2.0
    else:
        request.initial_pose.position.x = float(spawn_x)
    if spawn_y is None:
        request.initial_pose.position.y = (warehouse_y_limit_min + warehouse_x_limit_max) / 2.0
    else:
        request.initial_pose.position.y = float(spawn_y)
    request.initial_pose.position.z = 0.01
    request.initial_pose.orientation.w = 1.0
    future = service.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    if not future.done():
        node.get_logger().error("Spawn request timed out")
        return False
    result = future.result()
    if result is None or not getattr(result, 'success', True):
        node.get_logger().error("Failed to spawn robot")
        return False
    node.get_logger().info("TB3 spawned successfully")
    return True

def entity_spawned(node, spawn_client, name, xml, x, y, z, yaw= 0.0):
    if not spawn_client.service_is_ready():
        return False
    request = SpawnEntity.Request()
    request.name = name
    request.xml = xml
    request.robot_namespace = ""
    request.reference_frame = "world"
    request.initial_pose = Pose()
    request.initial_pose.position = Point(x=x, y=y, z=z)
    request.initial_pose.orientation = Quaternion(z=math.sin(yaw / 2.0),w=math.cos(yaw / 2.0))
    future = spawn_client.call_async(request)
    try:
        rclpy.spin_until_future_complete(node, future, timeout_sec=3.0)
        if not future.done():
            return True 
        result = future.result()
        if result is None:
            return True  # Assume success
        return getattr(result,'success',True)
    except Exception as e:
        return False


def delete_entity(node, client, name):
    if not client.service_is_ready():
        return False
    request = DeleteEntity.Request()
    request.name = name
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=2.0) 
    return bool(future.result() and future.result().success)

def reset_robot_position(node, x = None, y= None, z= 0.1):
    if x is None:
        x = (warehouse_x_limit_max + warehouse_x_limit_min) / 2.0
    if y is None:
        y = (warehouse_y_limit_min + warehouse_x_limit_max) / 2.0
    cli = node.create_client(SetEntityState, "/set_entity_state")
    if not cli.wait_for_service(timeout_sec=1.0):
        return False
    
    request = SetEntityState.Request()
    request.state = EntityState()
    request.state.name = "tb3"
    request.state.pose.position.x = float(x)
    request.state.pose.position.y = float(y)
    request.state.pose.position.z = float(z)
    request.state.pose.orientation.w = 1.0
    try:
        cli.call_async(request)
        node.get_logger().info("Resetting position")
        return True
    except Exception as e:
        node.get_logger().debug("Resetting")
        return False

def docking_blue_box(node, client, task, x, y):
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
    blue_box_path = os.path.join(project_root, "gazebo_models", "blue_box", "model.sdf")
    if not os.path.exists(blue_box_path):
        blue_box_path = os.path.join(os.getcwd(), "gazebo_models", "blue_box", "model.sdf")
    if not os.path.exists(blue_box_path):
        return False
    with open(blue_box_path, "r") as f:
        xml = f.read()
    marker_name = "docking_marker_" + task
    return entity_spawned(node, client, marker_name, xml, x, y, 0.05, 0.0)

def delete_blue_box(node, client, task):
    marker_name = "docking_marker_" + task
    return delete_entity(node, client, marker_name)
