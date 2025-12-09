"""
Gazebo utilities for spawning, deleting entities, and robot management.
"""
import os
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, SetEntityState
from gazebo_msgs.msg import EntityState
from rl_nav.constants import X_MIN, X_MAX, Y_MIN, Y_MAX


def spawn_tb3(node: Node) -> bool:
    """
    Spawn TurtleBot3 in Gazebo.
    
    Args:
        node: ROS2 node instance
    
    Returns:
        bool: True if successful, False otherwise
    """
    model_path = os.path.expanduser(
        "~/turtlebot3_ws/install/turtlebot3_gazebo/share/"
        "turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf"
    )
    if not os.path.exists(model_path):
        node.get_logger().error(f"TB3 model not found: {model_path}")
        return False

    with open(model_path, "r") as f:
        sdf_xml = f.read()

    spawn_cli = node.create_client(SpawnEntity, "/spawn_entity")
    if not spawn_cli.wait_for_service(timeout_sec=10.0):
        node.get_logger().error("Service /spawn_entity not available")
        return False

    req = SpawnEntity.Request()
    req.name = "tb3"
    req.xml = sdf_xml
    req.robot_namespace = ""
    req.reference_frame = "world"
    # Spawn in workspace center
    req.initial_pose.position.x = (X_MIN + X_MAX) / 2.0
    req.initial_pose.position.y = (Y_MIN + Y_MAX) / 2.0
    req.initial_pose.position.z = 0.01
    yaw = 0.0
    req.initial_pose.orientation.z = math.sin(yaw / 2.0)
    req.initial_pose.orientation.w = math.cos(yaw / 2.0)

    future = spawn_cli.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is None:
        node.get_logger().error("Failed to spawn TB3")
        return False
    node.get_logger().info("TB3 spawned successfully")
    return True


def spawn_entity(node: Node, spawn_client, name: str, sdf_xml: str, 
                 x: float, y: float, z: float, yaw: float = 0.0) -> bool:
    """
    Spawn an entity in Gazebo.
    
    Args:
        node: ROS2 node instance
        spawn_client: SpawnEntity service client
        name: Entity name
        sdf_xml: SDF XML string
        x, y, z: Position coordinates
        yaw: Orientation yaw angle
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not spawn_client.service_is_ready():
        return False
    
    req = SpawnEntity.Request()
    req.name = name
    req.xml = sdf_xml
    req.robot_namespace = ""
    req.reference_frame = "world"
    req.initial_pose = Pose()
    req.initial_pose.position = Point(x=float(x), y=float(y), z=float(z))
    req.initial_pose.orientation = Quaternion(
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0)
    )
    
    future = spawn_client.call_async(req)
    try:
        rclpy.spin_until_future_complete(node, future, timeout_sec=3.0)
        if future.result() and future.result().success:
            return True
    except Exception as e:
        node.get_logger().warn(f"Exception spawning {name}: {e}")
    return False


def delete_entity(node: Node, delete_client, name: str) -> bool:
    """
    Delete an entity from Gazebo.
    
    Args:
        node: ROS2 node instance
        delete_client: DeleteEntity service client
        name: Entity name to delete
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not delete_client.service_is_ready():
        return False
    
    req = DeleteEntity.Request()
    req.name = name
    future = delete_client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=2.0)
    
    if future.result() and future.result().success:
        return True
    return False


def reset_robot_position(node: Node, x: float = None, y: float = None, z: float = 0.1) -> bool:
    """
    Reset robot to a specified position.
    
    Args:
        node: ROS2 node instance
        x, y: Position coordinates (defaults to workspace center)
        z: Z coordinate (default 0.1)
    
    Returns:
        bool: True if successful, False otherwise
    """
    if x is None:
        x = (X_MIN + X_MAX) / 2.0
    if y is None:
        y = (Y_MIN + Y_MAX) / 2.0
    
    reset_cli = node.create_client(SetEntityState, "/set_entity_state")
    if not reset_cli.wait_for_service(timeout_sec=1.0):
        node.get_logger().warn("Reset service not available - skipping reset")
        return False
    
    req = SetEntityState.Request()
    req.state = EntityState()
    req.state.name = "tb3"
    req.state.pose.position.x = float(x)
    req.state.pose.position.y = float(y)
    req.state.pose.position.z = float(z)
    req.state.pose.orientation.w = 1.0
    
    future = reset_cli.call_async(req)
    try:
        rclpy.spin_until_future_complete(node, future, timeout_sec=2.0)
    except Exception as e:
        node.get_logger().warn(f"Exception waiting for reset service: {e}")
        return False
    
    if not future.done():
        node.get_logger().warn("Reset service call timed out")
        return False
    
    try:
        result = future.result()
        if result and result.success:
            node.get_logger().info("Robot position reset successfully")
            return True
    except Exception as e:
        node.get_logger().warn(f"Exception getting reset result: {e}")
    return False


def spawn_blue_box_at_dock(node: Node, spawn_client, task_class: str, x: float, y: float) -> bool:
    """
    Spawn blue box marker at dock location for visual docking.
    
    Args:
        node: ROS2 node instance
        spawn_client: SpawnEntity service client
        task_class: Task class ('A', 'B', or 'C')
        x, y: Dock position coordinates
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Find project root by looking for gazebo_models directory
    # Start from this file's location and go up to find project root
    current_file = os.path.abspath(__file__)
    project_root = None
    
    # Try multiple possible paths
    search_paths = [
        # Relative to current file (src/rl_nav/rl_nav/gazebo_utils.py -> project root)
        os.path.join(os.path.dirname(current_file), "../../../"),
        # Common project directory names
        os.path.expanduser("~/MSML642FinalProject"),
        os.path.expanduser("~/MSML_642_FinalProject"),
    ]
    
    for path in search_paths:
        abs_path = os.path.abspath(path)
        blue_box_path = os.path.join(abs_path, "gazebo_models", "blue_box", "model.sdf")
        if os.path.exists(blue_box_path):
            project_root = abs_path
            break
    
    if not project_root:
        node.get_logger().error("Could not find project root with gazebo_models/blue_box/model.sdf")
        node.get_logger().error(f"Searched in: {search_paths}")
        return False
    
    blue_box_path = os.path.join(project_root, "gazebo_models", "blue_box", "model.sdf")
    
    with open(blue_box_path, "r") as f:
        sdf_xml = f.read()
    
    marker_name = f"docking_marker_{task_class}"
    return spawn_entity(node, spawn_client, marker_name, sdf_xml, x, y, 0.05, 0.0)


def delete_blue_box(node: Node, delete_client, task_class: str) -> bool:
    """
    Delete blue box marker after docking.
    
    Args:
        node: ROS2 node instance
        delete_client: DeleteEntity service client
        task_class: Task class ('A', 'B', or 'C')
    
    Returns:
        bool: True if successful, False otherwise
    """
    marker_name = f"docking_marker_{task_class}"
    return delete_entity(node, delete_client, marker_name)
