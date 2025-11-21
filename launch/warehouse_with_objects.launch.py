#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    

    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_warehouse_sim = get_package_share_directory('warehouse_sim')
    

    world_file = os.path.join(pkg_warehouse_sim, 'worlds', 'warehouse_empty.world')
    

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world': world_file,
            'verbose': 'true'
        }.items()
    )
    

    spawner_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='warehouse_sim',
                executable='custom_warehouse',
                name='warehouse_object_spawner',
                output='screen'
            )
        ]
    )
    
    return LaunchDescription([
        gazebo,
        spawner_node
    ])