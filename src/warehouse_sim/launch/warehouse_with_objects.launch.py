#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_dir = get_package_share_directory('warehouse_sim')
    world_file = os.path.join(pkg_dir, 'worlds', 'warehouse_empty.world')
    
    return LaunchDescription([
        # Launch Gazebo server
        ExecuteProcess(
            cmd=['gzserver', world_file, '--verbose',
                 '-s', 'libgazebo_ros_init.so',
                 '-s', 'libgazebo_ros_factory.so',
                 '-s', 'libgazebo_ros_force_system.so'],
            output='screen'
        ),
        
        # Launch Gazebo client
        ExecuteProcess(
            cmd=['gzclient', '--verbose'],
            output='screen'
        ),
        
        # Wait 5 seconds then launch spawner
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='warehouse_sim',
                    executable='warehouse_spawner',
                    name='warehouse_spawner',
                    output='screen'
                )
            ]
        ),
    ])
