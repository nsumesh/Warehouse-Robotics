#!/usr/bin/env python3
"""
Launch file for sorting demonstration

Launches the integrated sorting node that uses PPO for navigation.

Usage:
    ros2 launch rl_nav sorting_demo.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Integrated Sorting Node (includes PPO navigation)
        Node(
            package='rl_nav',
            executable='sorting_node',
            name='sorting_node',
            output='screen',
        ),
    ])

