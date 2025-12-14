from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rl_nav',
            executable='sorting_node',
            name='sorting_node',
            output='screen',
        ),
    ])

