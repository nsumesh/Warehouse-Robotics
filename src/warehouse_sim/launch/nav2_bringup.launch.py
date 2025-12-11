import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Dynamic path to user home directory to avoid "user1" errors
    home_dir = os.path.expanduser('~')
    
    # Define default paths
    default_map_path = os.path.join(home_dir, 'MSML_642_FinalProject', 'maps', 'warehouse_map_20251204_113438.yaml')
    default_params_path = os.path.join(home_dir, 'MSML_642_FinalProject', 'config', 'nav2_params.yaml')

    # Launch Configurations
    map_yaml_file = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        # 1. Arguments
        DeclareLaunchArgument(
            'use_sim_time', 
            default_value='true', 
            description='Use simulation clock'),

        DeclareLaunchArgument(
            'map', 
            default_value=default_map_path,
            description='Full path to map YAML file'),

        DeclareLaunchArgument(
            'params', 
            default_value=default_params_path,
            description='Full path to Nav2 parameters'),

        # 2. Map Server (Must run first)
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'yaml_filename': map_yaml_file}, {'use_sim_time': use_sim_time}]
        ),

        # 3. Nav2 Nodes (Delayed to allow map_server to initialize)
        TimerAction(
            period=2.0,
            actions=[
                # AMCL (Localization)
                Node(
                    package='nav2_amcl',
                    executable='amcl',
                    name='amcl',
                    output='screen',
                    parameters=[params_file]
                ),

                # Planner
                Node(
                    package='nav2_planner',
                    executable='planner_server',
                    name='planner_server',
                    output='screen',
                    parameters=[params_file]
                ),

                # Controller
                Node(
                    package='nav2_controller',
                    executable='controller_server',
                    name='controller_server',
                    output='screen',
                    parameters=[params_file]
                ),

                # Recoveries
                Node(
                    package='nav2_behaviors',
                    executable='behavior_server',
                    name='behavior_server',
                    output='screen',
                    parameters=[params_file]
                ),

                # Behavior Tree Navigator
                Node(
                    package='nav2_bt_navigator',
                    executable='bt_navigator',
                    name='bt_navigator',
                    output='screen',
                    parameters=[params_file]
                ),

                # Smoother
                Node(
                    package='nav2_smoother',
                    executable='smoother_server',
                    name='smoother_server',
                    output='screen',
                    parameters=[params_file]
                ),

                # Lifecycle Manager (Activates all nodes)
                Node(
                    package='nav2_lifecycle_manager',
                    executable='lifecycle_manager',
                    name='lifecycle_manager_navigation',
                    output='screen',
                    parameters=[{
                        'use_sim_time': use_sim_time,
                        'autostart': True,
                        'node_names': [
                            'map_server',
                            'amcl',
                            'controller_server',
                            'planner_server',
                            'behavior_server',
                            'bt_navigator',
                            'smoother_server'
                        ]
                    }]
                ),
            ]
        )
    ])

