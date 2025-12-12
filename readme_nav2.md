start warehouse : ./launch_warehouse.sh in terminal 1
Spawn robot : ros2 run gazebo_ros spawn_entity.py \
  -entity waffle_pi \
  -database turtlebot3_waffle_pi \
  -x 0.0 -y 0.0 -z 0.01

Publish static transformation in terminal 3 : ros2 run tf2_ros static_transform_publisher \
  0 0 0 0 0 0 map odom

Terminal 4: cd ~/MSML_642_FinalProject
source /opt/ros/humble/setup.bash
source install/setup.bash   # if present

ros2 launch nav2_bringup bringup_launch.py \
  use_sim_time:=true \
  map:=$(pwd)/maps/warehouse_map_final.yaml \
  params_file:=$(pwd)/config/nav2_params.yaml

Terminal 5 : ros2 launch nav2_bringup rviz_launch.py use_sim_time:=true

