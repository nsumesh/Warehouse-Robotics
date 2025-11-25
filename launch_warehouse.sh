#!/bin/bash
set -e

cd ~/MSML_642_FinalProject
source install/setup.bash

WORLD_FILE=$(ros2 pkg prefix warehouse_sim)/share/warehouse_sim/worlds/warehouse_empty.world
SPAWNER_BIN=$(ros2 pkg prefix warehouse_sim)/bin/warehouse_spawner

echo "=========================================="
echo "Starting Warehouse Simulation"
echo "=========================================="
echo "World file: $WORLD_FILE"
echo ""

##############################
# [1/4] Start Gazebo server
##############################
echo "[1/4] Starting Gazebo server..."
gzserver "$WORLD_FILE" -s libgazebo_ros_init.so -s libgazebo_ros_factory.so &
GZSERVER_PID=$!
sleep 2

##############################
# [2/4] Start Gazebo client
##############################
echo "[2/4] Starting Gazebo client..."
gzclient &
GZCLIENT_PID=$!
sleep 3

###############################################
# [3/4] Spawn warehouse objects (shelves/boxes)
###############################################
echo "[3/4] Spawning warehouse objects..."
"$SPAWNER_BIN" & sleep 2

###############################################
# [4/4] Spawn TurtleBot3 Waffle Pi robot
###############################################
echo "[4/4] Spawning TurtleBot3 Waffle Pi..."

ros2 run gazebo_ros spawn_entity.py \
  -entity tb3 \
  -file $(ros2 pkg prefix turtlebot3_gazebo)/share/turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf \
  -x -9 -y -9 -z 0.01

echo ""
echo "=========================================="
echo "Warehouse simulation + TB3 ready!"
echo "Use PPO Training, PPO Controller, or Sorting Node now."
echo "Press Ctrl+C to exit"
echo "=========================================="

trap "echo 'Shutting down...'; kill $GZSERVER_PID $GZCLIENT_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
