#!/bin/bash
set -e

cd ~/MSML_642_FinalProject
source install/setup.bash

WORLD_FILE=$(ros2 pkg prefix warehouse_sim)/share/warehouse_sim/worlds/warehouse_empty.world

echo "=========================================="
echo "Starting EMPTY Warehouse (NO OBJECTS, NO TB3)"
echo "For Stage 1 PPO Training"
echo "=========================================="
echo "World file: $WORLD_FILE"
echo ""

##############################
# [1/3] Start Gazebo server
##############################
echo "[1/3] Starting Gazebo server..."
gzserver "$WORLD_FILE" \
  -s libgazebo_ros_init.so \
  -s libgazebo_ros_factory.so \
  -s libgazebo_ros_state.so &
GZSERVER_PID=$!
sleep 2

##############################
# [2/3] Start Gazebo client
##############################
echo "[2/3] Starting Gazebo client..."
gzclient &
GZCLIENT_PID=$!
sleep 3

###############################################
# [3/3] DONE — NO OBJECTS, NO ROBOT
###############################################
echo ""
echo "=========================================="
echo "Empty warehouse ready for Stage 1 training!"
echo "No objects spawned - clear area for learning."
echo "Run: ros2 run rl_nav train_ppo --timesteps 10000"
echo "=========================================="

trap "echo 'Shutting down...'; kill $GZSERVER_PID $GZCLIENT_PID 2>/dev/null; exit" SIGINT SIGTERM
wait

