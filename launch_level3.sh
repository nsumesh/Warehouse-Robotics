#!/bin/bash
set -e

cd ~/MSML_642_FinalProject
source install/setup.bash

WORLD_FILE=$(ros2 pkg prefix warehouse_sim)/share/warehouse_sim/worlds/warehouse_stage3.world
SPAWNER_BIN=$(ros2 pkg prefix warehouse_sim)/bin/warehouse_spawner

echo "=========================================="
echo "Starting Level 3: Full Warehouse Training"
echo "=========================================="
echo "World file: $WORLD_FILE"
echo ""

##############################
# [1/4] Start Gazebo server
##############################
echo "[1/4] Starting Gazebo server..."
gzserver "$WORLD_FILE" \
  -s libgazebo_ros_init.so \
  -s libgazebo_ros_factory.so \
  -s libgazebo_ros_state.so &
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
"$SPAWNER_BIN" &
sleep 2

###############################################
# [4/4] DONE
###############################################
echo ""
echo "=========================================="
echo "Level 3 environment ready!"
echo "Run: ros2 run rl_nav train_ppo --level 3 --timesteps 20000"
echo "=========================================="

trap "echo 'Shutting down...'; kill $GZSERVER_PID $GZCLIENT_PID 2>/dev/null; exit" SIGINT SIGTERM
wait

