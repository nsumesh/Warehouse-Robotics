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

# Start Gazebo server
echo "[1/3] Starting Gazebo server..."
gzserver "$WORLD_FILE" -s libgazebo_ros_init.so -s libgazebo_ros_factory.so &
GZSERVER_PID=$!
sleep 2

# Start Gazebo client
echo "[2/3] Starting Gazebo client..."
gzclient &
GZCLIENT_PID=$!
sleep 3

# Start spawner using direct path
echo "[3/3] Spawning warehouse objects..."
"$SPAWNER_BIN"

echo ""
echo "=========================================="
echo "Warehouse simulation complete!"
echo "Press Ctrl+C to exit"
echo "=========================================="

trap "echo 'Shutting down...'; kill $GZSERVER_PID $GZCLIENT_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
