#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
source install/setup.bash

WORLD_FILE=$(ros2 pkg prefix warehouse_sim)/share/warehouse_sim/worlds/warehouse_empty.world
SPAWNER_BIN=$(ros2 pkg prefix warehouse_sim)/bin/warehouse_spawner

echo "World file: $WORLD_FILE"
echo ""

gzserver "$WORLD_FILE" \
  -s libgazebo_ros_init.so \
  -s libgazebo_ros_factory.so \
  -s libgazebo_ros_state.so &
GZSERVER_PID=$!
sleep 2

gzclient &
GZCLIENT_PID=$!
sleep 3

"$SPAWNER_BIN" &
sleep 2

trap "echo 'Shutting down...'; kill $GZSERVER_PID $GZCLIENT_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
