#!/bin/bash

# Set default build arg
BUILD_GAZEBO=${BUILD_GAZEBO_FROM_SOURCE:-false}

# Allow X11 forwarding
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xhost +local:docker 2>/dev/null
    echo "X11 forwarding enabled for Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    if ! pgrep -x "Xquartz" > /dev/null; then
        echo "Starting XQuartz..."
        open -a XQuartz
        sleep 2
    fi
    xhost +localhost 2>/dev/null
    echo "X11 forwarding enabled for Mac (XQuartz)"
fi

# Build and run with docker-compose
echo "Building Docker image (BUILD_GAZEBO_FROM_SOURCE=$BUILD_GAZEBO)..."
BUILD_GAZEBO_FROM_SOURCE=$BUILD_GAZEBO docker-compose up --build -d

# Enter container
echo "Entering container..."
echo "Note: All ROS 2 workspaces are already sourced in the container"
docker-compose exec ros-gazebo bash

# Cleanup on exit
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xhost -local:docker 2>/dev/null
elif [[ "$OSTYPE" == "darwin"* ]]; then
    xhost -localhost 2>/dev/null
fi
