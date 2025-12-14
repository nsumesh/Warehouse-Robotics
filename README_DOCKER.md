# Docker Setup for ROS 2 Sorting Project

## Prerequisites

### Linux
```bash
# Install Docker and Docker Compose
sudo apt-get update
sudo apt-get install docker.io docker-compose

# Allow X11 forwarding
xhost +local:docker
```

### Mac
```bash
# Install Docker Desktop
brew install --cask docker

# Install and start XQuartz
brew install --cask xquartz
open -a XQuartz
xhost +localhost
```

## Quick Start

### Option 1: Using Helper Script (Recommended)
```bash
# Make script executable
chmod +x run_docker.sh

# Run (will build and enter container)
./run_docker.sh

# For ARM architectures, build Gazebo from source:
BUILD_GAZEBO_FROM_SOURCE=true ./run_docker.sh
```

### Option 2: Using Docker Compose Directly
```bash
# Build (x86_64 - default)
docker-compose build

# Build (ARM - build Gazebo from source)
BUILD_GAZEBO_FROM_SOURCE=true docker-compose build

# Start container
docker-compose up -d

# Enter container
docker-compose exec ros-gazebo bash
```

### Option 3: Using Docker Directly
```bash
# Build image
sudo docker build --build-arg BUILD_GAZEBO_FROM_SOURCE=true -t ros-gazebo-project .

# Run container (Linux)
docker run -it --rm \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --network=host \
  --volume="$(pwd):/root/MSML642FinalProject" \
  ros-gazebo-project

# Run container (Mac)
docker run -it --rm \
  -e DISPLAY=host.docker.internal:0 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --volume="$(pwd):/root/MSML642FinalProject" \
  ros-gazebo-project
```

## Inside the Container

All ROS 2 workspaces are automatically sourced when you enter the container. You can verify:

```bash
# Check ROS 2 is sourced
echo $ROS_DISTRO  # Should output: humble

# Check Gazebo is installed
gzserver --version

# Test X11 forwarding
xeyes  # Should show eyes following cursor
```

## Running the Project

### 1. Launch Gazebo
```bash
# In first terminal
./launch_warehouse.sh
```

### 2. Run Sorting Node
```bash
# In second terminal (enter container again)
docker-compose exec ros-gazebo bash
ros2 run rl_nav sorting_node
```

### 3. Run Training
```bash
# Inside container
ros2 run rl_nav train_ppo --curriculum-stage 3 --timesteps 100000

# Or with custom log directory
ros2 run rl_nav train_ppo --curriculum-stage 3 --timesteps 100000 --logdir /root/MSML642FinalProject/ppo_runs
```

## Architecture Notes

### x86_64 (Intel/AMD)
- Gazebo Classic installs via apt (fast)
- No need to build from source
- Use: `docker-compose build`

### ARM (Apple Silicon, Raspberry Pi)
- Gazebo Classic may not be available via apt
- Build from source: `BUILD_GAZEBO_FROM_SOURCE=true docker-compose build`
- Takes 45-90 minutes to build Gazebo

## Troubleshooting

### X11 Forwarding Issues
**Linux:**
```bash
xhost +local:docker
```

**Mac:**
```bash
# Make sure XQuartz is running
open -a XQuartz
xhost +localhost
```

### Gazebo Not Displaying
```bash
# Check DISPLAY variable
echo $DISPLAY

# Test X11
xeyes

# Check graphics
glxinfo | grep "OpenGL version"
```

### Model Path Issues
```bash
# Check GAZEBO_MODEL_PATH
echo $GAZEBO_MODEL_PATH

# Should include: /root/MSML642FinalProject/gazebo_models
```

### Build Failures on ARM
If gazebo_ros_pkgs fails to build on ARM:
1. Ensure Gazebo Classic is built: `BUILD_GAZEBO_FROM_SOURCE=true`
2. Check build logs: `docker-compose build 2>&1 | tee build.log`
3. Verify Gazebo is installed: `which gzserver`

### Container Not Starting
```bash
# Check logs
docker-compose logs

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
```

## Useful Commands

```bash
# Stop container
docker-compose down

# View logs
docker-compose logs -f

# Rebuild without cache
docker-compose build --no-cache

# Remove everything and start fresh
docker-compose down -v
docker system prune -a
docker-compose build
```

## Project Structure in Container

```
/root/MSML642FinalProject/          # Your project
├── src/                            # ROS 2 packages
├── gazebo_models/                  # Gazebo models
├── ppo_runs/                       # Trained models
├── launch_warehouse.sh            # Launch script
└── install/                        # Built packages

/root/turtlebot3_ws/                # TurtleBot3 workspace
/root/gazebo_ros_pkgs_ws/          # Gazebo ROS packages workspace
```

## Notes

- The project directory is mounted as a volume, so changes persist
- All ROS 2 workspaces are automatically sourced in the container
- X11 forwarding is configured for GUI applications (Gazebo, RViz)
- GPU acceleration is available if `/dev/dri` exists on host
