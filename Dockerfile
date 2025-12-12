# -----------------------------------------------------------------------
# STAGE 1: BASE IMAGE
# -----------------------------------------------------------------------
FROM ubuntu:22.04

# Build argument to enable Gazebo Classic source build on ARM (takes ~30+ minutes)
ARG BUILD_GAZEBO_FROM_SOURCE=false

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV ROS_DISTRO=humble
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1

# -----------------------------------------------------------------------
# PHASE 1: INSTALL TOOLS & SETUP REPOSITORIES
# -----------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    locales \
    curl \
    gnupg2 \
    lsb-release \
    build-essential \
    cmake \
    git \
    wget \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    x11-xserver-utils \
    x11-apps \
    x11-utils \
    net-tools \
    python3-pip \
    python3-apt \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# Install colcon and rosdep via pip (available before ROS setup)
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    colcon-common-extensions \
    rosdep

# -----------------------------------------------------------------------
# PHASE 2: INSTALL ROS 2 HUMBLE
# -----------------------------------------------------------------------
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-dev-tools \
    python3-rosinstall-generator \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------
# PHASE 3: INSTALL GAZEBO CLASSIC (with ARM support)
# -----------------------------------------------------------------------
# Detect architecture and install Gazebo Classic appropriately
# On x86_64: install via apt
# On ARM: install build dependencies and build from source
RUN ARCH=$(dpkg --print-architecture) && \
    apt-get update && \
    if [ "$ARCH" = "amd64" ] || [ "$ARCH" = "i386" ]; then \
        # x86_64: Install Gazebo Classic via apt (standard packages available) \
        apt-get install -y \
            gazebo11 \
            libgazebo11-dev \
            gazebo11-plugins \
            gazebo11-plugin-base \
            gazebo11-common \
            libgazebo11-plugin-base || true; \
    else \
        # ARM architectures: Install build dependencies for building from source \
        echo "Detected ARM architecture ($ARCH), installing Gazebo Classic build dependencies..." && \
        # Install core build dependencies that are definitely available \
        apt-get install -y \
            libboost-all-dev \
            libtinyxml2-dev \
            liburdfdom-dev \
            libtar-dev \
            libcurl4-openssl-dev \
            libfreeimage-dev \
            libfreeimage3 \
            libprotobuf-dev \
            protobuf-compiler \
            protobuf-c-compiler \
            libprotoc-dev \
            libogre-1.9-dev \
            libxml2-dev \
            pkg-config \
            libqt5core5a \
            libqt5gui5 \
            libqt5widgets5 \
            qtbase5-dev \
            libqwt-qt5-dev \
            qtscript5-dev \
            libgts-dev \
            libavformat-dev \
            libavcodec-dev \
            libswscale-dev \
            libavdevice-dev \
            libusb-1.0-0-dev \
            libspnav-dev \
            libbluetooth-dev \
            libgstreamer1.0-dev \
            libgstreamer-plugins-base1.0-dev \
            libgstreamer-plugins-good1.0-dev \
            mercurial \
            cmake \
            build-essential && \
        # Try to install optional dependencies (may not be available on all ARM platforms) \
        apt-get install -y \
            libsimbody-dev \
            2>/dev/null || echo "libsimbody-dev not available, skipping" && \
        # Note: libsdformat, libignition-*, libdart, libqwt-dev may not be available on ARM \
        # Gazebo will build without some of these, though with limited functionality \
        echo "Core dependencies installed. Some optional packages may be unavailable on ARM."; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# Optional: Build Gazebo Classic from source on ARM if requested
# Usage: docker build --build-arg BUILD_GAZEBO_FROM_SOURCE=true ...
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$BUILD_GAZEBO_FROM_SOURCE" = "true" ] && \
       [ "$ARCH" != "amd64" ] && [ "$ARCH" != "i386" ] && \
       ! command -v gzserver >/dev/null 2>&1; then \
        echo "========================================" && \
        echo "Building Gazebo Classic from source on $ARCH" && \
        echo "This will take approximately 45-90 minutes..." && \
        echo "Note: Build dependencies should already be installed in Phase 3" && \
        echo "========================================" && \
        # Verify critical dependencies are installed \
        echo "Verifying dependencies..." && \
        dpkg -l | grep -E "(libprotobuf|libfreeimage|libboost)" || echo "Warning: Some dependencies may be missing" && \
        # Update apt and install any missing dependencies \
        apt-get update && apt-get install -y --fix-missing \
            libprotobuf-dev \
            protobuf-compiler \
            libfreeimage-dev \
            libfreeimage3 \
            pkg-config \
            libtinyxml-dev \
            ruby \
            ruby-dev && \
        # Install ignition-math6 (required by sdformat9) \
        echo "Building ignition-math6..." && \
        cd /tmp && \
        git clone --depth 1 --branch ign-math6 https://github.com/gazebosim/gz-math.git ignition-math && \
        cd ignition-math && mkdir build && cd build && \
        cmake .. -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Release && \
        make -j$(nproc) && make install && ldconfig && \
        cd /tmp && rm -rf ignition-math && \
        # Build sdformat9 (required by Gazebo) \
        echo "Building sdformat9..." && \
        cd /tmp && \
        git clone --depth 1 --branch sdf9 https://github.com/osrf/sdformat.git sdformat && \
        cd sdformat && mkdir build && cd build && \
        cmake .. -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Release && \
        make -j$(nproc) && make install && ldconfig && \
        cd /tmp && rm -rf sdformat && \
        # Now build Gazebo Classic \
        echo "Building Gazebo Classic..." && \
        cd /tmp && \
        git clone --depth 1 --branch gazebo11 https://github.com/osrf/gazebo.git gazebo-source && \
        if [ ! -d gazebo-source ]; then \
            echo "ERROR: Failed to clone Gazebo source repository" && exit 1; \
        fi && \
        cd gazebo-source && \
        mkdir -p build && cd build && \
        # Set PKG_CONFIG_PATH to help find packages (handle both single and multiarch) \
        ARCH_DEB=$(dpkg-architecture -q DEB_BUILD_MULTIARCH 2>/dev/null || dpkg --print-architecture) && \
        export PKG_CONFIG_PATH=/usr/lib/pkgconfig:/usr/lib/${ARCH_DEB}/pkgconfig:/usr/share/pkgconfig:$PKG_CONFIG_PATH && \
        # Verify packages are findable \
        echo "Checking for protobuf..." && (pkg-config --exists protobuf && echo "protobuf found" || echo "WARNING: protobuf not in pkg-config, cmake may need help finding it") && \
        echo "Checking for freeimage..." && (pkg-config --exists freeimage && echo "freeimage found" || echo "WARNING: freeimage not in pkg-config, cmake may need help finding it") && \
        cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX=/usr \
            -DENABLE_SCREEN_TESTS=OFF \
            -DCMAKE_PREFIX_PATH=/usr && \
        make -j$(nproc) && \
        make install && \
        ldconfig && \
        cd / && rm -rf /tmp/gazebo-source && \
        rm -rf /var/lib/apt/lists/* && \
        echo "Gazebo Classic built and installed successfully" || \
        (echo "Gazebo Classic build failed" && exit 1); \
    elif [ "$ARCH" != "amd64" ] && [ "$ARCH" != "i386" ] && ! command -v gzserver >/dev/null 2>&1; then \
        echo "========================================" && \
        echo "Note: Gazebo Classic not installed on $ARCH" && \
        echo "To build from source, use: --build-arg BUILD_GAZEBO_FROM_SOURCE=true" && \
        echo "========================================"; \
    fi

# -----------------------------------------------------------------------
# PHASE 4: INITIALIZE ROSDEP
# -----------------------------------------------------------------------
RUN rosdep init && rosdep update

# -----------------------------------------------------------------------
# PHASE 5: INSTALL TURTLEBOT3 PACKAGES
# -----------------------------------------------------------------------
WORKDIR /root/turtlebot3_ws/src
RUN git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3.git \
    && git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git

# Install TurtleBot3 dependencies
WORKDIR /root/turtlebot3_ws
RUN . /opt/ros/humble/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y --rosdistro humble || true

# Build TurtleBot3 packages
RUN . /opt/ros/humble/setup.bash && \
    colcon build --symlink-install

# -----------------------------------------------------------------------
# PHASE 6: FETCH AND BUILD GAZEBO_ROS_PKGS
# -----------------------------------------------------------------------
WORKDIR /root/gazebo_ros_pkgs_ws/src
RUN git clone https://github.com/ros-simulation/gazebo_ros_pkgs.git -b humble

# Install gazebo_ros_pkgs dependencies with ROS_DISTRO set
WORKDIR /root/gazebo_ros_pkgs_ws

# Create a script to handle rosdep installation with better error handling
RUN cat > /tmp/install_deps.sh << 'EOFSCRIPT'
#!/bin/bash
set +e
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble 2>&1 | tee /tmp/rosdep.log
ROSDEP_EXIT=${PIPESTATUS[0]}
if [ $ROSDEP_EXIT -ne 0 ]; then
    if grep -q "Cannot locate rosdep definition" /tmp/rosdep.log || grep -q "Unable to locate package" /tmp/rosdep.log; then
        echo "Warning: Some rosdep keys could not be resolved (this may be normal on ARM architectures)"
        echo "Continuing with available dependencies..."
    else
        echo "Error: rosdep installation failed with unexpected errors"
        exit $ROSDEP_EXIT
    fi
fi
exit 0
EOFSCRIPT
RUN chmod +x /tmp/install_deps.sh

RUN /tmp/install_deps.sh

# Verify Gazebo installation status
RUN ARCH=$(dpkg --print-architecture) && \
    if ! command -v gzserver >/dev/null 2>&1; then \
        echo "========================================" && \
        echo "WARNING: Gazebo Classic is not installed" && \
        echo "Architecture: $ARCH" && \
        echo "========================================" && \
        echo "On ARM architectures, Gazebo Classic may not be available via apt." && \
        echo "You may need to:" && \
        echo "  1. Build Gazebo Classic from source" && \
        echo "  2. Use a multi-stage Docker build with a pre-built Gazebo image" && \
        echo "  3. Use Gazebo Garden (newer version) if compatible" && \
        echo "========================================"; \
    else \
        echo "Gazebo Classic is installed: $(which gzserver)"; \
    fi

# Build gazebo_ros_pkgs
# This may fail on ARM if Gazebo Classic is not installed
RUN ARCH=$(dpkg --print-architecture) && \
    . /opt/ros/humble/setup.bash && \
    if colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | tee /tmp/build.log; then \
        echo "gazebo_ros_pkgs built successfully"; \
    else \
        BUILD_EXIT=${PIPESTATUS[0]}; \
        if [ "$ARCH" != "amd64" ] && [ "$ARCH" != "i386" ]; then \
            echo "========================================" && \
            echo "Build failed on $ARCH architecture" && \
            echo "This is likely because Gazebo Classic is not installed" && \
            echo "========================================" && \
            echo "To fix this on ARM, you need to either:" && \
            echo "1. Build Gazebo Classic from source (see Gazebo documentation)" && \
            echo "2. Use a Docker multi-stage build with a Gazebo base image" && \
            echo "3. Use an alternative simulation platform" && \
            echo "========================================"; \
        else \
            echo "Build failed on $ARCH - unexpected error"; \
        fi; \
        exit $BUILD_EXIT; \
    fi

# -----------------------------------------------------------------------
# PHASE 7: INSTALL PYTHON DEPENDENCIES
# -----------------------------------------------------------------------
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    "numpy<2" \
    gymnasium==0.29.1 \
    stable-baselines3==2.3.0 \
    torch \
    opencv-python \
    matplotlib \
    tensorboard

# -----------------------------------------------------------------------
# PHASE 8: SETUP PROJECT WORKSPACE
# -----------------------------------------------------------------------
WORKDIR /root/MSML642FinalProject

# Copy project files into container
# Note: You can also mount the workspace as a volume instead for development
COPY . /root/MSML642FinalProject/

# Build the project packages
RUN . /opt/ros/humble/setup.bash && \
    . /root/turtlebot3_ws/install/setup.bash && \
    . /root/gazebo_ros_pkgs_ws/install/setup.bash && \
    colcon build --symlink-install || echo "Build completed with some warnings/errors"

# -----------------------------------------------------------------------
# FINAL CONFIGURATION
# -----------------------------------------------------------------------
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /root/turtlebot3_ws/install/setup.bash" >> /root/.bashrc && \
    echo "source /root/gazebo_ros_pkgs_ws/install/setup.bash" >> /root/.bashrc && \
    echo "source /root/MSML642FinalProject/install/setup.bash" >> /root/.bashrc && \
    echo "export TURTLEBOT3_MODEL=waffle_pi" >> /root/.bashrc && \
    echo "export GAZEBO_MODEL_PATH=/root/MSML642FinalProject/gazebo_models:\$GAZEBO_MODEL_PATH" >> /root/.bashrc

WORKDIR /root/MSML642FinalProject
CMD ["bash"]
