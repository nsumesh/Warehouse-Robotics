"""
Docking utilities for color-based visual servoing.
"""
import time
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rl_nav.constants import (
    BLUE_LOWER, BLUE_UPPER, BLUE_MARKER_DETECTION_THRESHOLD,
    BLUE_MARKER_COMPLETE_THRESHOLD, BLUE_MARKER_CENTERED_THRESHOLD,
    DOCKING_STABLE_DURATION
)


def detect_blue_marker(frame):
    """
    Detect blue marker in camera frame.
    
    Args:
        frame: BGR image frame
    
    Returns:
        dict: Detection results with keys:
            - detected: bool
            - area: float (pixels²)
            - centroid_x: int
            - error_x: int (pixel error from center)
            - centered: bool
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Blue color range
    lower = np.array(BLUE_LOWER)
    upper = np.array(BLUE_UPPER)
    
    mask = cv2.inRange(hsv, lower, upper)
    moments = cv2.moments(mask)
    
    frame_center_x = frame.shape[1] // 2
    
    if moments["m00"] > BLUE_MARKER_DETECTION_THRESHOLD:
        cx = int(moments["m10"] / moments["m00"])
        error_x = cx - frame_center_x
        centered = abs(error_x) < BLUE_MARKER_CENTERED_THRESHOLD
        
        return {
            'detected': True,
            'area': moments["m00"],
            'centroid_x': cx,
            'error_x': error_x,
            'centered': centered
        }
    else:
        return {
            'detected': False,
            'area': 0,
            'centroid_x': 0,
            'error_x': 0,
            'centered': False
        }


def process_camera_image(bridge, msg, phase):
    """
    Process camera image for blue marker detection.
    
    Args:
        bridge: CvBridge instance
        msg: Image message
        phase: Current phase (only processes if "GO_DOCKING")
    
    Returns:
        dict: Detection results (same as detect_blue_marker) or None if phase mismatch
    """
    if phase != "GO_DOCKING":
        return None
    
    try:
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
        return detect_blue_marker(frame)
    except Exception as e:
        return {'error': str(e)}


def is_docking_complete(marker_area, marker_centered, stable_time, stable_duration=None):
    """
    Check if docking is complete based on marker state.
    
    Args:
        marker_area: Marker area in pixels²
        marker_centered: Whether marker is centered
        stable_time: Time when marker first met completion criteria (or None)
        stable_duration: Required stability duration in seconds
    
    Returns:
        tuple: (is_complete: bool, new_stable_time: float or None)
    """
    if stable_duration is None:
        stable_duration = DOCKING_STABLE_DURATION
    
    # Check if very close (large area = close to marker)
    very_close = marker_area > BLUE_MARKER_COMPLETE_THRESHOLD
    
    # Docking complete if: large marker, centered, and stable
    if very_close and marker_centered:
        now = time.time()
        if stable_time is None:
            return False, now
        elif now - stable_time >= stable_duration:
            return True, stable_time
        else:
            return False, stable_time
    else:
        # Reset stability timer if conditions not met
        return False, None


def calculate_docking_control(marker_detected, error_x, search_angular_vel=0.3):
    """
    Calculate control commands for docking.
    
    Args:
        marker_detected: Whether blue marker is detected
        error_x: Horizontal error in pixels
        search_angular_vel: Angular velocity when searching (no marker)
    
    Returns:
        tuple: (linear_x: float, angular_z: float)
    """
    if marker_detected:
        # Marker detected: move forward and center
        linear_x = 0.1
        # Angular control based on horizontal error (proportional)
        angular_z = -float(error_x) / 300.0
    else:
        # No marker: search by rotating slowly
        linear_x = 0.0
        angular_z = search_angular_vel
    
    return linear_x, angular_z
