import time
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rl_nav.constants import (lower_hsv_mask, upper_hsv_mask, initial_detection_threshold, final_docking_threshold, alignment_threshold, docking_success_duration)

def detect_blue_marker(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    
    mask = cv2.inRange(hsv, np.array(lower_hsv_mask), np.array(upper_hsv_mask))
    moments = cv2.moments(mask)
    frame_center_x = frame.shape[1] // 2
    if moments["m00"] > initial_detection_threshold:
        cx = int(moments["m10"] / moments["m00"])
        error_x = cx - frame_center_x
        centered = abs(error_x) < alignment_threshold
        return {'detected': True,'area': moments["m00"],'centroid_x': cx,'error_x': error_x,'centered': centered}
    else:
        return {'detected': False,'area': 0,'centroid_x': 0,'error_x': 0,'centered': False}

def process_camera_image(bridge, msg, phase):
    if phase != "final_dock":
        return None
    try:
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
        return detect_blue_marker(frame)
    except Exception as e:
        return {'error': str(e)}

def docking_complete(marker_area, marker_centered, stable_time, required_duration=None):
    if required_duration is None:
        stable_duration = docking_success_duration
    close = marker_area > final_docking_threshold
    if close and marker_centered:
        now = time.time()
        if stable_time is None:
            return False, now
        elif now - stable_time >= stable_duration:
            return True, stable_time
        else:
            return False, stable_time
    else:
        return False, None

def docking_control(marker_detected, error_x, search_rate=0.3):
    if marker_detected:
        velocity = 0.1
        angular_velocity = -float(error_x) / 300.0
    else:
        velocity = 0.0
        angular_velocity = search_rate
    return velocity, angular_velocity
