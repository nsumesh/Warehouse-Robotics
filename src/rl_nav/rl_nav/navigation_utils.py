"""
Navigation utilities for distance calculation, goal detection, collision, and stuck detection.
"""
import math
import time
import numpy as np
from sensor_msgs.msg import LaserScan
from rl_nav.constants import MAX_RANGE, NUM_SCAN_BINS


def distance_to_goal(pose, goal):
    """
    Calculate Euclidean distance to goal.
    
    Args:
        pose: Robot pose [x, y, yaw] or [x, y]
        goal: Goal position [x, y]
    
    Returns:
        float: Distance to goal
    """
    if goal is None:
        return float('inf')
    dx, dy = (np.array(goal) - np.array(pose[:2]))
    return float(math.hypot(dx, dy))


def goal_reached(pose, goal, threshold, goal_reached_time=None, stability_duration=3.0):
    """
    Check if goal is reached with stability check.
    
    Args:
        pose: Robot pose [x, y, yaw]
        goal: Goal position [x, y]
        threshold: Distance threshold for goal reached
        goal_reached_time: Previous time when goal was first reached (for stability)
        stability_duration: Required stability duration in seconds
    
    Returns:
        tuple: (is_reached: bool, new_goal_reached_time: float or None)
    """
    dist = distance_to_goal(pose, goal)
    if dist < threshold:
        now = time.time()
        if goal_reached_time is None:
            return False, now
        elif now - goal_reached_time >= stability_duration:
            return True, goal_reached_time
        else:
            return False, goal_reached_time
    else:
        return False, None


def check_collision(scan, max_range=None, threshold=0.15):
    """
    Check if robot is too close to obstacles.
    
    Args:
        scan: Normalized LiDAR scan array (0-1 range)
        max_range: Maximum range in meters (for denormalization)
        threshold: Collision threshold in meters
    
    Returns:
        bool: True if collision detected
    """
    if scan is None:
        return False
    
    if max_range is None:
        max_range = MAX_RANGE
    
    # Ensure scan is valid (not all NaN or invalid)
    valid_scan = scan[~np.isnan(scan)]
    if len(valid_scan) == 0:
        return False
    
    # Check if any LiDAR reading is very close (collision threshold)
    min_dist = np.min(valid_scan) * max_range  # Convert normalized to meters
    return min_dist < threshold


def check_stuck(current_dist, last_stuck_check_time, last_stuck_dist, 
                improvement_threshold=0.2, check_interval=15.0, close_threshold=1.5):
    """
    Check if robot is stuck (distance not improving).
    
    Args:
        current_dist: Current distance to goal
        last_stuck_check_time: Last time stuck check was performed
        last_stuck_dist: Distance at last stuck check
        improvement_threshold: Required improvement in meters
        check_interval: Time interval between checks in seconds
        close_threshold: Distance below which to use more lenient threshold
    
    Returns:
        tuple: (is_stuck: bool, new_check_time: float, new_dist: float)
    """
    # Don't check for stuck if very close to goal
    if current_dist < 1.0:
        return False, last_stuck_check_time, last_stuck_dist
    
    if last_stuck_check_time is None:
        return False, time.time(), current_dist
    
    # Check every check_interval seconds
    now = time.time()
    if now - last_stuck_check_time > check_interval:
        # If robot is close to goal, be more lenient
        if current_dist < close_threshold:
            improvement_threshold = 0.1
        else:
            improvement_threshold = 0.2
        
        # If distance hasn't improved by threshold, consider stuck
        if abs(current_dist - last_stuck_dist) < improvement_threshold:
            return True, now, current_dist
        return False, now, current_dist
    
    return False, last_stuck_check_time, last_stuck_dist


def process_scan_to_bins(scan_msg, num_bins=None, max_range=None):
    """
    Process LiDAR scan message into normalized bins.
    
    Args:
        scan_msg: LaserScan message
        num_bins: Number of bins to create
        max_range: Maximum range for normalization
    
    Returns:
        tuple: (scan_bins: np.array, collision: bool, min_dist: float)
    """
    if num_bins is None:
        num_bins = NUM_SCAN_BINS
    if max_range is None:
        max_range = MAX_RANGE
    
    ranges = scan_msg.ranges
    n = len(ranges)
    if n == 0:
        return np.ones(num_bins, dtype=np.float32), False, max_range
    
    step = max(1, n // num_bins)
    bins = []
    min_dist = max_range
    collision = False
    
    for i in range(0, n, step):
        v = ranges[i]
        if math.isnan(v) or v <= 0.0:
            v = max_range
        if v < min_dist:
            min_dist = v
        if v < 0.12:  # Match collision threshold (10cm + margin)
            collision = True
        bins.append(min(v, max_range) / max_range)
        if len(bins) == num_bins:
            break
    
    if len(bins) < num_bins:
        bins.extend([1.0] * (num_bins - len(bins)))
    
    return np.array(bins, dtype=np.float32), collision, float(min_dist)
