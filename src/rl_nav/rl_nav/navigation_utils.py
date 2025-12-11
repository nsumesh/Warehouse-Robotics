"""
Navigation utilities for distance calculation, goal detection, collision, and stuck detection.
"""
import math
import time
import numpy as np
from sensor_msgs.msg import LaserScan
from rl_nav.constants import max_clamp_range, lidar_bins

def euclidean_distance(pose, goal):
    px, py = pose[:2]
    gx, gy= goal
    return math.hypot(gx-px, gy-py)


def goal_reached(pose, goal, threshold, goal_reached_time=None, duration=3.0):
    dist = euclidean_distance(pose, goal)
    if dist < threshold:
        now = time.time()
        if goal_reached_time is None:
            return False, now
        return (now-goal_reached_time >= duration), goal_reached_time
    else:
        return False, None


def check_collision(scan, max_range=None, threshold=0.15):
    if scan is None:
        return False
    if max_range is None:
        max_range = max_clamp_range
    
    valid_scan = scan[~np.isnan(scan)]
    if len(valid_scan) == 0:
        return False
    min_dist = np.min(valid_scan) * max_range  
    return min_dist < threshold


def process_scan_to_bins(scan_msg, num_bins=None, max_range=None):
    if num_bins is None:
        num_bins = lidar_bins
    if max_range is None:
        max_range = max_clamp_range
    
    ranges = scan_msg.ranges
    n = len(ranges)
    if n == 0:
        return np.ones(num_bins, dtype=np.float32), False, max_range
    invalid = ~np.isfinite(ranges) | (ranges <= 0.0)
    ranges[invalid] = max_range
    step = max(1, n // num_bins)
    sample = ranges[::step][:num_bins]
    if sample.size<num_bins:
        padding = num_bins-sample.size
        sample = np.concatenate([sample, np.full(padding, max_range, dtype=np.float32)])
    min_dist = float(sample.min())
    collision = min_dist<0.12
    normalized = sample/max_range
    return normalized.astype(np.float32), collision, min_dist


# def check_stuck(current_dist, last_stuck_check_time, last_stuck_dist, 
#                 improvement_threshold=0.001, check_interval=15.0, close_threshold=1.5):
#     """
#     Check if robot is stuck (distance not improving).
    
#     Args:
#         current_dist: Current distance to goal
#         last_stuck_check_time: Last time stuck check was performed
#         last_stuck_dist: Distance at last stuck check
#         improvement_threshold: Required improvement in meters
#         check_interval: Time interval between checks in seconds
#         close_threshold: Distance below which to use more lenient threshold
    
#     Returns:
#         tuple: (is_stuck: bool, new_check_time: float, new_dist: float)
#     """
#     # Don't check for stuck if very close to goal
#     if current_dist < 1.0:
#         return False, last_stuck_check_time, last_stuck_dist
    
#     if last_stuck_check_time is None:
#         return False, time.time(), current_dist
    
#     # Check every check_interval seconds
#     now = time.time()
#     if now - last_stuck_check_time > check_interval:
#         # If robot is close to goal, be more lenient
#         if current_dist < close_threshold:
#             improvement_threshold = 0.1
#         else:
#             improvement_threshold = 0.2
        
#         # If distance hasn't improved by threshold, consider stuck
#         if last_stuck_dist-current_dist < improvement_threshold:
#             return True, now, current_dist
#         return False, now, current_dist
    
#     return False, last_stuck_check_time, last_stuck_dist
