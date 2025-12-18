'''
navigation_functions.py : this file provides navigation and vision functions for the robot, it processes LiDAR scan data and detects collisions and determins goal reaching conditions.
'''

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

    ranges = np.array(scan_msg.ranges, dtype=np.float32)  
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

