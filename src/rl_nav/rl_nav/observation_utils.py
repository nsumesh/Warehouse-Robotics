"""
Observation building utilities for PPO training and execution.
"""
import numpy as np
from rl_nav.constants import NUM_SCAN_BINS


def encode_task_class_onehot(task_class, phase=None):
    """
    Encode task class as one-hot vector.
    
    Args:
        task_class: Task class ('A', 'B', 'C', or None)
        phase: Current phase (for determining encoding strategy)
    
    Returns:
        np.array: One-hot encoded task class [3 dims]
    """
    # During GO_PICKUP or no task class: use dummy encoding
    if phase == "GO_PICKUP" or task_class is None:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    # During GO_DROPOFF: use actual task class
    if task_class == 'A':
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    elif task_class == 'B':
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    elif task_class == 'C':
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def build_observation(scan, pose, goal, task_class=None, phase=None):
    """
    Build 30-dim observation: 24 scan + [dx, dy, yaw] + task_class one-hot.
    
    Args:
        scan: LiDAR scan bins [24 dims]
        pose: Robot pose [x, y, yaw]
        goal: Goal position [x, y]
        task_class: Task class ('A', 'B', 'C', or None)
        phase: Current phase (for task class encoding)
    
    Returns:
        np.array: Observation vector [30 dims] or None if inputs invalid
    """
    if scan is None or goal is None:
        return None
    
    # LiDAR bins (24 dims)
    scan_array = np.array(scan, dtype=np.float32)
    
    # Goal relative pose (3 dims)
    dx, dy = (np.array(goal) - np.array(pose[:2]))
    tail = np.array([dx, dy, pose[2]], dtype=np.float32)
    
    # Task class one-hot (3 dims)
    task_onehot = encode_task_class_onehot(task_class, phase)
    
    return np.concatenate([scan_array, tail, task_onehot], axis=0)
