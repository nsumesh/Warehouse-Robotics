import numpy as np
from rl_nav.constants import lidar_bins


def encode_task(task, phase=None):
    if phase == "pickup" or task is None:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    if task == 'A':
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    elif task == 'B':
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    elif task == 'C':
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def observation(scan, pose, goal, task=None, phase=None):
    if scan is None or goal is None:
        return None
    scan_array = np.array(scan, dtype=np.float32)
    dx, dy = (np.array(goal) - np.array(pose[:2]))
    tail = np.array([dx, dy, pose[2]], dtype=np.float32)
    encoded_task = encode_task(task, phase)
    return np.concatenate([scan_array, tail, encoded_task], axis=0)
