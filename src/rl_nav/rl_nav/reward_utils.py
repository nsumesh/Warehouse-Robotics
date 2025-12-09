"""
Reward calculation utilities and constants.
"""
import math
from rl_nav.constants import SUCCESS_RADIUS, CLOSE_RADIUS, DOCK_A, DOCK_B, DOCK_C, PICKUP


def angle_diff(a: float, b: float) -> float:
    """
    Smallest signed difference between two angles in [-pi, pi].
    
    Args:
        a: First angle in radians
        b: Second angle in radians
    
    Returns:
        float: Signed angle difference in [-pi, pi]
    """
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


# Reward constants - tuned for Stage 3 two-phase training
K_PROGRESS = 3.0  # Progress reward multiplier
K_TIME = 0.003    # Time penalty per step
K_COLLISION = 5.0  # Collision penalty
K_SUCCESS = 15.0  # Success reward
K_PICKUP_SUCCESS = 7.0  # Pickup success reward
K_CLOSE_ZONE = 1.0  # One-time bonus for entering close zone
K_WRONG_DOCK = 5.0  # Penalty for reaching wrong dock


def calculate_progress_reward(prev_dist, current_dist, k_progress=None):
    """
    Calculate progress reward based on distance improvement.
    
    Args:
        prev_dist: Previous distance to goal
        current_dist: Current distance to goal
        k_progress: Progress reward multiplier
    
    Returns:
        float: Progress reward
    """
    if k_progress is None:
        k_progress = K_PROGRESS
    
    if prev_dist is None:
        return 0.0
    
    delta = prev_dist - current_dist
    return k_progress * delta


def check_close_zone_bonus(dist, in_close_zone, k_bonus=None):
    """
    Check if robot should receive close zone bonus.
    
    Args:
        dist: Current distance to goal
        in_close_zone: Whether robot was previously in close zone
        k_bonus: Bonus amount
    
    Returns:
        tuple: (reward: float, new_in_close_zone: bool)
    """
    if k_bonus is None:
        k_bonus = K_CLOSE_ZONE
    
    # Intermediate bonus for getting close to goal (one-time when entering)
    if dist < CLOSE_RADIUS and dist >= SUCCESS_RADIUS:
        if not in_close_zone:
            return k_bonus, True  # One-time bonus for entering close zone
        return 0.0, True
    elif dist >= CLOSE_RADIUS:
        return 0.0, False  # Reset when leaving close zone
    
    return 0.0, in_close_zone


def check_dock_success(pose, task_class, success_radius=None):
    """
    Check if robot reached the correct dock.
    
    Args:
        pose: Robot pose [x, y, yaw]
        task_class: Expected task class ('A', 'B', or 'C')
        success_radius: Success radius threshold
    
    Returns:
        tuple: (success: bool, wrong_dock: bool, closest_dock: str)
    """
    if success_radius is None:
        success_radius = SUCCESS_RADIUS
    
    # Check which dock is closest
    dist_to_a = math.hypot(pose[0] - DOCK_A[0], pose[1] - DOCK_A[1])
    dist_to_b = math.hypot(pose[0] - DOCK_B[0], pose[1] - DOCK_B[1])
    dist_to_c = math.hypot(pose[0] - DOCK_C[0], pose[1] - DOCK_C[1])
    
    # Find closest dock
    closest_dock = min([('A', dist_to_a), ('B', dist_to_b), ('C', dist_to_c)], key=lambda x: x[1])
    
    if closest_dock[1] < success_radius:
        if closest_dock[0] == task_class:
            return True, False, closest_dock[0]  # Correct dock
        else:
            return False, True, closest_dock[0]  # Wrong dock
    
    return False, False, closest_dock[0]  # Not at any dock
