'''
reward_function.py : this file sets up the reward shaping function for PPO training. It provides appropriatre rewards for progress, task complettion, collision and incorrect actions
'''

import math
from rl_nav.constants import success_region, sucess_close_bonus, dockA, dockB, dockC, pickup

progress = 3.0  # Progress reward multiplier
time_penalty = 0.003    # Time penalty per step
collision_penalty = 5.0  # Collision penalty
final_success_reward = 15.0  # Success reward
pickup_reward = 7.0  # Pickup success reward
close_bonus = 1.0  # One-time bonus for entering close zone
wrong_dock = 5.0  # Penalty for reaching wrong dock

def progress_reward(previous_distance, current_distance, progress_toward=3):
    if previous_distance is None:
        return 0.0 
    diff = previous_distance - current_distance
    return progress_toward * diff


def close_zone_bonus(distance, close_zone, bonus=close_bonus):
    if sucess_close_bonus>distance>=success_region:
        if not close_zone:
            return bonus, True
    return 0.0, close_zone


def docking_success(pose, task, success_radius=success_region):
    distA = math.hypot(pose[0] - dockA[0], pose[1] - dockA[1])
    distB = math.hypot(pose[0] - dockB[0], pose[1] - dockB[1])
    distC = math.hypot(pose[0] - dockC[0], pose[1] - dockC[1])
    closest_dock, distance = min([('A', distA), ('B', distB), ('C', distC)], key=lambda x: x[1])
    if distance < success_radius:
        if closest_dock == task:
            return True, False, closest_dock
        return False, True, closest_dock
    return False, False, closest_dock
