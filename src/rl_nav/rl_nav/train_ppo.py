import os
import math
import time
import random
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SetEntityState, SpawnEntity, DeleteEntity
from gazebo_msgs.msg import EntityState
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import torch

from rl_nav.constants import warehouse_x_limit_max, warehouse_x_limit_min, warehouse_y_limit_max, warehouse_y_limit_min, dockA, dockB, dockC, pickup, success_region, robot_actions, max_clamp_range, lidar_bins
from rl_nav.gazebo_functions import robot_initilization, entity_spawned, delete_entity
from rl_nav.box_functions import generate_item, get_item_color
from rl_nav.navigation_functions import process_scan_to_bins
from rl_nav.observation_functions import observation
from rl_nav.reward_function import progress, time_penalty, collision_penalty, final_success_reward, pickup_reward, close_bonus, wrong_dock, progress_reward, close_zone_bonus, docking_success

step_time = 0.15

class Tb3Env(Node):
    def __init__(self, curriculum_stage=1):
        super().__init__("tb3_env")
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_into_bins, 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.update_orientation, 10)
        self.reset_cli = self.create_client(SetEntityState, "/set_entity_state")
        self.max_range = max_clamp_range
        self.num_scan_bins = lidar_bins
        self.scan = None
        self.pose = np.zeros(3, dtype=np.float32)
        self.goal = np.array([0.0, 6.0], dtype=np.float32)
        self.collision = False
        self.previous_dist = None
        self.episode_steps = 0
        self.max_steps = 200 if curriculum_stage == 1 else (400 if curriculum_stage == 2 else 600)
        self.in_close_zone = False
        self.actions = robot_actions
        self.curriculum_stage = curriculum_stage
        self.task = None
        self.phase = None
        self.dropoff_goal = None
        self.pickup_reached_time = None
        self.spawn_client = None
        self.delete_client = None
        self.current_item_id = None
        self.item_counter = {'A': 0, 'B': 0, 'C': 0}
        if curriculum_stage == 3:
            self.spawn_client = self.create_client(SpawnEntity, "/spawn_entity")
            self.delete_client = self.create_client(DeleteEntity, "/delete_entity")

    def scan_into_bins(self, msg):
        self.scan, self.collision, _ = process_scan_to_bins(msg, self.num_scan_bins, self.max_range)

    def update_orientation(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose[:] = (x, y, yaw)

    def pickup_item_spawning(self, task):
        if not self.spawn_client or not self.spawn_client.service_is_ready():
            return None
        self.item_counter[task] += 1
        item_id = "item_" + task + "_" + str(self.item_counter[task])
        color = get_item_color(task)
        item_sdf = generate_item(item_id, color)
        if entity_spawned(self, self.spawn_client, item_id, item_sdf, pickup[0], pickup[1], 0.15):
            return item_id
        return None

    def start_and_goal(self):
        if self.curriculum_stage == 1:
            dock_x, dock_y = dockA
            sx = max(warehouse_x_limit_min, min(warehouse_x_limit_max, dock_x + random.uniform(-0.5, 0.5)))
            sy = max(warehouse_y_limit_min, min(warehouse_y_limit_max, dock_y + random.uniform(-0.5, 0.5)))
            yaw = random.uniform(-math.pi, math.pi)
            self.task = None
            return (sx, sy, yaw), dockA
        elif self.curriculum_stage == 2:
            sx = random.uniform(warehouse_x_limit_min, warehouse_x_limit_max)
            sy = random.uniform(warehouse_y_limit_min, warehouse_y_limit_max)
            yaw = random.uniform(-math.pi, math.pi)
            self.task = None
            return (sx, sy, yaw), dockA
        sx = random.uniform(warehouse_x_limit_min, warehouse_x_limit_max)
        sy = random.uniform(warehouse_y_limit_min, warehouse_y_limit_max)
        yaw = random.uniform(-math.pi, math.pi)
        self.task = random.choice(['A', 'B', 'C'])
        docks = {'A': dockA, 'B': dockB, 'C': dockC}
        self.dropoff_goal = docks[self.task]
        self.phase = "pickup"
        self.pickup_reached_time = None
        return (sx, sy, yaw), pickup

    def build_observation(self):
        scan = self.scan if self.scan is not None else np.ones(self.num_scan_bins, dtype=np.float32)
        task_class = self.task if (self.curriculum_stage == 3 and self.phase == "dropoff") else None
        return observation(scan, self.pose, self.goal, task_class, self.phase)

    def reset(self):
        if self.curriculum_stage == 3 and self.current_item_id:
            delete_entity(self, self.delete_client, self.current_item_id)
            self.current_item_id = None
        self.episode_steps = 0
        self.collision = False
        if self.curriculum_stage == 3:
            self.pickup_reached_time = None
        (sx, sy, syaw), (gx, gy) = self.start_and_goal()
        self.goal[:] = (gx, gy)
        if self.curriculum_stage == 3 and self.task:
            if self.spawn_client:
                self.spawn_client.wait_for_service(timeout_sec=2.0)
            self.current_item_id = self.pickup_item_spawning(self.task)
        if self.reset_cli.wait_for_service(timeout_sec=5.0):
            request = SetEntityState.Request()
            request.state = EntityState()
            request.state.name = "tb3"
            request.state.pose.position.x = float(sx)
            request.state.pose.position.y = float(sy)
            request.state.pose.position.z = 0.01
            request.state.pose.orientation.z = math.sin(syaw / 2.0)
            request.state.pose.orientation.w = math.cos(syaw / 2.0)
            self.reset_cli.call_async(request)
        start_time = time.time()
        self.scan = None
        while (self.scan is None) and (time.time() - start_time < 2.0):
            rclpy.spin_once(self, timeout_sec=0.05)
        self.previous_dist = float(math.hypot(self.goal[0] - self.pose[0], self.goal[1] - self.pose[1]))
        self.in_close_zone = False
        return self.build_observation()
    
    def step(self, action):
        v, w = self.actions[int(action)]
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_pub.publish(twist)
        start_time = time.time()
        while (time.time() - start_time) < step_time:
            rclpy.spin_once(self, timeout_sec=0.01)
        self.episode_steps += 1
        obs = self.build_observation()
        current_distance = float(math.hypot(self.goal[0] - self.pose[0], self.goal[1] - self.pose[1]))
        reward = progress_reward(self.previous_dist, current_distance, progress)
        self.previous_dist = current_distance
        reward -= time_penalty
        bonus, self.in_close_zone = close_zone_bonus(current_distance, self.in_close_zone, close_bonus)
        reward += bonus
        done = False
        success = False
        if self.collision:
            reward -= collision_penalty
            done = True
        if self.curriculum_stage == 3 and self.task and not done:
            if self.phase == "pickup":
                pickup_distance = math.hypot(self.pose[0] - pickup[0], self.pose[1] - pickup[1])
                if pickup_distance < success_region and not self.collision:
                    curr_time = time.time()
                    if self.pickup_reached_time is None:
                        self.pickup_reached_time = curr_time
                    elif curr_time - self.pickup_reached_time >= 1.0:
                        reward += pickup_reward
                        self.phase = "dropoff"
                        self.goal[:] = self.dropoff_goal
                        if self.current_item_id:
                            delete_entity(self, self.delete_client, self.current_item_id)
                        self.previous_dist = float(math.hypot(self.goal[0] - self.pose[0], self.goal[1] - self.pose[1]))
                        self.in_close_zone = False
                        self.pickup_reached_time = None
                elif pickup_distance >= success_region:
                    self.pickup_reached_time = None
            elif self.phase == "dropoff":
                if current_distance < success_region and not self.collision:
                    dock_success, wrong_dock_flag, dock_id = docking_success(self.pose, self.task, success_region)
                    if dock_success:
                        reward += final_success_reward
                        success = True
                        done = True
                    elif wrong_dock_flag:
                        reward -= wrong_dock
                        done = True
        elif current_distance < success_region and not self.collision and not done:
            reward += final_success_reward
            success = True
            done = True
        if self.episode_steps >= self.max_steps:
            done = True
        current_status = {"Distance to Goal": current_distance, "Stage": self.curriculum_stage, "Episode Steps": self.episode_steps, "Success (Binary)": success, "Task type": self.task}
        return obs, reward, done, current_status

class GymEnvironment(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.node.actions))

    def reset(self, *, seed=None, options=None):
        return self.node.reset(), {}

    def step(self, action):
        obs, reward, done, status = self.node.step(int(action))
        terminated = done
        truncated = (self.node.episode_steps >= self.node.max_steps and not terminated)
        return obs, reward, terminated, truncated, status

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--logdir", type=str, default="ppo_runs")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--curriculum-stage", type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()
    rclpy.init()
    log_dir = os.path.abspath(args.logdir)
    os.makedirs(args.logdir, exist_ok=True)
    node = Tb3Env(curriculum_stage=args.curriculum_stage)
    node.get_logger().info("Training of stage " + str(args.curriculum_stage))
    if robot_initilization(node)==False:
        node.get_logger().error("Waffle Pi initialization failed")
        rclpy.shutdown()
        return
    environment = GymEnvironment(node)
    model = None
    if args.resume:
        resume_path = args.resume if os.path.isabs(args.resume) else os.path.join(log_dir, args.resume)
        resume_path = os.path.expanduser(resume_path)
        if os.path.exists(resume_path):
            node.get_logger().info("Resuming from checkpoint: " + resume_path)
            model = PPO.load(resume_path, env=environment)
        else:
            node.get_logger().warn("Checkpoint not found: " + resume_path + ". Training will start fresh.")
    if model is None:
        model = PPO("MlpPolicy", environment, verbose=1, tensorboard_log=args.logdir, n_steps=256, batch_size=64, n_epochs=10, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, policy_kwargs=dict(net_arch=[64, 64], activation_fn=torch.nn.Tanh))
        node.get_logger().info("New training starting")
    model.learn(total_timesteps=args.timesteps)
    model_name = "ppo_stage1.zip" if args.curriculum_stage == 1 else ("ppo_stage2.zip" if args.curriculum_stage == 2 else "ppo_stage3_sorting.zip")
    model_path = os.path.join(log_dir, model_name)
    model.save(model_path)
    node.get_logger().info("Policy saved to " + model_path)
    print("Policy saved to " + model_path)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
