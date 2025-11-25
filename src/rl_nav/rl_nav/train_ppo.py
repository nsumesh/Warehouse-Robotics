# train_ppo.py
import os
import math
import time
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState

import gymnasium as gym
from gymnasium import spaces


class Tb3Env(Node):
    """
    TurtleBot3 PPO environment in the warehouse world.

    - Spawns robot near the pallet area at the BOTTOM of the map
    - Random goals are placed INSIDE the warehouse aisles
    - Reward is potential-based: positive when moving closer to goal
    - Strong penalty for collisions and being too close to obstacles
    """

    def __init__(self):
        super().__init__('tb3_env')

        # --- ROS I/O ---
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self._on_scan, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._on_odom, 10
        )
        self.reset_cli = self.create_client(SetEntityState, '/set_entity_state')

        # --- State ---
        self.max_range = 3.5
        self.scan = None  # 24-dim downsampled scan
        self.pose = np.zeros(3, dtype=np.float32)  # x, y, yaw
        self.goal = np.array([0.0, 6.0], dtype=np.float32)
        self.collision = False

        # rollout / timing
        self.step_time = 0.15
        self.episode_steps = 0
        self.max_steps = 250

        # Potential-based reward
        self.prev_dist = None
        self.no_progress_steps = 0

        # Discrete actions: (linear v, angular w)
        self.actions = [
            (0.15,  0.8),
            (0.15,  0.0),
            (0.15, -0.8),
            (0.00,  0.8),
            (0.00, -0.8),
        ]

        # Metrics for logging
        self.episode_idx = 0
        self.episode_collisions = 0
        self.min_obstacle_dist = self.max_range

        # CSV metrics file
        self.metrics_path = os.path.expanduser(
            "~/MSML_642_FinalProject/ppo_runs/episode_metrics.csv"
        )
        if not os.path.exists(os.path.dirname(self.metrics_path)):
            os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        # Write header once
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w") as f:
                f.write(
                    "episode,steps,final_dist,collisions,min_obst_dist,success\n"
                )

    # ------------------------------------------------------------------ callbacks

    def _on_scan(self, msg: LaserScan):
        rng = np.array(msg.ranges, dtype=np.float32)
        n = 24
        inds = np.linspace(0, len(rng) - 1, n).astype(int)
        r = np.clip(rng[inds], 0.0, self.max_range)

        # consider NaNs as max range
        r[np.isnan(r)] = self.max_range

        self.collision = bool((r < 0.18).any())
        self.scan = r / self.max_range

        current_min = float(np.nanmin(r))
        if current_min < self.min_obstacle_dist:
            self.min_obstacle_dist = current_min

    def _on_odom(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.pose[:] = (x, y, yaw)

    # ------------------------------------------------------------------ env API

    def _sample_start_and_goal(self):
        """
        Choose a start near the pallet area at the bottom of the map,
        and a goal somewhere inside the warehouse aisles.
        Adjust numbers only if your world scale is very different.
        """

        # Start: bottom region / pallet zone
        start_x = np.random.uniform(-2.0, 2.0)
        start_y = np.random.uniform(-9.0, -7.5)
        # Face roughly "up" into the aisles
        start_yaw = np.random.uniform(math.pi/2 - 0.4, math.pi/2 + 0.4)

        # Goal: inside aisle region
        goal_x = np.random.uniform(-2.0, 2.0)
        goal_y = np.random.uniform(3.0, 7.0)

        return (start_x, start_y, start_yaw), (goal_x, goal_y)

    def reset(self):
        # Before resetting, log previous episode metrics if any
        if self.episode_steps > 0 and self.prev_dist is not None:
            final_dist = float(np.linalg.norm(self.goal - self.pose[:2]))
            success = int(final_dist < 0.4 and not self.collision)
            with open(self.metrics_path, "a") as f:
                f.write(
                    f"{self.episode_idx},"
                    f"{self.episode_steps},"
                    f"{final_dist:.3f},"
                    f"{self.episode_collisions},"
                    f"{self.min_obstacle_dist:.3f},"
                    f"{success}\n"
                )

        self.episode_idx += 1
        self.episode_steps = 0
        self.episode_collisions = 0
        self.min_obstacle_dist = self.max_range
        self.no_progress_steps = 0
        self.collision = False

        # Sample new start + goal
        start, goal = self._sample_start_and_goal()
        self.goal[:] = goal

        # Teleport robot
        if not self.reset_cli.service_is_ready():
            self.reset_cli.wait_for_service(timeout_sec=5.0)

        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = 'tb3'
        req.state.pose.position.x = float(start[0])
        req.state.pose.position.y = float(start[1])
        req.state.pose.position.z = 0.01
        cy, sy = math.cos(start[2] * 0.5), math.sin(start[2] * 0.5)
        req.state.pose.orientation.z = sy
        req.state.pose.orientation.w = cy
        self.reset_cli.call_async(req)

        # Wait for scan + odom to be ready
        t0 = time.time()
        self.scan = None
        while (self.scan is None or np.allclose(self.pose, 0.0)) and \
                (time.time() - t0 < 2.0):
            rclpy.spin_once(self, timeout_sec=0.05)

        # Initialize potential
        self.prev_dist = float(np.linalg.norm(self.goal - self.pose[:2]))

        return self._obs()

    def step(self, a_idx: int):
        # --- Apply action ---
        v, w = self.actions[int(a_idx)]
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_pub.publish(twist)

        # Spin for step_time
        t_end = self.get_clock().now().nanoseconds + int(self.step_time * 1e9)
        while self.get_clock().now().nanoseconds < t_end:
            rclpy.spin_once(self, timeout_sec=0.02)

        self.episode_steps += 1

        # --- Observation & distances ---
        obs = self._obs()
        dist = float(np.linalg.norm(self.goal - self.pose[:2]))

        # Potential-based reward: positive if closer than previous step
        delta = self.prev_dist - dist
        self.prev_dist = dist

        # Base reward
        rew = 0.5 * delta - 0.01  # small time penalty

        # Obstacle proximity penalty
        if self.scan is not None:
            obs_min = float(np.min(self.scan) * self.max_range)
            if obs_min < 0.5:
                rew -= 0.2 * (0.5 - obs_min)  # stronger penalty when very close

        done = False

        # Collision penalty
        if self.collision:
            rew -= 5.0
            done = True
            self.episode_collisions += 1

        # Success when close to goal
        if dist < 0.4:
            rew += 8.0
            done = True

        # Stuck detection: no progress for many steps
        if delta < 0.01:  # not getting closer
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0

        if self.no_progress_steps > 30:
            rew -= 2.0
            done = True

        # Max episode length
        if self.episode_steps >= self.max_steps:
            done = True

        info = {
            "distance_to_goal": dist,
            "episode_steps": self.episode_steps,
        }
        return obs, rew, done, info

    # ------------------------------------------------------------------ helpers

    def _obs(self):
        scan = self.scan if self.scan is not None else np.zeros(24, dtype=np.float32)
        dx, dy = (self.goal - self.pose[:2]).tolist()
        tail = np.array([dx, dy, self.pose[2]], dtype=np.float32)
        return np.concatenate([scan, tail], axis=0)


class GymTb3(gym.Env):
    """
    Gymnasium wrapper around Tb3Env for stable-baselines3 PPO.
    Observation: 24 scan samples + (dx, dy, yaw) = 27
    Action: Discrete(5)
    """
    metadata = {"render_modes": []}

    def __init__(self, node: Tb3Env):
        super().__init__()
        self.node = node
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.node.actions))

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs = self.node.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.node.step(action)
        terminated = done
        truncated = (self.node.episode_steps >= self.node.max_steps) and not terminated
        return obs, reward, terminated, truncated, info


def main():
    import argparse
    from stable_baselines3 import PPO

    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=15000)
    parser.add_argument(
        '--logdir',
        type=str,
        default=os.path.expanduser('~/MSML_642_FinalProject/ppo_runs')
    )
    args = parser.parse_args()

    rclpy.init()
    node = Tb3Env()
    env = GymTb3(node)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=args.logdir,
        n_steps=512,
        batch_size=256,
        learning_rate=3e-4,
    )

    model.learn(total_timesteps=args.timesteps)

    os.makedirs(args.logdir, exist_ok=True)
    out = os.path.join(args.logdir, 'tb3_ppo.zip')
    model.save(out)
    print("Saved policy to", out)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
