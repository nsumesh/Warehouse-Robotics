# evaluate_ppo.py
import os
import numpy as np
import rclpy

from rl_nav.train_ppo import Tb3Env, GymTb3   # adjust if package name differs
from stable_baselines3 import PPO


def main():
    import argparse

    parser = argparse.ArgumentParser()
    # Try to find model in repo ppo_runs directory (relative to current working dir)
    default_model = os.path.abspath("ppo_runs/tb3_ppo.zip")
    
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="Path to trained PPO model .zip file (default: ppo_runs/tb3_ppo.zip in repo)",
    )
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    rclpy.init()
    node = Tb3Env()
    env = GymTb3(node)

    model = PPO.load(args.model)
    print(f"Loaded model from {args.model}")

    success_count = 0
    total_collisions = 0
    total_distance = 0.0
    total_steps = 0
    min_obstacle_dists = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                break

        # Read metrics from node
        success = node.success
        ep_collisions = node.episode_collisions
        ep_dist = node.episode_distance
        ep_steps = node.episode_steps
        ep_min_obs = node.min_obstacle_dist

        if success:
            success_count += 1
        total_collisions += ep_collisions
        total_distance += ep_dist
        total_steps += ep_steps
        min_obstacle_dists.append(ep_min_obs)

        print(
            f"[Episode {ep+1}] "
            f"success={success}  steps={ep_steps}  "
            f"distance={ep_dist:.2f} m  collisions={ep_collisions}  "
            f"min_obstacle_dist={ep_min_obs:.2f} m"
        )

    print("\n=== Aggregate metrics ===")
    print(f"Episodes: {args.episodes}")
    print(f"Success rate: {success_count / args.episodes:.2f}")
    print(f"Avg steps: {total_steps / args.episodes:.1f}")
    print(f"Avg distance: {total_distance / args.episodes:.2f} m")
    print(f"Total collisions: {total_collisions}")
    if min_obstacle_dists:
        print(
            f"Avg closest obstacle: {np.mean(min_obstacle_dists):.2f} m"
        )

    rclpy.shutdown()


if __name__ == "__main__":
    main()
