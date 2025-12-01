#!/usr/bin/env python3
"""
Evaluate trained PPO model on navigation tasks.
"""
import os
import rclpy
from rl_nav.train_ppo import Tb3Env, GymTb3
from stable_baselines3 import PPO


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained PPO model")
    default_model = os.path.abspath("ppo_runs/tb3_ppo.zip")
    parser.add_argument("--model", type=str, default=default_model,
                        help="Path to trained PPO model (default: ppo_runs/tb3_ppo.zip)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return

    rclpy.init()
    node = Tb3Env()
    env = GymTb3(node)
    model = PPO.load(args.model)
    print(f"Loaded model from {args.model}\n")

    success_count = 0
    total_steps = 0
    total_distance = 0.0

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        final_info = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            final_info = info

        if final_info:
            success = final_info.get("success", False)
            dist = final_info.get("distance_to_goal", 0.0)
            steps = final_info.get("episode_steps", 0)

        if success:
            success_count += 1
            total_steps += steps
            total_distance += dist

            print(f"[Episode {ep+1}] success={success}  steps={steps}  distance={dist:.2f}m")

    print("\n=== Results ===")
    print(f"Episodes: {args.episodes}")
    print(f"Success rate: {success_count / args.episodes:.2%}")
    print(f"Avg steps: {total_steps / args.episodes:.1f}")
    print(f"Avg final distance: {total_distance / args.episodes:.2f}m")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
