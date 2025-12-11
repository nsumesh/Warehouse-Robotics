#!/usr/bin/env python3
"""
Evaluate trained PPO policy for warehouse navigation with virtual sorting.

Example commands:
# Evaluate Stage 1 model (20 episodes)
ros2 run rl_nav eval_policy --model-path ppo_runs/ppo_stage1.zip --curriculum-stage 1 --episodes 20

# Evaluate Stage 2 model (20 episodes)
ros2 run rl_nav eval_policy --model-path ppo_runs/ppo_stage2.zip --curriculum-stage 2 --episodes 20

# Evaluate Stage 3 sorting model (20 episodes)
ros2 run rl_nav eval_policy --model-path ppo_runs/ppo_stage3_sorting.zip --curriculum-stage 3 --episodes 20
"""
import os
import rclpy
from rl_nav.train_ppo import Tb3Env, GymEnvironment
from rl_nav.gazebo_functions import robot_initilization
from stable_baselines3 import PPO


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained PPO policy")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained PPO model .zip file")
    parser.add_argument("--curriculum-stage", type=int, required=True, choices=[1, 2, 3],
                        help="Curriculum stage (must match training stage)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to evaluate (default: 10)")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return

    rclpy.init()
    
    # Create environment with specified curriculum stage
    node = Tb3Env(curriculum_stage=args.curriculum_stage)
    env = GymEnvironment(node)
    
    # Load model
    model = PPO.load(args.model_path)
    print(f"Loaded model from {args.model_path}")
    print(f"Evaluating with curriculum stage {args.curriculum_stage} for {args.episodes} episodes\n")

    if not robot_initilization(node):
        node.get_logger().error("TB3 spawn failed")
        rclpy.shutdown()
        return

    # Evaluation metrics
    success_count = 0
    collision_count = 0
    total_reward = 0.0
    total_steps = 0
    total_distance = 0.0

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        final_info = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1
            final_info = info

        if final_info:
            # Updated status dictionary keys to match train_ppo.py
            success_binary = final_info.get("Success (Binary)", 0)
            if success_binary == 1:
                success_count += 1
            
            stage = final_info.get("Stage", args.curriculum_stage)
            task_type = final_info.get("Task type", None)
            
            if stage == 3 and task_type:
                task_info = f" (task={task_type})"
            else:
                task_info = ""
            
            dist = final_info.get("Distance to Goal", 0.0)
            total_reward += episode_reward
            total_steps += episode_steps
            total_distance += dist

            print(f"Episode {ep+1}: reward={episode_reward:.2f} steps={episode_steps} "
                  f"final_dist={dist:.2f}m success={success_binary==1}{task_info}")

    # Print aggregate statistics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Episodes: {args.episodes}")
    print(f"Success rate: {success_count}/{args.episodes} ({100*success_count/args.episodes:.1f}%)")
    print(f"Collisions: {collision_count}")
    print(f"Average episode reward: {total_reward/args.episodes:.2f}")
    print(f"Average episode steps: {total_steps/args.episodes:.1f}")
    print(f"Average final distance: {total_distance/args.episodes:.2f}m")
    print("="*50)

    rclpy.shutdown()


if __name__ == "__main__":
    main()



