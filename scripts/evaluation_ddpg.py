import os
import argparse
import time
import numpy as np
import rospy

from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from franka_rl_env import FrankaRLEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments", "DDPG") 
MODELS_BASE_DIR = os.path.join(EXPERIMENTS_DIR, "final_models")

def find_latest_run_dir(base_dir):
    """Finds the most recent run directory."""
    if not os.path.exists(base_dir):
        return None
    run_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not run_dirs:
        return None
    run_dirs.sort()
    return run_dirs[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained DDPG agent for the Franka Robot.")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Specific run ID to load for evaluation. Defaults to the latest run.")
    args = parser.parse_args()

    rospy.init_node('DDPG_evaluator', anonymous=True)

    run_id_to_load = args.run_id if args.run_id else find_latest_run_dir(MODELS_BASE_DIR)

    if not run_id_to_load:
        rospy.logerr(f"[EVAL] No training runs found in '{MODELS_BASE_DIR}'. Cannot proceed.")
        exit()
        
    rospy.loginfo(f"[EVAL] Using run ID: {run_id_to_load}")

    model_load_dir = os.path.join(MODELS_BASE_DIR, run_id_to_load)
    model_path = os.path.join(model_load_dir, "ddpg_franka_model.zip")
    stats_path = os.path.join(model_load_dir, "vec_normalize_stats.pkl")

    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        rospy.logerr(f"[EVAL] Model or stats file not found in '{model_load_dir}'.")
        rospy.logerr("Please ensure both 'ddpg_franka_model.zip' and 'vec_normalize_stats.pkl' exist.")
        exit()

    eval_env = make_vec_env(lambda: FrankaRLEnv(), n_envs=1)
    
    rospy.loginfo(f"[EVAL] Loading normalization stats from: {stats_path}")
    eval_env = VecNormalize.load(stats_path, eval_env)
    eval_env.training = False 
    eval_env.norm_reward = False

    rospy.loginfo(f"[EVAL] Loading trained model from: {model_path}")
    model = DDPG.load(model_path, env=eval_env)

    num_eval_episodes = 100
    rospy.loginfo(f"[EVAL] Starting evaluation for {num_eval_episodes} episodes...")
    
    all_rewards = []
    all_steps = []
    success_count = 0
    failure_count = 0

    for i in range(num_eval_episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        collided = False

        while not done and not rospy.is_shutdown():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            if "collision" in info[0] and info[0]["collision"]:
                collided = True

        if collided:
            failure_count += 1
            rospy.logwarn(f"[EVAL] Episode {i+1}: FAIL (Collision) | Reward: {float(episode_reward):.2f} | Steps: {episode_steps}")
        else:
            success_count += 1
            rospy.loginfo(f"[EVAL] Episode {i+1}: SUCCESS (No Collision) | Reward: {float(episode_reward):.2f} | Steps: {episode_steps}")

        all_rewards.append(episode_reward)
        all_steps.append(episode_steps)
    
    eval_env.close()

    mean_reward = np.mean(all_rewards)
    mean_steps = np.mean(all_steps)
    success_rate = (success_count / num_eval_episodes) * 100.0
    
    print("\n" + "="*50)
    print("               Evaluation Summary")
    print("="*50)
    print(f"Total Episodes:   {num_eval_episodes}")
    print(f"Successes:        {success_count} ({success_rate:.2f}%)")
    print(f"Failures:         {failure_count} ({100 - success_rate:.2f}%)")
    print(f"Mean Reward:      {mean_reward:.2f}")
    print(f"Mean Steps:       {mean_steps:.2f}")
    print("="*50 + "\n")