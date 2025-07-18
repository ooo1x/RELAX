import rospy
import numpy as np
import tf2_ros
import os
import argparse
import time

from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from franka_rl_env import FrankaRLEnv
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

# --- Consistent Directory Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DDPG_dir = os.path.join(BASE_DIR, "experiments", "DDPG")
MODELS_BASE_DIR = os.path.join(DDPG_dir, "models")

end_effector_position = np.zeros(3, dtype=np.float32)

def find_latest_run_dir(base_dir):
    """Finds the most recent run directory in a given base directory."""
    if not os.path.exists(base_dir):
        return None
    
    run_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not run_dirs:
        return None
    
    run_dirs.sort()
    return run_dirs[-1]

def joint_state_callback(msg):
    """ROS subscriber callback to get the end-effector position via TF."""
    global end_effector_position
    try:
        # Ensure tfBuffer is available globally or passed correctly
        trans = tfBuffer.lookup_transform('world', 'panda_hand_tcp', rospy.Time(0))
        
        end_effector_position = np.array([
            trans.transform.translation.x,
            trans.transform.translation.y,
            trans.transform.translation.z
        ], dtype=np.float32)

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logwarn(f"Transform lookup failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained DDPG agent for the Franka Robot.")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Specific run ID to load for evaluation. Defaults to the latest run.")
    args = parser.parse_args()

    rospy.init_node('DDPG_evaluator', anonymous=True)

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    rospy.Subscriber("/joint_states", JointState, joint_state_callback, queue_size=1)
    
    rospy.loginfo("[EVAL] Waiting for TF transforms to become available...")
    time.sleep(2.0) # Give time for the TF buffer to fill

    # --- Find Model and Stats to Load ---
    run_id_to_load = args.run_id if args.run_id else find_latest_run_dir(MODELS_BASE_DIR)

    if not run_id_to_load:
        rospy.logerr("[EVAL] No training runs found in 'experiments/DDPG/models/'. Cannot proceed.")
        exit()
        
    rospy.loginfo(f"[EVAL] Using run ID: {run_id_to_load}")

    model_load_dir = os.path.join(MODELS_BASE_DIR, run_id_to_load)
    model_path = os.path.join(model_load_dir, "ddpg_franka_model.zip")
    stats_path = os.path.join(model_load_dir, "vec_normalize_stats.pkl")

    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        rospy.logerr(f"[EVAL] Model or stats file not found in '{model_load_dir}'.")
        rospy.logerr("Please ensure both 'ddpg_franka_model.zip' and 'vec_normalize_stats.pkl' exist.")
        exit()

    # --- Create Environment and Load Normalization Stats ---
    # 1. Create the base environment
    env_lambda = lambda: FrankaRLEnv()
    eval_env = make_vec_env(env_lambda, n_envs=1)

    # 2. Load the normalization stats and wrap the environment
    # Set training=False to prevent the stats from being updated during evaluation
    rospy.loginfo(f"[EVAL] Loading normalization stats from: {stats_path}")
    eval_env = VecNormalize.load(stats_path, eval_env)
    eval_env.training = False 
    eval_env.norm_reward = False

    # --- Load the Model ---
    rospy.loginfo(f"[EVAL] Loading trained model from: {model_path}")
    model = DDPG.load(model_path, env=eval_env)

    # --- Run Evaluation Loop ---
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

            info = info[0]
            if info["collision"]:
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

    # --- Print Summary ---
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    mean_steps = np.mean(all_steps)

    success_rate = (success_count / num_eval_episodes) * 100.0
    rospy.loginfo("="*50)
    rospy.loginfo("               Evaluation Summary")
    rospy.loginfo("="*50)
    rospy.loginfo(f"Episodes:         {num_eval_episodes}")
    rospy.loginfo(f"Success:          {success_count} ({success_rate:.2f}%)")
    rospy.loginfo(f"Failures:         {failure_count} ({100 - success_rate:.2f}%)")
    rospy.loginfo("="*50)