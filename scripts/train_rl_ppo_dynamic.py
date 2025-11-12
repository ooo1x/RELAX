import os
import re
import time
import argparse
import rospy

from stable_baselines3 import PPO 
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure

from franka_rl_env_dynamic import FrankaRLEnv
from std_msgs.msg import Int32

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments", "PPO") 

MODELS_DIR = os.path.join(EXPERIMENTS_DIR, "final_models")
CHECKPOINTS_DIR = os.path.join(EXPERIMENTS_DIR, "checkpoints")
TENSORBOARD_LOG_DIR = os.path.join(EXPERIMENTS_DIR, "tensorboard_logs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)


# --- UTILITY FUNCTIONS ---
def find_latest_run_dir(base_dir):
    if not os.path.exists(base_dir):
        return None
    run_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not run_dirs:
        return None
    run_dirs.sort()
    return run_dirs[-1]

def find_latest_checkpoint(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None, None
    model_files = os.listdir(ckpt_dir)
    pattern = re.compile(r'ppo_franka_(\d+)_steps\.zip') 
    steps = []
    for file in model_files:
        match = pattern.match(file)
        if match:
            steps.append(int(match.group(1)))
    if not steps:
        return None, None
    latest_step = max(steps)
    model_path = os.path.join(ckpt_dir, f'ppo_franka_{latest_step}_steps.zip')
    vec_path = os.path.join(ckpt_dir, f'ppo_franka_vecnormalize_{latest_step}_steps.pkl') 
    if os.path.exists(model_path) and os.path.exists(vec_path):
        return model_path, vec_path
    return None, None


# --- ROS & CALLBACKS (Unchanged) ---
class RosStateTracker:
    def __init__(self):
        self.current_cpp_episode = 0
        rospy.Subscriber("/episode", Int32, self._cpp_episode_callback)
    def _cpp_episode_callback(self, msg):
        self.current_cpp_episode = msg.data
    def get_cpp_episode(self):
        return self.current_cpp_episode

class StopWhenCppDone(BaseCallback):
    def __init__(self, get_cpp_episode_func, max_cpp_episodes, verbose=0):
        super().__init__(verbose)
        self.get_cpp_episode = get_cpp_episode_func
        self.max_cpp_episodes = max_cpp_episodes
    def _on_step(self) -> bool:
        rospy.sleep(0.001)
        return self.get_cpp_episode() < self.max_cpp_episodes


# --- MAIN SCRIPT ---
if __name__ == "__main__":
    rospy.init_node('PPO_trainer', anonymous=True) 

    parser = argparse.ArgumentParser(description="Train a PPO agent for the Franka robot.") 
    parser.add_argument("--new", action="store_true", help="Force a new training run, ignoring all existing models.")
    parser.add_argument("--cl", action="store_true", help="Continue training from the latest completed run's final model.")
    parser.add_argument("--lcr", type=str, metavar="RUN_ID", help="Load the latest checkpoint from a specific run ID (e.g., 20250723-173706).")
    args = parser.parse_args()

    # --- Configuration ---
    MAX_CPP_EPISODES = 5000
    
    # --- Variable Initialization (Unchanged) ---
    load_model_path = None
    load_stats_path = None
    is_resuming_from_checkpoint = False
    run_id = time.strftime("%Y%m%d-%H%M%S")
    
    # --- Mode Selection Logic (Unchanged) ---
    if args.new:
        rospy.loginfo("Mode: Starting a completely new training run.")
    elif args.cl:
        rospy.loginfo("Mode: Continuing from the latest completed run.")
        latest_run = find_latest_run_dir(MODELS_DIR)
        if latest_run:
            run_id = latest_run
            load_model_path = os.path.join(MODELS_DIR, run_id, "ppo_franka_model.zip") # <--- Changed
            load_stats_path = os.path.join(MODELS_DIR, run_id, "vec_normalize_stats.pkl")
            rospy.loginfo(f"Found latest run '{run_id}'. Will load final model and stats.")
        else:
            rospy.logwarn("No completed runs found to continue from. Starting a new run instead.")
    elif args.lcr:
        rospy.loginfo(f"Mode: Loading latest checkpoint from run '{args.lcr}'.")
        run_id = args.lcr
        checkpoint_folder = os.path.join(CHECKPOINTS_DIR, run_id)
        load_model_path, load_stats_path = find_latest_checkpoint(checkpoint_folder)
        if load_model_path:
            is_resuming_from_checkpoint = True
            rospy.loginfo(f"Found checkpoint to load: {os.path.basename(load_model_path)}")
        else:
            rospy.logerr(f"Could not find any valid checkpoints in '{checkpoint_folder}'. Exiting.")
            exit()
    else:
        rospy.loginfo("Mode: Defaulting to a new training run (no flags specified).")

    # --- Setup Paths and Environment --
    model_save_dir = os.path.join(MODELS_DIR, run_id)
    checkpoint_save_dir = os.path.join(CHECKPOINTS_DIR, run_id)
    tensorboard_log_dir = os.path.join(TENSORBOARD_LOG_DIR, run_id)
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(checkpoint_save_dir, exist_ok=True)
    model_final_save_path = os.path.join(model_save_dir, "ppo_franka_model.zip") # <--- Changed
    stats_final_save_path = os.path.join(model_save_dir, "vec_normalize_stats.pkl")

    rospy.loginfo("Initializing FrankaRLEnv...")
    vec_env = make_vec_env(lambda: FrankaRLEnv(), n_envs=1)
    norm_vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, gamma=0.99)

    # --- Load or Create Model ---
    is_model_loaded = load_model_path and os.path.exists(load_model_path) and os.path.exists(load_stats_path)

    if is_model_loaded:
        rospy.loginfo(f"Loading VecNormalize stats from: {load_stats_path}")
        norm_vec_env = VecNormalize.load(load_stats_path, vec_env)
        rospy.loginfo(f"Loading PPO model from: {load_model_path}") 
        model = PPO.load(load_model_path, env=norm_vec_env)
        rospy.loginfo("Reusing existing TensorBoard logs for continuous tracking.")
        new_logger = configure(tensorboard_log_dir, ["tensorboard"])
        model.set_logger(new_logger)
    else:
        rospy.loginfo("Creating a new PPO model.") 
        model = PPO("MlpPolicy", norm_vec_env, verbose=1, 
                    gamma=0.99,
                    n_steps=512, 
                    batch_size=64, 
                    tensorboard_log=TENSORBOARD_LOG_DIR)

    # --- Setup Callbacks ---
    ros_tracker = RosStateTracker()
    checkpoint_callback = CheckpointCallback(
        save_freq=800,
        save_path=checkpoint_save_dir,
        name_prefix="ppo_franka", 
        save_vecnormalize=True
    )
    stop_callback = StopWhenCppDone(
        get_cpp_episode_func=ros_tracker.get_cpp_episode,
        max_cpp_episodes=MAX_CPP_EPISODES
    )
    callbacks = CallbackList([checkpoint_callback, stop_callback])

    # --- Train The Model ---
    rospy.loginfo(f"Starting PPO training. Run ID: {run_id}") 
    rospy.loginfo(f"Checkpoints will be saved in: {checkpoint_save_dir}")
    rospy.loginfo(f"TensorBoard logs are in: {tensorboard_log_dir}")
    reset_timesteps = not is_model_loaded
    rospy.loginfo(f"Starting model.learn() with reset_num_timesteps={reset_timesteps}")
    
    model.learn(total_timesteps=10000000, callback=callbacks, reset_num_timesteps=reset_timesteps)
    
    # --- Final Save ---
    rospy.loginfo("Training finished. Saving final model and normalization stats...")
    model.save(model_final_save_path)
    norm_vec_env.save(stats_final_save_path)
    rospy.loginfo(f"Final model saved to: {model_final_save_path}")
    rospy.loginfo(f"Final stats saved to: {stats_final_save_path}")
    rospy.loginfo("Exiting.")