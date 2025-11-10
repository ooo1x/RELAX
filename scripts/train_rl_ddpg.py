import os
import re
import time
import argparse
import numpy as np
import rospy

from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure

from franka_rl_env import FrankaRLEnv # Assuming this file is in the same directory or accessible
from std_msgs.msg import Int32

# --- DIRECTORY SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments", "DDPG")

MODELS_DIR = os.path.join(EXPERIMENTS_DIR, "final_models")
CHECKPOINTS_DIR = os.path.join(EXPERIMENTS_DIR, "checkpoints")
TENSORBOARD_LOG_DIR = os.path.join(EXPERIMENTS_DIR, "tensorboard_logs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)


# --- UTILITY FUNCTIONS ---
def find_latest_run_dir(base_dir):
    """Finds the most recent directory in a given base directory."""
    if not os.path.exists(base_dir):
        return None
    run_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not run_dirs:
        return None
    run_dirs.sort()
    return run_dirs[-1]

def find_latest_checkpoint(ckpt_dir):
    """Finds the latest checkpoint file and VecNormalize stats in a directory."""
    if not os.path.isdir(ckpt_dir):
        return None, None
        
    model_files = os.listdir(ckpt_dir)
    pattern = re.compile(r'ddpg_franka_(\d+)_steps\.zip')
    steps = []

    for file in model_files:
        match = pattern.match(file)
        if match:
            steps.append(int(match.group(1)))

    if not steps:
        return None, None

    latest_step = max(steps)
    model_path = os.path.join(ckpt_dir, f'ddpg_franka_{latest_step}_steps.zip')
    vec_path = os.path.join(ckpt_dir, f'ddpg_franka_vecnormalize_{latest_step}_steps.pkl')
    
    if os.path.exists(model_path) and os.path.exists(vec_path):
        return model_path, vec_path
    return None, None


# --- ROS & CALLBACKS ---
class RosStateTracker:
    """Tracks the current episode number published by the C++ node."""
    def __init__(self):
        self.current_cpp_episode = 0
        rospy.Subscriber("/episode", Int32, self._cpp_episode_callback)

    def _cpp_episode_callback(self, msg):
        self.current_cpp_episode = msg.data

    def get_cpp_episode(self):
        return self.current_cpp_episode

class StopWhenCppDone(BaseCallback):
    """Stops training when the C++ node has completed its episodes."""
    def __init__(self, get_cpp_episode_func, max_cpp_episodes, verbose=0):
        super().__init__(verbose)
        self.get_cpp_episode = get_cpp_episode_func
        self.max_cpp_episodes = max_cpp_episodes
        self.verbose = verbose

    def _on_step(self) -> bool:
        # A small sleep to prevent this from overwhelming ROS spin
        rospy.sleep(0.001)
        # The C++ loop starts at 1, so we check if it has reached the max
        return self.get_cpp_episode() < self.max_cpp_episodes


# --- MAIN SCRIPT ---
if __name__ == "__main__":
    rospy.init_node('DDPG_trainer', anonymous=True)

    parser = argparse.ArgumentParser(description="Train a DDPG agent for the Franka robot.")
    parser.add_argument("--new", action="store_true", help="Force a new training run, ignoring all existing models.")
    parser.add_argument("--cl", action="store_true", help="Continue training from the latest completed run's final model.")
    parser.add_argument("--lcr", type=str, metavar="RUN_ID", help="Load the latest checkpoint from a specific run ID (e.g., 20250723-173706).")
    args = parser.parse_args()

    # --- Configuration ---
    GAMMA = 0.99
    MAX_CPP_EPISODES = 2  # Set the total number of episodes for the C++ node to run
    
    # --- Variable Initialization ---
    load_model_path = None
    load_stats_path = None
    is_resuming_from_checkpoint = False
    
    # The run_id determines the folder names for saving models, checkpoints, and logs.
    # When resuming, we reuse the old run_id. When starting new, we create a new one.
    run_id = time.strftime("%Y%m%d-%H%M%S")
    
    # --- Mode Selection Logic ---
    if args.new:
        rospy.loginfo("Mode: Starting a completely new training run.")
    
    elif args.cl:
        rospy.loginfo("Mode: Continuing from the latest completed run.")
        latest_run = find_latest_run_dir(MODELS_DIR)
        if latest_run:
            run_id = latest_run # Reuse the old run_id for continuous logging
            load_model_path = os.path.join(MODELS_DIR, run_id, "ddpg_franka_model.zip")
            load_stats_path = os.path.join(MODELS_DIR, run_id, "vec_normalize_stats.pkl")
            rospy.loginfo(f"Found latest run '{run_id}'. Will load final model and stats.")
        else:
            rospy.logwarn("No completed runs found to continue from. Starting a new run instead.")

    elif args.lcr:
        rospy.loginfo(f"Mode: Loading latest checkpoint from run '{args.lcr}'.")
        run_id = args.lcr # Reuse the old run_id
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

    # --- Setup Paths and Environment ---
    model_save_dir = os.path.join(MODELS_DIR, run_id)
    checkpoint_save_dir = os.path.join(CHECKPOINTS_DIR, run_id)
    tensorboard_log_dir = os.path.join(TENSORBOARD_LOG_DIR, run_id)
    
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(checkpoint_save_dir, exist_ok=True)
    
    model_final_save_path = os.path.join(model_save_dir, "ddpg_franka_model.zip")
    stats_final_save_path = os.path.join(model_save_dir, "vec_normalize_stats.pkl")

    rospy.loginfo("Initializing FrankaRLEnv...")
    vec_env = make_vec_env(lambda: FrankaRLEnv(), n_envs=1)
    norm_vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, gamma=GAMMA)

    # --- Load or Create Model ---
    is_model_loaded = load_model_path and os.path.exists(load_model_path) and os.path.exists(load_stats_path)

    if is_model_loaded:
        rospy.loginfo(f"Loading VecNormalize stats from: {load_stats_path}")
        norm_vec_env = VecNormalize.load(load_stats_path, vec_env)
        
        rospy.loginfo(f"Loading DDPG model from: {load_model_path}")
        model = DDPG.load(load_model_path, env=norm_vec_env)
        
        rospy.loginfo("Reusing existing TensorBoard logs for continuous tracking.")
        new_logger = configure(tensorboard_log_dir, ["tensorboard"])
        model.set_logger(new_logger)
    else:
        rospy.loginfo("Creating a new DDPG model.")
        n_actions = norm_vec_env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = DDPG("MlpPolicy", norm_vec_env, action_noise=action_noise, verbose=1, 
                     gamma=GAMMA, tensorboard_log=TENSORBOARD_LOG_DIR,
                     buffer_size=100000, learning_starts=800)

    # --- Setup Callbacks ---
    ros_tracker = RosStateTracker()
    
    checkpoint_callback = CheckpointCallback(
        save_freq=400,  # Save a checkpoint every 800 steps
        save_path=checkpoint_save_dir, 
        name_prefix="ddpg_franka",
        save_replay_buffer=True,
        save_vecnormalize=True 
    )
    stop_callback = StopWhenCppDone(
        get_cpp_episode_func=ros_tracker.get_cpp_episode, 
        max_cpp_episodes=MAX_CPP_EPISODES
    )
    callbacks = CallbackList([checkpoint_callback, stop_callback])

    # --- Train The Model ---
    rospy.loginfo(f"Starting DDPG training. Run ID: {run_id}")
    rospy.loginfo(f"Checkpoints will be saved in: {checkpoint_save_dir}")
    rospy.loginfo(f"TensorBoard logs are in: {tensorboard_log_dir}")
    
    # If resuming from a checkpoint, we don't reset the step counter.
    # Otherwise (new or continued from final model), we do.
    reset_timesteps = not is_resuming_from_checkpoint
    
    model.learn(total_timesteps=10000000, callback=callbacks, reset_num_timesteps=reset_timesteps)
    
    # --- Final Save ---
    rospy.loginfo("Training finished. Saving final model and normalization stats...")
    model.save(model_final_save_path)
    norm_vec_env.save(stats_final_save_path)
    
    rospy.loginfo(f"Final model saved to: {model_final_save_path}")
    rospy.loginfo(f"Final stats saved to: {stats_final_save_path}")
    rospy.loginfo("Exiting.")