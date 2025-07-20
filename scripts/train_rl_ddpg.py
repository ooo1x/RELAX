from stable_baselines3 import DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from franka_rl_env import FrankaRLEnv
import rospy
from std_msgs.msg import Int32, Float32MultiArray, Bool
import os
import numpy as np
import tf2_ros
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped
import time
from stable_baselines3.common.logger import configure
from rosgraph_msgs.msg import Log
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DDPG_dir = os.path.join(BASE_DIR, "experiments", "DDPG")

MODELS_BASE_DIR = os.path.join(DDPG_dir, "models")
CHECKPOINTS_BASE_DIR = os.path.join(DDPG_dir, "checkpoints")
TENSORBOARD_LOG_DIR = os.path.join(DDPG_dir, "tensorboard_logs")

os.makedirs(MODELS_BASE_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_BASE_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)


class RosStateTracker:
    def __init__(self):
        self.current_cpp_episode = 0
        rospy.Subscriber("/episode", Int32, self._cpp_episode_callback)

    def _cpp_episode_callback(self, msg):
        self.current_cpp_episode = msg.data

    def get_cpp_episode(self):
        return self.current_cpp_episode
    
def find_latest_run_dir(base_dir):
    if not os.path.exists(base_dir):
        return None
    
    run_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not run_dirs:
        return None
    
    run_dirs.sort()
    return run_dirs[-1]

class PrintTimestepsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        rospy.loginfo(f"[TRAIN] Total timesteps: {self.num_timesteps}")
        return True  

class StopWhenCppDone(BaseCallback):
    def __init__(self, get_cpp_episode, max_cpp_episodes, verbose=0):
        super().__init__(verbose)
        self.get_cpp_episode = get_cpp_episode
        self.max_cpp_episodes = max_cpp_episodes
        self.verbose = verbose

    def _on_step(self) -> bool:
        rospy.sleep(0.001)
        current_episode = self.get_cpp_episode()
        return current_episode < self.max_cpp_episodes


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train DDPG agent for Franka Robot, with options to continue or start new.")
    parser.add_argument("--new", action="store_true", 
                        help="Force a new training run, ignoring any existing models.")
    args = parser.parse_args()

    rospy.init_node('DDPG_trainer', anonymous=True)
    ros_tracker = RosStateTracker()

    current_run_id = time.strftime("%Y%m%d-%H%M%S")
    rospy.loginfo(f"Current run ID for saving is: {current_run_id}")

    latest_run_id_to_load = find_latest_run_dir(MODELS_BASE_DIR)
    load_model_path = None
    load_stats_path = None

    if not args.new and latest_run_id_to_load:
        rospy.loginfo(f"Continue mode: Found latest run to load from: {latest_run_id_to_load}")
        load_model_path = os.path.join(MODELS_BASE_DIR, latest_run_id_to_load, "ddpg_franka_model.zip")
        load_stats_path = os.path.join(MODELS_BASE_DIR, latest_run_id_to_load, "vec_normalize_stats.pkl")
    elif args.new:
        rospy.logwarn("New run mode: --new flag was used. Ignoring existing models.")
    else:
        rospy.loginfo("New run mode: No previous run found. Starting a fresh training.")

    save_checkpoint_dir = os.path.join(CHECKPOINTS_BASE_DIR, current_run_id)
    model_save_dir = os.path.join(MODELS_BASE_DIR, current_run_id)
    os.makedirs(model_save_dir, exist_ok=True) 
    model_save_path = os.path.join(model_save_dir, "ddpg_franka_model.zip")
    stats_save_path = os.path.join(model_save_dir, "vec_normalize_stats.pkl")

    rospy.loginfo("[TRAIN] Initializing FrankaRLEnv...")
    GAMMA = 0.99
    vec_env = make_vec_env(lambda: FrankaRLEnv(), n_envs=1)
    norm_vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, gamma=GAMMA)

    n_actions = norm_vec_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    MAX_CPP_EPISODES = 3

    is_model_loaded = load_model_path and os.path.exists(load_model_path) and os.path.exists(load_stats_path)
    tensorboard_run_log_dir = os.path.join(TENSORBOARD_LOG_DIR, current_run_id)

    if is_model_loaded:
        rospy.loginfo(f"[TRAIN] Loading existing model from {load_model_path}")
        norm_vec_env = VecNormalize.load(load_stats_path, vec_env)
        model = DDPG.load(load_model_path, env=norm_vec_env)
        rospy.loginfo("[TRAIN] Model and normalization stats loaded successfully.")
        new_logger = configure(tensorboard_run_log_dir, ["tensorboard"])
        model.set_logger(new_logger)
    else:
        rospy.loginfo("[TRAIN] No loadable model found or --new flag specified. Creating a new one.")
        model = DDPG("MlpPolicy", norm_vec_env, action_noise=action_noise, verbose=1, 
                     gamma=GAMMA, tensorboard_log=TENSORBOARD_LOG_DIR,
                     buffer_size=100000, learning_starts=1000)

    checkpoint_callback = CheckpointCallback(
        save_freq=800,
        save_path=save_checkpoint_dir, 
        name_prefix="ddpg_franka",
        save_replay_buffer=True,
        save_vecnormalize=True 
    )

    stop_callback = StopWhenCppDone(
        get_cpp_episode=ros_tracker.get_cpp_episode, 
        max_cpp_episodes=MAX_CPP_EPISODES, 
        verbose=1
    )
    callbacks = CallbackList([checkpoint_callback, stop_callback, PrintTimestepsCallback()])
    
    rospy.loginfo("[TRAIN] Starting DDPG training...")
    model.learn(total_timesteps=10000000, callback=callbacks, reset_num_timesteps=not is_model_loaded)
    
    rospy.loginfo("[TRAIN] Training finished. Saving final model and normalization stats...")
    model.save(model_save_path)
    norm_vec_env.save(stats_save_path)
    
    rospy.loginfo(f"[TRAIN] Final model saved to {model_save_path}")
    rospy.loginfo(f"[TRAIN] Final stats saved to {stats_save_path}")
    rospy.loginfo("[TRAIN] Exiting.")


    #1 set delay for each action, e.g. 5s
    #2 visualize EE position, use a rostopic to publish the position
    #3 calculate distance between EE and designed fixed points (consider faulty actions)

#Next step
    #1 set different fault parameters
    #2 random points