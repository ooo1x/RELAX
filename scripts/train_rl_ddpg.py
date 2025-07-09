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

# current_state = 0
# fault_flag = 0
end_effector_position = np.zeros(3, dtype=np.float32)
# start_signal_received = False
# round_end_flag = False
current_cpp_episode = 0

def find_latest_run_dir(base_dir):
    if not os.path.exists(base_dir):
        return None
    
    run_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not run_dirs:
        return None
    
    run_dirs.sort()
    return run_dirs[-1]

def cpp_episode_callback(msg):
    global current_cpp_episode
    current_cpp_episode = msg.data

def start_signal_callback(msg):
    global start_signal_received
    if msg.data:
        rospy.loginfo("[TRAIN] Start signal received")
        dummy_action = Float32MultiArray(data=[0.0, 0.0, 0.0])
        rospy.loginfo("[TRAIN] Sending dummy action to unblock C++...")
        action_pub.publish(dummy_action)
        start_signal_received = True

def pose_state_callback(msg):
    global current_state, round_end_flag
    current_state = msg.data
    if msg.data == 404:
        round_end_flag = True
    else:
        round_end_flag = False

def fault_flag_callback(msg):
    global fault_flag
    fault_flag = msg.data

def joint_state_callback(msg):
    global end_effector_position
    try:
        trans = tfBuffer.lookup_transform('world', 'panda_hand_tcp', rospy.Time(0))
        panda_pose = PoseStamped()
        panda_pose.header.frame_id = 'world'
        panda_pose.header.stamp = rospy.Time.now()
        panda_pose.pose.position.x = trans.transform.translation.x
        panda_pose.pose.position.y = trans.transform.translation.y
        panda_pose.pose.position.z = trans.transform.translation.z

        end_effector_position = (panda_pose.pose.position.x, panda_pose.pose.position.y, panda_pose.pose.position.z)

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rospy.logwarn("Transform lookup failed!")

class RosLogCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        if not rospy.core.is_initialized():
            rospy.init_node("DDPG_logger", anonymous=True)

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards")[0]
        if reward is not None:
            self.logger.record("reward/step", float(reward))
        rospy.loginfo(f"timesteps: {self.num_timesteps}")

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
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    # Publishers
    action_pub = rospy.Publisher("/rl_action", Float32MultiArray, queue_size=10, latch=True)

    # Subscribers
    #rospy.Subscriber("/start_signal", Bool, start_signal_callback)
    #rospy.Subscriber("/pose_state", Int32, pose_state_callback)
    #rospy.Subscriber("/fault_flag", Int32, fault_flag_callback)
    rospy.Subscriber("/joint_states", JointState, joint_state_callback)
    rospy.Subscriber("/episode", Int32, cpp_episode_callback)

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
    env_lambda = lambda: FrankaRLEnv(get_ee_position=lambda: end_effector_position)
    vec_env = make_vec_env(env_lambda, n_envs=1)
    norm_vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, gamma=GAMMA)

    n_actions = norm_vec_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    MAX_CPP_EPISODES = 499 

    is_model_loaded = load_model_path and os.path.exists(load_model_path) and os.path.exists(load_stats_path)

    if is_model_loaded:
        rospy.loginfo(f"[TRAIN] Loading existing model from {load_model_path}")
        norm_vec_env = VecNormalize.load(load_stats_path, vec_env)
        model = DDPG.load(load_model_path, env=norm_vec_env)
        rospy.loginfo("[TRAIN] Model and normalization stats loaded successfully.")
    else:
        rospy.loginfo("[TRAIN] No loadable model found or --new flag specified. Creating a new one.")
        model = DDPG("MlpPolicy", norm_vec_env, action_noise=action_noise, verbose=1, 
                     gamma=0.99, tensorboard_log=TENSORBOARD_LOG_DIR,
                     buffer_size=200000, learning_starts=30000)

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=save_checkpoint_dir, 
        name_prefix="ddpg_franka",
        save_replay_buffer=True,
        save_vecnormalize=True 
    )

    stop_callback = StopWhenCppDone(
        get_cpp_episode=lambda: current_cpp_episode, 
        max_cpp_episodes=MAX_CPP_EPISODES, 
        verbose=1
    )
    callbacks = CallbackList([RosLogCallback(), checkpoint_callback, stop_callback])
    
    rospy.loginfo("[TRAIN] Starting DDPG training...")
    model.learn(total_timesteps=10000000, callback=callbacks, reset_num_timesteps=not is_model_loaded)
    
    rospy.loginfo("[TRAIN] Training finished. Saving final model and normalization stats...")
    model.save(model_save_path)
    norm_vec_env.save(stats_save_path)
    
    rospy.loginfo(f"[TRAIN] Final model saved to {model_save_path}")
    rospy.loginfo(f"[TRAIN] Final stats saved to {stats_save_path}")
    rospy.loginfo("[TRAIN] Exiting.")

    