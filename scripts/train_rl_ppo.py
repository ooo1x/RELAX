from stable_baselines3 import PPO
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


run_id = time.strftime("%Y%m%d-%H%M%S")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
experement_dir = os.path.join(BASE_DIR, "experiments")
ppo_dir = os.path.join(experement_dir, "ppo")
checkpoint_dir = os.path.join(ppo_dir, "ppo_checkpoints", run_id)
model_save_path = os.path.join(ppo_dir, "ppo_franka_model", "ppo_franka_model.zip")
tensorboard_log_dir = os.path.join(ppo_dir, "ppo_franka_tensorboard")
evaluation_dir = os.path.join(ppo_dir, "ppo_franka_evaluation")


# current_state = 0
# fault_flag = 0
end_effector_position = np.zeros(3, dtype=np.float32)
# start_signal_received = False
# round_end_flag = False
current_cpp_episode = 0

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
            rospy.init_node("ppo_logger", anonymous=True)

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards")
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

    rospy.init_node('ppo_trainer', anonymous=True)
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

    rospy.loginfo("[TRAIN] Waiting for TF transforms to be available...")
    rospy.sleep(2.0) 
    rospy.loginfo("[TRAIN] Initializing FrankaRLEnv...")
    
    env = FrankaRLEnv(get_ee_position=lambda: end_effector_position,
       )


    # check_env(env, warn=True)
    MAX_CPP_EPISODES = 199

    if os.path.exists(model_save_path):
        model = PPO.load(model_save_path, env=env, tensorboard_log=tensorboard_log_dir)
        new_logger = configure(tensorboard_log_dir, ["stdout", "tensorboard"])
        model.set_logger(new_logger)
    else:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir)

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=checkpoint_dir,
        name_prefix="ppo_franka"
    )

    callbacks = CallbackList([
        RosLogCallback(),
        StopWhenCppDone(get_cpp_episode=lambda: current_cpp_episode, max_cpp_episodes=MAX_CPP_EPISODES, verbose=1),
        checkpoint_callback
    ])

    model.learn(total_timesteps=100000000, callback=callbacks, reset_num_timesteps=False)
    rospy.loginfo("[TRAIN] PPO training finished, saving model...")
    model.save(model_save_path)
    rospy.loginfo("[TRAIN] Model saved, exiting.")
