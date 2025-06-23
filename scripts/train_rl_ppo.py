from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from franka_rl_env import FrankaRLEnv
import rospy
from std_msgs.msg import Int32
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
checkpoint_dir = os.path.join(BASE_DIR, "ppo_checkpoints")

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
    def __init__(self, max_cpp_episodes=100, verbose=0):
        super().__init__(verbose)
        self.max_cpp_episodes = max_cpp_episodes
        self.current_cpp_episode = 0

        if not rospy.core.is_initialized():
            rospy.init_node("rl_callback_listener", anonymous=True)

        rospy.Subscriber("/episode", Int32, self.episode_callback)

    def episode_callback(self, msg):
        self.current_cpp_episode = msg.data
        if self.verbose:
            rospy.loginfo(f"Received C++ episode: {self.current_cpp_episode}")

    def _on_step(self) -> bool:
        return self.current_cpp_episode < self.max_cpp_episodes

env = FrankaRLEnv()
check_env(env, warn=True)
MAX_CPP_EPISODES = 100 

#model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_franka_tensorboard/")

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./ppo_checkpoints/",
    name_prefix="ppo_franka"
)

callbacks = CallbackList([
    RosLogCallback(),
    StopWhenCppDone(MAX_CPP_EPISODES, verbose=1),
    checkpoint_callback
])

model = PPO.load("ppo_franka_model", env=env, tensorboard_log="./ppo_franka_tensorboard/")
model.learn(total_timesteps=1000000, callback=callbacks, reset_num_timesteps=False )
model.save("ppo_franka_model")

env.close()

#   update:
    # 1, publish EE location position, write a rostopic
    # 2, calculate realtime distance
    # 3, 3 balls, scenarios design, radius
    # 4, rewrite RL