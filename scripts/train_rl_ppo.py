from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from franka_rl_env import FrankaRLEnv

env = FrankaRLEnv()
check_env(env,warn=True)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_franka_tensorboard/")
model.learn(total_timesteps=1000)
model.save("ppo_franka_model")

env.close()


