# evaluate.py

import rospy
import numpy as np
import tf2_ros
import os
import argparse  

from stable_baselines3 import PPO
from franka_rl_env import FrankaRLEnv  
from std_msgs.msg import Int32
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped
import time
from stable_baselines3.common.logger import configure


run_id = time.strftime("%Y%m%d-%H%M%S")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
experement_dir = os.path.join(BASE_DIR, "experiments")
ppo_dir = os.path.join(experement_dir, "ppo")
checkpoint_dir = os.path.join(ppo_dir, "ppo_checkpoints", run_id)
tensorboard_log_dir = os.path.join(ppo_dir, "ppo_franka_tensorboard")
evaluation_dir = os.path.join(ppo_dir, "ppo_franka_evaluation")

model_save_path = os.path.join(ppo_dir, "ppo_franka_model", "ppo_franka_model.zip")


end_effector_position = np.zeros(3, dtype=np.float32)
tfBuffer = None 

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

if __name__ == "__main__":
    rospy.init_node('ppo_evaluator', anonymous=True)

    if not os.path.exists(model_save_path):
        rospy.logerr(f"Model file not found at path: {model_save_path}")
        rospy.logerr("Please open evaluate.py and update the MODEL_TO_EVALUATE variable.")
        exit()

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    rospy.Subscriber("/joint_states", JointState, joint_state_callback, queue_size=1)
    rospy.loginfo("[EVAL] Waiting for TF transforms...")
    rospy.sleep(2.0)

    eval_logger = configure(evaluation_dir, ["stdout", "tensorboard"])

    env = FrankaRLEnv(get_ee_position=lambda: end_effector_position)

    rospy.loginfo(f"[EVAL] Loading trained model from: {model_save_path}")
    model = PPO.load(model_save_path, env=env)

    num_eval_episodes = 50
    rospy.loginfo(f"[EVAL] Starting evaluation for {num_eval_episodes} episodes...")
    
    all_rewards = []
    all_steps = []

    for eval_ep in range(num_eval_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and not rospy.is_shutdown():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        rospy.loginfo(f"[EVAL] Episode {eval_ep + 1}/{num_eval_episodes} Finished: Total Reward = {total_reward:.2f}")
        all_rewards.append(total_reward)
        all_steps.append(steps)

        eval_logger.record("eval/episode_reward", total_reward)
        eval_logger.record("eval/episode_length", steps)
        eval_logger.dump(step=eval_ep)


    # --- 打印最终评估总结 ---
    rospy.loginfo("="*50)
    rospy.loginfo("Evaluation Summary")
    rospy.loginfo(f"Average Reward over {num_eval_episodes} episodes: {np.mean(all_rewards):.2f} +/- {np.std(all_rewards):.2f}")
    rospy.loginfo("="*50)

    env.close()
    rospy.loginfo("Evaluation script finished.")