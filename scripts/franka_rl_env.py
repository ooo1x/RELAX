import gym
import rospy
import numpy as np
from gym import spaces
from moveit_commander import MoveGroupCommander, RobotCommander, roscpp_initialize, roscpp_shutdown
from geometry_msgs.msg import Pose

class FrankaRLEnv(gym.Env):
    def __init__(self):
        super(FrankaRLEnv, self).__init__()
        roscpp_initialize([])
        rospy.init_node('franka_rl_env', anonymous=True)
        rospy.loginfo("Initializing FrankaRLEnv...")

        # 初始化 MoveIt
        self.robot = RobotCommander()
        self.arm = MoveGroupCommander("panda_arm")
        self.arm.set_max_velocity_scaling_factor(0.2)
        self.arm.set_max_acceleration_scaling_factor(0.1)

        # Set 2 goals
        self.goal1 = np.array([0.5, -0.2, 0.72])
        self.goal2 = np.array([0.5,  0.2, 0.72])

        # Set state space: ee_x, ee_y, ee_z, dx1, dy1, dz1, dx2, dy2, dz2, fault_flag
        obs_high = np.array([2.0] * 10, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # Set action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.fault_flag = 0  
        self.reset()
        rospy.loginfo("FrankaRLEnv initialized")

    def _get_ee_position(self):
        pose = self.arm.get_current_pose().pose
        return np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)

    def _get_observation(self):
        ee_position = self._get_ee_position()
        delta1 = self.goal1 - ee_position
        delta2 = self.goal2 - ee_position
        return np.concatenate((ee_position, delta1, delta2, [self.fault_flag]), dtype=np.float32)

    def compute_reward(self, obs):
        distance_to_goal1 = np.linalg.norm(obs[3:6])
        distance_to_goal2 = np.linalg.norm(obs[6:9])
        reward = -min(distance_to_goal1, distance_to_goal2)
        if distance_to_goal1 < 0.02 or distance_to_goal2 < 0.02:
            reward -= 5.0  
        return reward

    def _is_done(self, obs):
        distance_to_goal1 = np.linalg.norm(obs[3:6])
        distance_to_goal2 = np.linalg.norm(obs[6:9])
        return distance_to_goal1 < 0.01 or distance_to_goal2 < 0.01

    def reset(self):
        self.arm.set_named_target("ready")
        self.arm.go(wait=True)
        self.arm.stop()
        return self._get_observation()

    def step(self, action):
        current_position = self._get_ee_position()
        target_position = current_position + action 

        pose = Pose()
        pose.position.x = target_position[0]
        pose.position.y = target_position[1]
        pose.position.z = target_position[2]
        pose.orientation.w = 1.0
        self.arm.set_pose_target(pose)
        success = self.arm.go(wait=True)
        self.arm.stop()

        obs = self._get_observation()
        reward = self.compute_reward(obs)
        done = self._is_done(obs)
        info = {"success": success}

        return obs, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        roscpp_shutdown()
