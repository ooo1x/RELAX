import rospy
import numpy as np
from moveit_commander import MoveGroupCommander, RobotCommander, roscpp_initialize, roscpp_shutdown
from std_msgs.msg import Float32MultiArray, Int32, Bool
from gymnasium import Env,spaces
import math
from rosgraph_msgs.msg import Log


class FrankaRLEnv(Env):
    def __init__(self, get_current_state, get_fault_flag, get_ee_position, get_round_end):
        super(FrankaRLEnv, self).__init__()
        roscpp_initialize([])
        rospy.loginfo("Initializing FrankaRLEnv...")

        # Initialize MoveIt
        self.robot = RobotCommander()
        self.arm = MoveGroupCommander("panda_arm")
        self.arm.set_max_velocity_scaling_factor(0.2)
        self.arm.set_max_acceleration_scaling_factor(0.1)

        # Set 2 goals
        self.goal1 = np.array([0.6, -0.5, 1.5])
        self.goal2 = np.array([0.6,  0.5, 1.42])
        self.goal3 = np.array([0.6, 0.0, 1.7])

        # Set state space: ee_x, ee_y, ee_z, dx1, dy1, dz1, dx2, dy2, dz2, fault_flag
        obs_high = np.array([2.0] * 7, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # Set action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.MAX_DELTA = 0.05

        # Initialize variables
        self.get_current_state = get_current_state
        self.get_fault_flag = get_fault_flag
        self.get_ee_position = get_ee_position
        self.get_round_end = get_round_end
        self.planning_failed_flag = False

        # ROS communication
        self.action_pub = rospy.Publisher("/rl_action",Float32MultiArray, queue_size=1)
        rospy.Subscriber("/rosout", Log, self._rosout_callback)

        self.MAX_EPISODE_STEPS = 500
        self.current_episode_steps = 0

        rospy.loginfo("FrankaRLEnv initialized successfully.")

    def _rosout_callback(self, msg):
        if msg.name == "/move_group":
            error_keywords = [
                "No motion plan found",
                "controller failed",
                "ABORTED"
            ]
            if any(keyword in msg.msg for keyword in error_keywords):
                rospy.logwarn(f"[RL ENV] MoveIt failed: {msg.msg}")
                self.planning_failed_flag = True 

    def calculate_distance(self,point1, point2):
        distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)
        return distance
    
    def _get_observation(self):
        ee_position = np.array(self.get_ee_position(), dtype=np.float32)
        d1 = self.calculate_distance(self.goal1, ee_position)
        d2 = self.calculate_distance(self.goal2, ee_position)
        d3 = self.calculate_distance(self.goal3, ee_position)
        fault = self.get_fault_flag()
        return np.concatenate((ee_position, [d1, d2, d3, fault]), dtype=np.float32)


    def _compute_reward(self, obs, step_count=0):
        d1, d2, d3 = obs[3], obs[4], obs[5]
        min_dist = min(d1, d2, d3)

        safe_range = 0.2 
        reward = 0.0

        if min_dist >= safe_range:
            reward = 2.0 * min(min_dist, 0.5) / 0.5
        else:
            reward = -3.0 * (1 - (min_dist / safe_range))
            reward = max(reward, -3.0)

        if self._is_terminated_by_collision(obs):
            reward -= 5.0
            rospy.loginfo("[STEP] Collision detected! Terminating episode.")

        if step_count >= self.MAX_EPISODE_STEPS:
            reward -= 1.0
            rospy.loginfo(f"[STEP] Max steps reached ({self.MAX_EPISODE_STEPS}), truncating.")

        print(f"[REWARD] step: {step_count}, min_dist: {min_dist:.3f}, reward: {reward:.2f}")
        return reward
    
    def _is_terminated_by_collision(self, obs):
        d1 = obs[3]
        d2 = obs[4]
        d3 = obs[5]
        collision_threshold = 0.2
        return bool(d1 < collision_threshold or d2 < collision_threshold or d3 < collision_threshold)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rospy.loginfo("Resetting FrankaRLEnv...")
        rospy.sleep(0.5)
        self.current_episode_steps = 0
        self.planning_failed_flag = False
        obs = self._get_observation()
        return obs, {}  
    
    def step(self, action):
        wait_time = 0.0
        wait_dt = 0.01
        max_wait = 5.0  

        while self.get_current_state() not in (3, 4, 5, 6) and wait_time < max_wait:
            rospy.sleep(wait_dt)
            wait_time += wait_dt

        if self.get_current_state() not in (3, 4, 5, 6):
            rospy.logwarn(f"[RL ENV] Timeout waiting for state 3~6, current state: {self.get_current_state()}")

        self.current_episode_steps += 1

        if self.get_current_state() in (3, 4, 5, 6):
            delta = np.clip(action, -1.0, 1.0) * self.MAX_DELTA
            msg = Float32MultiArray(data=delta.astype(np.float32).tolist())
            self.action_pub.publish(msg)
        
        rospy.sleep(0.02)

        obs = self._get_observation()
        reward = float(self._compute_reward(obs, self.current_episode_steps))
        
        if self.planning_failed_flag: 
            reward -= 5.0
            rospy.loginfo("[STEP] MoveIt planning failed! Applying penalty.")
            self.planning_failed_flag = False 
 
        terminated = self._is_terminated_by_collision(obs)
        truncated = self.current_episode_steps >= self.MAX_EPISODE_STEPS
        info = {}

        return obs, reward, terminated, truncated, info


    def render(self, mode='human'):
        pass

    def close(self):
        roscpp_shutdown()
