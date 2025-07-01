import rospy
import numpy as np
from moveit_commander import MoveGroupCommander, RobotCommander, roscpp_initialize, roscpp_shutdown
from std_msgs.msg import Float32MultiArray, Int32, Bool
from gymnasium import Env,spaces
import math
from rosgraph_msgs.msg import Log
from geometry_msgs.msg import Pose


class FrankaRLEnv(Env):
    def __init__(self, get_ee_position, ):
        super(FrankaRLEnv, self).__init__()
        roscpp_initialize([])
        rospy.loginfo("Initializing FrankaRLEnv...")

        # Initialize MoveIt
        self.robot = RobotCommander()
        self.arm = MoveGroupCommander("panda_arm")
        self.arm.set_max_velocity_scaling_factor(0.2)
        self.arm.set_max_acceleration_scaling_factor(0.1)

        # Set obstacles and goals
        self.obstacle1 = np.array([0.6, -0.5, 1.5])
        self.obstacle2 = np.array([0.6,  0.5, 1.42])
        self.obstacle3 = np.array([0.6, 0.0, 1.7])
        self.obstacles = [self.obstacle1, self.obstacle2, self.obstacle3]

        # Set state space: ee_x, ee_y, ee_z, dx1, dy1, dz1, dx2, dy2, dz2, dx3, dy3, dz3
        obs_high = np.array([2.0] * 3 + [2.0] * 3 + [5.0] * 3, dtype=np.float32)
        obs_low = np.array([-2.0] * 3 + [-2.0] * 3 + [0.0] * 3, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Set action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.MAX_DELTA = 0.05

        # Initialize variables
        #self.get_current_state = get_current_state
        #self.get_fault_flag = get_fault_flag
        self.get_ee_position = get_ee_position
        #self.get_round_end = get_round_end
        self.planning_failed_flag = False

        # ROS communication
        self.resolved_pose_pub = rospy.Publisher("/rl/action_resolved", Pose, queue_size=1)
        rospy.Subscriber("/rosout", Log, self._rosout_callback)

        self.MAX_EPISODE_STEPS = 800
        self.current_episode_steps = 0
        self.current_request = None

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

        if self.current_request:
            target_position = np.array([
                self.current_request.position.x,
                self.current_request.position.y,
                self.current_request.position.z
            ], dtype=np.float32)
        else:
            target_position = np.zeros(3, dtype=np.float32)

        dist_to_obs1 = self.calculate_distance(self.obstacle1, ee_position)
        dist_to_obs2 = self.calculate_distance(self.obstacle2, ee_position)
        dist_to_obs3 = self.calculate_distance(self.obstacle3, ee_position)
        
        return np.concatenate((ee_position, target_position, [dist_to_obs1, dist_to_obs2, dist_to_obs3]), dtype=np.float32)


    def _compute_reward(self):
        final_ee_pos = np.array(self.get_ee_position())
        
        target_pos = np.array([
            self.current_request.position.x,
            self.current_request.position.y,
            self.current_request.position.z
        ])

        distance_to_target = np.linalg.norm(final_ee_pos - target_pos)
        reward_target = math.exp(-10 * distance_to_target)  

        min_dist_to_obstacle = min(np.linalg.norm(final_ee_pos - obs) for obs in self.obstacles)
        
        penalty_obstacle = 0.0
        SAFETY_MARGIN = 0.25
        COLLISION_THRESHOLD = 0.2

        if min_dist_to_obstacle < COLLISION_THRESHOLD:
            penalty_obstacle = -1.0
        elif min_dist_to_obstacle < SAFETY_MARGIN:
            penalty_obstacle = -1.0 * ((SAFETY_MARGIN - min_dist_to_obstacle) / SAFETY_MARGIN)**2
            penalty_obstacle = max(penalty_obstacle, -1.0)

        success_bonus = 0.0
        TARGET_THRESHOLD = 0.05
        if distance_to_target < TARGET_THRESHOLD:
            success_bonus = 1.0

        total_reward = reward_target + penalty_obstacle + success_bonus

        return total_reward

    
    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)
        rospy.loginfo("Env Reset. Waiting for the first action request from C++ for the new episode...")
        
        self.current_episode_steps = 0
        self.planning_failed_flag = False
        self.current_request = None

        try:
            self.current_request = rospy.wait_for_message("/rl/action_request", Pose, timeout=60.0)
            rospy.loginfo("Received initial request for the new episode.")
        except rospy.ROSException:
            rospy.logerr("Timeout waiting for initial action request on reset. Is the C++ node running and starting a new round?")
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, {'error': 'reset_timeout'}

        obs = self._get_observation()
        
        return obs, {}


    def step(self, action):
       
        if self.current_request is None:
            rospy.logerr("Cannot perform step: current_request is None. The episode may have failed on reset.")
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, -100, True, False, {'error': 'no_request_in_step'}

        final_pose = self.current_request
        delta = np.clip(action, -1.0, 1.0) * self.MAX_DELTA
        final_pose.position.x += delta[0]
        final_pose.position.y += delta[1]
        final_pose.position.z += delta[2]

        self.resolved_pose_pub.publish(final_pose)
        rospy.loginfo(f"Step {self.current_episode_steps + 1}")
        
        try:
            self.current_request = rospy.wait_for_message("/rl/action_request", Pose, timeout=60.0)
            rospy.loginfo("Move complete. Received next action request.")
            episode_is_over = False
        except rospy.ROSException:
            rospy.loginfo("Timeout waiting for the next request. Assuming this RL episode part is done.")
            episode_is_over = True

        reward = self._compute_reward() 
        obs = self._get_observation()
        
        self.current_episode_steps += 1

        terminated = self.planning_failed_flag
        if terminated:
            rospy.logwarn("Terminating episode due to planning failure.")
            reward -= 5

        truncated = episode_is_over or (self.current_episode_steps >= self.MAX_EPISODE_STEPS)
        
        info = {}
        
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        roscpp_shutdown()
