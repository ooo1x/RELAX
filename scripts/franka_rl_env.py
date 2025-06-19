import rospy
import numpy as np
from moveit_commander import MoveGroupCommander, RobotCommander, roscpp_initialize, roscpp_shutdown
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Float32MultiArray, Int32
from gymnasium import Env,spaces



class FrankaRLEnv(Env):
    def __init__(self):
        super(FrankaRLEnv, self).__init__()
        roscpp_initialize([])
        rospy.init_node('franka_rl_env', anonymous=True)
        rospy.loginfo("Initializing FrankaRLEnv...")

        # Initialize MoveIt
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

        self.ee_position = np.zeros(3, dtype=np.float32)

        self.fault_flag = 0  

        # ROS communication
        self.action_pub = rospy.Publisher("/rl_action",Float32MultiArray, queue_size=1)
        rospy.Subscriber("/ee_pose", PoseStamped, self.ee_callback)
        rospy.Subscriber("/fault_flag", Int32, self.fault_callback)

        self.reset()
        rospy.loginfo("FrankaRLEnv initialized")
    
    def ee_callback(self,msg):
        self.ee_position = np.array((msg.pose.position.x,
                                    msg.pose.position.y,
                                    msg.pose.position.z), dtype=np.float32)
    
    def fault_callback(self, msg):
        self.fault_flag = msg.data

    def _get_observation(self):
        ee_position = self.ee_position
        delta1 = self.goal1 - ee_position
        delta2 = self.goal2 - ee_position
        return np.concatenate((ee_position, delta1, delta2, [self.fault_flag]), dtype=np.float32)

    def _compute_reward(self, obs):
        distance_to_goal1 = np.linalg.norm(obs[3:6])
        distance_to_goal2 = np.linalg.norm(obs[6:9])
        reward = -min(distance_to_goal1, distance_to_goal2)
        if distance_to_goal1 < 0.02 or distance_to_goal2 < 0.02:
            reward -= 5.0  
        if self.fault_flag > 0:
            reward -= 1.0
        return reward

    def _is_done(self, obs):
        distance_to_goal1 = np.linalg.norm(obs[3:6])
        distance_to_goal2 = np.linalg.norm(obs[6:9])
        return distance_to_goal1 < 0.01 or distance_to_goal2 < 0.01


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rospy.loginfo("Resetting FrankaRLEnv...")
        rospy.sleep(0.5)
        obs = self._get_observation()
        return obs, {}  
    
    def step(self, action):
        msg = Float32MultiArray(data=action.tolist())
        self.action_pub.publish(msg)
        rospy.sleep(0.1)

        obs = self._get_observation()
        reward = float(self._compute_reward(obs)) 
        terminated = bool(self._is_done(obs))
        truncated = False  
        info = {}

        return obs, reward, terminated, truncated, info


    def render(self, mode='human'):
        pass

    def close(self):
        roscpp_shutdown()
