import rospy
import numpy as np
from moveit_commander import MoveGroupCommander, RobotCommander, roscpp_initialize, roscpp_shutdown
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Float32MultiArray, Int32
from gymnasium import Env,spaces
import math
import tf2_ros
from sensor_msgs.msg import JointState

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
        self.goal1 = np.array([0.6, -0.5, 1.5])
        self.goal2 = np.array([0.6,  0.5, 1.42])
        self.goal3 = np.array([0.6, 0.0, 1.7])

        # Set state space: ee_x, ee_y, ee_z, dx1, dy1, dz1, dx2, dy2, dz2, fault_flag
        obs_high = np.array([2.0] * 7, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # Set action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.MAX_DELTA = 0.05

        # Initialize end-effector position
        self.end_effector_position = np.zeros(3, dtype=np.float32)
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.joint_state_sub=rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)

        self.fault_flag = 0  

        # ROS communication
        self.action_pub = rospy.Publisher("/rl_action",Float32MultiArray, queue_size=1)
        rospy.Subscriber("/fault_flag", Int32, self.fault_callback)

        self.current_state = 0  # Initial state(in RELAX DEMO)
        rospy.Subscriber("/pose_state", Int32, self.state_callback)
        self.dummy_sent = False

        self.reset()
        rospy.loginfo("FrankaRLEnv initialized")

    def fault_callback(self, msg):
        self.fault_flag = msg.data

    def state_callback(self, msg):
        self.current_state = msg.data

    def joint_state_callback(self,joint_state_msg):
        trans = self.tfBuffer.lookup_transform('world', 'panda_hand_tcp', rospy.Time(0))
        panda_pose = PoseStamped()
        panda_pose.header.frame_id = 'world'
        panda_pose.header.stamp = rospy.Time.now()
        panda_pose.pose.position.x = trans.transform.translation.x
        panda_pose.pose.position.y = trans.transform.translation.y
        panda_pose.pose.position.z = trans.transform.translation.z

        self.end_effector_position = (panda_pose.pose.position.x, panda_pose.pose.position.y, panda_pose.pose.position.z)

    def calculate_distance(self,point1, point2):
        distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)
        return distance
    
    def _get_observation(self):
        ee_position = self.end_effector_position
        d1 = self.calculate_distance(self.goal1, ee_position)
        d2 = self.calculate_distance(self.goal2, ee_position)
        d3 = self.calculate_distance(self.goal3, ee_position)
        return np.concatenate((ee_position, [d1, d2, d3, self.fault_flag]), dtype=np.float32)


    def _compute_reward(self, obs):
        d1 = obs[3]
        d2 = obs[4]
        d3 = obs[5]
        danger_threshold = 0.5
        reward = 0.1
        min_dist = min(d1, d2, d3)
        if min_dist < danger_threshold:
            reward -= (danger_threshold - min_dist) * 2.0
        if self.fault_flag > 0:
            reward -= 0.5
        return reward

    def _is_done(self, obs):
        d1 = obs[3]
        d2 = obs[4]
        d3 = obs[5]
        collision_threshold = 0.2
        return d1 < collision_threshold or d2 < collision_threshold or d3 < collision_threshold

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rospy.loginfo("Resetting FrankaRLEnv...")
        rospy.sleep(0.5)
        obs = self._get_observation()
        return obs, {}  
    
    def step(self, action):
        if self.current_state in (3, 4, 5, 6):
            delta = np.clip(action, -1.0, 1.0) * self.MAX_DELTA
            msg = Float32MultiArray(data=delta.astype(np.float32).tolist())
            self.action_pub.publish(msg)
        else:
            if not self.dummy_sent:
                self.action_pub.publish(Float32MultiArray(data=[0.0, 0.0, 0.0]))
                self.dummy_sent = True

        rospy.sleep(0.02)

        obs = self._get_observation()
        reward = float(self._compute_reward(obs)) 
        terminated = bool(self._is_done(obs))
        truncated = False  
        info = {}

        if terminated:
            rospy.loginfo("Collision detected! Episode terminated.")
            reward -= 2.0

        return obs, reward, terminated, truncated, info


    def render(self, mode='human'):
        pass

    def close(self):
        roscpp_shutdown()
