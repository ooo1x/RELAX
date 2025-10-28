import rospy
import numpy as np
from moveit_commander import MoveGroupCommander, RobotCommander, roscpp_initialize, roscpp_shutdown
from std_msgs.msg import Float32MultiArray, Int32, Bool, Float32
from gymnasium import Env,spaces
import math
from geometry_msgs.msg import Pose, PoseStamped
from copy import deepcopy
from moveit_msgs.msg import MoveGroupActionResult
import tf2_ros
import threading
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from moveit_commander import PlanningSceneInterface
import random
from sensor_msgs.msg import JointState

class FrankaRLEnv(Env):
    def __init__(self):
        super(FrankaRLEnv, self).__init__()
        roscpp_initialize([])
        rospy.loginfo("Initializing FrankaRLEnv...")
        self.move_group = MoveGroupCommander("panda_manipulator")

        self.faulty_indicator = np.array([0,0,0,1,0,0,0], dtype=np.float32)  
        self.num_obstacles = 3
        self.num_joints = 7

        # Set state space: current_joint_states(7)
        obs_dim = 7 + self.num_obstacles * 3
        high_bounds = np.array([np.inf] * obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high_bounds, high=high_bounds, dtype=np.float32)

        # Set action space
        #把 action_space 设为 7 维（每个关节的绝对目标值）：
        self.joint_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float32)
        self.joint_upper = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973], dtype=np.float32)
        #self.action_space = spaces.Box(low=self.joint_lower, high=self.joint_upper, dtype=np.float32)
        joint_4_lower = self.joint_lower[3]
        joint_4_upper = self.joint_upper[3]
        self.action_space = spaces.Box(low=joint_4_lower, high=joint_4_upper, shape=(1,), dtype=np.float32)

        self.request_event = threading.Event()
        self.step_result_event = threading.Event() 
        self.reward_received_event = threading.Event()

        self.max_episode_steps = 300
        self.current_episode_steps = 0
        self.current_request = None
        self.previous_distance_to_target = None
        self.pose_state = 0
        self.ready_for_rl_request = False 
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.had_collision = False
        self.had_planning_failure = False
        #self._add_obstacles_to_scene()
        self.visited_states = set()
        self.faulty_start_j4 = 0.0
        self.execution_aborted_flag = False
        self.trajectory_reward = 0.0
        self.full_observation = np.zeros(obs_dim, dtype=np.float32)

        # ROS communication
        self.corrected_start_pub = rospy.Publisher("/rl/corrected_start_joints", Float32MultiArray, queue_size=10)
        self.ready_pub = rospy.Publisher('/rl/ready_for_next', Bool, queue_size=1)
        rospy.Subscriber('/rl/faulty_joint_states', Float32MultiArray, self._observation_callback, queue_size=1)
        rospy.Subscriber('/rl/step_result', Bool, self._step_result_callback, queue_size=10)
        rospy.Subscriber('/rl/trajectory_reward', Float32, self._reward_callback, queue_size=10)
        rospy.Subscriber('pose_state', Int32, self._pose_state_callback, queue_size=10)
        rospy.loginfo("FrankaRLEnv initialized successfully.")
   
    def _reward_callback(self, msg: Float32):
        """接收来自C++的、基于整个轨迹计算的奖励值"""
        self.trajectory_reward = msg.data
        self.reward_received_event.set()

    def _step_result_callback(self, msg: Bool):
        """
        接收来自C++的、关于上一步执行结果的回调。
        """
        self.step_succeeded = msg.data
        self.step_result_event.set() 
    

    def _observation_callback(self, msg: Float32MultiArray):
        expected_len = self.num_joints + self.num_obstacles * 3
        if len(msg.data) == expected_len:
            self.full_observation = np.array(msg.data, dtype=np.float32)
            self.request_event.set()
        else:
            rospy.logwarn(f"Received observation with unexpected length: {len(msg.data)}.")
    
    def _pose_state_callback(self, msg):
            self.pose_state = msg.data
            self.visited_states.add(self.pose_state)
            
            if self.pose_state in {4, 5}: 
                self.ready_for_rl_request = True
            
            if self.pose_state == 9:
                self.episode_is_over = True

    def _move_group_result_cb(self, msg: MoveGroupActionResult):
        self.last_move_status = msg.status.status
        self.move_result_received = True
    
    # def _add_obstacles_to_scene(self):
    #     #rospy.loginfo("Adding obstacles to the planning scene.")
        
    #     for obs_name in ["obstacle1", "obstacle2", "obstacle3"]:
    #         self.scene.remove_world_object(obs_name)
        
    #     rospy.sleep(1) 

    #     obstacle_radius = 0.2

    #     for i, obs_pos in enumerate(self.obstacles):
    #         obs_name = f"obstacle{i+1}"
    #         collision_object = CollisionObject()
    #         collision_object.header.frame_id = "world"
    #         collision_object.id = obs_name

    #         primitive = SolidPrimitive()
    #         primitive.type = primitive.SPHERE
    #         primitive.dimensions = [obstacle_radius]

    #         pose = Pose()
    #         pose.position.x = obs_pos[0]
    #         pose.position.y = obs_pos[1]
    #         pose.position.z = obs_pos[2]
    #         pose.orientation.w = 1.0 

    #         collision_object.primitives.append(primitive)
    #         collision_object.primitive_poses.append(pose)
            
    #         collision_object.operation = collision_object.ADD

    #         self.scene.add_object(collision_object)

    #     rospy.sleep(1) 
    #     #rospy.loginfo("Obstacles added to the planning scene successfully.")
    
    def get_ee_position(self):
        try:
            trans = self.tfBuffer.lookup_transform('world', 'panda_hand_tcp', rospy.Time(0))
            panda_pose = PoseStamped()
            panda_pose.header.frame_id = 'world'
            panda_pose.header.stamp = rospy.Time.now()
            panda_pose.pose.position.x = trans.transform.translation.x
            panda_pose.pose.position.y = trans.transform.translation.y
            panda_pose.pose.position.z = trans.transform.translation.z

            end_effector_position = (panda_pose.pose.position.x, panda_pose.pose.position.y, panda_pose.pose.position.z)
            return end_effector_position

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Transform lookup failed!")

    def calculate_distance(self,point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)
    
    def _get_observation(self):
        return self.full_observation


    def _compute_reward(self, planning_failed, cpp_provided_reward):

        # PENALTY_PLANNING_FAILURE = -200.0
        
        # if planning_failed:
        #     rospy.logwarn("Planning failed. Applying penalty.")
        #     return PENALTY_PLANNING_FAILURE,True

        trajectory_reward = cpp_provided_reward
        
        total_reward = trajectory_reward

        return total_reward, False
    
    def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            
            self.had_collision = False
            self.had_planning_failure = False
            self.episode_is_over = False
            self.visited_states = set() 
            self.ready_for_rl_request = False
            self.current_episode_steps = 0
            self.current_request = None
            self.step_result_event.clear()
            self.reward_received_event.clear()
            self.trajectory_reward = 0.0
            
            rospy.loginfo("Env Reset completed. Ready for the first step.")

            return self._get_observation(), {}

    def step(self, action):
        if self.current_episode_steps == 0:
            rospy.logdebug("First step of episode, ensuring visited_states is cleared.")
            self.visited_states = set()
        
        self.request_event.clear()
        self.step_result_event.clear()
        self.reward_received_event.clear()

        while not self.request_event.wait(timeout=1.0): # Wait for 1 second
            if rospy.is_shutdown():
                rospy.logwarn("Shutdown signal received while waiting for request. Terminating episode.")
                return np.zeros(self.observation_space.shape), 0.0, True, False, {}

        #action = np.asarray(action, dtype=np.float32)
        #action = np.clip(action, self.joint_lower, self.joint_upper)

        # --- 取当前关节 ---
        current_joint_states = self.full_observation[:self.num_joints]  # shape (7,)

        # --- 应用动作到 j4（绝对值方案）---
        a = float(action[0])  # 1维动作
        a = np.clip(a, self.joint_lower[3], self.joint_upper[3])  # 用 j4 的上下界 clip

        corrected = current_joint_states.copy()
        corrected[3] = a  # 只改 j4，其余 6 个不变

        # --- 再对整条7维做一次界限裁剪（防御性）---
        corrected = np.clip(corrected, self.joint_lower, self.joint_upper)

        # --- 发布完整7维 ---
        self.corrected_start_pub.publish(Float32MultiArray(data=corrected.tolist()))
                
        if not self.step_result_event.wait(timeout=10.0):
            rospy.logerr("Timeout! C++ did not return a step result in time.")
            self.step_succeeded = False 
        
        if not self.reward_received_event.wait(timeout=10.0):
            rospy.logerr("Timeout! C++ did not return a reward value in time.")
            self.trajectory_reward = 0 if not self.step_succeeded else 0.0
        
        # rospy.loginfo(f"Move result received with status: {self.last_move_status}")

        planning_failed = (not self.step_succeeded)

        reward, collision = self._compute_reward(planning_failed, self.trajectory_reward) 
        obs = self._get_observation()

        if collision:
            self.had_collision = True
        if planning_failed:
            self.had_planning_failure = True
        
        self.current_episode_steps += 1

        REQUIRED_STATES = {4, 5}
        if REQUIRED_STATES.issubset(self.visited_states):
            if not self.had_planning_failure:
                reward += 0
        
        terminated = self.episode_is_over

        truncated = self.current_episode_steps >= self.max_episode_steps
       
        info = {'collision': collision, 'planning_failed': planning_failed}
        print(f"Step: {self.current_episode_steps}, Reward: {reward}")

        self.ready_pub.publish(True)

        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        roscpp_shutdown()