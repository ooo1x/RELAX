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
        self.move_group = MoveGroupCommander("panda_arm")

        # Set obstacles and goals
        self.obstacle1 = np.array([0.75, -0.5, 1.48])
        self.obstacle2 = np.array([0.75,  0.5, 1.48])
        # self.obstacle3 = np.array([0.75, 0.0, 1.64])
        self.obstacles = [self.obstacle1, self.obstacle2]
        self.scene = PlanningSceneInterface()

        # Set state space: ee_pos(3), vec_to_obs1-3(9), current_joint_states(7), falty_joint4_value(1)
        obs_dim = 17
        high_bounds = np.array([np.inf] * obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high_bounds, high=high_bounds, dtype=np.float32)

        # Set action space
        self.action_space = spaces.Box(low=-0.5, high=0, shape=(1,), dtype=np.float32)
        
        self.request_event = threading.Event()
        self.step_result_event = threading.Event() 
        self.reward_received_event = threading.Event()

        self.MAX_EPISODE_STEPS = 300
        self.current_episode_steps = 0
        self.current_request = None
        self.previous_distance_to_target = None
        self.pose_state = 0
        self.ready_for_rl_request = False 
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.had_collision = False
        self.had_planning_failure = False
        self._add_obstacles_to_scene()
        self.visited_states = set()
        self.faulty_start_j4 = 0.0
        self.execution_aborted_flag = False
        self.trajectory_reward = 0.0

        # ROS communication
        self.corrected_start_j4_pub = rospy.Publisher("/rl/corrected_start_joint4", Float32, queue_size=10)
        self.request_sub = rospy.Subscriber('/rl/faulty_start_joint4', Float32, self._faulty_start_joint_callback, queue_size=1)
        rospy.Subscriber('/pose_state', Int32, self._pose_state_callback, queue_size=1)
        self.ready_pub = rospy.Publisher('/rl/ready_for_next', Bool, queue_size=1)
        self.joint_states = np.zeros(7) # Panda7个关节
        self.joint_states_sub = rospy.Subscriber('/faulty_joint_states', JointState, self._joint_states_cb, queue_size=1)
        rospy.Subscriber('/rl/step_result', Bool, self._step_result_callback, queue_size=1)
        rospy.Subscriber('/rl/trajectory_reward', Float32, self._reward_callback, queue_size=1)
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
        self.step_result_event.set() # 收到结果，设置事件，让step函数可以继续

    def _joint_states_cb(self, msg):
        try:
            arm_joint_names = [f'panda_joint{i+1}' for i in range(7)]
            joint_positions = []
            for name in arm_joint_names:
                if name in msg.name:
                    idx = msg.name.index(name)
                    joint_positions.append(msg.position[idx])
            
            if len(joint_positions) == 7:
                self.joint_states = np.array(joint_positions, dtype=np.float32)
        except Exception as e:
            rospy.logwarn(f"Could not extract joint states: {e}")
    
    def _faulty_start_joint_callback(self, msg: Float32):

        if self.request_event.is_set():
            return
        
        self.faulty_start_j4 = msg.data
        
        self.request_event.set()
    
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
    
    def _add_obstacles_to_scene(self):
        #rospy.loginfo("Adding obstacles to the planning scene.")
        
        for obs_name in ["obstacle1", "obstacle2"]:
            self.scene.remove_world_object(obs_name)
        
        rospy.sleep(1) 

        obstacle_radius = 0.05

        for i, obs_pos in enumerate(self.obstacles):
            obs_name = f"obstacle{i+1}"
            collision_object = CollisionObject()
            collision_object.header.frame_id = "world"
            collision_object.id = obs_name

            primitive = SolidPrimitive()
            primitive.type = primitive.SPHERE
            primitive.dimensions = [obstacle_radius]

            pose = Pose()
            pose.position.x = obs_pos[0]
            pose.position.y = obs_pos[1]
            pose.position.z = obs_pos[2]
            pose.orientation.w = 1.0 

            collision_object.primitives.append(primitive)
            collision_object.primitive_poses.append(pose)
            
            collision_object.operation = collision_object.ADD

            self.scene.add_object(collision_object)

        rospy.sleep(1) 
        #rospy.loginfo("Obstacles added to the planning scene successfully.")
    
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
        ee_position = np.array(self.get_ee_position(), dtype=np.float32)
        
        vec_to_obs1 = self.obstacle1 - ee_position
        vec_to_obs2 = self.obstacle2 - ee_position
        #vec_to_obs3 = self.obstacle3 - ee_position

        return np.concatenate([
            ee_position, 
            vec_to_obs1, 
            vec_to_obs2, 
            self.joint_states,
            np.array([self.faulty_start_j4], dtype=np.float32),
        ], dtype=np.float32)


    def _compute_reward(self, planning_failed, cpp_provided_reward):

        PENALTY_PLANNING_FAILURE = -200.0
        
        if planning_failed:
            rospy.logwarn("Planning failed. Applying penalty.")
            return PENALTY_PLANNING_FAILURE,True

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
        while not self.request_event.wait(timeout=1.0): # Wait for 1 second
            if rospy.is_shutdown():
                rospy.logwarn("Shutdown signal received while waiting for request. Terminating episode.")
                return np.zeros(self.observation_space.shape), 0.0, True, False, {}

        correction_value = action[0]
        if not self.faulty_start_j4:
            rospy.logerr("Step function started but faulty_joint4_waypoints is empty!")
            # 发生这种情况，返回惩罚并结束
            return self._get_observation(), 0, True, False, {'error': 'no waypoints received'}
        
        corrected_start_j4 = self.faulty_start_j4 + correction_value
        msg_to_cpp = Float32()
        msg_to_cpp.data = corrected_start_j4
        self.corrected_start_j4_pub.publish(msg_to_cpp)
        # rospy.loginfo(f"Published resolved pose: {self.resolved_pose.position.x:.3f}, {self.resolved_pose.position.y:.3f}, {self.resolved_pose.position.z:.3f}")
        
        if not self.step_result_event.wait(timeout=10.0):
            rospy.logerr("Timeout! C++ did not return a step result in time.")
            self.step_succeeded = False # 超时就算失败
        
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
                reward += 200
        
        terminated = self.episode_is_over

        truncated = self.current_episode_steps >= self.MAX_EPISODE_STEPS
       
        info = {'collision': collision, 'planning_failed': planning_failed}
        print(f"Step: {self.current_episode_steps}, Reward: {reward}")

        self.ready_pub.publish(True)

        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        roscpp_shutdown()