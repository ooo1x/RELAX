import rospy
import numpy as np
from moveit_commander import MoveGroupCommander, RobotCommander, roscpp_initialize, roscpp_shutdown
from std_msgs.msg import Float32MultiArray, Int32, Bool
from gymnasium import Env,spaces
import math
from rosgraph_msgs.msg import Log
from geometry_msgs.msg import Pose, PoseStamped
from copy import deepcopy
from moveit_msgs.msg import MoveGroupActionResult
from actionlib_msgs.msg import GoalStatus
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
        self.obstacle1 = np.array([0.75, -0.25, 1.48])
        self.obstacle2 = np.array([0.75,  0.25, 1.48])
        self.obstacle3 = np.array([0.75, 0.0, 1.64])
        self.obstacles = [self.obstacle1, self.obstacle2, self.obstacle3]
        self.scene = PlanningSceneInterface()

        # Set state space: ee_pos(3), vec_to_target(3), vec_to_obs1-3(9), current_joint_states(7) 
        obs_dim = 22
        high_bounds = np.array([np.inf] * obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high_bounds, high=high_bounds, dtype=np.float32)

        # Set action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.MAX_DELTA = 0.05

        # ROS communication
        self.resolved_pose_pub = rospy.Publisher("/rl/action_resolved", Pose, queue_size=1, latch=True)
        self.last_move_status = None
        rospy.Subscriber('/move_group/result',MoveGroupActionResult,self._move_group_result_cb,queue_size=1)
        #self.distances_pub = rospy.Publisher('/distances_to_obstacles', Float32MultiArray, queue_size=10)
        rospy.Subscriber('/pose_state', Int32, self._pose_state_callback, queue_size=1)
        self.request_sub = rospy.Subscriber('/rl/action_request', Pose, self._request_callback, queue_size=1)
        self.ready_pub = rospy.Publisher('/rl/ready_for_next', Bool, queue_size=1)
        self.joint_states = np.zeros(7) # Panda7个关节
        self.joint_states_sub = rospy.Subscriber('/joint_states', JointState, self._joint_states_cb, queue_size=1)
    
        self.request_event = threading.Event()

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

        rospy.loginfo("FrankaRLEnv initialized successfully.")

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
    
    def _request_callback(self, msg: Pose):

        if self.request_event.is_set():
            return 
        self.current_request = msg
        self.request_event.set()
    
    def _pose_state_callback(self, msg):
            self.pose_state = msg.data
            self.visited_states.add(self.pose_state)
            
            if self.pose_state in {4, 5}: 
                self.ready_for_rl_request = True

    def _move_group_result_cb(self, msg: MoveGroupActionResult):
        self.last_move_status = msg.status.status
        self.move_result_received = True
    
    def _add_obstacles_to_scene(self):
        #rospy.loginfo("Adding obstacles to the planning scene.")
        
        for obs_name in ["obstacle1", "obstacle2", "obstacle3"]:
            self.scene.remove_world_object(obs_name)
        
        rospy.sleep(1) 

        obstacle_radius = 0.2 

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

        if self.current_request:
            target_position = np.array([
                self.current_request.position.x,
                self.current_request.position.y,
                self.current_request.position.z
            ], dtype=np.float32)
        else:
            target_position = np.zeros(3, dtype=np.float32)
        
        vec_to_target = target_position - ee_position
        vec_to_obs1 = self.obstacle1 - ee_position
        vec_to_obs2 = self.obstacle2 - ee_position
        vec_to_obs3 = self.obstacle3 - ee_position
    
        return np.concatenate([
            ee_position, 
            vec_to_target, 
            vec_to_obs1, 
            vec_to_obs2, 
            vec_to_obs3,
            self.joint_states 
        ], dtype=np.float32)


    def _compute_reward(self, planning_failed):
        TARGET_REWARD_WEIGHT = 50.0      
        COLLISION_THRESHOLD = 0.20        
        PENALTY_COLLISION = -200          
        PENALTY_PLANNING_FAILURE = -100    
        JOINT_ERROR_WEIGHT = 200.0 

        if planning_failed:
            rospy.logwarn("Planning failed. Applying penalty.")
            return PENALTY_PLANNING_FAILURE, False

        ee_position = np.array(self.get_ee_position(), dtype=np.float32)
        target_pos_np = np.array([
            self.current_request.position.x,
            self.current_request.position.y,
            self.current_request.position.z
        ])

        distance_to_target = np.linalg.norm(ee_position - target_pos_np)

        if self.previous_distance_to_target is None:
            delta_dist = 0.0
        else:
            delta_dist = self.previous_distance_to_target - distance_to_target
       
        target_reward = delta_dist * TARGET_REWARD_WEIGHT 

        self.previous_distance_to_target = distance_to_target

        min_dist_to_obstacle = min([np.linalg.norm(ee_position - obs_pos) for obs_pos in self.obstacles])
    
        if min_dist_to_obstacle < COLLISION_THRESHOLD:
            rospy.logwarn(f"Collision detected! Distance to obstacle: {min_dist_to_obstacle:.3f} < {COLLISION_THRESHOLD}")
            return PENALTY_COLLISION, True
        
        joint_error_penalty = 0.0
        try:
            original_target_pose = deepcopy(self.current_request)
            
            self.move_group.set_pose_target(original_target_pose)
            print(f"Original target pose: {original_target_pose.position.x:.3f}, {original_target_pose.position.y:.3f}, {original_target_pose.position.z:.3f}")
            ideal_joint_values = self.move_group.get_joint_value_target()
            print(f"Ideal joint values: {ideal_joint_values}")
            ideal_joint_4 = ideal_joint_values[3]
            print(f"Ideal joint 4 value: {ideal_joint_4:.3f}")
            current_joint_4 = self.joint_states[3]
            print(f"Current joint 4 value: {current_joint_4:.3f}")
            error_j4 = abs(current_joint_4 - ideal_joint_4)
            joint_error_penalty = -error_j4 * JOINT_ERROR_WEIGHT
            
        except Exception as e:
            rospy.logwarn(f"Could not compute joint error penalty: {e}")

        total_reward = target_reward + joint_error_penalty
        
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
            
            self.previous_distance_to_target = None

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

        ee_position_before_move = self.get_ee_position()

        target_pos_np = np.array([
            self.current_request.position.x,
            self.current_request.position.y,
            self.current_request.position.z
        ])
        self.previous_distance_to_target = np.linalg.norm(ee_position_before_move - target_pos_np)

        self.original_target_pose = deepcopy(self.current_request) 

        self.resolved_pose = deepcopy(self.original_target_pose)      
        delta = np.clip(action, -1.0, 1.0) * self.MAX_DELTA
        self.resolved_pose.position.x += delta[0]
        self.resolved_pose.position.y += delta[1]
        self.resolved_pose.position.z += delta[2]
        
        self.move_result_received = False 
        self.last_move_status = None
        self.resolved_pose_pub.publish(self.resolved_pose)
        # rospy.loginfo(f"Published resolved pose: {self.resolved_pose.position.x:.3f}, {self.resolved_pose.position.y:.3f}, {self.resolved_pose.position.z:.3f}")

        rate = rospy.Rate(50)
        timeout_sec = 10
        start_time = rospy.Time.now().to_sec()

        while not rospy.is_shutdown() and not self.move_result_received:
            if rospy.Time.now().to_sec() - start_time > timeout_sec:
                rospy.logwarn("Timeout waiting for move result!")
                break
            rate.sleep()
        
        # rospy.loginfo(f"Move result received with status: {self.last_move_status}")

        planning_failed = (self.last_move_status == GoalStatus.ABORTED)

        reward, collision = self._compute_reward(planning_failed) 
        obs = self._get_observation()

        if collision:
            self.had_collision = True
        if planning_failed:
            self.had_planning_failure = True
        
        self.current_episode_steps += 1

        terminated = (self.pose_state == 9)

        truncated = self.current_episode_steps >= self.MAX_EPISODE_STEPS

        REQUIRED_STATES = {4, 5}
        if self.pose_state == 9:
            if REQUIRED_STATES.issubset(self.visited_states):
                if not self.had_planning_failure:
                    reward += 300
       
        info = {'collision': collision, 'planning_failed': planning_failed}
        print(f"Step: {self.current_episode_steps}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

        self.ready_pub.publish(True)

        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        roscpp_shutdown()