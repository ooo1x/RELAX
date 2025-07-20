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

class FrankaRLEnv(Env):
    def __init__(self):
        super(FrankaRLEnv, self).__init__()
        roscpp_initialize([])
        rospy.loginfo("Initializing FrankaRLEnv...")

        # Set obstacles and goals
        self.obstacle1 = np.array([0.7, -0.25, 1.48])
        self.obstacle2 = np.array([0.7,  0.25, 1.48])
        self.obstacle3 = np.array([0.7, 0.0, 1.64])
        self.obstacles = [self.obstacle1, self.obstacle2, self.obstacle3]

        # Set state space: ee_x, ee_y, ee_z, t_x,t_y,t_z, [dx1, dy1, dz1], [dx2, dy2, dz2], [dx3, dy3, dz3]
        obs_dim = 15
        high_bounds = np.array([np.inf] * obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high_bounds, high=high_bounds, dtype=np.float32)

        # Set action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.MAX_DELTA = 0.05

        # ROS communication
        self.resolved_pose_pub = rospy.Publisher("/rl/action_resolved", Pose, queue_size=1, latch=True)
        self.last_move_status = None
        rospy.Subscriber('/move_group/result',MoveGroupActionResult,self._move_group_result_cb,queue_size=1)
        self.distances_pub = rospy.Publisher('/distances_to_obstacles', Float32MultiArray, queue_size=10)
        rospy.Subscriber('/pose_state', Int32, self._pose_state_callback, queue_size=1)
        self.request_sub = rospy.Subscriber('/rl/action_request', Pose, self._request_callback, queue_size=1)
        self.ready_pub = rospy.Publisher('/rl/ready_for_next', Bool, queue_size=1)

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

        rospy.loginfo("FrankaRLEnv initialized successfully.")
    
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
            if self.pose_state == 404: 
                self.episode_is_over = True

    def _move_group_result_cb(self, msg: MoveGroupActionResult):
        self.last_move_status = msg.status.status
        self.move_result_received = True
    
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

        try:
            dist1 = np.linalg.norm(ee_position - self.obstacle1)
            dist2 = np.linalg.norm(ee_position - self.obstacle2)
            dist3 = np.linalg.norm(ee_position - self.obstacle3)

            distances_msg = Float32MultiArray()
            distances_msg.data = [dist1, dist2, dist3]
            
            self.distances_pub.publish(distances_msg)
        except Exception as e:
            rospy.logwarn_throttle(10, f"[FrankaRLEnv] Could not publish distances: {e}")

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
            vec_to_obs3
        ], dtype=np.float32)


    def _compute_reward(self, planning_failed):
        TARGET_REWARD_WEIGHT = 100.0      
        COLLISION_THRESHOLD = 0.20        
        PENALTY_COLLISION = -200          
        PENALTY_PLANNING_FAILURE = -20    

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

        return target_reward, False
    
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
        
        REQUIRED_STATES = {1, 2, 4, 5, 9}
        if self.episode_is_over:
            if REQUIRED_STATES.issubset(self.visited_states):
                rospy.loginfo("Episode ended with full state sequence.")
                if not self.had_collision:
                    reward += 300

        terminated = self.episode_is_over
        truncated = self.current_episode_steps >= self.MAX_EPISODE_STEPS
       
        info = {'collision': collision, 'planning_failed': planning_failed}
        print(f"Step: {self.current_episode_steps}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

        self.ready_pub.publish(True)

        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        roscpp_shutdown()