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
        self.MAX_DELTA = 0.1

        # ROS communication
        self.resolved_pose_pub = rospy.Publisher("/rl/action_resolved", Pose, queue_size=1)
        self.last_move_status = None
        rospy.Subscriber('/move_group/result',MoveGroupActionResult,self._move_group_result_cb,queue_size=1)
        self.distances_pub = rospy.Publisher('/distances_to_obstacles', Float32MultiArray, queue_size=10)
        rospy.Subscriber('/pose_state', Int32, self._pose_state_callback, queue_size=1)
        self.request_sub = rospy.Subscriber('/rl/action_request', Pose, self._request_callback, queue_size=1)

        self.request_event = threading.Event()

        self.MAX_EPISODE_STEPS = 300
        self.current_episode_steps = 0
        self.current_request = None
        self.previous_distance_to_target = None
        self.pose_state = 0
        self.ready_for_rl_request = False 
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        rospy.loginfo("FrankaRLEnv initialized successfully.")
    
    def _request_callback(self, msg: Pose):
        rospy.logwarn(f">>> Received RL action request <<< pose: {msg.position.x}, {msg.position.y}, {msg.position.z}")
        
        self.current_request = msg
        self.request_event.set()
    
    def _pose_state_callback(self, msg):
            self.pose_state = msg.data
            if self.pose_state == 4:
                self.ready_for_rl_request = True
            elif self.pose_state == 404:
                self.episode_is_over = True

    def _move_group_result_cb(self, msg: MoveGroupActionResult):
        rospy.loginfo(f"MoveIt result callback: status={msg.status.status}, text='{msg.status.text}'")
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
        distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)
        return distance
    
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
        GOAL_THRESHOLD = 0.05             

        PENALTY_COLLISION = -200          
        REWARD_GOAL_REACHED = 300         
        PENALTY_PLANNING_FAILURE = -50    

        if planning_failed:
            rospy.logwarn("Planning failed. Applying penalty.")
            return PENALTY_PLANNING_FAILURE, False

        ee_position = np.array(self.get_ee_position(), dtype=np.float32)
        target_pos_np = np.array([
            self.resolved_pose.position.x,
            self.resolved_pose.position.y,
            self.resolved_pose.position.z
        ])

        distance_to_target = np.linalg.norm(ee_position - target_pos_np)

        target_reward = -distance_to_target * TARGET_REWARD_WEIGHT

        min_dist_to_obstacle = min([np.linalg.norm(ee_position - obs_pos) for obs_pos in self.obstacles])
    
        if min_dist_to_obstacle < COLLISION_THRESHOLD:
            rospy.logwarn("Collision detected! Applying penalty.")
            return PENALTY_COLLISION, True

        success_reward = 0.0
        if distance_to_target < GOAL_THRESHOLD:
            rospy.loginfo("Goal reached! Applying big reward.")
            success_reward = REWARD_GOAL_REACHED

        total_reward = target_reward + success_reward

        return total_reward, False
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        self.episode_is_over = False
        self.ready_for_rl_request = False
        self.current_episode_steps = 0
        self.current_request = None
        self.previous_distance_to_target = None

        rospy.loginfo("Env Reset.")

        while not self.ready_for_rl_request and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        while self.request_sub.get_num_connections() == 0 and not rospy.is_shutdown():
            # rospy.logwarn_throttle(2, "Waiting for C++ publisher to connect to /rl/action_request...")
            rospy.sleep(0.2)
        
        self.request_event.clear()
        event_is_set = self.request_event.wait(timeout=15.0)

        if not event_is_set:
            # rospy.logerr("Timeout on reset. Publisher is connected, but did not send a request in time.")
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, {'error': 'reset_timeout'}

        # rospy.loginfo("Received initial request for the new episode.")
        obs = self._get_observation()
        return obs, {}

    def step(self, action):

        rospy.loginfo(f"self.current_request: {self.current_request}")

        if self.current_request is None:
            rospy.logwarn("No action request available. Waiting for C++ up to 10s...")

        self.original_target_pose = deepcopy(self.current_request) 

        self.resolved_pose = deepcopy(self.original_target_pose)      
        delta = np.clip(action, -1.0, 1.0) * self.MAX_DELTA
        self.resolved_pose.position.x += delta[0]
        self.resolved_pose.position.y += delta[1]
        self.resolved_pose.position.z += delta[2]
        
        self.move_result_received = False 
        self.last_move_status = None
        self.resolved_pose_pub.publish(self.resolved_pose)

        # rospy.logwarn(f"Publishing resolved pose: {self.resolved_pose.position.x}, {self.resolved_pose.position.y}, {self.resolved_pose.position.z}")

        while not self.move_result_received and not rospy.is_shutdown():
            rospy.sleep(2)
        
        # rospy.loginfo(f"Move result received with status: {self.last_move_status}")

        planning_failed = (self.last_move_status == GoalStatus.ABORTED)

        reward, collision = self._compute_reward(planning_failed) 
        obs = self._get_observation()
        
        self.current_episode_steps += 1
        terminated = self.episode_is_over
        truncated = self.current_episode_steps >= self.MAX_EPISODE_STEPS

        if self.episode_is_over and not (collision or planning_failed):
            reward += 200
        
        info = {'collision': collision, 'planning_failed': planning_failed}
        print(f"Step: {self.current_episode_steps}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        roscpp_shutdown()
