import rospy
import numpy as np
from moveit_commander import MoveGroupCommander, RobotCommander, roscpp_initialize, roscpp_shutdown
from std_msgs.msg import Float32MultiArray, Int32, Bool
from gymnasium import Env,spaces
import math
from rosgraph_msgs.msg import Log
from geometry_msgs.msg import Pose
from copy import deepcopy
from moveit_msgs.msg import MoveGroupActionResult
from actionlib_msgs.msg import GoalStatus

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

        # Initialize variables
        #self.get_current_state = get_current_state
        #self.get_fault_flag = get_fault_flag
        self.get_ee_position = get_ee_position
        #self.get_round_end = get_round_end
        self.planning_failed_flag = False

        # ROS communication
        self.resolved_pose_pub = rospy.Publisher("/rl/action_resolved", Pose, queue_size=1)
        self.last_move_status = None
        rospy.Subscriber(
            '/move_group/result',
            MoveGroupActionResult,
            self._move_group_result_cb,
            queue_size=1
        )

        self.distances_pub = rospy.Publisher('/distances_to_obstacles', Float32MultiArray, queue_size=10)

        self.MAX_EPISODE_STEPS = 800
        self.current_episode_steps = 0
        self.current_request = None

        rospy.loginfo("FrankaRLEnv initialized successfully.")

    def _move_group_result_cb(self, msg: MoveGroupActionResult):
        self.last_move_status = msg.status.status

        if self.last_move_status == GoalStatus.SUCCEEDED:
            rospy.loginfo("[RL ENV] MoveIt ActionResult: SUCCEEDED")
        elif self.last_move_status == GoalStatus.ABORTED:
            rospy.logwarn("[RL ENV] MoveIt ActionResult: ABORTED")
        else:
            rospy.loginfo(f"[RL ENV] MoveIt ActionResult: status={self.last_move_status}")

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


    def _compute_reward(self):
        PENALTY_COLLISION = -15
        PENALTY_PLANNING_FAILURE = -10.0  
        WEIGHT_DEVIATION = -2.0           
        WEIGHT_OBSTACLE_CLEARANCE = 8.0   

        SAFE_DISTANCE = 0.25              
        COLLISION_THRESHOLD = 0.2   
        REWARD_SURVIVAL = 10

        if self.planning_failed_flag:
            return PENALTY_PLANNING_FAILURE, False

        original_pos = self.original_target_pose.position
        resolved_pos = self.resolved_rl_pose.position

        original_pos_np = np.array([original_pos.x, original_pos.y, original_pos.z])
        resolved_pos_np = np.array([resolved_pos.x, resolved_pos.y, resolved_pos.z])

        deviation = self.calculate_distance(original_pos_np, resolved_pos_np)
        deviation_penalty = WEIGHT_DEVIATION * deviation

        min_dist_to_obstacle = min(
            [self.calculate_distance(obs_pos, resolved_pos_np) for obs_pos in self.obstacles]
        )

        collision = min_dist_to_obstacle < COLLISION_THRESHOLD
        if collision:
            obstacle_reward = PENALTY_COLLISION
            total_reward = deviation_penalty + obstacle_reward
        else:
            clearance_bonus = min(min_dist_to_obstacle, SAFE_DISTANCE)
            obstacle_reward = WEIGHT_OBSTACLE_CLEARANCE * clearance_bonus
            total_reward = deviation_penalty + obstacle_reward + REWARD_SURVIVAL
        
        return total_reward, collision

    
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
        self.last_move_status = None
        self.planning_failed_flag = False
       
        if self.current_request is None:
            rospy.logerr("Cannot perform step: current_request is None. The episode may have failed on reset.")
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, -100, True, False, {'error': 'no_request_in_step'}
        self.original_target_pose = deepcopy(self.current_request) 
        resolved_pose = deepcopy(self.original_target_pose)      
        delta = np.clip(action, -1.0, 1.0) * self.MAX_DELTA
        resolved_pose.position.x += delta[0]
        resolved_pose.position.y += delta[1]
        resolved_pose.position.z += delta[2]
        self.resolved_rl_pose = resolved_pose

        self.resolved_pose_pub.publish(resolved_pose)
        rospy.loginfo(f"Step {self.current_episode_steps + 1}")

        if self.last_move_status == GoalStatus.ABORTED:
            rospy.logwarn("[RL ENV] Detected MoveGroupActionResult: ABORTED")
            self.planning_failed_flag = True
        
        try:
            self.current_request = rospy.wait_for_message("/rl/action_request", Pose, timeout=60.0)
            episode_is_over = False
        except rospy.ROSException:
            episode_is_over = True

        reward, collision = self._compute_reward() 
        obs = self._get_observation()
        
        self.current_episode_steps += 1

        # terminated = collision or self.planning_failed_flag 
        # if terminated:
        #     if collision:
        #         rospy.logwarn("Episode FAILED due to collision.")
        #     if self.planning_failed_flag:
        #         rospy.logwarn("Episode FAILED due to planning failure.")

        terminated = False

        truncated = episode_is_over or (self.current_episode_steps >= self.MAX_EPISODE_STEPS)
        
        info = {"collision": collision}
        
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        roscpp_shutdown()
