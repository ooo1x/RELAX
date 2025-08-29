#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray, Bool
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
from moveit_msgs.msg import RobotState
import gym
from gym import spaces


class FrankaRLEnv(gym.Env):
    """
    RL environment for Franka with ROS + MoveIt
    - Observation = faulty joints (7) + fault vector (7) + last point (7) = 21-dim vector
    - Action = desired joint positions (7) = 7-dim vector
    - Reward: 检查 trajectory 是否进入距离 fixed_point < 0.15m 的区域
    """

    def __init__(self):
        super(FrankaRLEnv, self).__init__()

        # 初始化 ROS 节点
        # rospy.init_node("franka_rl_env", anonymous=True)

        # Observation space = 21-dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )

        # Action space = 7-dim (desired joint positions) with Panda joint limits
        self.action_space = spaces.Box(
            low=np.array([
                -2.8973,   # A1
                -1.7628,   # A2
                -2.8973,   # A3
                -3.0718,   # A4
                -2.8973,   # A5
                -0.0175,   # A6
                -2.8973    # A7
            ], dtype=np.float32),
            high=np.array([
                2.8973,    # A1
                1.7628,    # A2
                2.8973,    # A3
                -0.0698,   # A4
                2.8973,    # A5
                3.7525,    # A6
                2.8973     # A7
            ], dtype=np.float32),
            dtype=np.float32
        )

        # RL 状态订阅
        self._latest_msg = None
        rospy.Subscriber("/rl_state", Float64MultiArray, self._callback)

        # 动作发布
        self._action_pub = rospy.Publisher("/rl_action", Float64MultiArray, queue_size=10)

        # MoveIt FK service
        rospy.loginfo("Waiting for /compute_fk service...")
        rospy.wait_for_service("/compute_fk")
        self.fk_client = rospy.ServiceProxy("/compute_fk", GetPositionFK)

        # Panda joint names
        self.joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3",
            "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
        ]
 
        rospy.loginfo("FrankaRLEnv initialized, waiting for /rl_state...")

    def _callback(self, msg):
        """Callback to receive ROS message"""
        self._latest_msg = np.array(msg.data, dtype=np.float32)

        if len(self._latest_msg) >= 21:
            rospy.loginfo("=== Received new RL state, triggering step() ===")
            # ⚠️ 这里目前用随机动作测试，可以替换成 agent.select_action(state)
            random_action = self.action_space.sample()
            self.step(random_action)
        else:
            rospy.logwarn("Received rl_state length < 21")

    def _get_obs(self):
        if self._latest_msg is None:
            return np.zeros(21, dtype=np.float32)
        return self._latest_msg

    def _fk(self, joint_positions):
        """调用 MoveIt FK 服务，把关节角度转成末端执行器坐标"""
        req = GetPositionFKRequest()
        req.header.frame_id = "world"
        req.fk_link_names = ["panda_hand_tcp"]

        joint_state = JointState()
        joint_state.name = self.joint_names
        joint_state.position = joint_positions.tolist()

        robot_state = RobotState()
        robot_state.joint_state = joint_state
        req.robot_state = robot_state

        try:
            res = self.fk_client(req)
            if res.error_code.val == 1:  # SUCCESS
                pose = res.pose_stamped[0].pose
                return np.array([pose.position.x, pose.position.y, pose.position.z])
            else:
                rospy.logwarn("FK failed with code %d", res.error_code.val)
                return None
        except rospy.ServiceException as e:
            rospy.logerr("FK service call failed: %s", e)
            return None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._get_obs()
        return obs, {}
    
    def step(self, action):
        """
        action: np.array of shape (7,) -> desired joint positions
        """
        # 发布 action
        action_msg = Float64MultiArray()
        action_msg.data = action.tolist()
        self._action_pub.publish(action_msg)
        rospy.loginfo(f"Published action (desired joints): {action}")

        # === 获取上一次状态的 last_point 作为轨迹终点 ===
        obs = self._get_obs()
        if len(obs) < 21:
            rospy.logwarn("Invalid state received, returning default reward.")
            return obs, -100.0, False, False, {}

        last_point = obs[14:21]
        start_joints = np.array(action, dtype=np.float32)
        goal_joints = np.array(last_point, dtype=np.float32)

        # === 三次插补，生成轨迹 ===
        n_points = 20
        trajectory = []
        for t in np.linspace(0, 1, n_points):
            q_t = (2*t**3 - 3*t**2 + 1) * start_joints + \
                (-2*t**3 + 3*t**2) * goal_joints
            trajectory.append(q_t)
        trajectory = np.array(trajectory)

        # === 检查轨迹的末端是否进入危险区域 ===
        fixed_point = np.array([0.4, 0.25, 1.05])
        too_close = False
        min_dist = 999.0

        rospy.loginfo("=== New Trajectory EE positions ===")
        for i, q in enumerate(trajectory):
            ee_pos = self._fk(q)
            if ee_pos is None:
                continue

            rospy.loginfo(f"Point {i:02d}: x={ee_pos[0]:.3f}, y={ee_pos[1]:.3f}, z={ee_pos[2]:.3f}")

            dist = np.linalg.norm(ee_pos - fixed_point)
            min_dist = min(min_dist, dist)
            if dist < 0.15:
                too_close = True
                break

        # === 奖励函数 ===
        if too_close:
            reward = -100.0
            rospy.logwarn(f"Unsafe trajectory! min_dist={min_dist:.3f} m")
        else:
            reward = -min_dist  # 越远越好

        terminated = False  # 单步版本，不负责 episode
        truncated = False

        return obs, reward, terminated, truncated, {}



if __name__ == "__main__":
    env = FrankaRLEnv()
    rospy.spin()   # 事件驱动，不再自己循环

