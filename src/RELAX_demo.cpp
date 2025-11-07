#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>

#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/CollisionObject.h>

#include <moveit_visual_tools/moveit_visual_tools.h>

// TF2
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

// ROS
#include <ros/ros.h>

//Time Parametrization
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include "std_msgs/Bool.h"
#include "std_msgs/Int32.h"
// The circle constant tau = 2*pi. One tau is one rotation in radians.
const double tau = 2 * M_PI;

#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float32MultiArray.h>
#include "std_msgs/Float32.h"
#include <iomanip>
#include <trajectory_msgs/JointTrajectoryPoint.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <sensor_msgs/JointState.h>
#include <angles/angles.h>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>

std::string g_csv_file_j4;                 // 例如 $HOME/rl_joint4_logs.csv
std::mutex g_rec_mutex;                    // 保护 g_recorded_joint_states

bool g_python_is_ready = false;
bool g_is_recording = false; 
ros::Publisher g_step_result_pub;          // 发布执行成功/失败结果给RL
ros::Publisher g_pose_state_pub;           // 发布机器人的当前位姿状态
double g_current_panda_joints = 0.0;
std::vector<sensor_msgs::JointState> g_recorded_joint_states;
ros::Publisher g_joint_trajectory_command_pub;
ros::Publisher g_reward_pub;

ros::Publisher g_faulty_joints_pub;         // 发布“有问题的”规划起始关节状态给RL
std::vector<double> g_corrected_joints;     // 用于存储完整的7个修正后的关节值
bool g_corrected_joints_received = false; // 标志位

static inline bool fileExists(const std::string& p){ std::ifstream f(p.c_str()); return f.good(); }
void ensureJ4Header(const std::string& path){
  if (fileExists(path)) return;
  std::ofstream ofs(path, std::ios::out);
  ofs << "phase,point_idx,t_from_start,joint4\n";
  ofs.close();
}

int findJ4Index(const std::vector<std::string>& names){
  for (size_t i=0;i<names.size();++i) if (names[i]=="panda_joint4") return static_cast<int>(i);
  return -1;
}

// 仅保存 plan 的 joint4
void appendPlanToCsv(const std::string& path,
                       const std::string& phase,                // "initial_plan" / "corrected_plan"
                       const moveit_msgs::RobotTrajectory& traj)
{
  const auto& jt = traj.joint_trajectory;
  if (jt.joint_names.empty() || jt.points.empty()) return;
  int j4 = findJ4Index(jt.joint_names);
  if (j4 < 0) return;

  ensureJ4Header(path);
  std::ofstream ofs(path, std::ios::app);
  for (size_t p=0; p<jt.points.size(); ++p){
    const auto& pt = jt.points[p];
    double j4_val = (j4 < (int)pt.positions.size()) ? pt.positions[j4] : std::numeric_limits<double>::quiet_NaN();
    double t = pt.time_from_start.toSec();
    ofs << phase << "," << p << "," << t << "," << std::setprecision(9) << j4_val << "\n";
  }
  ofs.close();
}

// 保存执行时的 joint4（/joint_states）
void appendExecutedJ4ToCsv(const std::string& path,
                           const std::vector<sensor_msgs::JointState>& data)
{
  if (data.empty()) return;
  ensureJ4Header(path);

  ros::Time t0 = data.front().header.stamp;
  std::ofstream ofs(path, std::ios::app);

  int j4_hint = findJ4Index(data.front().name);

  for (size_t i=0; i<data.size(); ++i){
    const auto& msg = data[i];
    int idx = j4_hint;
    if (idx < 0 || idx >= (int)msg.position.size()){
      idx = findJ4Index(msg.name);
    }
    double j4_val = (idx >= 0 && idx < (int)msg.position.size()) ? msg.position[idx] : std::numeric_limits<double>::quiet_NaN();
    double dt = (msg.header.stamp - t0).toSec(); if (dt < 0) dt = 0.0;

    ofs << "executed_joint_states," << i << "," << dt << "," << std::setprecision(9) << j4_val << "\n";
  }
  ofs.close();
}


// 接收完整的7个关节值
void correctedJointsCallback(const std_msgs::Float32MultiArray::ConstPtr& msg)
{
  // 检查收到的数据是否包含7个元素
  if (msg->data.size() == 7)
  {
    g_corrected_joints.assign(msg->data.begin(), msg->data.end());
    g_corrected_joints_received = true;
  }
  else
  {
ROS_WARN("Received array with incorrect size (was %zu), message ignored.", msg->data.size());
}
}

std::vector<Eigen::Vector3d> g_obstacles = {
    Eigen::Vector3d(0.75, -0.35, 1.2),
    Eigen::Vector3d(0.75, 0.25, 1.2),
    Eigen::Vector3d(0.75, 0.0, 1.64)
};
const double COLLISION_THRESHOLD = 0.20;

void pythonReadyCallback(const std_msgs::BoolConstPtr& msg)
{
    if (msg->data)
    {
        g_python_is_ready = true;
    }
}

float computeTrajectoryReward(const moveit_msgs::RobotTrajectory& trajectory,
                              const moveit::core::RobotModelConstPtr& robot_model,
                              const std::vector<double>& start_joints,
                              const geometry_msgs::Pose& target_pose)
{
    const float TARGET_REWARD_WEIGHT = 100.0f;
    const float PENALTY_COLLISION = -200.0f;
    const float SUCCESSFULLY_REWARD = 200.0f;

    if (trajectory.joint_trajectory.points.empty())
    {
        ROS_WARN("Planned trajectory is empty, reward is 0.");
        return 0.0f;
    }
    
    moveit::core::RobotState robot_state(robot_model);
    for (const auto& point : trajectory.joint_trajectory.points)
    {
        if (point.positions.empty()) continue;

        robot_state.setJointGroupPositions("panda_manipulator", point.positions);
        const Eigen::Isometry3d& ee_pose = robot_state.getGlobalLinkTransform("panda_hand_tcp");
        Eigen::Vector3d ee_position = ee_pose.translation();

        for (const auto& obs_pos : g_obstacles)
        {
            double dist = (ee_position - obs_pos).norm();
            if (dist <= COLLISION_THRESHOLD)
            {
                ROS_WARN("Trajectory failed safety check. Applying collision penalty.");
                return PENALTY_COLLISION;
            }
        }
    }

    ROS_INFO("Trajectory passed safety check. Calculating target reward.");

    moveit::core::RobotState start_state(robot_model);
    start_state.setJointGroupPositions("panda_manipulator", start_joints);
    const Eigen::Isometry3d& start_ee_pose = start_state.getGlobalLinkTransform("panda_hand_tcp");
    Eigen::Vector3d start_ee_position = start_ee_pose.translation();

    Eigen::Vector3d target_position(target_pose.position.x, target_pose.position.y, target_pose.position.z);

    double initial_distance_to_target = (start_ee_position - target_position).norm();
    float target_reward = initial_distance_to_target * TARGET_REWARD_WEIGHT;
    float total_reward = target_reward + SUCCESSFULLY_REWARD;

    return target_reward;
}


bool createManualPlan(const std::vector<double>& start_joints,
                      const std::vector<double>& end_joints,
                      moveit::planning_interface::MoveGroupInterface::Plan& plan,
                      moveit::planning_interface::MoveGroupInterface& move_group)
{
    moveit::core::RobotState start_state(move_group.getRobotModel());
    start_state.setJointGroupPositions(move_group.getName(), start_joints);
    
    moveit::core::RobotState end_state(move_group.getRobotModel());
    end_state.setJointGroupPositions(move_group.getName(), end_joints);

    robot_trajectory::RobotTrajectory robot_traj(move_group.getRobotModel(), move_group.getName());

    double distance = start_state.distance(end_state);
    
    int steps = static_cast<int>(std::ceil(distance / 0.05));
    if (steps < 2) steps = 2; // 保证至少有起点和终点

    ROS_INFO("Distance between states is %.2f, interpolating with %d steps.", distance, steps);

    for (int i = 0; i <= steps; ++i)
    {
        moveit::core::RobotState interpolated_state(start_state);
        double ratio = static_cast<double>(i) / static_cast<double>(steps);
        start_state.interpolate(end_state, ratio, interpolated_state);
        robot_traj.addSuffixWayPoint(interpolated_state, 0.01);
    }

    // 4. 时间参数化
    trajectory_processing::TimeOptimalTrajectoryGeneration totg;
    if (!totg.computeTimeStamps(robot_traj, 0.3, 0.3)) {
        ROS_ERROR("[ManualPlan] Failed to re-time the trajectory with TOTG!");
        return false;
    }
    
    robot_traj.getRobotTrajectoryMsg(plan.trajectory_);
    
    ROS_INFO("[ManualPlan] Successfully created a DENSE plan with %zu points.", plan.trajectory_.joint_trajectory.points.size());
    
    return true;
}

bool performRLStep(moveit::planning_interface::MoveGroupInterface& move_group, const geometry_msgs::Pose& final_target_pose)
{
    g_corrected_joints_received = false;

    // 步骤 1: 规划一次路径，以获取初始的、“有问题的”关节状态
    move_group.setStartStateToCurrentState();
    moveit::planning_interface::MoveGroupInterface::Plan initial_plan;
    move_group.setPoseTarget(final_target_pose);

    if (move_group.plan(initial_plan) != moveit::core::MoveItErrorCode::SUCCESS)
    {
        std_msgs::Bool result_msg; result_msg.data = false; g_step_result_pub.publish(result_msg);
        std_msgs::Float32 reward_msg; reward_msg.data = -200.0f; g_reward_pub.publish(reward_msg); // 对规划失败施加惩罚
        return false;
    }

    if (initial_plan.trajectory_.joint_trajectory.points.empty()) {
        std_msgs::Bool result_msg; result_msg.data = false; g_step_result_pub.publish(result_msg);
        std_msgs::Float32 reward_msg; reward_msg.data = -200.0f; g_reward_pub.publish(reward_msg);
        return false;
    }

    // MoveIt规划出的完整起始关节状态数组
    std::vector<double> faulty_start_joints = initial_plan.trajectory_.joint_trajectory.points.front().positions;
    std::vector<double> final_joints = initial_plan.trajectory_.joint_trajectory.points.back().positions;

    {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(4);
        ss << "[";
        for (size_t i = 0; i < faulty_start_joints.size(); ++i) {
            ss << faulty_start_joints[i] << (i < faulty_start_joints.size() - 1 ? ", " : "");
        }
        ss << "]";
        ROS_INFO_STREAM("[RLStep] faulty_joints: " << ss.str());
    }
    appendPlanToCsv(g_csv_file_j4, "initial_plan", initial_plan.trajectory_);


    // 步骤 2: 将“有问题的”关节值发送给Python，并等待修正动作
    std_msgs::Float32MultiArray msg_to_rl;
    for (const auto& joint_value : faulty_start_joints)
    {
        msg_to_rl.data.push_back(static_cast<float>(joint_value));
    }
    
    ros::Rate r(10);
    ros::Time start_time = ros::Time::now();
    while(ros::ok() && !g_corrected_joints_received)
    {
        g_faulty_joints_pub.publish(msg_to_rl);
        ros::spinOnce();
        r.sleep();
        if ((ros::Time::now() - start_time).toSec() > 5.0) {
            std_msgs::Bool result_msg; result_msg.data = false; g_step_result_pub.publish(result_msg);
            std_msgs::Float32 reward_msg; reward_msg.data = 0; g_reward_pub.publish(reward_msg);
            return false;
        }
    }

    // 步骤 3: 直接应用从RL接收到的修正 
    std::vector<double> corrected_joints = g_corrected_joints;
    {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(4);
        ss << "[";
        for (size_t i = 0; i < corrected_joints.size(); ++i) {
            ss << corrected_joints[i] << (i < corrected_joints.size() - 1 ? ", " : "");
        }
        ss << "]";
        ROS_INFO_STREAM("[RL step]  corrected_joints: " << ss.str());
    }
    
    // 步骤 4: 创建一个从修正后的起始状态到最终目标的新规划
    moveit::planning_interface::MoveGroupInterface::Plan final_plan;
    if (!createManualPlan(corrected_joints, final_joints, final_plan, move_group)) {
        std_msgs::Bool result_msg; result_msg.data = false; g_step_result_pub.publish(result_msg);
        std_msgs::Float32 reward_msg; reward_msg.data = 0; g_reward_pub.publish(reward_msg);
        return false;
    }
    appendPlanToCsv(g_csv_file_j4, "corrected_plan", final_plan.trajectory_);

    std_msgs::Float32 reward_msg;
    reward_msg.data = computeTrajectoryReward(final_plan.trajectory_, move_group.getRobotModel(), corrected_joints, final_target_pose);
    
    // 步骤 5: 执行规划
    moveit::core::MoveItErrorCode result = move_group.execute(final_plan);
    
    std_msgs::Bool result_msg;
    result_msg.data = (result == moveit::core::MoveItErrorCode::SUCCESS);

    // 将结果和奖励发布回RL
    g_reward_pub.publish(reward_msg);
    g_step_result_pub.publish(result_msg);

    ros::Duration(0.1).sleep();
    return result_msg.data;
}


void initPose(moveit::planning_interface::MoveGroupInterface& move_group)
{ 
  // We can plan a motion for this group to a desired pose for the
  // end-effector.
  geometry_msgs::Pose target_pose_init;

  //Convert Orienation from RPY to Quaternion
  tf2::Quaternion orientation;
  orientation.setRPY(-tau/2, 0, -tau/8);

  target_pose_init.orientation = tf2::toMsg(orientation);
  
  target_pose_init.position.x = 0.5;
  target_pose_init.position.y = 0.0;
  target_pose_init.position.z = 1.5;
  move_group.setPoseTarget(target_pose_init);

  move_group.move();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
//Position, Orientation, Planning, Exectuion hoverPose

void hoverPose(moveit::planning_interface::MoveGroupInterface& move_group)
{ 
  // We can plan a motion for this group to a desired pose for the
  // end-effector.
  geometry_msgs::Pose target_pose_hover;

  //Convert Orienation from RPY to Quaternion
  tf2::Quaternion orientation;
  orientation.setRPY(-tau/2, 0, -tau/8);

  target_pose_hover.orientation = tf2::toMsg(orientation);
  
  target_pose_hover.position.x = 0.502;  //Random points near. Python script: 1, sub joint_states, 
  //                                                                          2, calculate real time EE locations
  //                                                                          3, set a/multiple fixed points
  //                                                                          4, design reward function: lower 20cm. 
  //                                                                          5, define a new topic: output from RL
  //                                                                          6, control code sub this new topic                       
  target_pose_hover.position.y = -0.2;
  target_pose_hover.position.z = 1.5;
  move_group.setPoseTarget(target_pose_hover);

  move_group.move();
  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
//Position, Orientation, Planning, Exectuion pickPose as Cartesian Motion

void pickPose(moveit::planning_interface::MoveGroupInterface& move_group_interface, std::string direction){
  moveit::planning_interface::MoveGroupInterface::Plan cartesianPlan;
  move_group_interface.setStartStateToCurrentState();

  move_group_interface.setMaxVelocityScalingFactor(1);
  move_group_interface.setMaxAccelerationScalingFactor(1);

  std::vector<geometry_msgs::Pose> waypoints;

  geometry_msgs::Pose target_pose_pick = move_group_interface.getCurrentPose().pose;
  target_pose_pick.position.x += 0.0;
  target_pose_pick.position.y += 0.0;
  if (direction == "down"){
    target_pose_pick.position.z -= 0.26;//0.26
  }
  else if (direction == "up"){
    target_pose_pick.position.z += 0.26;
  }
  waypoints.push_back(target_pose_pick); 

  moveit_msgs::RobotTrajectory trajectory_msg;
  move_group_interface.setPlanningTime(10.0);
  
 
  double fraction = move_group_interface.computeCartesianPath(waypoints,
                                               0.01,  // eef_step
                                               0.0,   // jump_threshold
                                               trajectory_msg, false);
  
  // Modify trajectory for adjusting speed
  
  // Create robot trajectory object
  robot_trajectory::RobotTrajectory rt(move_group_interface.getCurrentState()->getRobotModel(), "panda_arm");

  // Get robot trajectory
  rt.setRobotTrajectoryMsg(*move_group_interface.getCurrentState(), trajectory_msg);
 
  // Create a IterativeParabolicTimeParameterization object
  trajectory_processing::IterativeParabolicTimeParameterization iptp;

  //Compute TimeStamps
  iptp.computeTimeStamps(rt, 0.1, 0.1);
  
  // Get RobotTrajectory_msg from RobotTrajectory
  rt.getRobotTrajectoryMsg(trajectory_msg);

  cartesianPlan.trajectory_ = trajectory_msg;
 
  move_group_interface.execute(cartesianPlan);  

  

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
//Position, Orientation, Planning, Exectuion hoverPlacePose

void hoverPlacePose(moveit::planning_interface::MoveGroupInterface& move_group)
{ 
  // We can plan a motion for this group to a desired pose for the
  // end-effector.
  geometry_msgs::Pose pose_hover_place;

  //Convert Orienation from RPY to Quaternion
  tf2::Quaternion orientation;
  orientation.setRPY(-tau/2, 0, -tau/8);

  pose_hover_place.orientation = tf2::toMsg(orientation);
  
  pose_hover_place.position.x = 0.502;
  pose_hover_place.position.y = 0.2;
  pose_hover_place.position.z = 1.5;
  move_group.setPoseTarget(pose_hover_place);

  move_group.move();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//Position, Orientation, Planning, Exectuion PlacePose

void PlacePose(moveit::planning_interface::MoveGroupInterface& move_group_interface, std::string direction)
{ 
  moveit::planning_interface::MoveGroupInterface::Plan cartesianPlan;
  move_group_interface.setStartStateToCurrentState();

  move_group_interface.setMaxVelocityScalingFactor(1);
  move_group_interface.setMaxAccelerationScalingFactor(1);

  std::vector<geometry_msgs::Pose> waypoints;

  geometry_msgs::Pose target_pose_place = move_group_interface.getCurrentPose().pose;
  target_pose_place.position.x += 0.0;
  target_pose_place.position.y += 0.0;
  if (direction == "down"){
    target_pose_place.position.z -= 0.26;
  }
  else if (direction == "up"){
    target_pose_place.position.z += 0.26;
  }
  waypoints.push_back(target_pose_place); 

  moveit_msgs::RobotTrajectory trajectory_msg;
  move_group_interface.setPlanningTime(2);
 
  double fraction = move_group_interface.computeCartesianPath(waypoints,
                                               0.01,  // eef_step
                                               0.0,   // jump_threshold
                                               trajectory_msg, false);
  // Modify trajectory for adjusting speed
  
  // Create robot trajectory object
  robot_trajectory::RobotTrajectory rt(move_group_interface.getCurrentState()->getRobotModel(), "panda_arm");

  // Get robot trajectory
  rt.setRobotTrajectoryMsg(*move_group_interface.getCurrentState(), trajectory_msg);
 
  // Create a IterativeParabolicTimeParameterization object
  trajectory_processing::IterativeParabolicTimeParameterization iptp;

  //Compute TimeStamps
  iptp.computeTimeStamps(rt, 0.1, 0.1);
  
  // Get RobotTrajectory_msg from RobotTrajectory
  rt.getRobotTrajectoryMsg(trajectory_msg);

  cartesianPlan.trajectory_ = trajectory_msg;
 
 
  move_group_interface.execute(cartesianPlan);  

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// close Gripper for moveit msg grasp

void closedGripper(trajectory_msgs::JointTrajectory& posture)
{

  // Add both finger joints
  posture.joint_names.resize(2);
  posture.joint_names[0] = "panda_finger_joint1";
  posture.joint_names[1] = "panda_finger_joint2";

  //set closed position
  posture.points.resize(1);
  posture.points[0].positions.resize(2);
  posture.points[0].positions[0] = 0.0065;
  posture.points[0].positions[1] = 0.0065;
  posture.points[0].effort.resize(2);
  posture.points[0].effort[0] = 5.00;
  posture.points[0].effort[1] = 5.00;
  
  //Additonal try if force is possible
  //posture.points[0].effort.resize(1);
  //posture.points[0].effort[0] = 1;
  
  
  posture.points[0].time_from_start = ros::Duration(0.5);

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//Little movement for grasping with grasp msg

void pick(moveit::planning_interface::MoveGroupInterface& move_group)
{
  //Create Vector for grasp approaches (only need 1)
  std::vector<moveit_msgs::Grasp> grasps;
  grasps.resize(1);
  
  grasps[0].pre_grasp_approach.direction.header.frame_id = "panda_link0";

  // Setting grasp pose
  geometry_msgs::PoseStamped current_pose = move_group.getCurrentPose();
  grasps[0].grasp_pose = current_pose;
  grasps[0].grasp_pose.pose.position.z -= 0.001;
  grasps[0].grasp_pose.header.frame_id = "panda_link0";
 
  //std::cout << "------------Pick Pose----------" << std::endl;
  //std::cout << current_pose << std::endl;

  // Setting pre-grasp approach

  // Direction is set as negative z axis, approach from above the object
  grasps[0].pre_grasp_approach.direction.vector.z = -1.0;
  grasps[0].pre_grasp_approach.min_distance = 0.001;
  grasps[0].pre_grasp_approach.desired_distance = 0.002;

  //Close gripper to grasp object
  closedGripper(grasps[0].grasp_posture);

  // Set support surface as table1.
  move_group.setSupportSurfaceName("table1");
  // Call pick to pick up the object using the grasps given
  move_group.pick("cylinder1", grasps);
  
}

///////////////////////////////////////////////////////////////////////////////////////
// Plan and execute open hand

void openHand(moveit::planning_interface::MoveGroupInterface& move_group_interface_hand)
{ 
  // Open the gripper
  move_group_interface_hand.setJointValueTarget(move_group_interface_hand.getNamedTargetValues("open"));

  //Move the robot
  // ROS_WARN("START EXECUTION");
  move_group_interface_hand.move();

}

//planning scene
///////////////////////////////////////////////////////////////////////////////////////////////////////////
//Objects for planing scene including table, wall back, wall left

void addCollisionObjects(moveit::planning_interface::PlanningSceneInterface& planning_scene_interface)
{

  // Create vector to hold 3 collision objects.
  std::vector<moveit_msgs::CollisionObject> collision_objects;
  collision_objects.resize(3);

  // Add the first table where the cylinder will originally be kept.
  collision_objects[0].id = "table1";
  collision_objects[0].header.frame_id = "panda_link0";

  // Define the primitive and its dimensions. 
  collision_objects[0].primitives.resize(1);
  collision_objects[0].primitives[0].type = collision_objects[0].primitives[0].BOX;
  collision_objects[0].primitives[0].dimensions.resize(3);
  collision_objects[0].primitives[0].dimensions[0] = 1;
  collision_objects[0].primitives[0].dimensions[1] = 1.8;
  collision_objects[0].primitives[0].dimensions[2] = 0;

  // Define the pose of the table. 
  collision_objects[0].primitive_poses.resize(1);
  collision_objects[0].primitive_poses[0].position.x = 0.2;
  collision_objects[0].primitive_poses[0].position.y = 0;
  collision_objects[0].primitive_poses[0].position.z = -0.01;
  collision_objects[0].primitive_poses[0].orientation.w = 1.0;


  collision_objects[0].operation = collision_objects[0].ADD;

  // Add the wall at the back of the robot.
  collision_objects[1].id = "wallback";
  collision_objects[1].header.frame_id = "panda_link0";

  // Define the primitive and its dimensions. 
  collision_objects[1].primitives.resize(1);
  collision_objects[1].primitives[0].type = collision_objects[1].primitives[0].BOX;
  collision_objects[1].primitives[0].dimensions.resize(3);
  collision_objects[1].primitives[0].dimensions[0] = 0;
  collision_objects[1].primitives[0].dimensions[1] = 1.8;
  collision_objects[1].primitives[0].dimensions[2] = 1.0;

  // Define the pose of the wall. 
  collision_objects[1].primitive_poses.resize(1);
  collision_objects[1].primitive_poses[0].position.x = -0.3;
  collision_objects[1].primitive_poses[0].position.y = 0;
  collision_objects[1].primitive_poses[0].position.z = 0.5;
  collision_objects[1].primitive_poses[0].orientation.w = 1.0;


  collision_objects[1].operation = collision_objects[1].ADD;

  // Add the second wall on the left hand side
  collision_objects[2].id = "wallleft";
  collision_objects[2].header.frame_id = "panda_link0";

  // Define the primitive and its dimensions. 
  collision_objects[2].primitives.resize(1);
  collision_objects[2].primitives[0].type = collision_objects[1].primitives[0].BOX;
  collision_objects[2].primitives[0].dimensions.resize(3);
  collision_objects[2].primitives[0].dimensions[0] = 1.2;
  collision_objects[2].primitives[0].dimensions[1] = 0;
  collision_objects[2].primitives[0].dimensions[2] = 1.0;

  // Define the pose of the wall on the left. 
  collision_objects[2].primitive_poses.resize(1);
  collision_objects[2].primitive_poses[0].position.x = 0.2;
  collision_objects[2].primitive_poses[0].position.y = -0.9;
  collision_objects[2].primitive_poses[0].position.z = 0.5;
  collision_objects[2].primitive_poses[0].orientation.w = 1.0;


  collision_objects[2].operation = collision_objects[2].ADD;

  planning_scene_interface.applyCollisionObjects(collision_objects);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  ros::init(argc, argv, "own_pick_place_V4");
  ros::NodeHandle nh;
  const char* home = std::getenv("HOME"); if (!home) home = "/tmp";
  g_csv_file_j4 = std::string(home) + "/rl_joint4_logs.csv";
  ROS_WARN_STREAM("J4 CSV -> " << g_csv_file_j4);

  //Get information about robot state
  ros::AsyncSpinner spinner(2);
  spinner.start();

  ros::Publisher start_signal_pub = nh.advertise<std_msgs::Bool>("/start_signal", 1, true);
  ros::Subscriber python_ready_sub = nh.subscribe("/rl/ready_for_next", 1, pythonReadyCallback);
  ros::Publisher pose_state_pub = nh.advertise<std_msgs::Int32>("pose_state", 1000); 
  ros::Publisher episode_pub = nh.advertise<std_msgs::Int32>("/episode", 10);
  g_step_result_pub = nh.advertise<std_msgs::Bool>("/rl/step_result", 10);
  g_joint_trajectory_command_pub = nh.advertise<trajectory_msgs::JointTrajectory>("/panda_arm_controller/command", 1);
  g_reward_pub = nh.advertise<std_msgs::Float32>("/rl/trajectory_reward", 10); 
  ros::Subscriber corrected_action_sub = nh.subscribe("/rl/corrected_start_joints", 1, correctedJointsCallback);
  g_faulty_joints_pub = nh.advertise<std_msgs::Float32MultiArray>("/rl/faulty_joint_states", 10);
    
  ROS_INFO(" Publishing start signal...");
  std_msgs::Bool start_msg;
  start_msg.data = true;
  start_signal_pub.publish(start_msg);

  ros::WallDuration(1.0).sleep();
  

  // use for planning scene
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

  //planning interface
  moveit::planning_interface::MoveGroupInterface group_arm("panda_manipulator");
  moveit::planning_interface::MoveGroupInterface group_hand("panda_hand");
  
  // Set parameters for group like planner, speed, acceleration
  group_arm.setPlannerId("RRTConnect");
  group_arm.setMaxVelocityScalingFactor(0.3);
  group_arm.setMaxAccelerationScalingFactor(0.3);
  group_arm.setNumPlanningAttempts(2);
  group_arm.setGoalJointTolerance(0.01);

  for (int i = 1; i < 2 ;i = i + 1)
  { 
        
    // Add Objects to the envoirement
    addCollisionObjects(planning_scene_interface);

    //Create Cylinder
    shape_msgs::SolidPrimitive primitive;

    moveit_msgs::CollisionObject object_to_attach;
    object_to_attach.id = "cylinder1";

    shape_msgs::SolidPrimitive cylinder_primitive;
    cylinder_primitive.type = primitive.CYLINDER;
    cylinder_primitive.dimensions.resize(2);
    cylinder_primitive.dimensions[primitive.CYLINDER_HEIGHT] = 0.145;
    cylinder_primitive.dimensions[primitive.CYLINDER_RADIUS] = 0.013;
    
    // define the frame/pose for this cylinder
    object_to_attach.header.frame_id = "panda_link0";
    geometry_msgs::Pose grab_pose;
    grab_pose.orientation.w = 1.0;
    grab_pose.position.x = 0.5;
    grab_pose.position.y = -0.2;
    grab_pose.position.z = 0.0725;

    // First, we add the object to the world (without using a vector)
    object_to_attach.primitives.push_back(cylinder_primitive);
    object_to_attach.primitive_poses.push_back(grab_pose);
    object_to_attach.operation = object_to_attach.ADD;
    planning_scene_interface.applyCollisionObject(object_to_attach);

    // Wait a bit for ROS things to initialize
    ros::WallDuration(1.0).sleep();

    std_msgs::Int32 state;
    state.data = 1;
    pose_state_pub.publish(state);
    hoverPose(group_arm);
    ROS_INFO("Task 1: Hover Pose done");
    // { 
    //   geometry_msgs::Pose target_pose_hover;
    //   tf2::Quaternion orientation;
    //   orientation.setRPY(-tau/2, 0, -tau/8);
    //   target_pose_hover.orientation = tf2::toMsg(orientation);
    //   target_pose_hover.position.x = 0.502;
    //   target_pose_hover.position.y = -0.2;
    //   target_pose_hover.position.z = 1.5;
      
    //   performRLStep(group_arm, target_pose_hover);
    //   ROS_INFO("Task 1: Hover Pose done.");
    // }
    
    state.data = 2;
    pose_state_pub.publish(state);
    pickPose(group_arm , "down");
    ROS_INFO("Task 2: Pick Pose done");

    // geometry_msgs::Pose state2_pose = group_arm.getCurrentPose().pose;
    // ROS_INFO("Saved return position after Task 2.");

    // state.data = 2;
    // pose_state_pub.publish(state);
    // {
    //   geometry_msgs::Pose original_target = group_arm.getCurrentPose().pose;
    //   original_target.position.z -= 0.26; 
    //   performRLStep(group_arm, original_target);
    //   ROS_INFO("Task 2: Pick Pose done");
    // }
    
    // state.data = 3;
    // pose_state_pub.publish(state);
    // pick(group_arm);
    // ROS_INFO("Task 3: Pick done");
    
    //ros::WallDuration(2.0).sleep();
    // state.data = 4;
    // pose_state_pub.publish(state);
    // pickPose(group_arm , "up");

    // state.data = 5;
    // pose_state_pub.publish(state);
    // hoverPlacePose(group_arm);
    
    // for (int j = 0; j < 6; j++)
    // {
      state.data = 4;
      pose_state_pub.publish(state);
      {
        geometry_msgs::Pose original_target = group_arm.getCurrentPose().pose;
        original_target.position.z += 0.26; 
        performRLStep(group_arm, original_target);
        ROS_INFO("Task 4: Lift up done");
        }

      state.data = 5;
      pose_state_pub.publish(state);
      {
        geometry_msgs::Pose original_target;
        tf2::Quaternion orientation;
        orientation.setRPY(-tau/2, 0, -tau/8);
        original_target.orientation = tf2::toMsg(orientation);
        original_target.position.x = 0.502; 
        original_target.position.y = 0.2;
        original_target.position.z = 1.5;
        performRLStep(group_arm, original_target);
        ROS_INFO("Task 5: Hover Place Pose done");
     }

    //   ROS_INFO("Returning to state 2 position...");
    //   group_arm.setPoseTarget(state2_pose);
    //   group_arm.move();
    //   ROS_INFO("Returned to state 2 position successfully.");
      
    // }

    // state.data = 6;
    // pose_state_pub.publish(state);
    // PlacePose(group_arm , "down");
    // ROS_INFO("Task 6: Place Pose (down) done.");

  //   {
  //     geometry_msgs::Pose original_target = group_arm.getCurrentPose().pose;
  //     original_target.position.z -= 0.26;
  //     performRLStep(group_arm, original_target);
  //     ROS_INFO("Task 6: Place Pose (down) done.");
  // }

    // state.data = 7;
    // pose_state_pub.publish(state);
    // openHand(group_hand);
    // group_arm.detachObject(object_to_attach.id);
    // ROS_INFO("Task 7: Open Hand done");
    
    // state.data = 8;
    // pose_state_pub.publish(state);
    // PlacePose(group_arm , "up");
    // ROS_INFO("Task 8: Place Pose (up) done.");
    

  //   {
  //     geometry_msgs::Pose original_target = group_arm.getCurrentPose().pose;
  //     original_target.position.z += 0.26;
  //     performRLStep(group_arm, original_target);
  //     ROS_INFO("Task 8: Place Pose (up) done.");
  // }
    
    state.data = 9;
    pose_state_pub.publish(state);
    initPose(group_arm);
    ROS_INFO("Task 9: Init Pose done.");
   
    // {
    //     geometry_msgs::Pose target_pose_init;
    //     tf2::Quaternion orientation;
    //     orientation.setRPY(-tau/2, 0, -tau/8);
    //     target_pose_init.orientation = tf2::toMsg(orientation);
    //     target_pose_init.position.x = 0.5;
    //     target_pose_init.position.y = 0.0;
    //     target_pose_init.position.z = 1.5;

    //     performRLStep(group_arm, target_pose_init);
    //     ROS_INFO("Task 9: Init Pose done.");
    // }
    
    
    state.data = 404;
    pose_state_pub.publish(state);
    // ros::WallDuration(2.0).sleep();
    ROS_WARN("round end");

    std_msgs::Int32 episodeMsg;
    episodeMsg.data = i;
    episode_pub.publish(episodeMsg);
    ROS_INFO("Episode %d published", i);
  
  }
  ros::shutdown();
  return 0;
}