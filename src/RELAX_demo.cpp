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

bool g_python_is_ready = false;
bool g_is_recording = false; 
ros::Publisher g_faulty_start_j4_pub;      // 发布有问题的 joint4 值给RL
double g_corrected_start_j4 = 0.0;         // 存储RL返回的修正值
bool g_corrected_j4_received = false;      // 标志位，表示已接收到修正值
ros::Publisher g_step_result_pub;          // 发布执行成功/失败结果给RL
ros::Publisher g_pose_state_pub;           // 发布机器人的当前位姿状态
double g_current_panda_joint4 = 0.0;
std::vector<sensor_msgs::JointState> g_recorded_joint_states;
ros::Publisher g_joint_trajectory_command_pub;
ros::Publisher g_reward_pub;

std::vector<Eigen::Vector3d> g_obstacles = {
    Eigen::Vector3d(0.75, -0.5, 1.48),
    Eigen::Vector3d(0.75, 0.5, 1.48),
    // Eigen::Vector3d(0.75, 0.0, 1.64)
};
const double COLLISION_THRESHOLD = 0.20;

void correctedStartJointCallback(const std_msgs::Float32::ConstPtr& msg)
{
    g_corrected_start_j4 = msg->data;
    g_corrected_j4_received = true;
    ROS_INFO("Received corrected start J4 value from RL: %f", g_corrected_start_j4);
}

// // createStitchedTrajectory 函数，它现在使用真实的状态作为起点
// moveit_msgs::RobotTrajectory createStitchedTrajectory(
//     const moveit::planning_interface::MoveGroupInterface::Plan& original_plan,
//     moveit::planning_interface::MoveGroupInterface& move_group,
//     const std::vector<double>& start_joints)
// {
//     ROS_INFO("[Stitcher] Creating a new stitched trajectory...");

//     moveit::core::RobotState start_state(move_group.getRobotModel());
//     start_state.setJointGroupPositions(move_group.getName(), start_joints);
    
//     robot_trajectory::RobotTrajectory stitched_trajectory(start_state.getRobotModel(), move_group.getName());
//     stitched_trajectory.addSuffixWayPoint(start_state, 0.0);
    
//     double original_duration = 0.0;
//     for (const auto& original_point : original_plan.trajectory_.joint_trajectory.points)
//     {
//         if (original_point.positions.empty()) continue;
//         moveit::core::RobotState temp_state(start_state.getRobotModel());
//         temp_state.setJointGroupPositions(move_group.getName(), original_point.positions);
//         original_duration = original_point.time_from_start.toSec();
//         stitched_trajectory.addSuffixWayPoint(temp_state, original_duration);
//     }
    
//     trajectory_processing::IterativeParabolicTimeParameterization iptp;
//     bool success = iptp.computeTimeStamps(stitched_trajectory, 1.0, 1.0);
//     if (!success)
//     {
//         ROS_ERROR("[Stitcher] Failed to re-time the stitched trajectory!");
//         return moveit_msgs::RobotTrajectory();
//     }

//     moveit_msgs::RobotTrajectory stitched_trajectory_msg;
//     stitched_trajectory.getRobotTrajectoryMsg(stitched_trajectory_msg);

//     ROS_INFO("[Stitcher] Stitched trajectory created with %zu points.", stitched_trajectory_msg.joint_trajectory.points.size());
//     return stitched_trajectory_msg;
// }

void faultyJointStatesCallback(const sensor_msgs::JointState::ConstPtr& msg)
{
    // 更新当前的joint4值
    for (size_t i = 0; i < msg->name.size(); ++i) {
        if (msg->name[i] == "panda_joint4") {
            g_current_panda_joint4 = msg->position[i];
            break; 
        }
    }
    
    // 把当前收到的这帧数据存起来
    if (g_is_recording) {
        g_recorded_joint_states.push_back(*msg);
    }
}

void pythonReadyCallback(const std_msgs::BoolConstPtr& msg)
{
    if (msg->data)
    {
        g_python_is_ready = true;
    }
}

float computeTrajectoryReward(const moveit_msgs::RobotTrajectory& trajectory, const moveit::core::RobotModelConstPtr& robot_model)
{
    float total_reward = 0.0;
    const double MIN_DIST_REWARD_WEIGHT = 50.0;
    const double CLOSE_DIST_PENALTY = -100.0;
    const double DIST_THRESHOLD = 0.2;
    
    if (trajectory.joint_trajectory.points.empty()) {
        ROS_WARN("Planned trajectory is empty, reward is 0.");
        return 0.0;
    }
    
    moveit::core::RobotState robot_state(robot_model);
    
    for (const auto& point : trajectory.joint_trajectory.points) {
        if (point.positions.empty()) continue;
        
        robot_state.setJointGroupPositions("panda_manipulator", point.positions);
        
        const Eigen::Isometry3d& ee_pose = robot_state.getGlobalLinkTransform("panda_hand_tcp");
        Eigen::Vector3d ee_position = ee_pose.translation();
        
        double min_dist_to_obstacle = std::numeric_limits<double>::max();
        for (const auto& obs_pos : g_obstacles) {
            double dist = (ee_position - obs_pos).norm();
            if (dist < min_dist_to_obstacle) {
                min_dist_to_obstacle = dist;
            }
        }
        
        double dist_reward = 0.0;
        if (min_dist_to_obstacle > DIST_THRESHOLD) {
            dist_reward = MIN_DIST_REWARD_WEIGHT * (min_dist_to_obstacle - DIST_THRESHOLD);
        } else {
            dist_reward = CLOSE_DIST_PENALTY;
        }
        
        total_reward += dist_reward;
    }
    
    ROS_INFO("Computed total reward from planned trajectory: %f", total_reward);
    return total_reward;
}

bool createManualPlan(const std::vector<double>& start_joints,
                      const std::vector<double>& end_joints,
                      moveit::planning_interface::MoveGroupInterface::Plan& plan,
                      moveit::planning_interface::MoveGroupInterface& move_group,
                      double duration_sec = 2.0)
{
    ROS_INFO("[ManualPlan] Creating plan from start to end...");
    moveit::core::RobotState start_state(move_group.getRobotModel());
    start_state.setJointGroupPositions(move_group.getName(), start_joints);
    
    robot_trajectory::RobotTrajectory robot_traj(start_state.getRobotModel(), move_group.getName());
    robot_traj.addSuffixWayPoint(start_state, 0.0);
    
    moveit::core::RobotState end_state(move_group.getRobotModel());
    end_state.setJointGroupPositions(move_group.getName(), end_joints);
    robot_traj.addSuffixWayPoint(end_state, duration_sec);
    
    trajectory_processing::IterativeParabolicTimeParameterization iptp;
    if (!iptp.computeTimeStamps(robot_traj, 1.0, 1.0)) {
        ROS_ERROR("[ManualPlan] Failed to re-time the trajectory!");
        return false;
    }
    
    robot_traj.getRobotTrajectoryMsg(plan.trajectory_);
    return true;
}

bool performRLStep(moveit::planning_interface::MoveGroupInterface& move_group, const geometry_msgs::Pose& final_target_pose)
{
    g_corrected_j4_received = false;
    
    // 步骤1: 规划一次以获取 faulty_start_j4
    move_group.setStartStateToCurrentState();
    moveit::planning_interface::MoveGroupInterface::Plan initial_plan;
    move_group.setPoseTarget(final_target_pose);
    if (move_group.plan(initial_plan) != moveit::core::MoveItErrorCode::SUCCESS)
    {
        ROS_ERROR("[RLStep] Initial planning to trigger RL failed!");
        std_msgs::Bool result_msg; result_msg.data = false; g_step_result_pub.publish(result_msg);
        std_msgs::Float32 reward_msg; reward_msg.data = 0; g_reward_pub.publish(reward_msg);
        return false;
    }
    if (initial_plan.trajectory_.joint_trajectory.points.empty()) {
        ROS_ERROR("[RLStep] Initial plan is empty, cannot extract starting joint4.");
        std_msgs::Bool result_msg; result_msg.data = false; g_step_result_pub.publish(result_msg);
        std_msgs::Float32 reward_msg; reward_msg.data = 0; g_reward_pub.publish(reward_msg);
        return false;
    }
    double faulty_start_j4 = initial_plan.trajectory_.joint_trajectory.points.front().positions[3];
    std::vector<double> final_joints = initial_plan.trajectory_.joint_trajectory.points.back().positions;
    ROS_INFO("[RLStep] Initial planned faulty joint4: %f", faulty_start_j4);
    

    // 步骤2: 与Python交互获取修正值
    std_msgs::Float32 msg_to_rl; msg_to_rl.data = faulty_start_j4;
    ros::Rate r(10);
    ros::Time start_time = ros::Time::now();
    while(ros::ok() && !g_corrected_j4_received)
    {
        g_faulty_start_j4_pub.publish(msg_to_rl);
        ros::spinOnce();
        r.sleep();
        if ((ros::Time::now() - start_time).toSec() > 5.0) {
            ROS_ERROR("[RLStep] Timeout waiting for RL correction!");
            std_msgs::Bool result_msg; result_msg.data = false; g_step_result_pub.publish(result_msg);
            std_msgs::Float32 reward_msg; reward_msg.data = 0; g_reward_pub.publish(reward_msg);
            return false;
        }
    }

    // 步骤3: 执行修正动作
    std::vector<double> current_joints = move_group.getCurrentJointValues();
    std::vector<double> corrected_joints = current_joints;
    int joint4_index = -1;
    const auto& joint_names = move_group.getJointNames();
    for(size_t i = 0; i < joint_names.size(); ++i) {
        if(joint_names[i] == "panda_joint4") {
            joint4_index = i;
            break;
        }
    }
    if (joint4_index != -1) {
        corrected_joints[joint4_index] = g_corrected_start_j4;
    } else {
        ROS_ERROR("[RLStep] Could not find panda_joint4 index!");
        std_msgs::Bool result_msg; result_msg.data = false; g_step_result_pub.publish(result_msg);
        std_msgs::Float32 reward_msg; reward_msg.data = -0; g_reward_pub.publish(reward_msg);
        return false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan correction_plan;
    if (!createManualPlan(current_joints, corrected_joints, correction_plan, move_group, 1.0)) {
        ROS_ERROR("[RLStep] Failed to create manual plan for correction move!");
        std_msgs::Bool result_msg; result_msg.data = false; g_step_result_pub.publish(result_msg);
        std_msgs::Float32 reward_msg; reward_msg.data = 0; g_reward_pub.publish(reward_msg);
        return false;
    }
    if (move_group.execute(correction_plan) != moveit::core::MoveItErrorCode::SUCCESS) {
        ROS_ERROR("[RLStep] Failed to execute the correction move!");
        std_msgs::Bool result_msg; result_msg.data = false; g_step_result_pub.publish(result_msg);
        std_msgs::Float32 reward_msg; reward_msg.data = -500.0; g_reward_pub.publish(reward_msg);
        return false;
    }
    ROS_INFO("[RLStep] Correction move completed. Ready for final motion.");
    
    // 步骤4: 主轨迹动作 - 手动插补并执行到 final_target_pose
    moveit::planning_interface::MoveGroupInterface::Plan final_plan;
    if (!createManualPlan(corrected_joints, final_joints, final_plan, move_group, 2.0)) {
        ROS_ERROR("[RLStep] Failed to create manual plan for final move!");
        std_msgs::Bool result_msg; result_msg.data = false; g_step_result_pub.publish(result_msg);
        std_msgs::Float32 reward_msg; reward_msg.data = 0; g_reward_pub.publish(reward_msg);
        return false;
    }
    
    // 步骤 5: 计算并发送奖励
    float trajectory_reward = computeTrajectoryReward(final_plan.trajectory_, move_group.getRobotModel());
    std_msgs::Float32 reward_msg;
    reward_msg.data = trajectory_reward;
    g_reward_pub.publish(reward_msg);
    
    // 步骤 6: 执行最终规划
     ROS_INFO("--- Final Trajectory Details ---");
    if (final_plan.trajectory_.joint_trajectory.points.size() > 1) {
        ROS_INFO("Final Traj Start J4: %f", final_plan.trajectory_.joint_trajectory.points.front().positions[joint4_index]);
        ROS_INFO("Final Traj End J4:   %f", final_plan.trajectory_.joint_trajectory.points.back().positions[joint4_index]);
    }
    ROS_INFO("--------------------------------");
    
    moveit::core::MoveItErrorCode result = move_group.execute(final_plan);
    
    std_msgs::Bool result_msg;
    result_msg.data = (result == moveit::core::MoveItErrorCode::SUCCESS);
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

  //Get information about robot state
  ros::AsyncSpinner spinner(2);
  spinner.start();

  ros::Publisher start_signal_pub = nh.advertise<std_msgs::Bool>("/start_signal", 1, true);
  ros::Subscriber python_ready_sub = nh.subscribe("/rl/ready_for_next", 1, pythonReadyCallback);
  g_faulty_start_j4_pub = nh.advertise<std_msgs::Float32>("/rl/faulty_start_joint4", 10);
  ros::Subscriber corrected_j4_sub = nh.subscribe("/rl/corrected_start_joint4", 1, correctedStartJointCallback);
  // ros::Publisher goal_pub = nh.advertise<std_msgs::Bool>("goal_state", 1000);  
  ros::Publisher pose_state_pub = nh.advertise<std_msgs::Int32>("pose_state", 1000); 
  ros::Publisher episode_pub = nh.advertise<std_msgs::Int32>("/episode", 10);
  g_step_result_pub = nh.advertise<std_msgs::Bool>("/rl/step_result", 1, true);
  g_joint_trajectory_command_pub = nh.advertise<trajectory_msgs::JointTrajectory>("/panda_arm_controller/command", 1);
  ros::Subscriber faulty_joint_states_sub = nh.subscribe<sensor_msgs::JointState>("/faulty_joint_states", 10, faultyJointStatesCallback);
  g_reward_pub = nh.advertise<std_msgs::Float32>("/rl/trajectory_reward", 1, true); // 新增的奖励发布者

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

  for (int i = 1; i < 1500 ;i = i + 1)
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
    
    state.data = 4;
    pose_state_pub.publish(state);
   {
    geometry_msgs::Pose original_target = group_arm.getCurrentPose().pose;
    original_target.position.z += 0.26; 
    performRLStep(group_arm, original_target);
    ROS_INFO("Task 4: Lift up done");
    }
    //ros::WallDuration(2.0).sleep();

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