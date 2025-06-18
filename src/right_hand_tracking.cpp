#include <moveit/move_group_interface/move_group_interface.h>

#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/CollisionObject.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
// #include "opencv_services/box_and_target_position.h"
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <moveit_msgs/CollisionObject.h>

// TF2
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <cmath> 

#include <sstream>
#include <vector>
#include <Eigen/Dense>
#include <algorithm>

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <numeric> 

#include <limits>
#include <random>
#include <matplotlibcpp.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include <chrono>
#include <locale>
#include "std_msgs/Bool.h"
#include "std_msgs/Int32.h"


namespace plt = matplotlibcpp;

// The circle constant tau = 2*pi. One tau is one rotation in radians.
const double tau = 2 * M_PI;
std::vector<Eigen::Vector3d> human_joint_positions;
std::string expFolderPath;
geometry_msgs::Point target_position;

void PersonCallback(const sensor_msgs::JointState::ConstPtr& msg) {
    if (msg->position.size() == 25 && msg->velocity.size() == 25 && msg->effort.size() == 25) {
        // 清除之前的关节位置数据
        human_joint_positions.clear();

        for (size_t i = 0; i < 25; ++i) {
            // 检查是否为 NaN
            // if (std::isnan(msg->position[i]) || std::isnan(msg->velocity[i]) || std::isnan(msg->effort[i])) {
            //     continue; // 如果任一维度是 NaN，则跳过这个点
            // }

            Eigen::Vector3d joint_position(msg->position[i], msg->velocity[i], msg->effort[i]);
            // 将新的关节位置加入到向量中
            human_joint_positions.push_back(joint_position);
            target_position.x = std::round(msg->position[0] * 100.0) / 100.0 - 0.2;
            target_position.y = std::round(msg->velocity[0] * 100.0) / 100.0;
            target_position.z = std::round(msg->effort[0] * 100.0) / 100.0 + 1.015;
            ROS_INFO("Target Position - x: [%f], y: [%f], z: [%f]", target_position.x, target_position.y, target_position.z);
        }
    ros::Rate rate(5);  // 1 Hz = 1 message per second
    rate.sleep();

    } else {
        ROS_WARN("JointState message does not contain 25 points for each field.");
    }
}  
    
///////////////////////////////////////////////////////////////////////////////////////////////////////////
//Position, Orientation, Planning, Exectuion initPose

void initPose(moveit::planning_interface::MoveGroupInterface& move_group)
{ 
  moveit::core::RobotStatePtr current_state = move_group.getCurrentState();
  //
  // Next get the current set of joint values for the group.
  std::vector<double> joint_group_positions;
  // Raw pointers are frequently used to refer to the planning group for improved performance.
  const moveit::core::JointModelGroup* joint_model_group =
  move_group.getCurrentState()->getJointModelGroup("panda_manipulator");
  current_state->copyJointGroupPositions(joint_model_group, joint_group_positions);
  joint_group_positions[0] = 0;
  joint_group_positions[1] = -tau / 8;  // -1/8 turn in radians
  joint_group_positions[2] = 0;
  joint_group_positions[3] = -3 * tau / 8;  // -8/8 turn in radians
  joint_group_positions[4] = 0;
  joint_group_positions[5] = tau / 4 + 0.03;  // 1/4 turn in radians
  joint_group_positions[6] = tau / 8;  // 1/8 turn in radians
  move_group.setJointValueTarget(joint_group_positions);

  move_group.setMaxVelocityScalingFactor(0.2); // default 0.05
  move_group.setMaxAccelerationScalingFactor(0.2); // default 0.05
  //
  move_group.move();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
//Position, Orientation, Planning, Exectuion hoverPose

void hoverPose(moveit::planning_interface::MoveGroupInterface& move_group_interface)
{ 
  moveit::planning_interface::MoveGroupInterface::Plan cartesianPlan;
  move_group_interface.setStartStateToCurrentState();

  move_group_interface.setMaxVelocityScalingFactor(0.2);
  move_group_interface.setMaxAccelerationScalingFactor(0.2);
  geometry_msgs::Pose target_pose_hover = move_group_interface.getCurrentPose().pose;
  std::vector<geometry_msgs::Pose> waypoints;

  //Convert Orienation from RPY to Quaternion
  tf2::Quaternion orientation;
  orientation.setRPY(-tau/2, 0, 0);

  target_pose_hover.orientation = tf2::toMsg(orientation);
  
  target_pose_hover.position.x = target_position.x;
  target_pose_hover.position.y = target_position.y;
  target_pose_hover.position.z = target_position.z;

  ROS_INFO("Conbtrol Target Position - x: [%f], y: [%f], z: [%f]", target_pose_hover.position.x, target_pose_hover.position.y, target_pose_hover.position.z);

  waypoints.push_back(target_pose_hover);

  moveit_msgs::RobotTrajectory trajectory_msg;
  move_group_interface.setPlanningTime(10.0);
  double fraction = move_group_interface.computeCartesianPath(waypoints,
                                               0.01,  // eef_step
                                               0.0,   // jump_threshold
                                               trajectory_msg, false);
  robot_trajectory::RobotTrajectory rt(move_group_interface.getCurrentState()->getRobotModel(), "panda_manipulator");

  rt.setRobotTrajectoryMsg(*move_group_interface.getCurrentState(), trajectory_msg);
  trajectory_processing::IterativeParabolicTimeParameterization iptp;
  iptp.computeTimeStamps(rt, 0.1, 0.1);
  rt.getRobotTrajectoryMsg(trajectory_msg);
  cartesianPlan.trajectory_ = trajectory_msg;
  
  // move_group.setPoseTarget(target_pose_hover);
  move_group_interface.execute(cartesianPlan);  

  // Plan and move the robot to the target pose
  // moveit::planning_interface::MoveGroupInterface::Plan my_plan;
  // bool success = (move_group_interface.plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

  // if (success) {
      
  // } else {
  //     ROS_WARN("Planning to target position failed.");
  // }
}



void addCollisionObjects(moveit::planning_interface::PlanningSceneInterface& planning_scene_interface, float x, float y)
{

  // Create vector to hold 3 collision objects.
  std::vector<moveit_msgs::CollisionObject> collision_objects;
  collision_objects.resize(4);

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
  collision_objects[2].id = "wallright";
  collision_objects[2].header.frame_id = "panda_link0";

  // Define the primitive and its dimensions. 
  collision_objects[2].primitives.resize(1);
  collision_objects[2].primitives[0].type = collision_objects[1].primitives[0].BOX;
  collision_objects[2].primitives[0].dimensions.resize(3);
  collision_objects[2].primitives[0].dimensions[0] = 1.2;
  collision_objects[2].primitives[0].dimensions[1] = 0;
  collision_objects[2].primitives[0].dimensions[2] = 1.0;

  // Define the pose of the wall on the right. 
  collision_objects[2].primitive_poses.resize(1);
  collision_objects[2].primitive_poses[0].position.x = 0.2;
  collision_objects[2].primitive_poses[0].position.y = 0.9;
  collision_objects[2].primitive_poses[0].position.z = 0.5;
  collision_objects[2].primitive_poses[0].orientation.w = 1.0;


  collision_objects[2].operation = collision_objects[2].ADD;

 // Add the second wall on the left hand side
  collision_objects[3].id = "plate";
  collision_objects[3].header.frame_id = "panda_link0";

  // Define the primitive and its dimensions. 
  collision_objects[3].primitives.resize(1);
  collision_objects[3].primitives[0].type = collision_objects[1].primitives[0].CYLINDER;
  collision_objects[3].primitives[0].dimensions.resize(2);
  collision_objects[3].primitives[0].dimensions[0] = 0.03;
  collision_objects[3].primitives[0].dimensions[1] = 0.025;


  // Define the pose of the wall on the right. 
  collision_objects[3].primitive_poses.resize(1);
  collision_objects[3].primitive_poses[0].position.x = x;
  collision_objects[3].primitive_poses[0].position.y = y;
  collision_objects[3].primitive_poses[0].position.z = 0.01;
  collision_objects[3].primitive_poses[0].orientation.w = 1.0;


  collision_objects[3].operation = collision_objects[3].ADD;

  planning_scene_interface.applyCollisionObjects(collision_objects);
}



/////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{

  ros::init(argc, argv, "right_hand_tracking");
  ros::NodeHandle nh;

  //Get information about robot state
  ros::AsyncSpinner spinner(1);
  spinner.start();
  
  ros::Subscriber sub_per = nh.subscribe("robot_value", 1, PersonCallback);

  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

  //planning interface
  moveit::planning_interface::MoveGroupInterface group_manipulator("panda_manipulator");
  
  // Set parameters for group like planner, speed, acceleration
  group_manipulator.setPlannerId("RRTConnect");
  group_manipulator.setMaxVelocityScalingFactor(0.1);
  group_manipulator.setMaxAccelerationScalingFactor(0.1);

  ros::Rate loop_rate(1); // Control loop at 10 Hz
  while (ros::ok()) {
    // Move the robot to the latest target position
    hoverPose(group_manipulator);

    loop_rate.sleep();
  }

  ros::shutdown();
  return 0;
}