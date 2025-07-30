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

double RLAction[3] = {0.0, 0.0, 0.0};
int FaultFlag = 0;
bool   GotFirstAction = false; 

ros::Publisher g_rl_request_pub;      
geometry_msgs::Pose g_resolved_rl_pose; 
bool g_rl_pose_received = false;       
bool g_python_is_ready = false;

void pythonReadyCallback(const std_msgs::BoolConstPtr& msg)
{
    if (msg->data)
    {
        g_python_is_ready = true;
    }
}

void rlActionCallback(const std_msgs::Float32MultiArrayConstPtr& msg)
{
  if (!GotFirstAction && msg->data.size() >= 3)
  {
    RLAction[0] = msg->data[0];
    RLAction[1] = msg->data[1];
    RLAction[2] = msg->data[2];
    GotFirstAction = true;
  }
}
//Functions for Moving and grasping with robot
///////////////////////////////////////////////////////////////////////////////////////////////////////////
//Position, Orientation, Planning, Exectuion initPose

void resolvedPoseCallback(const geometry_msgs::PoseConstPtr& msg)
{
  g_resolved_rl_pose = *msg;
  g_rl_pose_received = true;
}

void faultFlagCallback(const std_msgs::Int32::ConstPtr& msg)
{
    FaultFlag = msg->data;     
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

void performRLStep(moveit::planning_interface::MoveGroupInterface& move_group, const geometry_msgs::Pose& target_pose)
{
  g_rl_pose_received = false;

  g_rl_request_pub.publish(target_pose);
  // ROS_INFO(" Published RL action request with target pose. Waiting for response...");

  ros::Rate r(100);
  double timeout_sec = 10.0; 
  ros::Time start_time = ros::Time::now();

  while (ros::ok() && !g_rl_pose_received)
  {
    if ((ros::Time::now() - start_time).toSec() > timeout_sec)
    {
      ROS_ERROR("Timeout waiting for RL resolved pose!");
      return; 
    }
    r.sleep();
  }

  if (g_rl_pose_received)
  {
    // ROS_INFO_STREAM("Received RL-resolved pose: " << g_resolved_rl_pose);
    move_group.setPoseTarget(g_resolved_rl_pose);
    move_group.move();
    // ROS_INFO("Move execution with RL-resolved pose complete.");
  }
  else
  {
    ROS_ERROR("Failed to receive RL pose in time or ROS is shutting down.");
  }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  ros::init(argc, argv, "own_pick_place_V4");
  ros::NodeHandle nh;

  //Get information about robot state
  ros::AsyncSpinner spinner(2);
  spinner.start();

  ros::Subscriber faultFlagSub =nh.subscribe("/fault_flag", 10, faultFlagCallback);
  ros::Publisher start_signal_pub = nh.advertise<std_msgs::Bool>("/start_signal", 1, true);
  ros::Subscriber rlResolvedSub = nh.subscribe("/rl/action_resolved", 1, resolvedPoseCallback);
  ros::Publisher request_pub = nh.advertise<geometry_msgs::Pose>("/rl/action_request", 1);
  g_rl_request_pub = request_pub;
  ros::Subscriber python_ready_sub = nh.subscribe("/rl/ready_for_next", 1, pythonReadyCallback);


  ROS_INFO(" Publishing start signal...");
  std_msgs::Bool start_msg;
  start_msg.data = true;
  start_signal_pub.publish(start_msg);

    
  // ros::Publisher goal_pub = nh.advertise<std_msgs::Bool>("goal_state", 1000);  
  ros::Publisher pose_state_pub = nh.advertise<std_msgs::Int32>("pose_state", 1000); 
  ros::Publisher episode_pub = nh.advertise<std_msgs::Int32>("/episode", 10);

  ros::WallDuration(1.0).sleep();
  

  // use for planning scene
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

  //planning interface
  moveit::planning_interface::MoveGroupInterface group_arm("panda_arm");
  moveit::planning_interface::MoveGroupInterface group_hand("panda_hand");
  
  // Set parameters for group like planner, speed, acceleration
  group_arm.setPlannerId("RRTConnect");
  group_arm.setMaxVelocityScalingFactor(1);
  group_arm.setMaxAccelerationScalingFactor(1);
  //group_arm.setNumPlanningAttempts(2);

  for (int i = 1; i < 1000 ;i = i + 1)
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

    // ROS_INFO("Waiting for Python node to be ready for the next step...");
    g_python_is_ready = false; 
    ros::Rate r(10); // 10 Hz
    while(ros::ok() && !g_python_is_ready)
    {
        // ros::spinOnce(); // AsyncSpinner正在处理回调，这里不需要
        r.sleep();
    }
    // ROS_INFO("Python is ready. Proceeding to Task 5.");

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