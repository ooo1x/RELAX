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


namespace plt = matplotlibcpp;

// The circle constant tau = 2*pi. One tau is one rotation in radians.
const double tau = 2 * M_PI;
std::vector<Eigen::Vector3d> human_joint_positions;
std::string expFolderPath;


//////////////////////////////////////////////////////////////////////////////////////////////////////

void createNextExpFolder(const std::string& basePath) {
    int expNumber = 1;

    while (true) {
        std::string potentialExpFolderPath = basePath + "exp" + std::to_string(expNumber) + "/";
        if (mkdir(potentialExpFolderPath.c_str(), 0777) == 0) { // 如果成功创建文件夹
            expFolderPath = potentialExpFolderPath;
            break;
        }
        expNumber++;
    }
}

// Generate DH parameters given joint variables
std::vector<std::vector<double>> dh_params(const std::vector<double>& joint_variables) 
{
    // Assuming joint_variables size is at least 7
    std::vector<std::vector<double>> dh = {
        {0,      0,      0.333,   joint_variables[0]},
        {-M_PI/2, 0,      0,       joint_variables[1]},
        {M_PI/2,  0,      0.316,   joint_variables[2]},
        {M_PI/2,  0.0825, 0,       joint_variables[3]},
        {-M_PI/2,-0.0825, 0.384,   joint_variables[4]},
        {M_PI/2,  0,      0,       joint_variables[5]},
        {M_PI/2,  0.088,  0.107+0.1034,   joint_variables[6]}
    };
    return dh;
}


std::vector<std::vector<double>> readCSV(const std::string& filename) 
{
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);
    std::string line;
    
    std::cout << "readCSV filename =" << filename << std::endl;

    std::getline(file, line);

    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            try {
                double number = std::stod(value);
                row.push_back(number);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid argument: '" << value << "' could not be converted to double." << std::endl;
                
                row.push_back(0.0);  
            }
        }
        data.push_back(row);
    }
    return data;

}

Eigen::Matrix4d TF_matrix(int i, const std::vector<std::vector<double>>& dh) 
{
 
    double alpha = dh[i][0];
    double a = dh[i][1];
    double d = dh[i][2];
    double q = dh[i][3];

    Eigen::Matrix4d TF;

    TF << cos(q), -sin(q), 0, a,
          sin(q) * cos(alpha), cos(q) * cos(alpha), -sin(alpha), -sin(alpha) * d,
          sin(q) * sin(alpha), cos(q) * sin(alpha), cos(alpha),  cos(alpha) * d,
          0, 0, 0, 1;

    // std::cout << "TF:" << std::endl << TF << std::endl;

    return TF;
}

std::vector<Eigen::Vector3d> calculateEEPosition(const std::string& csv_filename, std::vector<Eigen::Vector3d>& plan_trajectory_point) 
{
    auto data = readCSV(csv_filename);
    std::cout << "calculateEEPosition csv_filename =" << csv_filename << std::endl;

    std::cout << "CSV DATA" << std::endl;

    for (const auto& row:data) {
        for (const auto& value:row) {
          std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    for (const auto& row : data) {
        auto dh_parameters = dh_params(row);

        Eigen::Matrix4d T_01 = TF_matrix(0, dh_parameters);
        Eigen::Matrix4d T_12 = TF_matrix(1, dh_parameters);
        Eigen::Matrix4d T_23 = TF_matrix(2, dh_parameters);
        Eigen::Matrix4d T_34 = TF_matrix(3, dh_parameters);
        Eigen::Matrix4d T_45 = TF_matrix(4, dh_parameters);
        Eigen::Matrix4d T_56 = TF_matrix(5, dh_parameters);
        Eigen::Matrix4d T_67 = TF_matrix(6, dh_parameters);

        Eigen::Matrix4d T_07 = T_01 * T_12 * T_23 * T_34 * T_45 * T_56 * T_67;
        Eigen::Vector3d translations = T_07.block<3, 1>(0, 3);
        std::cout << "Location: X=" << translations[0] << " Y=" << translations[1] << " Z=" << translations[2] << std::endl;
        plan_trajectory_point.push_back(translations);
    }    
    return plan_trajectory_point;
}

double calculateDistance(const Eigen::Vector3d& point1, const Eigen::Vector3d& point2) {

    if (std::isnan(point1.x()) || std::isnan(point1.y()) || std::isnan(point1.z()) ||
        std::isnan(point2.x()) || std::isnan(point2.y()) || std::isnan(point2.z())) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return (point1 - point2).norm();
}

std::vector<int> range(int start, int stop) {
    std::vector<int> result;
    for (int i = start; i < stop; ++i) {
        result.push_back(i);
    }
    return result;
}


void plotMinDistances(const std::vector<double>& min_distances, const std::string& title, const std::string& expFolderPath) {
    plt::figure_size(1200, 800);

    std::vector<std::string> labels = {
        "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow",
        "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
        "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel",
        "RBigToe", "RSmallToe", "RHeel"
    };

    std::vector<double> distances_for_plot(min_distances.size());
    std::transform(min_distances.begin(), min_distances.end(), distances_for_plot.begin(), [](double d) {
        return std::isinf(d) ? std::numeric_limits<double>::quiet_NaN() : d; // Convert infinite values to NaN for skipping
    });

    std::vector<double> x(min_distances.size());
    std::iota(x.begin(), x.end(), 0); // Create x values for the plot

    // Plot each bar, setting the color conditionally
    for (size_t i = 0; i < distances_for_plot.size(); ++i) {
        if (std::isnan(distances_for_plot[i])) continue; // Skip NaN values

        std::string color = (distances_for_plot[i] < 0.2) ? "#E60000" : "#1E90FF";
        plt::bar(std::vector<double>{x[i]}, std::vector<double>{distances_for_plot[i]}, "black", "-", 1.0, 
                 {{"color", color}});
    }

    plt::xticks(x, labels, {{"rotation", "vertical"}}); // Rotate labels for better visibility
    plt::subplots_adjust({{"bottom", 0.2}}); // You might need to tweak this value
    plt::xlabel("Human Keypoints");
    plt::ylabel("Minimum Distance");
    plt::title("Minimum Distances between Robot and Human");

    double y_line = 0.2;
    // Draw the horizontal line to show threshold
    plt::plot({0, static_cast<double>(x.size())}, {y_line, y_line}, {{"color", "#66FF00"}, {"linestyle", "--"}});

    plt::save(expFolderPath + title + ".png");
}

// Function to calculate the minimum distances from robot trajectory to human joint positions
std::vector<double> calculateMinDistances(const std::string& filename, const std::vector<Eigen::Vector3d>& human_joint_positions,const std::string& title, std::vector<Eigen::Vector3d>& plan_trajectory_point){
    
    std::cout << "calculateMinDistances filename =" << filename << std::endl;
    std::vector<Eigen::Vector3d> plan_trajectory = calculateEEPosition(filename,plan_trajectory_point);
  
    std::vector<double> min_distances(human_joint_positions.size(), std::numeric_limits<double>::infinity());

    // Iterate through each point on the planned trajectory
    for (const auto& trajectory_point : plan_trajectory) {
        // Calculate the distance to each human joint position
        for (size_t i = 0; i < human_joint_positions.size(); ++i) {
            const auto& joint_position = human_joint_positions[i];
            double dist = calculateDistance(trajectory_point, joint_position);

            // Check if the calculated distance is the minimum distance found so far
            if (!std::isnan(dist) && dist < min_distances[i]) {
                min_distances[i] = dist;
            }
        }
    }
       
    plotMinDistances(min_distances,title,expFolderPath);
    return min_distances;

}

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
        }
    } else {
        ROS_WARN("JointState message does not contain 25 points for each field.");
    }
}  
    

std::string saveTrajectoryToCSV(const moveit_msgs::RobotTrajectory& trajectory_msg, const std::string& file_prefix) {
// std::string saveTrajectoryToCSV(const moveit_msgs::RobotTrajectory& trajectory_msg, const std::string& file_prefix, const std::string& file_path = csv_file_path) {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    std::tm* local_time = std::localtime(&now_time);

    std::stringstream ss;
    ss << expFolderPath << file_prefix << "_"
       << local_time->tm_mon + 1 << "_"  
       << local_time->tm_mday << "_"     
       << local_time->tm_hour << "_"     
       << local_time->tm_min << "_"      
       << local_time->tm_sec << ".csv";  

    std::string filename = ss.str();  

    std::ofstream csv_file(filename);
    if (!csv_file.is_open()) {
        ROS_ERROR_STREAM("Failed to open CSV file for writing: " << filename);
        return "";  
    }

    csv_file << "Joint1,Joint2,Joint3,Joint4,Joint5,Joint6,Joint7\n";
    for (const auto& point : trajectory_msg.joint_trajectory.points) {
        for (size_t i = 0; i < point.positions.size(); ++i) {
            csv_file << point.positions[i];
            if (i < point.positions.size() - 1) {
                csv_file << ",";
            }
        }
        csv_file << "\n";
    }

    csv_file.close();

    return filename; 
}

//Functions for Moving and grasping with robot
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
  //
  // We lower the allowed maximum velocity and acceleration to 5% of their maximum.
  // The default values are 10% (0.1).
  // Set your preferred defaults in the joint_limits.yaml file of your robot's moveit_config
  // or set explicit factors in your code if you need your robot to move faster.
  move_group.setMaxVelocityScalingFactor(0.2); // default 0.05
  move_group.setMaxAccelerationScalingFactor(0.2); // default 0.05
  //
  move_group.move();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
//Position, Orientation, Planning, Exectuion hoverPose

void hoverPose(moveit::planning_interface::MoveGroupInterface& move_group_interface, float x, float y)
{ 
  ROS_WARN("Start hoverPose");
  // We can plan a motion for this group to a desired pose for the
  // end-effector.

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
  
  target_pose_hover.position.x = x;
  target_pose_hover.position.y = y;
  target_pose_hover.position.z = 0.3;

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
  std::string saved_filename; 
  std::string title = "hover_pose";
  std::vector<Eigen::Vector3d> plan_trajectory_point;

  saved_filename = saveTrajectoryToCSV(trajectory_msg, title);
  std::cout << "saved_filename =" << saved_filename << std::endl;
    

  if (!saved_filename.empty()) {
            auto min_distances = calculateMinDistances(saved_filename, human_joint_positions, title, plan_trajectory_point);
            for (size_t i = 0; i < min_distances.size(); ++i) {
                std::cout << "Minimum distance between robot and human keypoints " << i << ": " << min_distances[i] << std::endl;
            }
  }
  move_group_interface.execute(cartesianPlan);  

  // move_group.move();
  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
//Position, Orientation, Planning, Exectuion pickPose as Cartesian Motion

void pickPose(moveit::planning_interface::MoveGroupInterface& move_group_interface, std::string direction, float x, float y)
{
  moveit::planning_interface::MoveGroupInterface::Plan cartesianPlan;
  move_group_interface.setStartStateToCurrentState();

  move_group_interface.setMaxVelocityScalingFactor(0.2);
  move_group_interface.setMaxAccelerationScalingFactor(0.2);

  std::vector<geometry_msgs::Pose> waypoints;

  geometry_msgs::Pose target_pose_pick = move_group_interface.getCurrentPose().pose;
  target_pose_pick.position.x = x;
  target_pose_pick.position.y = y;
  if (direction == "down"){
    // target_pose_pick.position.z -= 0.01;//0.26
    target_pose_pick.position.z = 0.005; //target position
  }
  else if (direction == "up"){
    target_pose_pick.position.z = 0.3;
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
  robot_trajectory::RobotTrajectory rt(move_group_interface.getCurrentState()->getRobotModel(), "panda_manipulator");

  // Get robot trajectory
  rt.setRobotTrajectoryMsg(*move_group_interface.getCurrentState(), trajectory_msg);
 
  // Create a IterativeParabolicTimeParameterization object
  trajectory_processing::IterativeParabolicTimeParameterization iptp;

  //Compute TimeStamps
  iptp.computeTimeStamps(rt, 0.1, 0.1);
  
  // Get RobotTrajectory_msg from RobotTrajectory
  rt.getRobotTrajectoryMsg(trajectory_msg);

  cartesianPlan.trajectory_ = trajectory_msg;

/////////////////////////////////////////////////////////////////////////////////////////////////////////


  if (direction == "down"){
    ROS_WARN("Start pick_down");
    std::string sub_title = "pick_down";
    std::string saved_filename; 
    std::vector<Eigen::Vector3d> plan_trajectory_point;

    saved_filename = saveTrajectoryToCSV(trajectory_msg,sub_title);

    if (!saved_filename.empty()) {
            auto min_distances = calculateMinDistances(saved_filename, human_joint_positions,sub_title,plan_trajectory_point);
            for (size_t i = 0; i < min_distances.size(); ++i) {
                std::cout << "Minimum distance between robot and human keypoints " << i << ": " << min_distances[i] << std::endl;
            }
        }
  }

  if (direction == "up"){
    ROS_WARN("Start pick_up");
    std::string sub_title = "pick_up";
    std::string saved_filename; 
    std::vector<Eigen::Vector3d> plan_trajectory_point;
    
    saved_filename = saveTrajectoryToCSV(trajectory_msg, sub_title);

    if (!saved_filename.empty()) {
            auto min_distances = calculateMinDistances(saved_filename, human_joint_positions,sub_title,plan_trajectory_point);
            for (size_t i = 0; i < min_distances.size(); ++i) {
                std::cout << "Minimum distance between robot and human keypoints " << i << ": " << min_distances[i] << std::endl;
            }
        }
  }

  // Execute the planned trajectory
  move_group_interface.execute(cartesianPlan);  

}


void hoverPlacePose(moveit::planning_interface::MoveGroupInterface& move_group_interface, float x, float y)
{ 
  ROS_WARN("Start hoverPlacePose");
  
  // We can plan a motion for this group to a desired pose for the
  // end-effector.

  moveit::planning_interface::MoveGroupInterface::Plan cartesianPlan;
  move_group_interface.setStartStateToCurrentState();

  move_group_interface.setMaxVelocityScalingFactor(0.2);
  move_group_interface.setMaxAccelerationScalingFactor(0.2);
  geometry_msgs::Pose pose_hover_place = move_group_interface.getCurrentPose().pose;
  std::vector<geometry_msgs::Pose> waypoints;

  //Convert Orienation from RPY to Quaternion
  tf2::Quaternion orientation;
  orientation.setRPY(-tau/2, 0, 0);

  pose_hover_place.orientation = tf2::toMsg(orientation);
  
  pose_hover_place.position.x = x;
  pose_hover_place.position.y = y;
  pose_hover_place.position.z = 0.3;

  waypoints.push_back(pose_hover_place);

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
  std::string saved_filename; 
  std::string title = "hover_place_hover";
  std::vector<Eigen::Vector3d> plan_trajectory_point;
  saved_filename = saveTrajectoryToCSV(trajectory_msg,title );

  if (!saved_filename.empty()) {
            auto min_distances = calculateMinDistances(saved_filename, human_joint_positions,title,plan_trajectory_point);
            for (size_t i = 0; i < min_distances.size(); ++i) {
                std::cout << "Minimum distance between robot and human keypoints " << i << ": " << min_distances[i] << std::endl;
            }
  }
  move_group_interface.execute(cartesianPlan);  

  // move_group.move();
  
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////
//Position, Orientation, Planning, Exectuion PlacePose

void PlacePose(moveit::planning_interface::MoveGroupInterface& move_group_interface, std::string direction,float x, float y)
{ 
  moveit::planning_interface::MoveGroupInterface::Plan cartesianPlan;
  move_group_interface.setStartStateToCurrentState();

  move_group_interface.setMaxVelocityScalingFactor(0.2);
  move_group_interface.setMaxAccelerationScalingFactor(0.2);

  std::vector<geometry_msgs::Pose> waypoints;

  geometry_msgs::Pose target_pose_place = move_group_interface.getCurrentPose().pose;
  target_pose_place.position.x = x;
  target_pose_place.position.y = y;
  if (direction == "down"){
    target_pose_place.position.z = 0.15; 
   
  }

  else if (direction == "up"){
    target_pose_place.position.z = 0.3;
  }
  waypoints.push_back(target_pose_place); 

  moveit_msgs::RobotTrajectory trajectory_msg;
  move_group_interface.setPlanningTime(10.0);
 
  double fraction = move_group_interface.computeCartesianPath(waypoints,
                                               0.01,  // eef_step
                                               0.0,   // jump_threshold
                                               trajectory_msg, false);
   
  // Create robot trajectory object
  robot_trajectory::RobotTrajectory rt(move_group_interface.getCurrentState()->getRobotModel(), "panda_manipulator");

  // Get robot trajectory
  rt.setRobotTrajectoryMsg(*move_group_interface.getCurrentState(), trajectory_msg);
 
  // Create a IterativeParabolicTimeParameterization object
  trajectory_processing::IterativeParabolicTimeParameterization iptp;

  //Compute TimeStamps
  iptp.computeTimeStamps(rt, 0.1, 0.1);
  
  // Get RobotTrajectory_msg from RobotTrajectory
  rt.getRobotTrajectoryMsg(trajectory_msg);
  
  cartesianPlan.trajectory_ = trajectory_msg;

  //save csv and calculate end effector position
  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  std::string saved_filename; 
  
  if (direction == "down"){
    
    ROS_WARN("Start place_down");
    std::string title = "place_down";
    std::vector<Eigen::Vector3d> plan_trajectory_point;
    saved_filename = saveTrajectoryToCSV(trajectory_msg, title);

    if (!saved_filename.empty()) {
            auto min_distances = calculateMinDistances(saved_filename, human_joint_positions,title,plan_trajectory_point);
            for (size_t i = 0; i < min_distances.size(); ++i) {
                std::cout << "Minimum distance between robot and human keypoints " << i << ": " << min_distances[i] << std::endl;
            }
        }
  }

  if (direction == "up"){
    ROS_WARN("Start place_up");
    std::string title = "place_up";
    std::vector<Eigen::Vector3d> plan_trajectory_point;
    saved_filename = saveTrajectoryToCSV(trajectory_msg, title);

    if (!saved_filename.empty()) {
            auto min_distances = calculateMinDistances(saved_filename, human_joint_positions,title,plan_trajectory_point);
            for (size_t i = 0; i < min_distances.size(); ++i) {
                std::cout << "Minimum distance between robot and human keypoints " << i << ": " << min_distances[i] << std::endl;
            }
        }
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////

  move_group_interface.execute(cartesianPlan);  

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// close Gripper for moveit msg grasp

void closedGripper(trajectory_msgs::JointTrajectory& posture)
{

  // Add both finger joints
  posture.header.stamp = ros::Time::now();
  posture.joint_names.resize(2);
  posture.joint_names[0] = "panda_finger_joint1";
  posture.joint_names[1] = "panda_finger_joint2";

  //set closed position
  posture.points.resize(1);
  posture.points[0].positions.resize(2);
  posture.points[0].positions[0] = 0.006;
  posture.points[0].positions[1] = 0.006;
  posture.points[0].effort.resize(2);
  posture.points[0].effort[0] = 1;
  posture.points[0].effort[1] = 1;
   
  
  posture.points[0].time_from_start = ros::Duration(1);

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//Little movement for grasping with grasp msg

void pick(moveit::planning_interface::MoveGroupInterface& move_group)
{
  //Create Vector for grasp approaches (only need 1)

  moveit_msgs::RobotTrajectory trajectory_msg;
  std::vector<moveit_msgs::Grasp> grasps;
  grasps.resize(1);
  
  grasps[0].pre_grasp_approach.direction.header.frame_id = "panda_link0";

  // Setting grasp pose
  geometry_msgs::PoseStamped current_pose = move_group.getCurrentPose();

  grasps[0].grasp_pose = current_pose;
  grasps[0].grasp_pose.pose.position.z -= 0.0001;
  grasps[0].grasp_pose.header.frame_id = "panda_link0";

  // Direction is set as negative z axis, approach from above the object
  grasps[0].pre_grasp_approach.direction.vector.z = -1.0;
  grasps[0].pre_grasp_approach.min_distance = 0.0001;
  grasps[0].pre_grasp_approach.desired_distance = 0.0002;

  //Close gripper to grasp object
  closedGripper(grasps[0].grasp_posture);

  // Set support surface as table1.
  move_group.setSupportSurfaceName("table1");
  // Call pick to pick up the object using the grasps given
  move_group.pick("tool", grasps);
}

void openHand(moveit::planning_interface::MoveGroupInterface& move_group_interface_hand)
{ 
  // Open the gripper
  move_group_interface_hand.setJointValueTarget(move_group_interface_hand.getNamedTargetValues("open"));

  //Move the robot
  move_group_interface_hand.move();

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
  ros::init(argc, argv, "own_pick_place_V4");
  ros::NodeHandle nh;

  //Get information about robot state
  ros::AsyncSpinner spinner(1);
  spinner.start();
  
  ros::Subscriber sub_per = nh.subscribe("robot_value", 1000, PersonCallback);

  std::string basePath = "/home/yuliang/franka_openpose/src/pick_trajectory/";
  createNextExpFolder(basePath);
  
  ros::WallDuration(1.0).sleep();
  
  // use for planning scene
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

  //planning interface
  moveit::planning_interface::MoveGroupInterface group_manipulator("panda_manipulator");
  moveit::planning_interface::MoveGroupInterface group_hand("panda_hand");
  moveit::planning_interface::MoveGroupInterface group_arm("panda_arm");
  
  // Set parameters for group like planner, speed, acceleration
  group_arm.setPlannerId("RRTConnect");
  group_arm.setMaxVelocityScalingFactor(0.1);
  group_arm.setMaxAccelerationScalingFactor(0.1);
  group_manipulator.setPlannerId("RRTConnect");
  group_manipulator.setMaxVelocityScalingFactor(0.1);
  group_manipulator.setMaxAccelerationScalingFactor(0.1);


  ros::Duration(5).sleep();

  //Create Box
  shape_msgs::SolidPrimitive primitive;
  moveit_msgs::CollisionObject object_to_attach;
  object_to_attach.id = "tool";

  // Wait a bit for ROS things to initialize
  ros::WallDuration(1.0).sleep();
  openHand(group_hand);

  hoverPose(group_manipulator, 0.4, 0.2);

  pickPose(group_manipulator  , "down", 0.4, 0.2);

  pick(group_arm);

  group_hand.attachObject(object_to_attach.id);

  pickPose(group_manipulator , "up", 0.4, 0.2);

  hoverPlacePose(group_manipulator, 0.4, -0.2);

  PlacePose(group_manipulator , "down", 0.4, -0.2);
  
  ros::WallDuration(1.0).sleep();
  openHand(group_hand);
  group_hand.detachObject(object_to_attach.id);
  
//   PlacePose(group_manipulator , "up", 0.4, -0.2);

  initPose(group_manipulator);

  ros::WallDuration(2.0).sleep();
  ROS_WARN("round end");
  
  ros::shutdown();
  return 0;
}