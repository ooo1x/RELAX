#include <ros/ros.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

std::vector<std::vector<double>> readCSV(const std::string& file_path) {
    std::vector<std::vector<double>> waypoints;
    std::ifstream file(file_path);
    std::string line;
    bool is_header = true;

    while (std::getline(file, line)) {
        if (is_header) {  // skip header
            is_header = false;
            continue;
        }

        std::vector<double> values;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            try {
                values.push_back(std::stod(cell));
            } catch (...) {
                ROS_WARN_STREAM("Failed to parse value: " << cell);
            }
        }

        if (values.size() == 7) {
            waypoints.push_back(values);
        } else {
            ROS_WARN_STREAM("Skipping row with " << values.size() << " values");
        }
    }

    return waypoints;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "joint_trajectory_player");
    ros::NodeHandle nh;

    ros::Publisher traj_pub = nh.advertise<trajectory_msgs::JointTrajectory>(
        "/panda_arm_trajectory_controller/command", 10);

    ros::Duration(1.0).sleep();  // Wait for publisher to connect

    // std::string file_path = "/home/yuliang/RELAX_franka_camera_pos1/src/pick_and_place/experiment/exp132/pick_up_8_5_15_17_38.csv";
    std::string file_path = "/home/yuliang/RELAX_franka_camera_pos1/src/pick_and_place/experiment/exp131/pick_up_8_5_15_15_34.csv";
    auto waypoints = readCSV(file_path);

    trajectory_msgs::JointTrajectory traj_msg;
    traj_msg.joint_names = {"panda_joint1", "panda_joint2", "panda_joint3",
                            "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"};

    double time_from_start = 0.0;
    double step_time = 10;  // 20Hz

    for (const auto& point : waypoints) {
        trajectory_msgs::JointTrajectoryPoint pt;
        pt.positions = point;
        time_from_start += step_time;
        pt.time_from_start = ros::Duration(time_from_start);
        traj_msg.points.push_back(pt);
    }

    ROS_INFO("Publishing trajectory...");
    traj_pub.publish(traj_msg);
    ros::spinOnce();

    return 0;
}

