#!/usr/bin/env python3
import math
import rospy
import tf2_ros
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32, Float32MultiArray
import numpy as np
# from get_model_info import get_model_state
from std_msgs.msg import Float32
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

def calculate_distance(point1, point2):
    distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)
    return distance

def joint_state_callback(joint_state_msg):

    # model_state = ModelState()
    # model_state.model_name = 'workpiece_clone'
    # rospy.wait_for_service('/gazebo/set_model_state')
    # set_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    try:
        trans = tfBuffer.lookup_transform('world', 'panda_hand_tcp', rospy.Time(0))
        panda_pose = PoseStamped()
        panda_pose.header.frame_id = 'world'
        panda_pose.header.stamp = rospy.Time.now()
        panda_pose.pose.position.x = trans.transform.translation.x
        panda_pose.pose.position.y = trans.transform.translation.y
        panda_pose.pose.position.z = trans.transform.translation.z

        end_effector_position = (panda_pose.pose.position.x, panda_pose.pose.position.y, panda_pose.pose.position.z)
        # print(f"end_effector_position",{end_effector_position})

        distances = [np.linalg.norm(end_effector_position - obs) for obs in obstacles]
        distance_msg = Float32MultiArray(data=distances)
        distances_pub.publish(distance_msg)
        print(','.join(map(str, distances)))

       
        # hand1_model_pose_position = get_model_state("hand_1")
        # hand2_model_pose_position = get_model_state("hand_2")
        # hand1_model_position = (hand1_model_pose_position.position.x, hand1_model_pose_position.position.y, hand1_model_pose_position.position.z)
        # hand2_model_position = (hand2_model_pose_position.position.x, hand2_model_pose_position.position.y, hand2_model_pose_position.position.z)

        # distance_1 = calculate_distance(end_effector_position, hand1_model_position)
        # # print('distance1:', distance_1)
        # distance_2 = calculate_distance(end_effector_position, hand2_model_position)
        # # print('distance2:', distance_2)
        # if distance_1 < 0.2 or distance_2 < 0.2:
        #     safety_violation = 1
        #     model_state.pose.position.x = 0.6    # Set your desired "safe" coordinates
        #     model_state.pose.position.y = 0
        #     model_state.pose.position.z = 0.1
        # else: 
        #     safety_violation = 0
        #     model_state.pose.position.x = 0.6  # Set your original/default coordinates
        #     model_state.pose.position.y = 0.0
        #     model_state.pose.position.z = -1.0

        # try:
        #     set_model_state_service(model_state)
        # except rospy.ServiceException as e:
        #     rospy.logerr(f"Service call failed: {e}")
        # distance1_pub.publish(distance_1)
        # distance2_pub.publish(distance_2)
        # safety_violation_pub.publish(safety_violation)

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rospy.logwarn("Transform lookup failed!")

def episode_callback(msg):
    """
    This function is called every time a message is published on the /episode topic.
    """
    # msg.data contains the integer value of the episode number
    episode_number = msg.data
    
    # Print a clear separator to the standard output
    print("="*60)
    print(f"--- EPISODE {episode_number} COMPLETE ---")
    print("="*60)

if __name__ == '__main__':
    rospy.init_node('end_effector_localization')
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    obstacles = [
        np.array([0.75, -0.35, 1.2]),
        np.array([0.75,  0.25, 1.2]),
        np.array([0.75,  0.0,  1.64])
    ]
   
    joint_state_sub=rospy.Subscriber('/joint_states', JointState, joint_state_callback)
    distances_pub = rospy.Publisher('/distances_to_obstacles', Float32MultiArray, queue_size=10)
    episode_sub = rospy.Subscriber("/episode", Int32, episode_callback)
    # distance1_pub = rospy.Publisher('/distance_between_end_effector_and_hand1', Float32, queue_size=100)
    # distance2_pub = rospy.Publisher('/distance_between_end_effector_and_hand2', Float32, queue_size=100)
    # safety_violation_pub = rospy.Publisher('/safety_violation', Int32, queue_size=100)
   
    rospy.spin()
