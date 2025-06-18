#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
import moveit_commander

class RLController:
    def __init__(self):
        rospy.init_node('rl_controller', anonymous=True)

        # Initialize MoveIt commander
        moveit_commander.roscpp_initialize([])
        self.arm = moveit_commander.MoveGroupCommander("panda_arm")
        
        # Subscribe to the joint states topic
        rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)
        rospy.loginfo("Subscribed to /joint_states topic")

        self.count = 0
    
    def joint_state_callback(self, msg):
        self.count+= 1
        if self.count % 10 != 0:
            return
        
        pose = self.arm.get_current_pose().pose
        x = pose.position.x
        y = pose.position.y
        z = pose.position.z
        rospy.loginfo("Current pose: x=%f, y=%f, z=%f", x, y, z)

        # Process the joint states
        rospy.loginfo("Received joint states: %s", msg.position)
        # Here you would typically implement your RL logic to control the robot

if __name__ == '__main__':
    try:
        controller = RLController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("An error occurred: %s", str(e))