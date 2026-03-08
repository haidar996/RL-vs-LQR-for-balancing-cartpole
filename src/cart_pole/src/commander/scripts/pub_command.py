#!/usr/bin/env python3
import rospy
import time
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import numpy as np
import math
from lqr import *

class PolePlacementController:
    def __init__(self):
        self.cart_pose = 0
        self.pole_angle = 0
        self.cart_v = 0
        self.pole_w = 0
        
       
        
        
        
        rospy.init_node('pole_placement_controller', anonymous=True)
        
        # Subscribe to joint states
        rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)
        
        # Publisher for cart control command
        self.pub_cart = rospy.Publisher('/cart_controller/command', Float64, queue_size=10)
        
        
    def joint_state_callback(self, data):
        """Callback for joint state updates"""
        try:
            # Get cart joint state
            ind = data.name.index('cart_joint')
            self.cart_pose = data.position[ind]
            self.cart_v = data.velocity[ind]
            
            # Get pole joint state
            ind = data.name.index('pole_joint')
            # Use unwrapped angle for control
            self.pole_angle = self.normalize_angle(data.position[ind])
            self.pole_w = data.velocity[ind]
            
        except ValueError as e:
            rospy.logwarn(f"Joint not found: {e}")
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def compute_control(self):
        effort=57.8868*self.pole_angle+31.818*self.pole_w+0.316*self.cart_pose+0.631*self.cart_v
        #effort=37.8868*self.pole_angle+11.818*self.pole_w+5*self.cart_pose+2*self.cart_v
        return effort
    
    def run(self):
        """Main control loop"""
        rate = rospy.Rate(30)  # 200 Hz control loop
        
        
        
        while not rospy.is_shutdown():
            # Get current time for integral calculation
            
            
            # Compute control effort
            control_effort = self.compute_control()
            
            # Publish control command
            msg = Float64()
            msg.data = float(control_effort)
            self.pub_cart.publish(msg)
            print(msg.data)
            
            
            
            rate.sleep()

if __name__ == '__main__':
    try:
        
        controller = PolePlacementController()
        controller.run()
        
    except rospy.ROSInterruptException:
        pass
