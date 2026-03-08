#!/usr/bin/env python3
import rospy
import time
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import numpy as np
import math
m1,m2=2.7,0.2
l=0.4895
b1,b2=0.00001,0.05
i=(1/12)*m2*l**2
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
        th=self.pole_angle
        x=self.cart_pose
        dth=self.pole_w
        dx=self.cart_v
        b=m1+m2
        c=0.5*m2*l*np.cos(th)
        a=-0.5*m2*l*np.sin(th)*dth**2
        d=i+0.25*m2*l**2
        e=0.5*m2*l*np.cos(th)
        f=-0.5*m2*9.81*l*np.sin(th)
        k1=(e/b)/(c*e/b -d)
        k2=(-k1*b1*dx-(e*a/b -f)+b2*dth)/(c*e/b-d)
        k=1
        E=0.5*(i+0.25*m2*l**2)*dth**2+0.5*m2*9.81*l*(np.cos(th)-1)
        if abs(th)<0.15:
            effort=57.8868*self.pole_angle+31.818*self.pole_w+0.0316*self.cart_pose+0.5631*self.cart_v
            return effort
        effort=-k*dth*E/((k1+0.0001)*(i+0.25*m2*l**2))# - (k2*(i+0.25*m2*l**2)-0.5*m2*9.81*l*np.sin(th))/((i+0.25*m2*l**2)*k1)
        effort=np.clip(effort,-70,70)
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