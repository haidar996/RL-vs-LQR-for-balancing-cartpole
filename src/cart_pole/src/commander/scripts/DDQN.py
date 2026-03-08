#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import numpy as np
from std_srvs.srv import Empty
import time
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import LinkStates
import tf.transformations as tf_trans
import threading
import logging
class cartpoleenv():
    def __init__(self):
        rospy.set_param('/rosconsole/rosout', 'DEBUG')
        self._max_retry=20
        self.force_mag = [0.1,0.5,1.0] # force applied to cart
        self.theta_threshold = 0.2 # ~12 degrees
        self.x_threshold = 0.9
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_state = rospy.ServiceProxy(
            '/gazebo/set_model_state', SetModelState
        )
        rospy.Subscriber(
        "/gazebo/link_states",
        LinkStates,
        self.link_state_callback,
        queue_size=1
    )

        self.dt = 0.002
        rospy.wait_for_service("/gazebo/pause_physics")
        rospy.wait_for_service("/gazebo/unpause_physics")
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)

        self.effort_pub=rospy.Publisher("/cart_controller/command",Float64,queue_size=1)
        
        self.spin_thread = threading.Thread(target=rospy.spin)
        self.spin_thread.daemon = True
        self.spin_thread.start()
        rospy.wait_for_service("/gazebo/reset_simulation")
        self.reset_sim = rospy.ServiceProxy(
            "/gazebo/reset_simulation",
            Empty
        )
        self.state = None
        self.rate = rospy.Rate(50)
        
    def hard_reset_model(self):
        state = ModelState()
        state.model_name = "cart_pole"  # <-- your model name in Gazebo
        state.pose.position.x = 0.0
        state.pose.position.y = 0.0
        state.pose.position.z = 1.225
        state.pose.orientation.w = 1.0
        state.twist.linear.x = 0.0
        state.twist.angular.z = 0.0
        state.reference_frame = "world"

        self.set_model_state(state)
    def link_state_callback(self,data):
        try:
            # Find the indices of the cart and pole links
            cart_index = data.name.index('cart_pole::cart_link')
            pole_index = data.name.index('cart_pole::pole_link')
            
            # Get cart position (x coordinate)
            cart_pose = data.pose[cart_index].position.x
            # Get cart linear velocity (x component)
            cart_v = data.twist[cart_index].linear.x
            
            # Get pole orientation (quaternion)
            pole_orientation = data.pose[pole_index].orientation
            # Convert quaternion to Euler angles
            euler = tf_trans.euler_from_quaternion([pole_orientation.x,
                                                    pole_orientation.y,
                                                    pole_orientation.z,
                                                    pole_orientation.w])
            # Get the pole angle (rotation around y-axis)
            pole_angle = self.normalize_angle(euler[1])  # index 1 is usually y-axis rotation
            
            # Get pole angular velocity (y component)
            self.pole_w = data.twist[pole_index].angular.y
            
            self.state = np.array([cart_pose, cart_v, pole_angle, self.pole_w], dtype=np.float32)
            
        except ValueError as e:
            rospy.logwarn("Could not find cart or pole in link states: %s", e)
        except IndexError as e:
            rospy.logwarn("Index error in link states: %s", e)
    
    def normalize_angle(self, angle):
        
        return (angle + np.pi) % (2 * np.pi) - np.pi
    def resetSimulation(self):
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_sim()
        except rospy.ServiceException as e:
            print ("/gazebo/reset_simulation service call failed")
    def reset(self):
        self.effort_pub.publish(Float64(0.0))
        
        
        
        
        self.resetSimulation()
        
        
        

        self.state = None
        
        self.unpauseSim()
        
        rospy.wait_for_message("/gazebo/link_states", LinkStates)
        
        while np.abs(self.state[0])<=0.00000001 and np.abs(self.state[2])<=0.0000001:
            
            pass
        self.pauseSim()
        
        return self.state.copy()
    def unpauseSim(self):
        
        unpaused_done = False
        counter = 0
        while not unpaused_done and not rospy.is_shutdown():
            if counter < self._max_retry:
                try:
                    
                    self.unpause()
                    unpaused_done = True
                    
                except rospy.ServiceException as e:
                    counter += 1
                    
            else:
                error_message = "Maximum retries done"+str(self._max_retry)+", please check Gazebo unpause service"
                rospy.logerr(error_message)
                assert False, error_message

        
    

    

    

    
    def pauseSim(self):
        
        paused_done = False
        counter = 0
        while not paused_done and not rospy.is_shutdown():
            if counter < self._max_retry:
                try:
                    
                    self.pause()
                    paused_done = True
                   
                except rospy.ServiceException as e:
                    counter += 1
                    
            else:
                error_message = "Maximum retries done"+str(self._max_retry)+", please check Gazebo pause service"
                rospy.logerr(error_message)
                assert False, error_message

        
    def step(self,action):
        
        
        

        # 1. Pause physics
        

        if action < 3:  # Left forces
            force = -self.force_mag[action]
        else:  # Right forces
            force = self.force_mag[action - 3]
        
        # Apply force
        self.effort_pub.publish(Float64(force))
        
        # Step physics with error handling
        
            # Unpause for time_step seconds
        
        self.unpauseSim()
        time.sleep(0.001)
        # Use precise timing
        time.sleep(0.05)
        self.pauseSim()
        # Pause physics
        
        
        # Brief pause for state propagation
        time.sleep(0.001)
            
        
            
        
        # Get state with error checking
        if self.state is None:
            rospy.logwarn("State is None, using zeros")
            self.state = np.zeros(4, dtype=np.float32)
        
        x, x_dot, theta, theta_dot = self.state
        done = abs(theta) > self.theta_threshold 

        reward = 1.0 - abs(theta) - 0.001*abs(self.pole_w) -abs(x) - 0.001*abs(x_dot)
        if done:
            reward = -10.0

        return self.state.copy(), reward, done

        

        

        

