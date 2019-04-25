import rospy
import roslaunch
import time
import numpy as np

import gym
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from gym.utils import seeding

from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import std_msgs.msg # Required for odom Empty message

from std_srvs.srv import Empty

class GazeboCircuit2TurtlebotLidarDdpgEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        gazebo_env.GazeboEnv.__init__(self, "GazeboDebug_v0.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.reward_range = (-np.inf, np.inf)
        self.goal = Point(-6, 0, 0)
        self.prev_dist = None

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def check_collision(self,data):
        min_range = 0.2
        done = False
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = True
        return data.ranges[0:14],done

    # Step the simulation forward in time
    def step(self, action):
        self.enable_physics()

        # Compute distance between Turtlebot and goal
        odom = self.get_odom()
        turtle_pos = odom.pose.pose.position

        dist = np.sqrt(np.power(turtle_pos.x - self.goal.x, 2) + np.power(turtle_pos.y - self.goal.y, 2))
            
        # Edge case for initializing previous distance to the goal
        if (self.prev_dist == None):
            self.prev_dist = dist

        delta_dist = dist - self.prev_dist

        max_ang_speed = 0.3
        ang_vel = (action[0][0]-10)*max_ang_speed*0.1 #from (-0.33 to + 0.33)

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.2
        vel_cmd.angular.z = ang_vel
        self.vel_pub.publish(vel_cmd)

        data = self.lidar_scan()

        self.pause_physics()

        state, done = self.check_collision(data)

        if not done:
            if (delta_dist < 0):
                reward = 5 
            if (delta_dist >= 0):
                reward = -5 
            # Straight reward = 5, Max angle reward = 0.5
            # reward = round(15* (max_ang_speed - abs(ang_vel) + 0.0335), 2)
            # print ("Action : "+str(action)+" Ang_vel : "+str(ang_vel)+" reward="+str(reward))
        else:
            reward = -200

        return np.asarray(state), reward, done, {}

    # Reset the episode
    def reset(self):
        self.reset_gazebo()

        self.enable_physics()

        self.reset_odom()

        data = self.lidar_scan()

        self.pause_physics()

        state, done = self.check_collision(data)

        return np.asarray(state)

###################
# EVENT CALLBACKS #
###################

    # Disable Gazebo physics
    def pause_physics(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

    # Enable Gazebo physics
    def enable_physics(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

    # Resets the state of the environment and returns an initial observation.
    def reset_gazebo(self):
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

    # Get depth camera data from Turtlebot
    def lidar_scan(self):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass
        return data

    # Retrieve odometry data
    def get_odom(self):
        odom = None
        while odom is None:
            try:
                odom = rospy.wait_for_message('/odom', Odometry, timeout=5)
            except:
                pass
        return odom

    # Reset odometry (required when calling reset_gazebo)
    def reset_odom(self):
        reset_odom = rospy.Publisher('/mobile_base/commands/reset_odometry', std_msgs.msg.Empty, queue_size=10)
        timer = time.time()
        # Takes some time to process Empty message and reset odometry
        while time.time() - timer < 0.25:
            reset_odom.publish(Empty())
