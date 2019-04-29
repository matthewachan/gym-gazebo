import rospy
import roslaunch
import time
import numpy as np
import tf

from numpy import cos, sin
import gym
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from gym.utils import seeding

from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import std_msgs.msg # Required for odom Empty message
from gazebo_msgs.srv import GetModelState

from std_srvs.srv import Empty

class GazeboCircuit2TurtlebotLidarDdpgEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Specify the map to load
        #gazebo_env.GazeboEnv.__init__(self, "GazeboCircuit2TurtlebotLidar_v0.launch")
        gazebo_env.GazeboEnv.__init__(self, "GazeboDebug_v0.launch")
        #gazebo_env.GazeboEnv.__init__(self, "GazeboEnv1.launch")

        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)

        # Gazebo services
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        self.reward_range = (-np.inf, np.inf)

        # Get Gazebo model info about the target (relative to green circle link)
        self.goal = None

        #self.prev_dist = np.sqrt()
        self.reached_goal = False

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
        return list(data.ranges),done

    # Input: the position of the robot
    # theta = 0 => don't need to transform
    def get_goal(self, x, y, t):
        delta_x = self.goal.x - x
        delta_y = self.goal.y - y
        real_x = delta_x * cos(t) + delta_y * sin(t)
        real_y = delta_y * cos(t) - delta_x * sin(t)
        return real_x,real_y


    # Step the simulation forward in time
    def step(self, action):

        self.enable_physics()


        # Compute distance between Turtlebot and goal
        odom = self.get_odom()

        # Compute goal position in the robot's frame
        turtle_pos = odom.pose.pose.position
        quaternion = odom.pose.pose.orientation
        explicit_quat = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        angle = tf.transformations.euler_from_quaternion(explicit_quat)
        
        stx, sty = self.get_goal(turtle_pos.x, turtle_pos.y, angle[2])
        dist = np.sqrt(np.power(turtle_pos.x - self.goal.x, 2) + np.power(turtle_pos.y - self.goal.y, 2))
        print turtle_pos

        # Change in distance from the goal
        delta_dist = dist - self.prev_dist
        self.prev_dist = dist

        # Get linear and angular action
        lin_action = action[0]
        ang_action = action[1]
        
        # Generated linear velocity action MUST be between 0 and 20
        max_lin_speed = 0.5
        # lin_vel = (lin_action / 20) * max_lin_speed
        lin_vel = lin_action * max_lin_speed

        # Generated angular velocity action MUST be between 0 and 20
        max_ang_speed = 1
        ang_vel = ang_action
        # ang_vel = (ang_action - 10) * max_ang_speed * 0.1 #from (-0.3 to + 0.3)

        # print (lin_vel, ang_vel)

        vel_cmd = Twist()
        vel_cmd.linear.x = lin_vel
        vel_cmd.angular.z = ang_vel # action / 3.0
        self.vel_pub.publish(vel_cmd)

        data = self.lidar_scan()

        self.pause_physics()

        state, done = self.check_collision(data)
        state += [stx, sty]

        # Set reward
        if not done:
            reward = -delta_dist * 200
        else:
            reward = -10

        # Check goal state
        if dist < 0.5:
            reward = 1000
            done = True

        return np.asarray(state), reward, done, {}

    # Reset the episode
    def reset(self):
        # Reset the simulation
        self.reset_gazebo()

        self.enable_physics()

        # Stop the robot
        self.vel_pub.publish(Twist())

        # Get the Target model's position relative to the ground plane
        self.goal = self.get_model_state('Target', 'ground_plane').pose.position
        # Reset Turtlebot's odometry
        self.reset_odom()

        # Retrieve the current state 
        odom = self.get_odom()
        turtle_pos = odom.pose.pose.position
        quaternion =odom.pose.pose.orientation
        explicit_quat = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        angle = tf.transformations.euler_from_quaternion(explicit_quat)
        stx, sty = self.get_goal(turtle_pos.x, turtle_pos.y, angle[2])
        self.prev_dist = np.sqrt(np.power(turtle_pos.x - self.goal.x, 2) + np.power(turtle_pos.y - self.goal.y, 2))
        
        data = self.lidar_scan()

        self.pause_physics()

        state, done = self.check_collision(data)
        state += [stx, sty]

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
            reset_odom.publish(std_msgs.msg.Empty())
