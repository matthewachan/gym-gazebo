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

from geometry_msgs.msg import Twist, Point, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import std_msgs.msg # Required for odom Empty message
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState

from std_srvs.srv import Empty

class GazeboCircuit2TurtlebotLidarDdpgEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Specify the map to load
        #gazebo_env.GazeboEnv.__init__(self, "GazeboCircuit2TurtlebotLidar_v0.launch")
        # gazebo_env.GazeboEnv.__init__(self, "GazeboDebug_v0.launch")
        # gazebo_env.GazeboEnv.__init__(self, "GazeboEnv1.launch")
        gazebo_env.GazeboEnv.__init__(self, "GazeboTest.launch")

        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)

        # Gazebo services
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.direction = 1
        self.reward_range = (-np.inf, np.inf)

        # Get Gazebo model info about the target (relative to green circle link)
        self.goal = None

        self.reached_goal = False

        self.prev_pose = None
        self._seed()

        self.target_list = [
                [-4, 3],
                [-3, -4],
                [4, -4],
                [2, 1.5],
                [0, 0]
        ]
        self.cur_index = 0
        self.total_dist = 0

    def move_dynamic_obstacles(self):
        wall1 = self.get_model_state('Wooden_Wall_Med_0_clone', 'ground_plane').pose

        increment = 0.05 * self.direction
        if (wall1.position.y > 3 and self.direction == 1 or wall1.position.y < -3 and self.direction == -1):
            self.direction *= -1

        wall1.position.y += increment

        timer = time.time()
        while time.time() - timer < 0.05:
            self.set_model_state(ModelState('Wooden_Wall_Med_0_clone', wall1, Twist(), ''))



    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def check_collision(self,data):
        min_range = 0.2
        MAX_DIST = 9999 # Distance reading to use instead of NaN
        done = False
        lidar_data = np.asarray(data.ranges)
        for i, item in enumerate(data.ranges):
            # Swap NaN values for MAX_DIST
            if (np.isinf(data.ranges[i])):
                lidar_data[i] = MAX_DIST
            if (min_range > data.ranges[i] > 0):
                done = True
        return list(lidar_data),done

    # Input: the position of the robot
    # theta = 0 => don't need to transform
    def get_goal(self, x, y, t):
        delta_x = self.goal.x - x
        delta_y = self.goal.y - y
        real_x = delta_x * cos(t) + delta_y * sin(t)
        real_y = delta_y * cos(t) - delta_x * sin(t)
        return real_x,real_y

    #the reward for the robot moving
    def get_dist_check(self, turtle_pos):
        move_dist = (self.prev_pose.x - turtle_pos.x)*(self.prev_pose.x - turtle_pos.x) + (self.prev_pose.y - turtle_pos.y)*(self.prev_pose.y - turtle_pos.y)
        move_dist = np.sqrt(move_dist)
        if(move_dist > 0.2):
            return 0, move_dist
        else:
            return -0.2 + move_dist, move_dist

    # Step the simulation forward in time
    def step(self, action):

        self.enable_physics()
        #self.move_dynamic_obstacles()

        ######################### get the speed  #################################

        # Get linear and angular action
        lin_action = action[0]
        ang_action = action[1]
        
        max_lin_speed = 0.5
        lin_vel = lin_action * max_lin_speed

        max_ang_speed = 0.3
        ang_vel = ang_action * max_ang_speed

        vel_cmd = Twist()
        vel_cmd.linear.x = lin_vel
        vel_cmd.angular.z = ang_vel
        self.vel_pub.publish(vel_cmd)

        data = self.lidar_scan()

        ############################ now get the corresponding rewards #############################
        # Compute distance between Turtlebot and goal
        odom = self.get_odom()

        # Compute goal position in the robot's frame
        turtle_pos = odom.pose.pose.position
        quaternion = odom.pose.pose.orientation
        explicit_quat = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        angle = tf.transformations.euler_from_quaternion(explicit_quat)
        
        stx, sty = self.get_goal(turtle_pos.x, turtle_pos.y, angle[2])
        dist = np.sqrt(np.power(turtle_pos.x - self.goal.x, 2) + np.power(turtle_pos.y - self.goal.y, 2))
        #print turtle_pos

        # Change in distance from the goal
        delta_dist = dist - self.prev_dist
        self.prev_dist = dist

        dist_reward, move_dist = self.get_dist_check(turtle_pos)
        # print str(move_dist)
        # + " " + str(turtle_pos) + " " + str(self.prev_pose)
        self.total_dist = move_dist
        dist_reward =  dist_reward * 5
        angle_diff = np.abs(self.prev_angle - angle[2]) * 5

        #just the z angle is needed
        self.prev_angle = angle[2]
        self.prev_pose = turtle_pos

        ############################ end of distance related rewards ##############################

        self.pause_physics()

        state, done = self.check_collision(data)
        # print("laser reading:")
        # print state
        state += [stx, sty]
        state += [lin_action, ang_action]
        self.collide = False
        # Set reward
        if not done:
            reward = -delta_dist * 15
        else:
            reward = -70
            self.collide = True

        # print "angle diff, dist reward, dist reward"
        # print angle_diff, dist_reward, reward
        # import IPython; IPython.embed()

        #currentlly disabled
        reward += dist_reward

        reward -= angle_diff

        

        # Check goal state
        if dist < 0.5:
            reward += 500
            if(self.cur_index == len(self.target_list)):
                done = True
                return np.asarray(state), reward, done, {}
            # done = True
            #instead, set it to a new state
            self.reset_target()
            self.enable_physics()
            self.vel_pub.publish(Twist())
            self.reset_param()
            self.pause_physics()

        # print("reward: " + str(reward))
        return np.asarray(state), reward, done, {}

    def get_stats(self):
        return self.collide, self.total_dist


    def reset_target(self):
        pose = Pose()

        #for debug_map
        # cord_low_x = -1
        # cord_high_x = 4
        # cord_low_y = -4
        # cord_high_y = 1

        #for env2
        cord_low_x = -4.5
        cord_high_x = 4.5
        cord_low_y = -4.5
        cord_high_y = 4.5

        coord = [np.random.uniform(cord_low_x, cord_high_x), np.random.uniform(cord_low_y, cord_high_y)]
        while self.validate_target(coord[0], coord[1]) == False:
            coord = [np.random.uniform(cord_low_x, cord_high_x), np.random.uniform(cord_low_y, cord_high_y)]

        # coord = [3,-3]
        coord = self.target_list[self.cur_index]
        self.cur_index += 1
        pose.position = Point(coord[0], coord[1], 0)
        print "Target at : " + str(coord[0]) + ", " + str(coord[1])

        timer = time.time()
        while time.time() - timer < 0.05:
            self.set_model_state(ModelState('Target', pose, Twist(), ''))

    #this function needs to be called with physics enabled
    def reset_param(self):
        # Get the Target model's position relative to the ground plane
        self.goal = self.get_model_state('Target', 'ground_plane').pose.position     

        # Retrieve the current state 
        odom = self.get_odom()
        turtle_pos = odom.pose.pose.position
        quaternion =odom.pose.pose.orientation
        explicit_quat = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        angle = tf.transformations.euler_from_quaternion(explicit_quat)
        self.prev_dist = np.sqrt(np.power(turtle_pos.x - self.goal.x, 2) + np.power(turtle_pos.y - self.goal.y, 2))
        self.prev_pose = turtle_pos
        self.prev_angle = angle[2]


    # Reset the episode
    def reset(self):
        # Reset the simulation

        self.reset_gazebo()

        #reset the position of the target
        self.cur_index = 0
        self.reset_target()

        self.enable_physics()

        # Stop the robot
        self.vel_pub.publish(Twist())

        # Reset Turtlebot's odometry
        self.reset_odom()

        #reset all the parameters
        self.reset_param()

        data = self.lidar_scan()

        self.pause_physics()

        stx, sty = self.get_goal(self.prev_pose.x, self.prev_pose.y, self.prev_angle)
        state, done = self.check_collision(data)
        state += [stx, sty]
        state += [0, 0]
        self.total_dist = 0

        return np.asarray(state)

    #this is just for env1
    def validate_target(self, x, y):
        if (-3 < x < 3 and -3 < y < -1):
            return False
        if (-3 < x < 3 and 1 < y < 3):
            return False
        if (-1 < x < 1 and 2 < y < 4):
            return False
        if (-1 < x < 1 and -4 < y < -2):
            return False
        return True

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
        #could set random pose here

        reset_odom = rospy.Publisher('/mobile_base/commands/reset_odometry', std_msgs.msg.Empty, queue_size=10)
        timer = time.time()
        # Takes some time to process Empty message and reset odometry
        while time.time() - timer < 0.25:
            reset_odom.publish(std_msgs.msg.Empty())
