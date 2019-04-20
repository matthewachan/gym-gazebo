import gym
import rospy
import roslaunch
import time
import numpy as np
import std_srvs.srv
import tf

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist, Point, Pose, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty

from sensor_msgs.msg import LaserScan

from gym.utils import seeding

class GazeboCircuit2TurtlebotLidarDdpgEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboCircuit2TurtlebotLidar_v0.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', std_srvs.srv.Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', std_srvs.srv.Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', std_srvs.srv.Empty)

        self.reward_range = (-np.inf, np.inf)
        self.prev_action = [0] * 2

        self.goal = Pose()
        # Goal position
        self.goal.position.x = -6
        # Identity quaternion (no rotation)
        self.goal.orientation.x = 1

        self._seed()

    def calculate_observation(self,data):
        min_range = 0.2
        done = False
        
        # Sensor readings broken up into 10 groups / buckets
        n_dof = 10
        bucket_len = len(data.ranges)/n_dof

        # Store only the minimum of each laser range bucket
        discretized_ranges = [np.inf] * 10
        for i, item in enumerate(data.ranges):
            # Determine current bucket
            bucket = int(i / bucket_len)

            # Check if we have collided
            if (min_range > data.ranges[i] > 0):
                done = True

            # Store new minimum
            if (data.ranges[i] < discretized_ranges[bucket]):
                discretized_ranges[bucket] = data.ranges[i]

        return discretized_ranges,done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # action has linear and angular vel components
        # lin_vel = action[0]
        lin_vel = 0.2
        ang_vel =0.2
        # ang_vel = action[1]

        max_ang_speed = 0.3
        # ang_vel = (action-10)*max_ang_speed*0.1 #from (-0.33 to + 0.33)

        vel_cmd = Twist()
        vel_cmd.linear.x = lin_vel
        vel_cmd.angular.z = ang_vel
        self.vel_pub.publish(vel_cmd)


        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        # Read odometry data
        odom = None
        while odom is None:
            try:
                odom = rospy.wait_for_message('/odom', Odometry, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state,done = self.calculate_observation(data)

        if not done:
            # Straight reward = 5, Max angle reward = 0.5
            reward = round(15*(max_ang_speed - abs(ang_vel) +0.0335), 2)
            # print ("Action : "+str(action)+" Ang_vel : "+str(ang_vel)+" reward="+str(reward))
        else:
            reward = -200


        # Compute polar coordinates

        # Get vector from robot to the goal
        position = odom.pose.pose.position
        robot_pos = [position.x, position.y]
        goal_pos = [self.goal.position.x, self.goal.position.y]
        to_goal = np.subtract(goal_pos, robot_pos)
        dist = np.linalg.norm(to_goal)
        print "DIST"
        print dist

        # Get the robot's forward vector
        orientation = odom.pose.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]
        print "YAW"
        print yaw

        forward = [1, 0]
        rot_mat = np.array([[np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]])
        robot_forward = rot_mat.dot(forward)
        print robot_forward

        # Get the angle 
        angle = np.arccos(robot_forward.dot(to_goal) / (1 * dist))
        print "ANGLE (in radians)"
        print angle
        

        # Build state
        print len(state)
        print len(action)
        state = np.concatenate((np.asarray(state), np.asarray(action), dist, angle), axis=None)
        print "STATE LEN"
        print len(state)

        self.prev_action = action

        return np.asarray(state), reward, done, {}

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Reset odometry
        reset_odom = rospy.Publisher('/mobile_base/commands/reset_odometry', Empty, queue_size=10)
        timer = time.time()
        while time.time() - timer < 0.25:
            reset_odom.publish(Empty())

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")
        #read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state,done = self.calculate_observation(data)

        return np.concatenate((np.asarray(state), self.prev_action), axis=None)
