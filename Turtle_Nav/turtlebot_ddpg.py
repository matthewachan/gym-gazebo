#!/usr/bin/env python

'''
Based on:
https://github.com/vmayoral/basic_reinforcement_learning
https://gist.github.com/wingedsheep/4199594b02138dd427c22a540d6d6b8d
'''
import gym
from gym import wrappers
import gym_gazebo
import time
from distutils.dir_util import copy_tree
import os
import json
import liveplot

import numpy as np
import random
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
import time

def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym')]

def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        print(file)
        os.unlink(file)

def print_stats(data):
    num_collisions = 0
    num_success = 0
    total_dist = 0
    total_time = 0
    total_steps = 0
    min_dist = data[0][2]
    for stat in data:
        if (stat[0] == True):
            num_collisions += 1
        else:
            num_success += 1
            total_dist += stat[1]
            total_steps += stat[3]
            total_time += stat[4]


    num_timed_out = 100 - len(data)
    avg_dist = total_dist / num_success
    avg_steps = total_steps / num_success
    avg_time = total_time / num_success

    print "Average distance traveled: " + str(avg_dist)
    print "Average time steps: " + str(avg_steps)
    print "Average time: " + str(avg_time)
    print "Success rate: " + str(num_success)
    print "Collision rate: " + str(num_collisions)
    print "Timed out rate: " + str(100 - len(data))
    print "Distance efficiency: " + str(avg_dist / min_dist)


if __name__ == '__main__':
    env = gym.make('GazeboCircuit2TurtlebotLidarDdpg-v0')
    outdir = '/tmp/gazebo_gym_experiments/'
    plotter = liveplot.LivePlot(outdir)

    continue_execution = 1
    train_indicator = 0

    #Parameters for DDPG agent
    BUFFER_SIZE = 200000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.05     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 2  # Angular vel + Linear vel
    state_dim = 24  # Num of features in state

    EXPLORE = 200.0*50
    episode_count = 1000 if (train_indicator) else 10
    max_steps = 500
    reward = 0
    done = False
    epsilon = 0.3 if (train_indicator) else 0.0
    indicator = 0

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    
    if continue_execution:
        #Now load the weight
        print("Now we load the weight")
        try:
            actor.model.load_weights("actormodel.h5")
            critic.model.load_weights("criticmodel.h5")
            actor.target_model.load_weights("actormodel.h5")
            critic.target_model.load_weights("criticmodel.h5")
            print("Weight load successfully")
        except:
            print("Cannot find the weight")

    if continue_execution:
        clear_monitor_files(outdir)

    env._max_episode_steps = max_steps # env returns done after _max_episode_steps
    env = gym.wrappers.Monitor(env, outdir,force=not continue_execution, resume=continue_execution)


    num_of_collision = 0
    minimum_distance = 22.7726
    stats = []

    # Training loop
    for i in range(episode_count):
        start_time = time.time()
        episode_duration = 0

        # print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        ob = env.reset()
        s_t = np.array(ob)
     
        total_reward = 0.

        done = False
        cur_distance = 0
        step = 0
        while not done:
            loss = 0 
            # epsilon -= 0.3 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            
            if np.random.random() > epsilon:
                a_type = "Exploit"
                a_t = actor.model.predict(s_t.reshape(1, s_t.shape[0]))*1 #rescalet
                a_t = a_t[0]

                if(np.any(np.isnan(a_t))):
                    print("Encountered a nan value by nueral network")
                    exit(0)
                    lin_vel = np.random.uniform(0, 1, size=1)[0]
                    ang_vel = np.random.uniform(-1, 1, size=1)[0]
                    a_t = [lin_vel, ang_vel]
            else:
                a_type = "Explore"
                lin_vel = np.random.uniform(0, 1, size=1)[0]
                ang_vel = np.random.uniform(-1, 1, size=1)[0]
                a_t = [lin_vel, ang_vel]

            ob, r_t, done, inf = env.step(a_t)
            step += 1
            collision, distance = env.get_stats()
            cur_distance += distance
            s_t1 = np.array(ob)
        
            buff.add(s_t, a_t, r_t, s_t1, done)      #Add replay buffer
            
            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[2] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t) 
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1
        
            # print("Episode", i, "Step", step, "Action", a_type, "Reward", r_t, "Loss", loss, "Epsilon", epsilon)
        
            if done and step < max_steps - 1:
                print "Step: " + str(step)
                if(collision):
                    num_of_collision+=1


                stats.append([collision, cur_distance, minimum_distance, step, time.time() - start_time])
                print(collision, cur_distance, minimum_distance, step, time.time() - start_time)

                if (i)%100==0:
                    # Save model weights and monitoring data every 100 epochs.
                    env._flush()

        if i % 100 == 0:
            plotter.plot(env)

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                critic.model.save_weights("criticmodel.h5", overwrite=True)

        # print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    print_stats(stats)
    env.close()
