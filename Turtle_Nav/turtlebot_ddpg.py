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
#from keras.engine.training import collect_trainable_weights
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

if __name__ == '__main__':

    

    #REMEMBER!: turtlebot_nn_setup.bash must be executed.
    #replace the action with the correct one
    env = gym.make('GazeboCircuit2TurtlebotLidarDdpg-v0')
    outdir = '/tmp/gazebo_gym_experiments/'
    plotter = liveplot.LivePlot(outdir)

    num_of_collision = 0
    total_distance = 0
    total_minimum_distance = 0

    continue_execution = 1
    train_indicator = 0
    #fill this if continue_execution=True

    #Parameters for the ddpg agent
    BUFFER_SIZE = 200000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.05     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 2  #angular vel + linear vel
    state_dim = 24  #num of features in state

    EXPLORE = 200.0*50
    episode_count = 1000 if (train_indicator) else 100
    # episode_count = 1000
    max_steps = 500
    reward = 0
    done = False
    step = 0
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

    #maybe delete this?
    if continue_execution:
        clear_monitor_files(outdir)

    env._max_episode_steps = max_steps # env returns done after _max_episode_steps
    env = gym.wrappers.Monitor(env, outdir,force=not continue_execution, resume=continue_execution)


    start_time = time.time()

    #potentially, we can use multi-threading here

    # for i in range(10):
    #     ob = env.reset()

    #     while not done:
    #         s_t = np.array(ob)
    #         a_t = np.random.uniform(0,20, size=action_dim)
    #         ob, r_t, done, info = env.step(a_t)
    #         s_t1 = np.array(ob)
    #         buff.add(s_t, a_t[0], r_t, s_t1, done)  


    #The training loop
    for i in range(episode_count):

        # print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        ob = env.reset()
        s_t = np.array(ob)
     
        total_reward = 0.

        done = False
        cur_distance = 0
        while not done:
            loss = 0 
            epsilon -= 0.3 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            
            #print(s_t)

            if np.random.random() > epsilon:
                a_type = "Exploit"
                a_t = actor.model.predict(s_t.reshape(1, s_t.shape[0]))*1 #rescalet
                a_t = a_t[0]
                # print("Exploit: ")
                #print(a_t)
                if(np.any(np.isnan(a_t))):
                    print("encountered a nan value by nueral network")
                    exit(0)
                    lin_vel = np.random.uniform(0, 1, size=1)[0]
                    ang_vel = np.random.uniform(-1, 1, size=1)[0]
                    a_t = [lin_vel, ang_vel]
            else:
                a_type = "Explore"
                lin_vel = np.random.uniform(0, 1, size=1)[0]
                ang_vel = np.random.uniform(-1, 1, size=1)[0]
                a_t = [lin_vel, ang_vel]
                #print a_t
                # a_t = np.random.uniform(0,20, size=action_dim)
                # a_t = np.asarray([lin_vel, ang_vel])
                # print("Explore: ")
                # print(a_t)
            # print("action: ")
            # print(a_t)
            ob, r_t, done, inf = env.step(a_t)
            collision, distance = env.get_stats()
            cur_distance = distance
            # cur_distance += distance
            # print("distance: "+ str(distance))
            # print("total distance: "+ str(cur_distance))
            s_t1 = np.array(ob)
        
            buff.add(s_t, a_t, r_t, s_t1, done)      #Add replay buffer
            
            # print "rewards: "
            # print r_t

            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[2] for e in batch])

            #print(actor.target_model.predict(new_states))
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
        
            step += 1
            if done:
                if(collision):
                    num_of_collision+=1
                else:
                    total_distance += cur_distance
                    total_minimum_distance += 22.7726

                print(num_of_collision, total_distance, total_minimum_distance)

                if (i)%100==0:
                    #save model weights and monitoring data every 100 epochs.
                    env._flush()

        if i % 100 == 0:
            plotter.plot(env)

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                critic.model.save_weights("criticmodel.h5", overwrite=True)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.close()
