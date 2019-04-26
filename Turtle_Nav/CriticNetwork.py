import numpy as np
import math
from keras.initializers import normal, identity,uniform
from keras.models import model_from_json
#from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation, concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

HIDDEN1_UNITS = 100
HIDDEN2_UNITS = 100

class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        
        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)  
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    # def create_critic_network(self, state_size,action_dim):
    #     print("Now we build the model")
    #     S = Input(shape=[state_size])  
    #     A = Input(shape=[action_dim],name='action2')   
    #     w = Dense(512, name="layer1", kernel_initializer='he_uniform',activation='relu')(S)
    #     h = concatenate([w,A])    
    #     h3 = Dense(512, name="layer2", kernel_initializer='he_uniform',activation='relu')(h)
    #     h4 = Dense(512, name="layer3", kernel_initializer='he_uniform',activation='relu')(h3)
    #     V = Dense(1,name="layer4",kernel_initializer=uniform(minval=-3e-3,maxval=3e-3,seed=None),activation='linear')(h4) 
    #     model = Model(input=[S,A],output=V)
    #     adam = Adam(lr=self.LEARNING_RATE)
    #     model.compile(loss='mse', optimizer=adam)
    #     return model, A, S 

    def create_critic_network(self, state_size,action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])  
        A = Input(shape=[action_dim],name='action2')   
        w = Dense(HIDDEN1_UNITS, kernel_initializer='he_uniform',activation='relu')(S)
        h = concatenate([w,A])    
        h3 = Dense(HIDDEN2_UNITS, kernel_initializer='he_uniform',activation='relu')(h)
        V = Dense(action_dim,kernel_initializer=uniform(minval=-3e-3,maxval=3e-3,seed=None),activation='linear')(h3)   
        model = Model(input=[S,A],output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S 
