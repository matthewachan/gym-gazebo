import numpy as np
import math
from keras.initializers import normal, identity, uniform
from keras.models import model_from_json
from keras.models import Sequential, Model
#from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation, Lambda, concatenate
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    # def create_actor_network(self, state_size,action_dim):
    #     #print("Now we build the model")
    #     model = Sequential()
    #     S = Input(shape=[state_size])   
    #     h0 = Dense(512, activation="relu", kernel_initializer="he_uniform")(S)
    #     h1 = Dense(512, activation="relu", kernel_initializer="he_uniform")(h0)
    #     h2 = Dense(512, activation="relu", kernel_initializer="he_uniform")(h1)
    #     # LinearV = Dense(1, activation='sigmoid', kernel_initializer=uniform(minval=-3e-3,maxval=3e-3,seed=None))(h2)
    #     AngleV = Dense(1, activation='sigmoid', kernel_initializer=uniform(minval=-3e-3,maxval=3e-3,seed=None))(h2)
    #     # F = concatenate([LinearV, AngleV])
    #     F = Lambda(lambda x: x * 20.0)(AngleV)
    #     model = Model(input=S,output=F)
    #     return model, model.trainable_weights, S

    def create_actor_network(self, state_size,action_dim):
        #print("Now we build the model")
        model = Sequential()
        S = Input(shape=[state_size])   
        h0 = Dense(512, activation="relu", kernel_initializer="he_uniform")(S)
        h1 = Dense(512, activation="relu", kernel_initializer="he_uniform")(h0)
        h2 = Dense(512, activation="relu", kernel_initializer="he_uniform")(h1)
        # #uniform = lambda shape, name: uniform(shape, scale=3e-3, name=name)
        # def my_init(shape, name=None):
        #     return uniform(shape, range = (0,0.01), name=name)
        LinVel = Dense(1, activation='sigmoid', kernel_initializer=uniform(minval=-3e-3,maxval=3e-3,seed=None))(h2)
        AngVel = Dense(1, activation='sigmoid', kernel_initializer=uniform(minval=-3e-3,maxval=3e-3,seed=None))(h2)

        F1 = Lambda(lambda x: x * 20.0)(LinVel)
        F2 = Lambda(lambda x: x * 20.0)(AngVel)

        F = concatenate([F1, F2])

        model = Model(input=S,output=F)

        return model, model.trainable_weights, S
