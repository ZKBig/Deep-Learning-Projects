# -*- coding: utf-8 -*-
# @Description: build the model of critic and actor network
# @author: victor
# @create time: 2021-05-18-9:22 上午


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import os

class Critic(keras.Model):
    def __init__(self, state_dims, action_dims, fc1_dims, fc2_dims, name, chkpt_dir = 'tmp/td3'):
        super(Critic, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = layers.Dense(self.action_dims+self.state_dims, activation='relu',
                                kernel_initializer=tf.keras.initializers.VarianceScaling(
                                    scale=1. / 3., distribution='uniform'))
        self.fc2 = layers.Dense(self.fc1_dims, activation='relu',
                                kernel_initializer=tf.keras.initializers.VarianceScaling(
                                    scale=1. / 3., distribution='uniform'))
        self.fc3 = layers.Dense(self.fc2_dims, activation='relu',
                                kernel_initializer=tf.keras.initializers.VarianceScaling(
                                    scale=1. / 3., distribution='uniform'))
        self.q_value = layers.Dense(1, kernel_initializer=tf.random_uniform_initializer(
            minval=-3e-3, maxval=3e-3))

    def call(self, state, action):
        # print(state)
        # print(action)
        x = self.fc1(tf.concat([state, action], axis=1))
        x = self.fc2(x)
        x = self.fc3(x)
        q_value = self.q_value(x)

        return q_value

class Actor(keras.Model):
    def __init__(self, state_dims, max_action, fc1_dims, fc2_dims, num_actions, name, chkpt_dir = 'tmp/td3'):
        super(Actor, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.state_dims = state_dims
        self.max_actions = max_action
        self.checkpoint_dir = chkpt_dir
        self.num_actions = num_actions
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = layers.Dense(self.state_dims, activation='relu',
                                kernel_initializer=tf.keras.initializers.VarianceScaling(
                                    scale=1. / 3., distribution='uniform'))
        self.fc2 = layers.Dense(self.fc1_dims, activation='relu',
                                kernel_initializer=tf.keras.initializers.VarianceScaling(
                                    scale=1. / 3., distribution='uniform'))
        self.fc3 = layers.Dense(self.fc2_dims, activation='relu',
                                kernel_initializer=tf.keras.initializers.VarianceScaling(
                                    scale=1. / 3., distribution='uniform'))
        self.actions = layers.Dense(self.num_actions,  activation='tanh',
                                kernel_initializer=tf.random_uniform_initializer(
                                    minval=-3e-3, maxval=3e-3))

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        actions = self.actions(x)
        actions = actions * self.max_actions

        return actions



