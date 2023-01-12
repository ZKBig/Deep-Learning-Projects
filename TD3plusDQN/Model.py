# -*- coding: utf-8 -*-
# @Description:
# @author: victor
# @create time: 2021-05-19-10:40 上午
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import os


class Critic(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, name, chkpt_dir='tmp/td3'):
        super(Critic, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = layers.Dense(self.fc1_dims, activation='relu')
        self.fc2 = layers.Dense(self.fc2_dims, activation='relu')
        self.q_value = layers.Dense(1, activation=None)

    def call(self, state, action):
        x = self.fc1(tf.concat([state, action], axis=1))
        x = self.fc2(x)
        q_value = self.q_value(x)

        return q_value


class Actor(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, num_actions, name, chkpt_dir='tmp/td3'):
        super(Actor, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.num_actions = num_actions
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = layers.Dense(self.fc1_dims, activation='relu')
        self.fc2 = layers.Dense(self.fc2_dims, activation='relu')
        self.actions = layers.Dense(self.num_actions, activation='tanh')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        actions = self.actions(x)

        return actions


class DuelingDQN(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, name, num_actions, chkpt_dir='tmp/td3'):
        super(DuelingDQN, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.num_actions = num_actions
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = layers.Dense(self.fc1_dims, activation='relu')
        self.fc2 = layers.Dense(self.fc2_dims, activation='relu')
        self.A_value = layers.Dense(self.num_actions, activation='relu')
        self.V_value = layers.Dense(1, activation='relu')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        A_value = self.A_value(x)
        V_value = self.V_value(x)

        # print(A_value)
        q_value = V_value + A_value - tf.reduce_mean(A_value, axis=1, keepdims=True)
        return q_value
