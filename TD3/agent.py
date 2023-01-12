# -*- coding: utf-8 -*-
# @Description:
# @author: victor
# @create time: 2021-05-18-9:52 上午

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
from model1 import Critic
from model1 import Actor
from replyBuffer import replyBuffer
import os

class Agent:
    def __init__(self,
                 theta,
                 omega,
                 input_dims,
                 tau,
                 gamma=0.99,
                 update_interval=2,
                 warmup=10000,
                 layer1_dims=400,
                 layer2_dims=300,
                 noise=0.1,
                 max_size=1000000,
                 num_actions=1,
                 batch_size=300,
                 max_action=None,
                 c=0.5
                 ):

        self.theta = theta
        self.omega = omega
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.noise = noise
        self.max_action = max_action
        self.c = c
        self.memory = replyBuffer(max_size, input_dims, num_actions)
        self.learn_step_counter = 0
        self.warmup = warmup
        self.time_step = 0
        self.num_actions = num_actions
        self.update_interval = update_interval
        self.input_dims = input_dims
        # self.update_target_network_parameters(tau=1)

        self.actor = Actor(self.input_dims, self.max_action, layer1_dims, layer2_dims, num_actions=num_actions, name='actor')
        self.target_actor = Actor(self.input_dims, self.max_action, layer1_dims, layer2_dims, num_actions=num_actions, name='target_actor')
        self.critic_1 = Critic(self.input_dims, self.num_actions, layer1_dims, layer2_dims, name='critic_1')
        self.critic_2 = Critic(self.input_dims, self.num_actions, layer1_dims, layer2_dims, name='critic_2')
        self.target_critic_1 = Critic(self.input_dims, self.num_actions, layer1_dims, layer2_dims, name='target_critic_1')
        self.target_critic_2 = Critic(self.input_dims, self.num_actions, layer1_dims, layer2_dims, name='target_critic_2')

    def choose_actions(self, observation):
        # if self.time_step < self.warmup:
        #     actions = np.random.normal(scale=self.noise, size=(self.num_actions,))
        # else:
        states = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor.call(states)[0]
        actions = actions + np.random.normal(loc=0, scale=self.noise)
        print(actions)
        actions = tf.clip_by_value(actions, -self.max_action, self.max_action)

        print(actions)

        self.time_step += 1

        return actions

    def remember(self, states, actions, reward, next_states):
        return self.memory.store_transition(states, actions, reward, next_states)

    def learn(self, done):
        if self.memory.counter < self.batch_size:
            return
        else:
            states, actions, reward, next_states = self.memory.sample_from_buffer(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor(next_states)
            target_actions = target_actions + tf.clip_by_value(np.random.normal(loc=0, scale=0.2), -self.c, self.c)
            target_actions = tf.clip_by_value(target_actions, -self.max_action, self.max_action)

            target_q1 = tf.squeeze(self.target_critic_1(next_states, target_actions), 1)
            target_q2 = tf.squeeze(self.target_critic_2(next_states, target_actions), 1)

            q1 = tf.squeeze(self.critic_1(states, actions), 1)
            q2 = tf.squeeze(self.critic_2(states, actions), 1)

            target_value_1 = reward + self.gamma * target_q1 * (1-done)
            target_value_2 = reward + self.gamma * target_q2 * (1-done)

            target_value = tf.math.minimum(target_value_1, target_value_2)

            critic_1_loss = keras.losses.Huber()(target_value, q1)
            critic_2_loss = keras.losses.Huber()(target_value, q2)

        self.critic_1_optimizer = keras.optimizers.Adam(learning_rate=self.omega)
        self.critic_2_optimizer = keras.optimizers.Adam(learning_rate=self.omega)

        critic_1_gradient = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_gradient = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)

        self.critic_1_optimizer.apply_gradients(zip(critic_1_gradient, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(critic_2_gradient, self.critic_2.trainable_variables))

        self.learn_step_counter += 1

        if self.learn_step_counter % self.update_interval != 0:
            return
        else:
            with tf.GradientTape() as tape:
                new_actions = self.actor(states)
                critic_1_q_value = self.critic_1(states, new_actions)
                actor_loss = -tf.math.reduce_mean(critic_1_q_value)

            self.actor_optimizer=keras.optimizers.Adam(learning_rate=self.theta)
            actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

            self.update_target_network_parameters()

    def update_target_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        self.target_actor.compile(optimizer=keras.optimizers.Adam(learning_rate=self.theta),
                                  loss='mean')
        self.target_critic_1.compile(optimizer=keras.optimizers.Adam(learning_rate=self.omega),
                                     loss='mean_squared_error')
        self.target_critic_2.compile(optimizer=keras.optimizers.Adam(learning_rate=self.omega),
                                     loss='mean_squared_error')

        weights = []
        target_weights = self.target_actor.weights
        for i,weight in enumerate(self.actor.weights):
            weights.append(weight * tau + target_weights[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        weights = []
        target_weights = self.target_critic_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights.append(weight * tau + target_weights[i] * (1 - tau))
        self.target_critic_1.set_weights(weights)

        weights = []
        target_weights = self.target_critic_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            weights.append(weight * tau + target_weights[i] * (1 - tau))
        self.target_critic_2.set_weights(weights)

    def save_models(self):
        print('.......saving models.......')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)
        self.critic_2.save_weights(self.critic_2.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic_1.save_weights(self.target_critic_1.checkpoint_file)
        self.target_critic_2.save_weights(self.target_critic_2.checkpoint_file)

    def load_models(self):
        print('.......loading models.......')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic_1.load_weights(self.critic_1.checkpoint_file)
        self.critic_2.load_weights(self.critic_2.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic_1.load_weights(self.target_critic_1.checkpoint_file)
        self.target_critic_2.load_weights(self.target_critic_2.checkpoint_file)




















