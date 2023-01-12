# -*- coding: utf-8 -*-
# @Description:
# @author: victor
# @create time: 2021-05-19-10:40 上午
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
from Model import Critic
from Model import Actor
from Model import DuelingDQN
from replyBufferTD3 import replyBufferTD3
from replyBufferDQN import replyBufferDQN
import os


class Agent:
    def __init__(self,
                 theta,
                 omega,
                 alpha,
                 input_dims,
                 tau,
                 gamma=0.99,
                 beta=0.95,
                 update_interval=3,
                 warmup=1000,
                 layer1_dims=300,
                 layer2_dims=400,
                 noise=0.1,
                 max_size=10000000,
                 num_continuous_actions=1,
                 num_discrete_actions=3,
                 batch_size=300,
                 c=3
                 ):

        self.theta = theta
        self.omega = omega
        self.tau = tau
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        self.batch_size = batch_size
        self.noise = noise
        self.c = c
        self.max_action = 10
        self.min_action = -10
        self.memoryTD3 = replyBufferTD3(max_size, input_dims, num_continuous_actions)
        self.memoryDQN = replyBufferDQN(max_size, input_dims, num_discrete_actions)
        self.learn_step_counter = 0
        self.warmup = warmup
        self.time_step = 0
        self.num_continuous_actions = num_continuous_actions
        self.num_discrete_actions = num_discrete_actions
        self.update_interval = update_interval
        # self.update_target_network_parameters(tau=1)
        self.lane_change = [-1.0, 0.0, 1.0]

        self.actor = Actor(layer1_dims, layer2_dims, num_actions=num_continuous_actions, name='actor')
        self.target_actor = Actor(layer1_dims, layer2_dims, num_actions=num_continuous_actions, name='target_actor')
        self.critic_1 = Critic(layer1_dims, layer2_dims, name='critic_1')
        self.critic_2 = Critic(layer1_dims, layer2_dims, name='critic_2')
        self.target_critic_1 = Critic(layer1_dims, layer2_dims, name='target_critic_1')
        self.target_critic_2 = Critic(layer1_dims, layer2_dims, name='target_critic_2')
        self.q_learning = DuelingDQN(layer1_dims, layer2_dims, num_actions=3, name='DQN')
        self.target_q_learning = DuelingDQN(layer1_dims, layer2_dims, num_actions=3, name='target_DQN')

        self.target_actor.compile(optimizer=keras.optimizers.Adam(learning_rate=self.theta),
                                  loss='mean')
        self.target_critic_1.compile(optimizer=keras.optimizers.Adam(learning_rate=self.omega),
                                     loss='mean_squared_error')
        self.target_critic_2.compile(optimizer=keras.optimizers.Adam(learning_rate=self.omega),
                                     loss='mean_squared_error')
        self.critic_1.compile(optimizer=keras.optimizers.Adam(learning_rate=self.omega),
                              loss='mean_squared_error')
        self.critic_2.compile(optimizer=keras.optimizers.Adam(learning_rate=self.omega),
                              loss='mean_squared_error')
        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=self.theta),
                           loss='mean')
        self.q_learning.compile(optimizer=keras.optimizers.Adam(learning_rate=self.alpha),
                                loss='mean_squared_error')
        self.target_q_learning.compile(optimizer=keras.optimizers.Adam(learning_rate=self.alpha),
                                       loss='mean_squared_error')

    def choose_actions(self, observation):
        if self.time_step < self.warmup:
            continuous_action = np.random.normal(scale=self.noise, size=(self.num_continuous_actions,))
            discrete_action = [np.random.randint(0, self.num_discrete_actions)]
        else:
            states = tf.convert_to_tensor([observation], dtype=tf.float32)
            continuous_action = self.actor.call(states)[0]
            actions_values = self.q_learning.call(states)
            action_index = np.argmax(actions_values)
            discrete_action = [self.lane_change[action_index]]

        continuous_action = continuous_action + np.random.normal(scale=self.noise)
        continuous_action = tf.clip_by_value(continuous_action, self.min_action, self.max_action)

        discrete_action = tf.convert_to_tensor(discrete_action, dtype=tf.float32)
        continuous_action = tf.cast(continuous_action, dtype=tf.float32)
        actions = tf.concat([continuous_action, discrete_action], axis=0)

        self.time_step += 1

        return actions

    def rememberTD3(self, states, actions, reward, next_states, done):
        return self.memoryTD3.store_transition(states, actions, reward, next_states, done)

    def rememberDQN(self, states, actions, reward, next_states, done):
        return self.memoryDQN.store_transition(states, actions, reward, next_states, done)

    def learn(self):
        if self.memoryTD3.counter < self.batch_size or self.memoryDQN.counter < self.batch_size:
            return
        else:
            states1, actions1, reward1, next_states1, done1 = self.memoryTD3.sample_from_buffer(self.batch_size)
            states2, actions2, reward2, next_states2, done2 = self.memoryDQN.sample_from_buffer(self.batch_size)

        states1 = tf.convert_to_tensor(states1, dtype=tf.float32)
        actions1 = tf.convert_to_tensor(actions1, dtype=tf.float32)
        next_states1 = tf.convert_to_tensor(next_states1, dtype=tf.float32)
        reward1 = tf.convert_to_tensor(reward1, dtype=tf.float32)

        states2 = tf.convert_to_tensor(states2, dtype=tf.float32)
        actions2 = tf.convert_to_tensor(actions2, dtype=tf.float32)
        next_states2 = tf.convert_to_tensor(next_states2, dtype=tf.float32)
        reward2 = tf.convert_to_tensor(reward2, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor(next_states1)
            target_actions = target_actions + tf.clip_by_value(np.random.normal(scale=0.2), -self.c, self.c)
            target_actions = tf.clip_by_value(target_actions, self.min_action, self.max_action)

            target_q1 = tf.squeeze(self.target_critic_1(next_states1, target_actions), 1)
            target_q2 = tf.squeeze(self.target_critic_2(next_states1, target_actions), 1)

            target_q3 = self.target_q_learning(next_states2)
            one_hot1 = tf.one_hot(np.argmax(target_q3, axis=1), depth=3, axis=-1)
            target_q3 = tf.reduce_sum(tf.multiply(target_q3, one_hot1), axis=1)

            q1 = tf.squeeze(self.critic_1(states1, actions1), 1)
            q2 = tf.squeeze(self.critic_2(states1, actions1), 1)

            q3 = self.q_learning(states2)
            one_hot2 = tf.one_hot(np.argmax(q3, axis=1), depth=3, axis=-1)
            q3 = tf.reduce_sum(tf.multiply(q3, one_hot2), axis=1)

            target_value_1 = reward1 + self.gamma * target_q1 * (1 - done1)
            target_value_2 = reward1 + self.gamma * target_q2 * (1 - done1)
            target_value_3 = reward2 + self.beta * target_q3 * (1 - done2)

            target_value = tf.math.minimum(target_value_1, target_value_2)

            critic_1_loss = keras.losses.MSE(target_value, q1)
            critic_2_loss = keras.losses.MSE(target_value, q2)
            dqn_loss = keras.losses.MSE(target_value_3, q3)

        critic_1_gradient = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_gradient = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        dqn_gradient = tape.gradient(dqn_loss, self.q_learning.trainable_variables)

        self.critic_1.optimizer.apply_gradients(zip(critic_1_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_gradient, self.critic_2.trainable_variables))
        self.q_learning.optimizer.apply_gradients(zip(dqn_gradient, self.q_learning.trainable_variables))

        self.learn_step_counter += 1

        if self.learn_step_counter % self.update_interval != 0:
            return
        else:
            with tf.GradientTape() as tape:
                new_actions = self.actor(states1)
                critic_1_q_value = self.critic_1(states1, new_actions)
                actor_loss = -tf.math.reduce_mean(critic_1_q_value)

            actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

            self.update_target_network_parameters()

    def update_target_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        target_weights = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
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

        weights = []
        target_weights = self.target_q_learning.weights
        for i, weight in enumerate(self.q_learning.weights):
            weights.append(weight * tau + target_weights[i] * (1 - tau))
        self.target_q_learning.set_weights(weights)

    def save_models(self):
        print('.......saving models.......')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)
        self.critic_2.save_weights(self.critic_2.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic_1.save_weights(self.target_critic_1.checkpoint_file)
        self.target_critic_2.save_weights(self.target_critic_2.checkpoint_file)
        self.target_q_learning.save_weights(self.target_q_learning.checkpoint_file)

    def load_models(self):
        print('.......loading models.......')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic_1.load_weights(self.critic_1.checkpoint_file)
        self.critic_2.load_weights(self.critic_2.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic_1.load_weights(self.target_critic_1.checkpoint_file)
        self.target_critic_2.load_weights(self.target_critic_2.checkpoint_file)
        self.target_q_learning.load_weights(self.target_q_learning.checkpoint_file)
