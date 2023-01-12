# -*- coding: utf-8 -*-
# @Description:
# @author: victor
# @create time: 2021-03-16-9:02 下午
import numpy as np
import tensorflow as tf
import gym

np.random.seed(2)
tf.set_random_seed(2)

class Critic(object):
    def _init_(self, sess, n_features, learning_rate=0.01, gamma=0.9):
        self.sess = sess
        self.n_features = n_features
        self.gamma = gamma
        self.learning_rate = learning_rate

    def _build_network(self):
        self.state = tf.placeholder(tf.float32, [None, self.n_features], "state")
        self.q_value2 = tf.placeholder(tf.float32, [None, 1], "q_value2")
        self.reward = tf.placeholder(tf.float32, None, "current_reward")

        with tf.variable_scope('Critic'):
            layer1 = tf.layers.dense(inputs=self.state, units=20, activation=tf.nn.relu, kernel_regularizer=tf.random_normal_initializer(0., .1),
                                 bias_initializer=tf.constant_initializer(0.1), name="layer1")
            self.q_value = tf.layer.dense(inputs=layer1, units=1, activation=None,  kernel_regularizer=tf.random_normal_initializer(0., .1),
                                 bias_initializer=tf.constant_initializer(0.1), name="q_value1")

        # build target function
        with tf.variable_scope("q_target"):
            q_target = self.reward + self.gamma * tf.reduce_max(self.q_value, name="Qmax")
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('squared_TD_error'):
            self.td_error = tf.reduce_mean(self.q_target-self.reward+self.gamma*self.q_value2)
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_value2, name="TD_error"))

        # train the loss function
        with tf.variable_scope("train"):
            self._train_operation = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def learn(self, state, reward, next_state):
        state=state[np.newaxis, :]
        next_state=next_state[np.newaxis, :]

        q_value2 = self.sess.run(self.q_value, {self.state: next_state})
        td_error, _ = self.sess.run([self.td_error, self._train_operation], {
            self.q_value2: q_value2, self.state: state, self.reward: reward
        })

        return td_error

class Actor(object):
    def _init_(self, sess, n_features, n_actions, learning_rate=0.001):
        self.sess = sess
        self.n_features = n_features
        self.n_actions = n_actions


    def _build_network(self):
        self.state = tf.placeholder(tf.float32, [None, self.n_features], "state")
        self.action = tf.placeholder(tf.int32, None, "action")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")

        with tf.variable_scope('Actor'):
            layer1 = tf.layers.dense(inputs=self.state, units=20, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0., .1),
                                 bias_initializer=tf.constant_initializer(0.1), name="layer1")
            self.acts_prob = tf.layer.dense(inputs=layer1, units=self.n_actions, activation=tf.nn.softmax, kernel_initializer=tf.random_normal_initializer(0., .1),
                                 bias_initializer=tf.constant_initializer(0.1), name="acts_prob")

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, state, action, td_error):
        state = state[np.newaxis, :]
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict={
            self.state: state, self.action: action, self.td_error: td_error
        })
        return exp_v

    def action_choice(self, state):
        state = state[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.state: state})
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())



