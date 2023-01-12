# -*- coding: utf-8 -*-
# @Description:
# @author: victor
# @create time: 2021-03-17-11:49 上午

import tensorflow as tf
import numpy as np
import gym
import time

np.random.seed(1)
tf.set_random_seed(1)

# hyper parameters
MAX_EPISODES = 300
MAX_EP_STEPS = 500
ACTOR_LEARNING_RATE = 0.001
CRITIC_LEARNING_RATE = 0.001
GAMMA = 0.9
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

# actor system
class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, update_target_iteration=500):
        self.sess = sess
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.update_target_iteration = update_target_iteration
        self.learn_step_counter = 0

        with tf.variable_scope('Actor'):
            # get the current action
            self.current_action = self._build_net(S, scope="value_net", trainable=True)
            # get the next action
            self.next_action = self._build_net(N_S, scope="target_net", trainable=False)

        self._update_target_parameters()

    def _build_net(self, state, scope, trainable):
        """
        construct the general actor neural network, the value of the units in hidden layer1 is fixed as 30

        :param state: the current state of the environment
        :param scope: the type of the network
        :param trainable: determine whether the network participates in the train or not
        :return: the scaled value of the action
        """
        with tf.variable_scope(scope):
            W = tf.random_normal_initializer(0., 0.3)
            b = tf.constant_initializer(0.1)

            layer1 = tf.layers.dense(state, 30, activation=tf.nn.relu, kernel_initializer=W,
                                     bias_initializer=b, name="layer1", trainable=trainable)

            with tf.variable_scope("action"):
                actions = tf.layers.dense(layer1, self.action_dim, activation=tf.nn.tanh, kernel_initializer=W,
                                          bias_initializer=b, name="action", trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name="scaled_action")
            return scaled_a

    def action_choice(self, state):
        """
        choose a determined action according to the input state

        :param state: the state
        :return: the value of one action
        """
        state = state[np.newaxis, :]
        action_value = self.sess.run(self.current_action, feed_dict={S: state})
        # print(action_value)
        return action_value[0]

    def _update_target_parameters(self):
        """
        update the target network parameters

        :return: None
        """
        self.value_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/value_net')
        self.target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')
        self.target_update_operation = [tf.assign(t, e) for t, e in zip(self.target_params, self.value_params)]

    def policy_grads(self, q_grads):
        """
        Implement the policy gradients by using back propagation

        :param q_grads: the value gradient obtained from the critic network
        :return: None
        """
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.current_action, xs=self.value_params, grad_ys=q_grads)

        with tf.variable_scope("Actor_train"):
            self.train_operation = tf.train.AdamOptimizer(-self.learning_rate).apply_gradients(
                zip(self.policy_grads, self.value_params))

    def learn(self, state):
        """
        execute the learning process, note that update the parameters of the target
        according to the count of the step

        :param state: the state that is needed to be trained
        :return: None
        """
        self.sess.run(self.train_operation, feed_dict={S: state})
        if self.learn_step_counter % self.update_target_iteration == 0:
            self.sess.run(self.target_update_operation)
            self.learn_step_counter += 1

# critic system
class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma,
                current_action, next_action, update_target_iteration=500):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.current_action = current_action
        self.next_action = next_action
        self.update_target_iteration = update_target_iteration
        self.learn_step_counter = 0

        with tf.variable_scope("Critic"):
            self.current_action = tf.stop_gradient(current_action)

            self.current_q = self._build_network(S, self.current_action, "value_net", trainable=True)
            self.next_q = self._build_network(N_S, self.next_action, "target_net", trainable=False)

        self._update_target_parameters()

        # calculate the target_q
        with tf.variable_scope("target_q"):
            self.target_q = R + self.gamma * self.next_q

        # calculate the TD error
        with tf.variable_scope("TD_error"):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.current_q))

        # train the critic neural network
        with tf.variable_scope('Critic_train'):
            self.train_operation = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # back propagate the critic neural network
        with tf.variable_scope('a_grad'):
            self.q_grads = tf.gradients(self.current_q, self.current_action)[0]

    def _build_network(self, state, action, scope, trainable):
        """
        construct the general actor neural network, the value of the units in hidden layer1 is fixed as 30

        :param state: the current state of the environment
        :param scope: the type of the network
        :param trainable: determine whether the network participates in the train or not
        :return: the scaled value of the action
        """
        with tf.variable_scope(scope):
            # build the first layer
            with tf.variable_scope("layer"):
                W_state = tf.get_variable("W_state", [self.state_dim, 30],
                                          initializer=tf.random_normal_initializer(0., 0.1), trainable=trainable)
                W_action = tf.get_variable("W_action", [self.action_dim, 30],
                                           initializer=tf.random_normal_initializer(0., 0.1), trainable=trainable)
                b = tf.get_variable('b1', [1, 30], initializer=tf.constant_initializer(0.1), trainable=trainable)
                layer1 = tf.nn.relu(tf.matmul(state, W_state) + tf.matmul(action, W_action) + b)

            # build the out layer
            with tf.variable_scope("q_value"):
                q = tf.layers.dense(layer1, 1, kernel_initializer=tf.random_normal_initializer(0., 0.1),
                                    bias_initializer=tf.constant_initializer(0.1), trainable=trainable)
            return q

    def _update_target_parameters(self):
        """
        update the target network parameters

        :return: None
        """
        self.value_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/value_net')
        self.target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')
        self.target_update_operation = [tf.assign(t, e) for t, e in zip(self.target_params, self.value_params)]

    def learn(self, state, action, reward, next_state):
        """
        execute the learning process, note that update the parameters of the target
        according to the count of the step

        :param state: the current state
        :param action: the current action in the state
        :param reward: the current reword of the action
        :param next_state: the next state
        :return: None
        """
        self.sess.run(self.train_operation, feed_dict={S: state, self.current_action: action,
                      R: reward, N_S: next_state})
        if self.learn_step_counter % self.update_target_iteration == 0:
            self.sess.run(self.target_update_operation)
            self.learn_step_counter += 1

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, current_state, current_action, reward, next_state):
        transition = np.hstack(( current_state, current_action, [reward], next_state))
        # replace the old memory with new memory
        index = self.pointer % self.capacity
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    env.seed(1)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high

    # construct placeholders
    with tf.name_scope('current_state'):
        S = tf.placeholder(tf.float32, shape=[None, state_dim], name='current_state')
    with tf.name_scope('reward'):
        R = tf.placeholder(tf.float32, [None, 1], name='reward')
    with tf.name_scope('next_state'):
        N_S = tf.placeholder(tf.float32, shape=[None, state_dim], name='next_state')

    sess = tf.Session()

    actor = Actor(sess, action_dim=action_dim, action_bound=action_bound, learning_rate=ACTOR_LEARNING_RATE)
    critic = Critic(sess, state_dim=state_dim, action_dim=action_dim, learning_rate=CRITIC_LEARNING_RATE, gamma=GAMMA,
                    current_action=actor.current_action, next_action=actor.next_action)
    actor.policy_grads(critic.q_grads)

    sess.run(tf.global_variables_initializer())

    M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)

    tf.summary.FileWriter("logs/", sess.graph)

    var = 3  # control exploration

    t = time.time()
    for i in range(MAX_EPISODES):
        s = env.reset()
        # print(s.shape)
        ep_reward = 0

        for j in range(MAX_EP_STEPS):
            env.render()

            # Add exploration noise
            a = actor.action_choice(s)
            a = np.clip(np.random.normal(a, var), -2, 2)  # add randomness to action selection for exploration
            s_, r, done, info = env.step(a)

            M.store_transition(s, a, r / 10, s_)

            if M.pointer > MEMORY_CAPACITY:
                var *= .9995  # decay the action randomness
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :state_dim]
                b_a = b_M[:, state_dim: state_dim + action_dim]
                b_r = b_M[:, -state_dim - 1: -state_dim]
                b_s_ = b_M[:, -state_dim:]

                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)

            s = s_
            ep_reward += r

            if j == MAX_EP_STEPS - 1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                # if ep_reward > -300:
                #     RENDER = True
                break

    print('Running time: ', time.time() - t)




