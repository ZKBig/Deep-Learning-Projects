# -*- coding: utf-8 -*-
# @Description:
# @author: victor
# @create time: 2021-03-16-9:50 上午
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
tf.set_random_seed(1)

class DQN_Model:
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.9, memory_size=500,
                 epsilon_greed_increment=0.001, epsilon_max=0.9, batch_size=32, output_graph=True, update_target_iteration=200):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = epsilon_greed_increment
        self.epsilon_max = epsilon_max
        self.epsilon = 0 if epsilon_greed_increment is not None else self.epsilon_max
        self.output_graph = output_graph
        self.update_target_iteration = update_target_iteration

        # total learning step
        self.learn_step_counter = 0

        # initialize the memory to store the transition values
        self.memory = np.zeros((self.memory_size, n_features*2+2))

        # build DQN
        self._build_network()

        # update the parameters
        self.update_target_parameters()

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        self.costs=[]

    def _build_network(self):
        # create placeholders
        self.state = tf.placeholder(tf.float32, [None, self.n_features], name="state")
        self.next_state = tf.placeholder(tf.float32, [None, self.n_features], name="next_state")
        self.reward = tf.placeholder(tf.float32, [None, ], name="reward")
        self.action = tf.placeholder(tf.int32, [None, ], name="action")

        w_initializer = tf.random_normal_initializer(0., 0.3)
        b_initializer = tf.constant_initializer(0.1)

        # build first q_value network
        with tf.variable_scope("q_value_net1"):
            state_value1 = tf.layers.dense(self.state, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer, name = "state_value1")
            self.q_value1 = tf.layers.dense(state_value1, self.n_actions, tf.nn.relu, kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer, name = "q_value1")

        # build next q_value network
        with tf.variable_scope("q_value_net2"):
            state_value2 = tf.layers.dense(self.next_state, 20, tf.nn.relu,  kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer, name = "state_value2")
            self.q_value2 = tf.layers.dense(state_value2, self.n_actions, tf.nn.relu, kernel_initializer=w_initializer,
                                            bias_initializer=b_initializer, name="q_value2")

        # build target function
        with tf.variable_scope("q_target"):
            q_target=self.reward + self.gamma * tf.reduce_max(self.q_value2, name="Qmax")
            self.q_target = tf.stop_gradient(q_target)

        # ?
        with tf.variable_scope("q_value"):
            a_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32), self.action], axis = 1)
            self.q_value = tf.gather_nd(params = self.q_value1, indices=a_indices)

        # build loss function
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_value, self.q_target, name="TD_error"))

        # train the loss function
        with tf.variable_scope("train"):
            self._train_operation = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def transition_store(self, state, action, reward, next_state):
        if not hasattr(self, "memory_counter"):
            self.memory_counter = 0

        transition = np.hstack((state, [action, reward], next_state))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter+=1

    def action_choice(self, observation):
        # add the dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform()<self.epsilon:
            # forward feed the observation and get q value for every actions
            action_values = self.sess.run(self.q_value1, feed_dict={self.state: observation})
            action=np.argmax(action_values)
        else:
            # choose the action randomly
            action = np.random.randint(0, self.n_actions)
        return action

    def update_target_parameters(self):
        p_value1_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="q_value_net1")
        p_value2_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="q_value_net2")
        self.target_update_operation = [tf.assign(t, e) for t, e in zip(p_value1_params, p_value2_params)]

    def learn(self):
        # check to replace target paramters
        if self.learn_step_counter % self.update_target_iteration==0:
            self.sess.run(self.target_update_operation)

        # sample batch from all memory
        if self.memory_counter>self.memory_size:
            sample_index = np.random.choice(self.memory_size, size = self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size = self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run([self._train_operation, self.loss], feed_dict={
            self.state: batch_memory[:, :self.n_features],
            self.action: batch_memory[:, self.n_features],
            self.reward: batch_memory[:, self.n_features+1],
            self.next_state: batch_memory[:, -self.n_features:]
        })
        self.costs.append(cost)

        # increase epsilon
        self.epsilon = self.epsilon+self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter +=1

    def plot_cost(self):
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()





