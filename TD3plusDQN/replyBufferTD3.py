# -*- coding: utf-8 -*-
# @Description:
# @author: victor
# @create time: 2021-05-19-10:40 上午
import numpy as np


class replyBufferTD3:
    def __init__(self, max_size, input_dims, num_actions):
        self.max_size = max_size
        self.counter = 0
        self.state_memory = np.zeros(shape=(self.max_size, *input_dims))
        self.next_state_memory = np.zeros(shape=(self.max_size, *input_dims))
        self.actions_memory = np.zeros(shape=(self.max_size, num_actions))
        self.reward_memory = np.zeros(shape=self.max_size)
        self.terminal_memory = np.zeros(shape=self.max_size, dtype=np.bool)

    def store_transition(self, state, actions, reward, next_state, done):
        index = self.counter % self.max_size
        self.state_memory[index] = state
        self.actions_memory[index] = actions
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = done
        self.counter += 1

    def sample_from_buffer(self, batch_size):
        max_memory_index = min(self.max_size, self.counter)
        batch = np.random.choice(max_memory_index, batch_size)

        states = self.state_memory[batch]
        actions = self.actions_memory[batch]
        reward = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        done = self.terminal_memory[batch]

        return states, actions, reward, next_states, done
