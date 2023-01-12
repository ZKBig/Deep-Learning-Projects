# -*- coding: utf-8 -*-
# @Description:
# @author: victor
# @create time: 2021-06-02-10:04 下午
import gym
import numpy as np
from agent import Agent

env = gym.make('Pendulum-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = Agent(input_dims=env.observation_space.shape[0],
           num_actions = env.action_space.shape[0],
           max_action=float(env.action_space.high[0]),
           theta=3e-4, omega=3e-4, tau=0.005, batch_size=128)

total_steps = 0
start_timesteps = 1e4

for i_episode in range(1000):

    observation = env.reset()
    ep_r = 0
    while True:
        env.render()
        if total_steps < start_timesteps:
            print("*****************")
            action = env.action_space.sample()
        else:
            action = np.array(RL.choose_actions(observation))
        print(action)
        # action = np.clip(np.random.normal(action, 3), -2, 2)

        observation_, reward, done, info = env.step(action)

        RL.remember(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 1000:
            RL.learn(done)

        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2))
            break

        observation = observation_
        total_steps += 1


