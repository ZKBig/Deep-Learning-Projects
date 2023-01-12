# -*- coding: utf-8 -*-
# @Description:
# @author: victor
# @create time: 2021-03-16-4:04 下午
import gym
from ReinforcementLearning.DQN_Model import DQN_Model

env = gym.make('CartPole-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DQN_Model(n_actions=env.action_space.n, n_features=env.observation_space.shape[0], learning_rate=0.01, epsilon_max=0.9,
               update_target_iteration=100, memory_size=2000, epsilon_greed_increment=0.001,)

total_steps = 0


for i_episode in range(1000):

    observation = env.reset()
    ep_r = 0
    while True:
        env.render()

        action = RL.action_choice(observation)

        observation_, reward, done, info = env.step(action)

        # the smaller theta and closer to center the better
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = r1 + r2

        RL.transition_store(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 1000:
            RL.learn()

        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1

RL.plot_cost()