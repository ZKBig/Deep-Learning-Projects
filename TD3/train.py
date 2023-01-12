# -*- coding: utf-8 -*-
# @Description:
# @author: victor
# @create time: 2021-05-18-12:00 下午
import gym
import numpy as np
from agent import Agent
import matplotlib.pyplot as plt

def train(load_checkpoint = False):
    env = gym.make('vissim-v0')
    agent = Agent(input_dims=env.observation_space.shape,
                      num_actions=env.action_space.shape[0], theta=0.001, omega=0.001, tau=0.005, batch_size=250)
    num_games=250

    best_score = env.reward_range[0]
    score_history = []

    if load_checkpoint:
        steps = 0
        while steps <= agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            action = np.array(action)
            next_observation, reward, done, info = env.step(action[0])
            agent.remember(observation, action, reward, next_observation, done)
            steps += 1
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False

    t_list = []
    result_list = []
    t = 0
    for i in range(num_games):
        print("****************Episode%d******************" %i)
        observation = env.reset()
        print(observation)
        done = False
        score = 0
        j = 1
        while not done:
            action = agent.choose_actions(observation)
            action = np.array(action)
            print(action)
            next_observation, reward, done, info = env.step(action[0])
            score += reward
            agent.remember(observation, action, reward, next_observation, done)
            if not load_checkpoint:
                agent.learn()
            observation = next_observation
            print('episode:', i, 'step:', j, 'reward: %.1f' %reward)
            j += 1
        env.sendAction(action[0])
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print('episode', i, 'score %.1f' %score, 'avg score %.1f' % avg_score)

        t += 1
        t_list.append(t)
        result_list.append(avg_score)
        plt.plot(t_list, result_list, c='r', ls='-', marker='o', mec='b', mfc='w')
        plt.pause(0.1)

def plot_learning_curve(x, score_history):
    plt.plot(x, score_history, 'o-', color='r', label="Reward_Accumulation")
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.show()

if __name__=='__main__':
    train(False)



