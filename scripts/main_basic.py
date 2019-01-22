"""
Project 3: Multiagent
Udacity Deep Reinforcement Learning Nanodegree
Brian McMahon
January 2019

Code inspired by implementation at https://github.com/danielnbarbosa/drlnd_collaboration_and_competition

Reimplementation of main runner file.
"""

import re
import time
import socket
import datetime
import numpy as np

import torch
# from unityagents import UnityEnvironment
from environment import Unity_Environment
from agents.MADDPG import MADDPG

PATH = "/Volumes/BC_Clutch/Dropbox/DeepRLND/rl_multiagent/"
timestamp = re.sub(r"\D","",str(datetime.datetime.now()))[:12]

def calc_runtime(seconds):
    """ Calculates runtime """
    h = int(seconds//(60*60))
    seconds = seconds - h*60*60
    m = int(seconds//60)
    s = int(round(seconds - m*60,0))
    return "{:02d}:{:02d}:{:02d}".format(h,m,s)

def train(PATH, environment, timestamp, n_episodes=10000, max_t=1000, score_threshold=0.5):
    """Train with MADDPG."""
    start = time.time()
    total_scores = []
    # env_path = PATH + f"data/{env_path}"
    # env = UnityEnvironment(file_name=env_path)
    # brain_name = env.brain_names[0]
    # brain = env.brains[brain_name]
    # env_info = env.reset(train_mode=True)[brain_name]
#     num_agents = len(env_info.agents)
#     states = env_info.vector_observations
#     state_size = states.shape[1]
#     action_size = brain.vector_action_space_size
    agent = MADDPG()
    for i_episode in range(1, n_episodes+1):
        scores = []
        states = environment.reset()
        # states = env_info.vector_observations
        # env_info = env.reset(train_mode=True)[brain_name]
#         states = env_info.vector_observations
#         scores = np.zeros(num_agents)
#         agent.reset()
        for t in range(max_t):
            if agent.evaluation_only:
                action = agent.act(states, add_noise=False)
            else:
                action = agent.act(states)
            actions = agent.act(states)
            # env_info = env.step(actions)[brain_name]        # send the action to the environment
            next_states, rewards, dones = environment.step(actions)
            # next_states = env_info.vector_observations
            # rewards = env_info.rewards                   # get the reward
            # dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            scores.append(rewards)
#             scores += env_info.rewards
            if np.any(dones):
                break
        score_length = len(total_scores) if len(total_scores)<100 else 100
        mean_score = np.mean(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        total_scores.append(mean_score)
        total_average_score = np.mean(total_scores[-score_length:])
        end = time.time()
        print(f'\rEpisode {i_episode}\tScore TAS/Mean/Max/Min: {total_average_score:.2f}/{mean_score:.2f}/{max_score:.2f}/{min_score:.2f}\t{calc_runtime(end-start)}',end=" ")
        per_agent_rewards = []
        for i in range(agent.num_agents):
            per_agent_rewards.append(np.sum(rewards[i]))

        save_name = f"../results/{timestamp}_episode_{i_episode}"
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tScore TAS/Mean/Max/Min: {total_average_score:.2f}/{mean_score:.2f}/{max_score:.2f}/{min_score:.2f}\t{calc_runtime(end-start)}')
            for save_agent in agent.agents:
                torch.save(save_agent.actor.state_dict(), save_name + "_actor.pth")
                torch.save(save_agent.critic.state_dict(), save_name + "_critic.pth")

        if total_average_score>score_threshold:
            print(f"Solved in {i_episode} and {calc_runtime(end-start)}")
            for i, save_agent in enumerate(agent.agents):
                torch.save(save_agent.actor.state_dict(), save_name + "solved_" + str(i) + "_actor.pth")
                torch.save(save_agent.critic.state_dict(), save_name + "solved_" + str(i) + "_critic.pth")
            break
    env.close()
    return total_scores

environment = Unity_Environment(evaluation_only=False)
train(PATH, environment, timestamp)
