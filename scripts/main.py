"""
Project 3: Multiagent Collaboration and Competition
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
from collections import deque

import torch
from environment import Unity_Multiagent
from agents.MADDPG import MADDPG
import statistics

PATH = "/Volumes/BC_Clutch/Dropbox/DeepRLND/rl_multiagent/"
timestamp = re.sub(r"\D","",str(datetime.datetime.now()))[:12]

def train(PATH, environment, agent, timestamp, n_episodes=10000, max_t=1000, score_threshold=0.5):
    """Train with MADDPG."""
    start = time.time()
    total_scores = deque(maxlen=100)
    stats = statistics.Stats()
    stats_format = "Buffer: {:6} NoiseW: {:.4}"

    for i_episode in range(1, n_episodes+1):
        scores = []
        states = environment.reset()
        for t in range(max_t):
            if agent.evaluation_only:
                action = agent.act(states, add_noise=False)
            else:
                action = agent.act(states)
            actions = agent.act(states)
            next_states, rewards, dones = environment.step(actions)
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            scores.append(rewards)
            if np.any(dones):
                break
        buffer_len = len(agent.memory)
        per_agent_rewards = []
        for i in range(agent.num_agents):
            per_agent_reward = 0
            for step in scores:
                per_agent_reward += step[i]
            per_agent_rewards.append(per_agent_reward)
        stats.update(t, [np.max(per_agent_rewards)], i_episode) # use max over all agents as episode reward
        stats.print_episode(i_episode, t, stats_format, buffer_len, agent.noise_weight,
                            agent.agents[0].critic_loss, agent.agents[1].critic_loss,
                            agent.agents[0].actor_loss, agent.agents[1].actor_loss,
                            agent.agents[0].noise_val, agent.agents[1].noise_val,
                            per_agent_rewards[0],per_agent_rewards[1])

        if i_episode % 500 == 0:
            stats.print_epoch(i_episode, stats_format, buffer_len, agent.noise_weight)
            save_name = f"../results/{timestamp}_episode_{i_episode}"
            for i,save_agent in enumerate(agent.agents):
                torch.save(save_agent.actor.state_dict(), save_name + f"A{i}_actor.pth")
                torch.save(save_agent.critic.state_dict(), save_name + f"A{i}_critic.pth")

        # if total_average_score>score_threshold:
        if stats.is_solved(i_episode, score_threshold):
            stats.print_solve(i_episode, stats_format, buffer_len, agent.noise_weight)
            save_name = f"../results/{timestamp}_solved_episode_{i_episode}"
            for i, save_agent in enumerate(agent.agents):
                torch.save(save_agent.actor.state_dict(), save_name + f"A{i}_actor.pth")
                torch.save(save_agent.critic.state_dict(), save_name + f"A{i}_critic.pth")
            break

environment = Unity_Multiagent(evaluation_only=False)
agent = MADDPG()
train(PATH, environment, agent, timestamp)
