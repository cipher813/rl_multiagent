"""
Project 3: Multiagent
Udacity Deep Reinforcement Learning Nanodegree
Brian McMahon
January 2019

Code inspired by implmentation at https://github.com/danielnbarbosa/drlnd_collaboration_and_competition
"""
import torch
import numpy as np
from statistics import Stats
from environment import UnityMLVectorMultiAgent
from agents.MADDPG import MADDPG

def train(environment, agent, n_episodes=10000, max_t=1000, solve_score=0.5):
    stats = Stats()
    stats_format = "Buffer: {:6}\tNoiseW: {:.4}"


    for i_episode in range(1, n_episodes+1):
        rewards = []
        state = environment.reset()
#         env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
#         states = env_info.vector_observations                  # get the current state (for each agent)
#         scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done = environment.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            rewards.append(reward)
            if any(done):
                break

        # every episode
        buffer_len = len(agent.memory)
        per_agent_rewards = []
        for i in range(agent.num_agents):
            per_agent_reward = 0
            for step in rewards:
                per_agent_reward += step[i]
            per_agent_rewards.append(per_agent_reward)
        stats.update(t,[np.max(per_agent_rewards)], i_episode) # use max over all agents as episode reward
        stats.print_episode(i_episode, t, stats_format, buffer_len, agent.noise_weight,
                            agent.agents[0].critic_loss, agent.agents[1].critic_loss,
                            agent.agents[0].actor_loss, agent.agents[1].actor_loss,
                            agent.agents[0].noise_val, agent.agents[1].noise_val,
                            per_agent_rewards[0], per_agent_rewards[1])

        # every epoch (100 episodes)
        if i_episode % 100 == 0:
            stats.print_epoch(i_episode, stats_format, buffer_len, agent.noise_weight)
            save_name = f"../checkpoints/episode_{i_episode}"
            for i, save_agent in enumerate(agent.agents):
                torch.save(save_agent.actor.state_dict(), save_name + str(i) + "_actor.pth")
                torch.save(save_agent.critic.state_dict(), save_name + str(i) + "_critic.pth")

        # if solved
        if stats.is_solved(i_episode, solve_score):
            stats.print_solve(i_episode, stats_format, buffer_len, agent.noise_weight)
            save_name = "../checkpoints/solved_"
            for i, save_agent in enumerate(agent.agents):
                torch.save(save_agent.actor.state_dict(),save_name+str(i)+"_actor.pth")
                torch.save(save_agent.critic.state_dict(),save_name+str(i)+"_critic.pth")
            break 

environment = UnityMLVectorMultiAgent()
agent = MADDPG()
train(environment, agent)
