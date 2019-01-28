<a name="report"></a>
# Reinforcement Learning Collaboration and Competition: Project Report

## Report Table of Contents

[RL Environment](#environment)

[Algorithm](#algorithm)

[Hyperparameters](#hyperparameters)

[Network Architecture](#network)

[Next Steps](#nextsteps)

<a name="environment"></a>
## The Reinforcement Learning (RL) Environment

For [Project 3](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet) of Udacity's [Deep Reinforcement Learning Nanodegree](https://github.com/udacity/deep-reinforcement-learning), we were tasked with teaching a pair of agents to play Tennis in an environment configured by Udacity on [Unity's ML-Agents platform](https://github.com/Unity-Technologies/ml-agents).  

According to the description provided by Udacity, two agents control rackets to bounce a ball over a net with the objective to keep a ball in play.  If the ball gets over the net, the agent receives a reward of +0.1.  If the ball hits the ground or goes out of bounds, the reward is then -0.1.  

The observation space _for each agent_ consists of 8 variables pertaining to the position and velocity of ball and racket.  Two continuous actions are available, corresponding to movement towards/away from the net and jumping.  

In order to solve the environment, agents must get an average score of +0.5 over the most recent 100 consecutive episodes, calculated by taking the maximum score over both agents per episode.  

For further information, see Udacity's [project github repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet).

<a name="algorithm"></a>
## The Algorithm

In this project, we explored the MADDPG policy, which is essentially a multiagent implementation of Deep Deterministic Policy Gradient ([DDPG](https://arxiv.org/abs/1509.02971)).  

The MADDPG algorithm successfully trained in 3257 episodes as determined by a running average of the scores of previous 100 episodes over 0.5.  

![alt text](https://github.com/cipher813/rl_multiagent/blob/master/charts/multiagent_results.png "Tennis Results with MADDPG")

**DDPG**

DDPG was introduced by DeepMind in 2016 as an adaptation of Deep Q-Learning (DQN) to the continuous action domain.  The algorithm is described as an "actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces."  While DQN solves problems with high-dimensional observation (state) spaces, it can only handle discrete, low-dimensional action spaces.  

For the multi-agent implementation, we can [share experience amongst agents to accelerate learning](https://ai.googleblog.com/2016/10/how-robots-can-acquire-new-skills-from.html).  We do this by using the same memory ReplayBuffer for all agents.

The base model in this repo utilizes the DDPG algorithm.  

**MADDPG**

[...]

<a name="hyperparameters"></a>
## Hyperparameters

Hyperparameters are found in the same file as the implementation in which it is deployed.  For [MADDPG](https://github.com/cipher813/rl_multiagent/blob/master/scripts/agents/MADDPG.py), key hyperparameters include:

[...]

<a name="network"></a>
# Neural Network Architecture

[...]

<a name="nextsteps"></a>
# Next Steps

[...]  
