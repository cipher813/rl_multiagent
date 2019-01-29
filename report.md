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

In order to solve the environment, agents must get an average score of +0.5 over the most recent 100 consecutive episodes, calculated by taking the maximum score over both agents per episode.  As in, the max of the sum of rewards for each agent is taken as the score for each episode.   

For further information, see Udacity's [project github repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet).

<a name="algorithm"></a>
## The Algorithm

In this project, we explored the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) policy as published by [OpenAI](https://arxiv.org/pdf/1706.02275.pdf), which is essentially a multiagent implementation of DeepMind's Deep Deterministic Policy Gradient ([DDPG](https://arxiv.org/abs/1509.02971)).  

The MADDPG algorithm successfully trained in 2676 episodes as determined by a running average of the scores of previous 100 episodes over 0.5.  

![alt text](https://github.com/cipher813/rl_multiagent/blob/master/charts/Average_Reward.png "Average Reward")

![alt text](https://github.com/cipher813/rl_multiagent/blob/master/charts/Reward.png "Actual Reward")


**DDPG**

DDPG was introduced by DeepMind in 2016 as an adaptation of Deep Q-Learning (DQN) to the continuous action domain.  The algorithm is described as an "actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces."  While DQN solves problems with high-dimensional observation (state) spaces, it can only handle discrete, low-dimensional action spaces.  

**MADDPG**

MADDPG with introduced by OpenAi in January 2018, adding mult-agent capability (in both cooperative and competitive capacities) to the DDPG algorithm.  While training, the agents cooperate by sharing memory of experiences in a shared ReplayBuffer.  Upon completion of training, the model then determines actions locally and independently.  

For this multi-agent implementation, we [share experience amongst agents to accelerate learning](https://ai.googleblog.com/2016/10/how-robots-can-acquire-new-skills-from.html) by using the same memory ReplayBuffer for all agents.

<a name="hyperparameters"></a>
## Hyperparameters

Hyperparameters are found in the same file as the implementation in which it is deployed.  For [MADDPG](https://github.com/cipher813/rl_multiagent/blob/master/scripts/agents/MADDPG.py), key hyperparameters include:

**Buffer Size.**  The ReplayBuffer memory size, as in the number of experiences that are remembered.   

**Batch Size.**  The size of each training batch sampled at a time.    

**Gamma.**  Discount factor for discounting past experiences (most recent experiences more highly rewarded as in less discount is applied).  

**Tau.**  For soft update of target parameters.  

**Learning Rate (Actor and Critic).**  Learning rates of actor and critic.  

**Weight Decay.**  L2 weight decay, which is not used (set at 0.0) as it negatively impacts the sparse data signal in this architecture.  

**Update Every** The frequency experiences are added to the ReplayBuffer, in timesteps.  

**Noise Start** Add noise to agent actions.  

**Noise Decay** The magnitude of the noise added.  

<a name="network"></a>
# Neural Network Architecture

Agent (as policy model) and critic (as value model) each use a DDPG architecture, with three layers including hidden layers of 400 and 300 nodes.  Adam is used as the optimizer, and ReLU is used as the per-layer activations.  

Each actor takes a state input for a single agent, whereas each critic takes a concatenation of states and actions from all agents.  Local and target models are initialized with the same weights by using the same seed.  

<a name="nextsteps"></a>
# Next Steps

**Environments**

I would like to implement the MADDPG algorithm in different environments, such as the [Unity Soccer environment](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet) provided by Udacity.  Further environments can also be developed for testing using Unity's [ML Agents platform](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Getting-Started-with-Balance-Ball.md).

**Algorithms**

I would also like to investigate other relevant multi-agent algorithms and subject the Tennis environment to these, to compare and contrast performance against this MADDPG implementation.  
