# Reinforcement Learning: Collaboration and Competition

## Repo Table of Contents

[Project Overview](#overview)

[Environment Setup](#setup)

[Model](#model)

[Resources](#resources)

[Report](https://github.com/cipher813/rl_multiagent/blob/master/report.md#report)

- [RL Environment](https://github.com/cipher813/rl_multiagent/blob/master/report.md#environment)

- [Algorithm](https://github.com/cipher813/rl_multiagent/blob/master/report.md#algorithm)

- [Hyperparameters](https://github.com/cipher813/rl_multiagent/blob/master/report.md#hyperparameters)

- [Network Architecture](https://github.com/cipher813/rl_multiagent/blob/master/report.md#network)

- [Next Steps](https://github.com/cipher813/rl_multiagent/blob/master/report.md#nextsteps)

<a name="overview"></a>
## Project Overview

For [Project 3](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet) of Udacity's [Deep Reinforcement Learning Nanodegree](https://github.com/udacity/deep-reinforcement-learning), we were tasked with [...] configured by Udacity on [Unity's ML-Agents platform](https://github.com/Unity-Technologies/ml-agents).  

For further information on the environment, see the accompanying project [Report](https://github.com/cipher813/rl_multiagent/blob/master/report.md) or Udacity's [project github repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet).  

In this project, we primarily explored the MADDPG algorithm, which is essentially the multi-agent version of Deep Deterministic Policy Gradient [DDPG](https://arxiv.org/abs/1509.02971).     

The algorithms are further explained in the accompanying [Report](https://github.com/cipher813/rl_multiagent/blob/master/report.md).

<a name="setup"></a>
## Environment Setup

To set up the python (conda) environment, in the root directory, type:

`conda env update --file=environment_drlnd.yml`

This requires installation of [OpenAI Gym](https://github.com/openai/gym) and Unity's [ML-Agents](https://github.com/Unity-Technologies/ml-agents).   

In the root directory, run `python setup.py` to set up directories and download specified environments.  When running this file, make sure you have the full path to your root repo folder readily available (and end the input with a "/").

If you need to further review and access environment implementation, visit the project repo [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet).

<a name="model"></a>
## The Model

The key files in this repo include:

### Scripts

[main.py](https://github.com/cipher813/rl_multiagent/tree/master/scripts)
Execute this script to train the MADDPG algorithm in the Unity Tennis environment.  

[environment.py](https://github.com/cipher813/rl_multiagent/tree/master/scripts)
[...]

[statistics.py](https://github.com/cipher813/rl_multiagent/tree/master/scripts)
[...]

[agents](https://github.com/cipher813/rl_multiagent/tree/master/scripts) folder
[...]
Contains the MADDPG algorithm.  See the accompanying [Report](https://github.com/cipher813/rl_multiagent/blob/master/report.md) for additional details on the agent implementation.

To train the agent, first open main.py in your favorite text editor (ie `nano main.py` or `vi main.py`).  Make sure the path to the root repo folder is correct and that the proper environments and agents (policies) are selected.  Then, in the command line run:

`source activate drlnd` # to activate python (conda) environment
`python main.py` # to train the environment and agent (policy)

To start tensorboard, in the root directory run `tensorboard --logdir=results`

### Notebooks

[rl3_results.ipynb](https://github.com/cipher813/rl_multiagent/tree/master/notebooks)

Charts the results from model results file.

### Results

Contains the "solved" [model weights](https://github.com/cipher813/rl_multiagent/tree/master/results).  

<a name="resources"></a>
## Resources

Credit goes to previous implementations of this project authored by:

[Udacity](https://github.com/udacity/deep-reinforcement-learning)

[danielnbarbosa](https://github.com/danielnbarbosa/drlnd_collaboration_and_competition)
