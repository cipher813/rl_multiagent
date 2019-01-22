"""
Project 3: Multiagent
Udacity Deep Reinforcement Learning Nanodegree
Brian McMahon
January 2019

Code inspired by DDPG implmentation at
https://github.com/partha746/DRLND_P2_Reacher_EnV
and MADDPG implementation at https://github.com/danielnbarbosa/drlnd_collaboration_and_competition
"""
import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG:
    """Interacts with and learns from the environment."""

    def __init__(self, action_size=2, random_seed=0, load_file=None, num_agents=2,
                 buffer_size=int(3e4),batch_size=128,gamma=0.99, update_every=2,
                 noise_start=1.0, noise_decay=1.0, evaluation_only=False):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.update_every = update_every
        self.gamma = gamma
        self.noise_weight = noise_start
        self.noise_decay = noise_decay
        self.timestep = 0
        self.evaluation_only = evaluation_only
        self.seed = random.seed(random_seed)

        # create 2 agents, each with own actor and critic
        models = [LowDim2x(num_agents=num_agents) for _ in range(num_agents)]
        self.agents = [DDPG(0,models[0],load_file=None),DDPG(1, models[1], load_file=None)]

        # shared replay buffer
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, random_seed)

    def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones):#, alpha=ALPHA, beta=BETA):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # reshape 2x24 into 1x48 dim vector
        all_states = all_states.reshape(1,-1)
        all_next_states = all_next_states.reshape(1,-1)

        self.memory.add(all_states, all_actions, all_rewards, all_next_states, all_dones)

        self.timestep = (self.timestep + 1) % self.update_every
        if self.timestep ==0 and self.evaluation_only==False:
            # if enough samples are available in memory, get random subset and learn
            if len(self.memory)>self.batch_size:
                # each agent does its own sampling from replay buffer
                experiences = [self.memory.sample() for _ in range(self.num_agents)]
                self.learn(experiences, self.gamma)

    def act(self, all_states, add_noise=True): # act
        """Returns actions for given state as per current policy."""
        all_actions = []

        for agent, state in zip(self.agents, all_states):
            action = agent.act(state, noise_weight=self.noise_weight, add_noise=True)
            self.noise_weight *= self.noise_decay
            all_actions.append(action)
        return np.array(all_actions).reshape(1,-1) # reshape 2x2 into 1x4 dim vector

    def learn(self, experiences, gamma): # train
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        # each agent uses its own actor to calculate next actions
        all_next_actions = []
        for i, agent in enumerate(self.agents):
            _, _, _, next_states, _ = experiences[i] # states, actions, rewards, next_states, dones
            agent_id = torch.tensor([i]).to(device)
            next_state = next_states.reshape(-1,2,24).index_select(1,agent_id).squeeze(1)
            next_action = agent.actor_target(next_state)
            all_next_actions.append(next_action)

        # each agent uses its own actor to calculate actions
        all_actions = []
        for i, agent in enumerate(self.agents):
            states, _, _, _, _ = experiences[i] # states, actions, rewards, next_states, dones
            agent_id = torch.tensor([i]).to(device)
            state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            action = agent.actor(state)
            all_actions.append(action)
        # each agent learns from experience sample
        for i, agent in enumerate(self.agents):
            agent.learn(i,experiences[i],gamma, all_next_actions, all_actions)

class DDPG():
    """Interacts with and learns from the environment."""

    def __init__(self, id, model, action_size=2, random_seed=0, load_file=None,
                 tau=1e-3, lr_actor=1e-4, lr_critic=1e-3, weight_decay=0.0):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        random.seed(random_seed)
        self.id = id
        self.action_size = action_size
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        # track stats for tensorboard logging
        self.critic_loss = 0
        self.actor_loss = 0
        self.noise_val = 0

        # Actor Network (w/ Target Network)
        self.actor = model.actor
        self.actor_target = model.actor_target
        # self.actor = Actor(state_size, action_size, random_seed).to(device) #max_action,
        # self.actor_target = Actor(state_size, action_size, random_seed).to(device) #max_action,
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic = model.critic
        self.critic_target = model.critic_target
        # self.critic = Critic(state_size, action_size, random_seed).to(device)
        # self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        # self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def act(self, state, noise_weight=1.0, add_noise=True): # act
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        if add_noise:
            self.noise_val = self.noise.sample()*noise_weight
            action += self.noise_val
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, agent_id, experiences, gamma, all_next_actions, all_actions): # train
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # get predicted next state actions and Q values from target models
        self.critic_optimizer.zero_grad()
        agent_id = torch.tensor([agent_id]).to(device)
        actions_next = torch.cat(all_next_actions, dim=1).to(device)
        with torch.no_grad():
            q_targets_next = self.critic_target(next_states, actions_next)
        # compute Q targets for current states (y_i)
        q_expected = self.critic(states, actions)

        # q_targets = reward of this timestep + discount * Q(st+1, at+1) from target network
        q_targets = rewards.index_select(1, agent_id) + (gamma * q_targets_next * (1 - dones.index_select(1, agent_id)))

        # compute critic loss
        critic_loss = F.mse_loss(q_expected, q_targets.detach())
        self.critic_loss = critic_loss.item() # for tensorboard logging

        # minimize loss
        critic_loss.backward()
        # nn.utils.clip_grad_norm_(self.critic.parameters(),1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # compute actor loss
        self.actor_optimizer.zero_grad()

        # detach actions from other agents
        actions_pred = [actions if i == self.id else actions.detach() for i, actions in enumerate(all_actions)]
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        actor_loss = -self.critic(states, actions_pred).mean()
        self.actor_loss = actor_loss.item() # calculate policy gradient

        # minimize loss

        # nn.utils.clip_grad_norm_(self.actor.parameters(),1)
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic, self.critic_target, self.tau)
        self.soft_update(self.actor, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        random.seed(seed)
        np.random.seed(seed)
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        # self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)#np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        random.seed(seed)
        np.random.seed(seed)
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        # self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):#, alpha=ALPHA, beta=BETA):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class LowDimActor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1u=400, fc2u=300): #max_action,
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(LowDimActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1u)
        self.fc2 = nn.Linear(fc1u,fc2u)
        self.fc3 = nn.Linear(fc2u, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights with near zero values."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        # x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class LowDimCritic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, seed, fc1u=400, fc2u=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(LowDimCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1u)
        # self.bn1 = nn.BatchNorm1d(fc1u)
        self.fc2 = nn.Linear(fc1u, fc2u)
        self.fc3 = nn.Linear(fc2u, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights with near zero values."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # xs = F.relu(self.bn1(self.fc1(state)))
        xs = torch.cat((states, actions),dim=1)
        x = F.relu(self.fc1(xs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LowDim2x():
    """Container for actor and critic along with respective target networks.
    Each actor takes a state input for a single agent.
    Each critic takes a concatenation of states and actions from all agents.
    Local and target models initialized with idential weights by using same seed.
    """
    def __init__(self, num_agents, state_size=24, action_size=2, seed=0):
        """
        Params
        ======
        """
        self.actor = LowDimActor(state_size, action_size, seed).to(device)
        self.actor_target = LowDimActor(state_size, action_size, seed).to(device)
        critic_input_size = (state_size+action_size)*num_agents
        self.critic = LowDimCritic(critic_input_size, seed).to(device)
        self.critic_target = LowDimCritic(critic_input_size, seed).to(device)

        # output model architecture
        # print(self.actor)
        print("Architecture Summary: Actor\n")
        summary(self.actor, (state_size,))
        # print(self.critic)
        print("Architecture Summary: Critic\n")
        summary(self.critic, (state_size,))
