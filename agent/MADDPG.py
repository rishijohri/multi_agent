import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
import random 

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MADDPG:
    
    def __init__(self, state_size, action_size, num_agents, seed, buffer_size=int(1e6), batch_size=128, gamma=0.99, tau=1e-3, lr_actor=1e-4, lr_critic=1e-3, weight_decay=0):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.num_agents = num_agents

        # Actor networks
        self.actor_local = [Actor(state_size, action_size, seed).to(device) for _ in range(num_agents)]
        self.actor_target = [Actor(state_size, action_size, seed).to(device) for _ in range(num_agents)]
        self.actor_optimizer = [optim.Adam(self.actor_local[i].parameters(), lr=lr_actor) for i in range(num_agents)]

        # Critic networks
        self.critic_local = Critic(state_size, action_size, num_agents, seed).to(device)
        self.critic_target = Critic(state_size, action_size, num_agents, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)

        # Noise process
        self.noise = OUNoise((num_agents, action_size), seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

        # Algorithm parameters
        self.gamma = gamma
        self.tau = tau

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        actions = np.zeros((self.num_agents, self.action_size))
        for i, state in enumerate(states):
            state = torch.from_numpy(state).float().to(device)
            self.actor_local[i].eval()
            with torch.no_grad():
                action = self.actor_local[i](state).cpu().data.numpy()
            self.actor_local[i].train()
            actions[i, :] = action
        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every update_every time steps.
        if len(self.memory) > BATCH_SIZE and len(self.memory) % UPDATE_EVERY == 0:
            for _ in range(UPDATE_TIMES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = [self.actor_target[i](next_states[:, i, :]) for i in range(self.num_agents)]
        actions_next = torch.cat(actions_next, dim=1)
        q_targets_next = self.critic_target(next_states.view(-1, self.state_size*self.num_agents), actions_next.view(-1, self.action_size*self.num_agents))
        
        # Compute Q targets for current states (y_i)
        q_targets = rewards.view(-1, self.num_agents) + (self.gamma * q_targets_next * (1 - dones.view(-1, self.num_agents)))
        
        # Compute critic loss
        q_expected = self.critic_local(states.view(-1, self.state_size*self.num_agents), actions.view(-1, self.action_size*self.num_agents))
        critic_loss = F.mse_loss(q_expected, q_targets.detach())
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = [self.actor_local[i](states[:, i, :]) for i in range(self.num_agents)]
        actions_pred = torch.cat(actions_pred, dim=1)
        actor_loss = -self.critic_local(states.view(-1, self.state_size*self.num_agents), actions_pred.view(-1, self.action_size*self.num_agents)).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)






