import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'communication', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
    
    def push(self, state=0, action=0, communication=None, reward=0, next_state=0, done=0):
        self.memory.append(Transition(state, action, communication, reward, next_state, done))
    
    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.stack(batch.state)
        # print(state_batch.shape, "STATE BATCH")
        # print(len(batch.action))
        action_batch = torch.stack(batch.action)
        # print(action_batch.shape, "ACTION BATCH")
        communication_batch = torch.stack(batch.communication)
        # print(communication_batch.shape, "COMMUNICATION BATCH")
        reward_batch = torch.stack(batch.reward)
        # print(reward_batch.shape, "REWARD BATCH")
        next_state_batch = torch.stack(batch.next_state)
        done_batch = torch.stack(batch.done).unsqueeze(1)
        # print(done_batch.shape, "DONE BATCH")
        return state_batch, action_batch, communication_batch, reward_batch, next_state_batch, done_batch
    
    def __len__(self):
        return len(self.memory)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, communication_size=0):
        super().__init__()
        self.communincation_size = communication_size
        self.conv1 = nn.Conv2d(state_size[0], 16, kernel_size=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2)
        self.bn3 = nn.BatchNorm2d(32)
        # fc1 has size equal to resulting flattened conv layer
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, action_size)
    
        #layers for communication
        # self.commconv1 = nn.Conv2d(state_size[0], 16, kernel_size=2)
        # self.commbn1 = nn.BatchNorm2d(16)
        # self.commconv2 = nn.Conv2d(16, 32, kernel_size=2)
        # self.commbn2 = nn.BatchNorm2d(32)
        # self.commconv3 = nn.Conv2d(32, 32, kernel_size=2)
        # self.commbn3 = nn.BatchNorm2d(32)
        # fc1 has size equal to resulting flattened conv layer
        self.commfc1 = nn.Linear(128, 256)
        self.commfc2 = nn.Linear(256, 1)
    def forward(self, state):
        # choose action based on state
        x = torch.relu(self.bn1(self.conv1(state)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        x_comm = torch.relu(self.commfc1(x))
        x_comm = torch.relu(self.commfc2(x_comm))
        x_comm = self.communincation_size*torch.sigmoid(x_comm)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.softmax(x, dim=-1)
        
        return x, x_comm

class Critic(nn.Module):
    def __init__(self, state_size, action_size, num_agents, communication_size=0):
        super().__init__()
        self.conv1 = nn.Conv2d(state_size[0]*num_agents, 16, kernel_size=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2)
        self.bn3 = nn.BatchNorm2d(32)
        # fc1 has size equal to resulting flattened conv layer
        self.fc0 = nn.Linear(num_agents+num_agents, 64)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, state, action, communication=None):
        x = torch.relu(self.bn1(self.conv1(state)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        y = torch.cat([action, communication], dim=-1)
        y = torch.relu(self.fc0(y))
        y = torch.relu(self.fc1(y))
        x = torch.cat([x, y], dim=-1)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        # x = torch.softmax(x, dim=-1)
        return x

class DDPGAgent:
    def __init__(self, 
                 state_size:tuple=(12, 5, 5),
                 action_size:int=5, 
                 buffer_size:int=1000, 
                 batch_size:int=4, 
                 gamma:float=0.99, 
                 tau:float=0.01, 
                 actor_lr:float=0.001,
                 critic_lr:float=0.003, 
                 num_agents:int=4,
                 communication_bits:int=0,
                 idd:int=0):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.num_agents = num_agents
        self.idd = idd
        self.communication_bits = communication_bits
        self.actor_model = Actor(state_size=state_size, 
                                 action_size=action_size, 
                                 communication_size=communication_bits)
        self.critic_model = Critic(state_size, 
                                   action_size, 
                                   num_agents,
                                   communication_size=communication_bits)
        self.target_actor_model = Actor(state_size=state_size, 
                                        action_size=action_size,
                                        communication_size=communication_bits)
        self.target_critic_model = Critic(state_size , 
                                          action_size, 
                                          num_agents,
                                          communication_size=communication_bits)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=critic_lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    def act(self, state, noise_scale=0.0):
        with torch.no_grad():
            action, msg = self.actor_model(state)
            action += noise_scale * torch.randn_like(action)
            action = torch.clamp(action, 0, 1)
            # print(msg, "TRUE MSG")
            # msg += noise_scale*torch.randn_like(msg)
        return action.squeeze(0), msg
    
    def select_action(self, state, noise_scale=0.0):
        action, msg = self.act(state, noise_scale)
        choice = torch.argmax(action)
        return action.cpu().detach().numpy(), choice.cpu().detach().item(), msg.cpu().detach().item()
    
    def update_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        state_batch, action_batch,communication_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        # compute target Q value
        target_next_actions = []
        target_next_communication = []
        state_shape = state_batch.shape
        next_state_shape = next_state_batch.shape
        for i in range(self.num_agents):
            start_index = int(state_shape[1]*(i)/self.num_agents)
            end_index = int(state_shape[1]*(i+1)/self.num_agents)
            action, msg = self.target_actor_model(next_state_batch[:,start_index:end_index, :, :])
            target_next_actions.append(action)
            target_next_communication.append(msg)
        target_next_actions = torch.stack(target_next_actions, dim=1)
        target_next_actions = torch.argmax(target_next_actions, dim=2)
        target_next_communication = torch.stack(target_next_communication, dim=1).squeeze(2)
        # print(target_next_actions.shape, "TARGET NEXT ACTIONS")
        # print(target_next_communication.shape, "TARGET NEXT COMMUNICATION")
        # target_next_actions += 0.1 * torch.randn_like(target_next_actions)
        # target_next_actions = torch.clamp(target_next_actions, 0, 1)

        target_q_values = self.target_critic_model(next_state_batch, target_next_actions, target_next_communication)
        target_q_values = reward_batch[:, self.idd].unsqueeze(1) + (1 - done_batch) * self.gamma * target_q_values
        # update critic model
        q_values = self.critic_model(state_batch, action_batch, communication_batch)
        critic_loss = nn.functional.mse_loss(q_values, target_q_values.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # update actor model
        actions = []
        communications = []
        for i in range(self.num_agents):
            start_index = int(state_shape[1]*(i)/self.num_agents)
            end_index = int(state_shape[1]*(i+1)/self.num_agents)
            action, msg = self.actor_model(state_batch[:,start_index:end_index, :, :])
            actions.append(action)
            communications.append(msg)
        actions = torch.stack(actions, dim=1)
        actions = torch.argmax(actions, dim=2)
        communications = torch.stack(communications, dim=1).squeeze(2)
        # print(actions.shape, "ACTIONS")
        # print(communications.shape, "COMMUNICATIONS")
        actor_loss = -self.critic_model(state_batch, actions, communications).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # print(critic_loss.item(), actor_loss.item())
        return critic_loss.item()
    
    def update_target_model(self):
        for target_param, param in zip(self.target_actor_model.parameters(), self.actor_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic_model.parameters(), self.critic_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def calc_loss(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        # compute target Q value
        target_next_actions = self.target_actor_model(next_state_batch)
        target_next_actions += 0.1 * torch.randn_like(target_next_actions)
        target_next_actions = torch.clamp(target_next_actions, 0, 1)
        target_q_values = self.target_critic_model(next_state_batch, target_next_actions)
        target_q_values = reward_batch + (1 - done_batch) * self.gamma * target_q_values
        # compute critic loss
        q_values = self.critic_model(state_batch, action_batch)
        critic_loss = nn.functional.mse_loss(q_values, target_q_values.detach())
        # compute actor loss
        actions = self.actor_model(state_batch)
        actor_loss = -self.critic_model(state_batch, actions).mean()
        return critic_loss.item(), actor_loss.item()
    
    def save_model(self, actor_file, critic_file):
        torch.save(self.actor_model.state_dict(), actor_file)
        torch.save(self.critic_model.state_dict(), critic_file)
    
    def load_model(self, actor_file, critic_file):
        self.actor_model.load_state_dict(torch.load(actor_file))
        self.critic_model.load_state_dict(torch.load(critic_file))
        self.target_actor_model.load_state_dict(torch.load(actor_file))
        self.target_critic_model.load_state_dict(torch.load(critic_file))   
        print("SUCCESSFULLY LOADED MODEL")
    def load_optim(self, actor_file, critic_file):
        self.actor_optimizer.load_state_dict(torch.load(actor_file))
        self.critic_optimizer.load_state_dict(torch.load(critic_file))
        print("SUCCESSFULLY LOADED OPTIMIZER")
