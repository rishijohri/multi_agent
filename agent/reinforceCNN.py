import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
import random 

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """
        Save a transition. Order of argument matters: state, action, reward, next_state, done 
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        '''
        sample a batch of transitions. To be used in training
        '''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_size[0], 16, kernel_size=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2)
        self.bn3 = nn.BatchNorm2d(32)
        # fc1 has size equal to resulting flattened conv layer
        self.fc1 = np.Linear()
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQNLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNLinear, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)
        self.fc7 = nn.Linear(16, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = self.fc7(x)
        return x

class RL_CNN():
    def __init__(self, 
                 input_size, 
                 output_size, 
                 linear=True, 
                 lr=0.001, 
                 gamma=0.99, 
                 replay_size=10000, 
                 batch_size=4):
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if linear:
            self.policy_net = DQNLinear(input_size, output_size).to(self.device)
            self.target_net = DQNLinear(input_size, output_size).to(self.device)
        else:
            self.policy_net = DQN(input_size, output_size).to(self.device)
            self.target_net = DQN(input_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayMemory(replay_size)
    
    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        q_values = self.policy_net(state)
        return q_values

    def calc_loss(self, batch):
        states = [b.state for b in batch]
        actions = [b.action for b in batch]
        rewards = [b.reward for b in batch]
        next_states = [b.next_state for b in batch]
        dones = [b.done for b in batch]
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        current_q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[1].unsqueeze(1)
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        # print(current_q_values.shape, 
        #       expected_q_values.shape, 
        #       next_q_values.shape, 
        #       rewards.shape, 
        #       dones.shape)
        loss = self.loss_fn(current_q_values, expected_q_values.detach())
        return loss

    def update_model(self):
        # sample a batch of transitions do not calc loss if memory is less than batch size
        if len(self.memory) < self.batch_size:
            return 0
        batch = self.memory.sample(self.batch_size)
        
        loss = self.calc_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state, epsilon:int=0.1):
        # returns the action and the q value associated with it
        q_values = self.policy_net(state)
        if random.random() < epsilon:
            # return random action and q value
            action = np.random.randint(self.output_size)
        else:
            action = q_values.max(1)[1].item()
        return action, q_values[0, action]
    
    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
    
    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))