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
        Save a transition. Order of argument matters: state, action, next_state, reward
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        '''
        sample a batch of transitions. To be used in training
        '''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class RandomAgent():
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
        self.memory = ReplayMemory(replay_size)
    def forward(self, state):
        return np.random.randint(self.output_size)
    
    def calc_loss(self, batch):
        pass

    def update_model(self):
        pass
    
    def update_target_model(self):
        pass

    def select_action(self, state, epsilon=0.1):
        return np.random.randint(self.output_size), 0