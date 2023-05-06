import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
import random 
from .replay_memory import ReplayMemory, Transition
    

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
    def act(self, state):
        return np.random.randint(self.output_size)
    
    def calc_loss(self, batch):
        pass

    def update_model(self):
        pass
    
    def update_target_model(self):
        pass

    def select_action(self, state, epsilon=0.1):
        return np.random.randint(self.output_size), 0