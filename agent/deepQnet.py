import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math

from helper import ReplayMemory, Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self,
                 env, 
                 n_observations:int, 
                 n_actions:int, 
                 eps_start:float, 
                 eps_end:float, 
                 eps_decay:float, 
                 batch_size:int):
        super(DQN, self).__init__()
        self.env = env
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(128, 128)
        
        self.layer3 = nn.Linear(32, n_actions)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = nn.Linear(32, 64)(x)
        x = F.relu(x)
        x = nn.Linear(64, 128)(x)
        x = F.relu(x)
        x = nn.Linear(128, 128)(x)
        x = F.relu(x)
        x = nn.Linear(128, 64)(x)
        x = F.relu(x)
        x = nn.Linear(64, 32)(x)
        x = F.relu(x)
        x = self.layer3(x)
        return x
    
    def select_action(self, output, steps_done):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * steps_done / self.eps_decay)
        steps_done += 1
        # print(f'sample: {sample}, eps_threshold: {eps_threshold}')
        if sample < eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return output.max(1)[1].view(1, 1), steps_done
        else:
            # print('random sampled')
            return torch.tensor([[self.env.single_action_space.sample()]], device=device, dtype=torch.long), steps_done
        

