import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math

from helper import ReplayMemory, Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepConv(nn.Module):
    def __init__(self,
                 env, 
                 input_shape:tuple, 
                 n_actions:int, 
                 eps_start:float, 
                 eps_end:float, 
                 eps_decay:float, 
                 batch_size:int):
        super(DeepConv, self).__init__()
        self.env = env
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.input_shape = input_shape
        self.actions = n_actions
        self.layer1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.layer3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.layer4 = nn.Linear(self.feature_size(), 512)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = nn.Flatten()(x)
        x = self.layer4(x)
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
        

