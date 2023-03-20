from collections import namedtuple, deque
import random 
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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