from typing import Type
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from helper import Transition, ReplayMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def huber_optimize(memory: ReplayMemory, batch_size:int, gamma:float , policy_net: Type[nn.Module], target_net: Type[nn.Module], optimizer):
    if len(memory) <batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions)) 
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                            if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()