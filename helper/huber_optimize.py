from typing import Type
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from helper import Transition, ReplayMemory


def huber_optimize(memory: ReplayMemory, 
                   batch_size:int, 
                   gamma:float,
                   num_agents:int, 
                   policy_net: Type[nn.Module], 
                   target_net: Type[nn.Module], 
                   optimizer,
                   device):
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
    reward_batch = torch.cat(batch.reward).unsqueeze(1)
    state_action_values = []
    for i in range(num_agents):
        # select and perform an action for each agent
        agent_state = state_batch[:, [i*2, i*2+1, 2*num_agents, 2*num_agents+1]]

        action_value = policy_net(agent_state).gather(1, action_batch)
        state_action_values.append(action_value)
    state_action_values = torch.cat(state_action_values, dim=1)
    # actions = np.array(actions, dtype=np.int32)
    # state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_action_values = torch.zeros(batch_size,num_agents, device=device)
    reward_batch = torch.ones(batch_size, num_agents, device=device)*reward_batch
    with torch.no_grad():
        next_state_action_list = []
        for i in range(num_agents):
            # select and perform an action for each agent
            try:
                agent_state = non_final_next_states[:, [i*2, i*2+1, 2*num_agents, 2*num_agents+1]]
                print(agent_state, "agent_state")
                action_value = target_net(agent_state).max(1)[0].unsqueeze(1)
                next_state_action_list.append(action_value)
            except:
                print(non_final_next_states.shape, action_batch.shape, batch.next_state)
                print(non_final_mask, "non_final_mask")
        next_state_action_values[non_final_mask] = torch.cat(next_state_action_list, dim=1)
        # todo: check if this is correct or not
        # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_action_values * gamma) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)/num_agents
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()