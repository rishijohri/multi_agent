import time  # for sleep
from collections import deque, namedtuple
from itertools import count

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from agent import DQN
from helper import ReplayMemory, Transition, huber_optimize
# custom packages
from world import dot_world

# for interactive plots
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()



# parameters for the environment
TAU = 0.005
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
LR = 1e-4
GAMMA = 0.99
NUM_EPISODES = 1000
NUM_AGENTS = 1
BATCH_SIZE = 2

env = dot_world.DotWorld(render_mode='human', size=5, agents=NUM_AGENTS, episode_length=50)
state, _ = env.reset()
n_actions = env.single_action_space.n
n_observations = 4 # considering single agent observation space
episode_durations = []
memory = ReplayMemory(10000)
policy_net = DQN(env, n_observations, n_actions, EPS_START, EPS_END, EPS_DECAY, 2).to(device)
target_net = DQN(env, n_observations, n_actions, EPS_START, EPS_END, EPS_DECAY, 2).to(device)
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
steps_done = 0

for i_episode in range(NUM_EPISODES):
    # Initialize the environment and get it's state
    state, info = env.reset()
    # print(state)
    state = np.concatenate((state['agent'], np.expand_dims(state['target'], 0)), axis=0)
    state = torch.tensor(state, dtype=torch.float32, device=device).flatten().unsqueeze(0)
    # print(state.shape)
    total_loss = []
    for t in count():
        actions = [ ]
        for i in range(NUM_AGENTS):
            # select and perform an action for each agent
            agent_state = state[:, [i*2, i*2+1, 2*NUM_AGENTS, 2*NUM_AGENTS+1]]
            action_space = policy_net(agent_state)
            action, _ = policy_net.select_action(action_space, steps_done)
            actions.append(action.item())
        actions = np.array(actions, dtype=np.int32)

        # Observe new state
        next_state, reward, terminated, truncated, _ = env.step(actions)
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated
        screen = env.render_frame_array(agent_id=1) # render the first agent
        # plt.imshow(screen.transpose(1, 2, 0))
        plt.pause(0.001)
        if terminated:
            next_state = None
        else:
            next_state = np.concatenate((next_state['agent'], np.expand_dims(next_state['target'], 0)), axis=0)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).flatten().unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        loss = huber_optimize(memory, BATCH_SIZE, GAMMA, NUM_AGENTS, policy_net, target_net, optimizer, device)
        if loss!=-1:
            total_loss.append(loss)
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        if done:
            episode_durations.append(t + 1)
            # plot_durations()
            break
    print(f"at Episode {i_episode} Loss is {round(sum(total_loss)/len(total_loss), 4)}", end="\r")
print('Complete')
# plot_durations(episode_durations=episode_durations, show_result=True, is_ipython=is_ipython)
plt.ioff()
plt.show() 