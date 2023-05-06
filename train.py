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

from agent import RL_CNN, RandomAgent
from helper import ReplayMemory, Transition
# custom packages
from world import PredatorPreyEnv

# for interactive plots
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# parameters for the environment
TAU = 0.005
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 3000
LR = 1e-4
GAMMA = 0.99
NUM_EPISODES = 50000
NUM_PRED = 5
NUM_PREY = 1
BATCH_SIZE = 2
VISION = 5
SIZE = 10
env = PredatorPreyEnv(
    render_mode='none',
    size=SIZE,
    predator=NUM_PRED,
    prey=NUM_PREY,
    episode_length=50,
    img_mode=True,
    vision=VISION
)
state, _ = env.reset()
n_actions = env.single_action_space.n
n_observations = 4 # considering single agent observation space
episode_durations = []
memory = ReplayMemory(10000)
steps_done = 0

# policy_net.load_state_dict(torch.load("./models/policy_net.dict"))
# target_net.load_state_dict(torch.load("./models/target_net.dict"))
losses = []
steps = []
rewards = []

#train using predator as RL_CNN and prey as random agent
# each predator has its own policy net and target net
# the observation space is the the vision of the predator
# the input to the policy net i is the vision of the predator i only
# each predator and prey has its own memory
# each predator and prey has its own optimizer
predators = [RL_CNN((12,VISION,VISION), n_actions, linear=False, lr=0.001, gamma=0.99) 
               for i in range(NUM_PRED)]

memories = [ReplayMemory(10000) for i in range(NUM_PRED)]
# prey is random agent
preys = [RandomAgent(VISION*VISION*3, n_actions, linear=True, lr=0.001, gamma=0.99) for i in range(NUM_PREY)]
losses = []
episode_durations = []
epsilon_predator = EPS_START
epsilon_prey = EPS_START

# load model
# for i in range(NUM_PRED):
#     predators[i].policy_net.load_state_dict(torch.load("./models/policy_net_"+str(i)+".dict"))
#     predators[i].target_net.load_state_dict(torch.load("./models/target_net_"+str(i)+".dict"))
# for i in range(NUM_PREY):
#     preys[i].policy_net.load_state_dict(torch.load("./models/policy_net_prey"+str(i)+".dict"))
#     preys[i].target_net.load_state_dict(torch.load("./models/target_net_prey"+str(i)+".dict"))

for i_episode in range(NUM_EPISODES):
    # Initialize the environment and get it's state
    state, info = env.reset()
    # print(state)
    for i_step in count():
        # Select and perform an action
        predator_actions = []
        if state is None:
            pred_actions = [0 for _ in range(NUM_PRED)]
            prey_actions = [0 for _ in range(NUM_PREY)]
            next_state, reward, done, info = env.step(pred_actions, prey_actions)
            steps_done+=1
            # Move to the next state
            state = next_state
            # if done or i == env.episode_length-1:
            if done:
                episode_durations.append(i_step + 1)
                break
            continue
        for i in range(NUM_PRED):
            state_i = [torch.tensor(s["predator"][i], dtype=torch.float32, device=device) for s in state]
            # print(len(state_i), state_i[0].shape, "BEFORE")
            state_i = torch.cat(state_i, dim=0).unsqueeze(0)
            # print(state_i.shape, "AFTER")
            # state_i = torch.tensor(state_i, dtype=torch.float32, device=device).unsqueeze(0)
            action_i, _ = predators[i].select_action(state_i, epsilon_predator)
            predator_actions.append(action_i)
        prey_actions = []
        for i in range(NUM_PREY):
            state_i = [torch.tensor(s["prey"][i], dtype=torch.float32, device=device) for s in state]
            action_i, _= preys[i].select_action(state_i, epsilon_prey)
            prey_actions.append(action_i)

        next_state, reward, done, info = env.step(predator_actions, prey_actions)
        # reward = torch.tensor([reward], device=device)


        # Store the transition in memory
        # todo: The state now contains history as well, refer the Environment to see structure of state. Write the training code to take history as well
        for i in range(NUM_PRED):
            state_i = [torch.tensor(s["predator"][i], dtype=torch.float32, device=device) for s in state]
            state_i = torch.cat(state_i, dim=0)
            next_state_i = [torch.tensor(s["predator"][i], dtype=torch.float32, device=device) for s in next_state]
            next_state_i = torch.cat(next_state_i, dim=0)
            reward_i = reward["predator"][i]
            action_i = predator_actions[i]
            predators[i].memory.push(state_i, action_i, reward_i, next_state_i,done)
        for i in range(NUM_PREY):
            state_i = [torch.tensor(s["prey"][i], dtype=torch.float32, device=device) for s in state]
            state_i = torch.cat(state_i, dim=0)
            next_state_i = [torch.tensor(s["prey"][i], dtype=torch.float32, device=device) for s in next_state]
            next_state_i = torch.cat(next_state_i, dim=0)
            reward_i = reward["prey"][i]
            action_i = predator_actions[i]
            predators[i].memory.push(state_i, action_i, reward_i, next_state_i,done)
        
        # update the policy net and target net
        for i in range(NUM_PRED):
            loss = predators[i].update_model()
            losses.append(loss)
            if i_episode % 10 == 0:
                predators[i].update_target_model()
        for i in range(NUM_PREY):
            preys[i].update_model()
            if i_episode % 20 == 0:
                preys[i].update_target_model()
        #decay epsilon
        epsilon_predator = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * steps_done / EPS_DECAY)
        epsilon_prey = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * steps_done / EPS_DECAY)
        steps_done+=1
        # Move to the next state
        state = next_state
        # if done or i == env.episode_length-1:
        if done:
            episode_durations.append(i_step + 1)
            break
    print(f"Episode {i_episode} finished after {i_step+1} steps epsilon {epsilon_predator:.2f} {epsilon_prey:.2f}")
    # save model in models folder
    for i in range(NUM_PRED):
        torch.save(predators[i].policy_net.state_dict(), f"./models/policy_net_{i}.dict")
        torch.save(predators[i].target_net.state_dict(), f"./models/target_net_{i}.dict")
        torch.save(predators[i].optimizer.state_dict(), f"./models/optimizer_{i}.dict")
plt.plot(episode_durations)
plt.show()
a = input("press any key to continue")