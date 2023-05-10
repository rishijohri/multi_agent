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

from agent import RL_CNN, RandomAgent, DDPGAgent
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
EPS_START = 0.01
EPS_END = 0.01
EPS_DECAY = 3000
LR = 1e-4
GAMMA = 0.99
NUM_EPISODES = 5000
NUM_PRED = 5
NUM_PREY = 1
BATCH_SIZE = 2
VISION = 5
SIZE = 10
LOAD = True
env = PredatorPreyEnv(
    render_mode='human',
    size=SIZE,
    predator=NUM_PRED,
    prey=NUM_PREY,
    episode_length=100,
    img_mode=True,
    vision=VISION
)
state, _ = env.reset()
n_actions = env.single_action_space.n
n_observations = 4 # considering single agent observation space
episode_durations = []
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
predators = [DDPGAgent((12,VISION,VISION), 
                       n_actions, 
                       num_agents=NUM_PRED, 
                       idd=i
                       ) for i in range(NUM_PRED)]
# prey is random agent
preys = [RandomAgent(VISION*VISION*3, 
                     n_actions,  
                     fix_pos=False) for i in range(NUM_PREY)]
fin_losses = []
episode_durations = []
epsilon_predator = EPS_START
epsilon_prey = EPS_START

#load model
if LOAD:
    for i in range(NUM_PRED):
        predators[i].load_model("./models/actor_model_"+str(i)+".dict", "./models/critic_model"+str(i)+".dict")
        predators[i].load_optim("./models/actor_optimizer_"+str(i)+".dict", "./models/critic_optimizer_"+str(i)+".dict")
      
for i_episode in range(NUM_EPISODES):
    # Initialize the environment and get it's state
    state, info = env.reset()
    # print(state)
    losses = []
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
            state_i = torch.cat(state_i, dim=0).unsqueeze(0)  # 12, 5, 5
            # print(state_i.shape, "AFTER")
            # state_i = torch.tensor(state_i, dtype=torch.float32, device=device).unsqueeze(0)
            action_i, action_choice = predators[i].select_action(state_i, epsilon_predator)
            predator_actions.append(action_choice)
        prey_actions = []
        for i in range(NUM_PREY):
            state_i = [torch.tensor(s["prey"][i], dtype=torch.float32, device=device) for s in state]
            action_i, _= preys[i].select_action(state_i, epsilon_prey)
            prey_actions.append(action_i)
        next_state, reward, done, info = env.step(predator_actions, prey_actions)
        
        # Store the transition in memory
        # todo: The state now contains history as well, refer the Environment to see structure of state. Write the training code to take history as well
        # total_state = []
        # total_next_state = []
        # reward_i = torch.tensor(reward["predator"], dtype=torch.float32, device=device)
        # action_i = torch.tensor(predator_actions, dtype=torch.float32, device=device)
        # torch_done = torch.tensor(done, dtype=torch.float32, device=device)
        # for i in range(NUM_PRED):
        #     state_i = [torch.tensor(s["predator"][i], dtype=torch.float32, device=device) for s in state]
        #     state_i = torch.cat(state_i, dim=0)
        #     next_state_i = [torch.tensor(s["predator"][i], dtype=torch.float32, device=device) for s in next_state]
        #     next_state_i = torch.cat(next_state_i, dim=0)
        #     total_state.append(state_i)
        #     total_next_state.append(next_state_i)
        # total_state = torch.cat(total_state, dim=0)
        # total_next_state = torch.cat(total_next_state, dim=0)

        # for i in range(NUM_PRED):
        #     predators[i].replay_buffer.push(total_state, action_i, reward_i, total_next_state, torch_done)
        # for i in range(NUM_PREY):
        #     state_i = [torch.tensor(snip["prey"][i], dtype=torch.float32, device=device) for snip in state]
        #     state_i = torch.cat(state_i, dim=0)
        #     next_state_i = [torch.tensor(snip["prey"][i], dtype=torch.float32, device=device) for snip in next_state]
        #     next_state_i = torch.cat(next_state_i, dim=0)
        #     reward_i = reward["prey"][i]
        #     action_i = prey_actions[i]
        #     preys[i].memory.push(state_i, action_i, reward_i, next_state_i, done)

        # update the policy net and target net
        # for i in range(NUM_PRED):
        #     loss = predators[i].update_model()
        #     if loss is not None:
        #         losses.append(loss)
        #     predators[i].update_target_model()
        # for i in range(NUM_PREY):
        #     preys[i].update_model()
        #     preys[i].update_target_model()
        #decay epsilon
        # epsilon_predator = EPS_END + (EPS_START - EPS_END) * \
        #     np.exp(-1. * steps_done / EPS_DECAY)
        # epsilon_prey = EPS_END + (EPS_START - EPS_END) * \
        #     np.exp(-1. * steps_done / EPS_DECAY)
        steps_done+=1
        # Move to the next state
        state = next_state
        # if done or i == env.episode_length-1:
        if done:
            episode_durations.append(i_step + 1)
            break
    print(f"Episode {i_episode} finished after {i_step+1} steps with Loss {(sum(losses)/(len(losses)+1)):.3f}, {len(losses)} ")
    # save model in models folder
    if (i_episode) % 10 == 0:
        for i in range(NUM_PRED):
            torch.save(predators[i].actor_model.state_dict(), f"./models/actor_model_{i}.dict")
            torch.save(predators[i].critic_model.state_dict(), f"./models/critic_model{i}.dict")
            torch.save(predators[i].actor_optimizer.state_dict(), f"./models/actor_optimizer_{i}.dict")
            torch.save(predators[i].critic_optimizer.state_dict(), f"./models/critic_optimizer_{i}.dict")
plt.plot(episode_durations)
plt.show()
a = input("press any key to continue")