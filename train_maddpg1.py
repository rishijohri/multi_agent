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
TAU = 0.05
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 5000
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.99
NUM_EPISODES = 5000
NUM_PRED = 5
NUM_PREY = 1
BATCH_SIZE = 2
VISION = 5
SIZE = 10
LOAD = True
HISTORY = 4
COMMUNICATION_BIT = 4
env = PredatorPreyEnv(
    render_mode='human',
    size=SIZE,
    predator=NUM_PRED,
    prey=NUM_PREY,
    episode_length=100,
    img_mode=True,
    vision=VISION,
    history_length=HISTORY,
    communication_bits=COMMUNICATION_BIT,
    success_reward=10,
    living_reward=-1.5,
    error_reward=-5,
    cooperate=1
)
state, _ = env.reset()
n_actions = env.single_action_space.n
n_observations = 4 # considering single agent observation space
episode_durations = []
steps_done = 0

losses = []
steps = []
rewards = []

predators = [DDPGAgent((HISTORY*4,VISION,VISION), 
                       n_actions, 
                       num_agents=NUM_PRED, 
                       idd=i,
                       gamma=GAMMA,
                       tau=TAU,
                       actor_lr=ACTOR_LR,
                       critic_lr=CRITIC_LR,
                       communication_bits=COMMUNICATION_BIT,
                       batch_size=BATCH_SIZE
                       ) for i in range(NUM_PRED)]

# prey is random agent
preys = [RandomAgent(VISION*VISION*3, 
                     n_actions,
                     fix_pos=False) for i in range(NUM_PREY)]
fin_losses = []
episode_durations = []
epsilon_predator = EPS_START
epsilon_prey = EPS_START
fin_rewards = []
#load model
if LOAD:
    for i in range(NUM_PRED):
        predators[i].load_model(f"./models2/actor_model_{i}.dict", "./models/critic_model"+str(i)+".dict")
        predators[i].load_optim(f"./models2/actor_optimizer_{i}.dict", f"./models2/critic_optimizer_{i}.dict")
      
for i_episode in range(NUM_EPISODES):
    # Initialize the environment and get it's state
    state, info = env.reset()
    # print(state)
    losses = []
    episode_reward = 0
    for i_step in count():
        # Select and perform an action
        predator_actions = []
        prey_actions = []
        pred_communication = [0 for _ in range(NUM_PRED)]
        prey_communication = [0 for _ in range(NUM_PREY)]
        if state is None:
            pred_actions = [0 for _ in range(NUM_PRED)]
            prey_actions = [0 for _ in range(NUM_PREY)]
            next_state, reward, done, info = env.step(pred_actions, 
                                                      prey_actions, 
                                                      pred_communication=pred_communication, 
                                                      prey_communication=prey_communication)
            steps_done+=1
            # Move to the next state
            state = next_state
            # if done or i == env.episode_length-1:
            if done:
                episode_durations.append(i_step + 1)
                break
            continue

        #CHOOSE ACTION
        for i in range(NUM_PRED):
            state_i = [torch.tensor(s["predator"][i], dtype=torch.float32, device=device) for s in state]
            state_i = torch.cat(state_i, dim=0).unsqueeze(0)  # 12, 5, 5
            # print(state_i.shape, "AFTER")
            # state_i = torch.tensor(state_i, dtype=torch.float32, device=device).unsqueeze(0)
            action_i, action_choice, msg_i = predators[i].select_action(state_i, epsilon_predator)
            predator_actions.append(action_choice)
            pred_communication[i] = msg_i
            # print(state_i[:4, :, :], "STATE")
            # exit()
            # print(msg_i, "RECIEVED_MSG")
        for i in range(NUM_PREY):
            state_i = [torch.tensor(s["prey"][i], dtype=torch.float32, device=device) for s in state]
            action_i, _= preys[i].select_action(state_i, epsilon_prey)
            prey_actions.append(action_i)
        
        # print(pred_communication, "pred_communication")
        #TAKE ACTION IN ENVIRONMENT
        next_state, reward, done, info = env.step(predator_actions, 
                                                  prey_actions, 
                                                  pred_communication=pred_communication)
        episode_reward += sum(reward["predator"])/len(reward["predator"])*GAMMA**i_step
        # Store the transition in memory
        total_state = []
        total_next_state = []
        reward_total = torch.tensor(reward["predator"], dtype=torch.float32, device=device)
        action_total = torch.tensor(predator_actions, dtype=torch.float32, device=device)
        pred_msg_total = torch.tensor(pred_communication, dtype=torch.float32, device=device)
        done_total = torch.tensor(done, dtype=torch.float32, device=device)

        for i in range(NUM_PRED):
            state_i = [torch.tensor(s["predator"][i], dtype=torch.float32, device=device) for s in state]
            state_i = torch.cat(state_i, dim=0)
            
            next_state_i = [torch.tensor(s["predator"][i], dtype=torch.float32, device=device) for s in next_state]
            next_state_i = torch.cat(next_state_i, dim=0)
            total_state.append(state_i)
            total_next_state.append(next_state_i)
        total_state = torch.cat(total_state, dim=0)
        total_next_state = torch.cat(total_next_state, dim=0)

        for i in range(NUM_PRED):
            predators[i].replay_buffer.push(total_state, action_total, pred_msg_total, reward_total, total_next_state, done_total)
        
        

        # update the Actor-Critic Network
        for i in range(NUM_PRED):
            loss = predators[i].update_model()
            if loss is not None:
                losses.append(loss)
            predators[i].update_target_model()
        for i in range(NUM_PREY):
            preys[i].update_model()
            preys[i].update_target_model()
        
        #decay epsilon
        epsilon_predator = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * steps_done / EPS_DECAY)
        epsilon_prey = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * steps_done / EPS_DECAY)
        steps_done+=1

        # Move to the next state
        state = next_state
#         print('happer')
        if done:
            episode_durations.append(i_step + 1)
            break
    LOSS = sum(losses)/(len(losses)+1)
    fin_losses.append(LOSS)
    fin_rewards.append(episode_reward)
    print(f"Episode {i_episode} finished after {i_step+1} steps with Loss {LOSS:.3f}, REWARD {episode_reward:.3f}")
    
    # save model in models folder
    if (i_episode) % 10 == 0:
        for i in range(NUM_PRED):
            torch.save(predators[i].actor_model.state_dict(), f"./models2/actor_model_{i}.dict")
            torch.save(predators[i].critic_model.state_dict(), f"./models2/critic_model{i}.dict")
            torch.save(predators[i].actor_optimizer.state_dict(), f"./models2/actor_optimizer_{i}.dict")
            torch.save(predators[i].critic_optimizer.state_dict(), f"./models2/critic_optimizer_{i}.dict")
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
        ax[0].plot(episode_durations)
        ax[0].set_xlabel('Episode')
        ax[0].set_ylabel('Duration')
        ax[0].set_title('EPISODE DURATIONS')
        ax[1].plot(fin_losses)
        ax[1].set_xlabel('Episode')
        ax[1].set_ylabel('Loss')
        ax[1].set_title('LOSS')
        ax[2].plot(fin_rewards)
        ax[2].set_xlabel('Episode')
        ax[2].set_ylabel('Reward')
        ax[2].set_title('AVG REWARD')
        fig.subplots_adjust(wspace=0.4)
        plt.savefig('figure.png', dpi=400)
        plt.close()
# a = input("press any key to continue")