from DQN import Agent, device
from environment import Building
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from vars import *
from collections import namedtuple
from itertools import count
import pickle as pkl
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

if __name__=='__main__':
    env = Building()
    scores = []
    brain = Agent(gamma=GAMMA, epsilon=EPSILON, batch_size=BATCH_SIZE, n_actions=N_ACTIONS,
                  input_dims=INPUT_DIMS,  lr = LEARNING_RATE, eps_dec = EPS_DECAY)
    start = time.time()
    for i_episode in range(NUM_EPISODES):
        # Initialize the environment and state
        state = torch.tensor(env.reset(),dtype=torch.float).unsqueeze(0).to(device)
        score = 0
        for t in count():
            # Select and perform an action
            action = brain.select_action(state).type(torch.FloatTensor)
            next_state, reward, done = env.step(action.item())
            score += reward
            reward = torch.tensor([reward],dtype=torch.float,device=device)

            if not done:
                next_state = torch.tensor(next_state,dtype=torch.float, device=device).unsqueeze(0)
            else:
                next_state = None

            # Store the transition in memory
            brain.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            brain.optimize_model()
            if done:
                scores.append(score)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            end = time.time()
            brain.target_net.load_state_dict(brain.policy_net.state_dict())
            print('------------------Episode Number {}------------------\n'.format(i_episode))
            print('After {} seconds for {} episodes'.format(end-start, TARGET_UPDATE))
            print('Current Reward {}'.format(score))
            start = time.time()

    with open(os.getcwd() + '/data/output/' + 'rewards_dqn.pkl', 'wb') as f:
        pkl.dump(scores,f)
    print('Complete')