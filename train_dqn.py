from DQN import DAgent
import sys
from environment import Building
from matplotlib import style
style.use('ggplot')
from vars import *
from itertools import count
import pickle as pkl
import os
import argparse
import sys
import torch
import pandas as pd

def train_dqn(ckpt,model_name,dynamic,soft):
    env = Building(dynamic)
    scores = []
    temperatures = []
    brain = DAgent(gamma=GAMMA, epsilon=EPSILON, batch_size=BATCH_SIZE, n_actions=N_ACTIONS,
                  input_dims=INPUT_DIMS,  lr = LEARNING_RATE, eps_dec = EPS_DECAY, ckpt=ckpt)
    for i_episode in range(NUM_EPISODES):
        # Initialize the environment.rst and state
        state = env.reset()
        temperatures_episode = [state[0]]
        state = torch.tensor(state,dtype=torch.float).to(device)
        # Normalizing data using an online algo
        brain.normalizer.observe(state)
        state = brain.normalizer.normalize(state).unsqueeze(0)
        score = 0
        for t in count():
            # Select and perform an action
            action = brain.select_action(state).type(torch.FloatTensor)
            next_state, reward, done = env.step(action.item())
            score += reward
            reward = torch.tensor([reward],dtype=torch.float,device=device)

            if not done:
                temperatures_episode.append(next_state[0])
                next_state = torch.tensor(next_state,dtype=torch.float, device=device)
                #normalize data using an online algo
                brain.normalizer.observe(next_state)
                next_state = brain.normalizer.normalize(next_state).unsqueeze(0)

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

        sys.stdout.write('Finished episode {} with reward {}\n'.format(i_episode, score))
        # Soft update for target network:

        if soft:
            brain.soft_update(TAU)

        # Update the target network, copying all weights and biases in DQN
        else:
            if i_episode % TARGET_UPDATE == 0:
                 brain.target_net.load_state_dict(brain.policy_net.state_dict())

        if i_episode % 1000 == 0:
            # Saving an intermediate model
            torch.save(brain, os.getcwd() + model_name + 'model.pt')

        temperatures.append(temperatures_episode)

    model_params = {'NUM_EPISODES':NUM_EPISODES,
                    'EPSILON':EPSILON,
                    'EPS_DECAY':EPS_DECAY,
                    'LEARNING_RATE_':LEARNING_RATE,
                    'GAMMA':GAMMA,
                    'TARGET_UPDATE':TARGET_UPDATE,
                    'BATCH_SIZE':BATCH_SIZE,
                     'TIME_STEP_SIZE':TIME_STEP_SIZE,
                    'NUM_HOURS':NUM_HOURS,
                    'E_PRICE':E_PRICE,
                    'COMFORT_PENALTY':COMFORT_PENALTY}

    scores.append(model_params)
    temperatures.append(model_params)
    with open(os.getcwd() + '/data/output/' + model_name + '_dynamic_' + str(dynamic) + '_rewards_dqn.pkl', 'wb') as f:
        pkl.dump(scores,f)

    with open(os.getcwd() + '/data/output/' + model_name + '_dynamic_' + str(dynamic) + '_temperatures_dqn.pkl', 'wb') as f:
        pkl.dump(temperatures,f)

    # Saving the final model
    torch.save(brain, os.getcwd() + model_name + 'model.pt')
    print('Complete')