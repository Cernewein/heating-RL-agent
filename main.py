from DQN import DAgent, device
from agent import Agent
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
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--model_name", default='')
    parser.add_argument("--dynamic", default=False, type=lambda x: (str(x).lower() == 'true'))
    return parser.parse_args()


def run(deep,ckpt,model_name,dynamic):

    env = Building(dynamic)

    if deep:
        scores = []
        temperatures = []
        brain = DAgent(gamma=GAMMA, epsilon=EPSILON, batch_size=BATCH_SIZE, n_actions=N_ACTIONS,
                      input_dims=INPUT_DIMS,  lr = LEARNING_RATE, eps_dec = EPS_DECAY, ckpt=ckpt)
        start = time.time()
        for i_episode in range(NUM_EPISODES):
            # Initialize the environment.rst and state
            state = env.reset()
            temperatures_episode = [state[0]]
            state = torch.tensor(state,dtype=torch.float).unsqueeze(0).to(device)
            score = 0
            for t in count():
                # Select and perform an action
                action = brain.select_action(state).type(torch.FloatTensor)
                next_state, reward, done = env.step(action.item())
                score += reward
                reward = torch.tensor([reward],dtype=torch.float,device=device)

                if not done:
                    temperatures_episode.append(next_state[0])
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

            if i_episode % 1000 == 0:
                # Saving an intermediate model
                torch.save(brain.policy_net.state_dict(), os.getcwd() + model_name + 'model.pt')

            temperatures.append(temperatures_episode)

        model_params = {'NUM_EPISODES':NUM_EPISODES,
                        'EPSILON':EPSILON,
                        'EPS_DECAY':EPS_DECAY,
                        'LEARNING_RATE':LEARNING_RATE,
                        'GAMMA':GAMMA,
                        'TARGET_UPDATE':TARGET_UPDATE,
                        'BATCH_SIZE':BATCH_SIZE,
                         'TIME_STEP_SIZE':TIME_STEP_SIZE}

        scores.append(model_params)
        temperatures.append(model_params)
        with open(os.getcwd() + '/data/output/' + model_name + '_dynamic_' + dynamic + '_rewards_dqn.pkl', 'wb') as f:
            pkl.dump(scores,f)

        with open(os.getcwd() + '/data/output/' + model_name + '_dynamic_' + dynamic + '_temperatures_dqn.pkl', 'wb') as f:
            pkl.dump(temperatures,f)

        # Saving the final model
        torch.save(brain.policy_net.state_dict(), os.getcwd() + model_name + 'model.pt')
        print('Complete')

    else: # If we are not in the deep case, we run the classic q-learning agent
        agent = Agent()
        agent.train(NUM_EPISODES, NUM_TIME_STEPS)

        plt.figure()
        plt.plot(agent.episode_rewards)
        plt.savefig(
            os.getcwd() + '/data/output/' + 'rewards_LR_' + str(LEARNING_RATE) + '_G_' + str(GAMMA) + '_EPS_' + str(
                EPSILON) + '_TS_' + str(TIME_STEP_SIZE) + '.png')

        plt.figure()
        plt.plot(agent.temperature_evolutions[0])
        plt.savefig(os.getcwd() + '/data/output/' + 'temperatures_initial_LR_' + str(LEARNING_RATE) + '_G_' + str(
            GAMMA) + '_EPS_' + str(EPSILON) + '_TS_' + str(TIME_STEP_SIZE) + '.png')

        plt.figure()
        plt.plot(agent.temperature_evolutions[-1])
        plt.savefig(os.getcwd() + '/data/output/' + 'temperatures_trained_LR_' + str(LEARNING_RATE) + '_G_' + str(
            GAMMA) + '_EPS_' + str(EPSILON) + '_TS_' + str(TIME_STEP_SIZE) + '.png')

        for t in range(T_BOUND_MIN, T_BOUND_MAX):
            print("\n Temperature {}".format(t))
            print(agent.q_table[t / 1.])

if __name__ == '__main__':
    args = parse_args()
    run(**vars(args))