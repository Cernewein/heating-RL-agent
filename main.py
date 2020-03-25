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
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--model_name", default='')
    parser.add_argument("--dynamic", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--soft", default=False,type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--eval", default=False, type=lambda x: (str(x).lower() == 'true'))
    return parser.parse_args()


def run(ckpt,model_name,dynamic,soft, eval):

    if not eval:
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
                        'LEARNING_RATE':LEARNING_RATE,
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

    else:
        if ckpt:
            brain = torch.load(ckpt,map_location=torch.device('cpu'))
            brain.epsilon = 0
            brain.eps_end = 0
            env = Building(dynamic=True, eval=True)
            inside_temperatures = [env.inside_temperature]
            ambient_temperatures = [env.ambient_temperature]
            prices = [env.price]
            actions = [0]
            rewards=[0]
            print('Starting evaluation of the model')
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float).to(device)
            # Normalizing data using an online algo
            brain.normalizer.observe(state)
            state = brain.normalizer.normalize(state).unsqueeze(0)
            for t_episode in range(NUM_TIME_STEPS):
                action = brain.select_action(state).type(torch.FloatTensor)
                prices.append(env.price) # Will be replaced with environment price in price branch
                actions.append(action.item())
                next_state, reward, done = env.step(action.item())
                rewards.append(reward)
                inside_temperatures.append(env.inside_temperature)
                ambient_temperatures.append(env.ambient_temperature)
                if not done:
                    next_state = torch.tensor(next_state, dtype=torch.float, device=device)
                    # normalize data using an online algo
                    brain.normalizer.observe(next_state)
                    next_state = brain.normalizer.normalize(next_state).unsqueeze(0)
                else:
                    next_state = None
                # Move to the next state
                state = next_state

            eval_data = pd.DataFrame()
            eval_data['Inside Temperatures'] = inside_temperatures
            eval_data['Ambient Temperatures'] = ambient_temperatures
            eval_data['Prices'] = prices
            eval_data['Actions'] = actions
            eval_data['Rewards'] = rewards
            #with open(os.getcwd() + '/data/output/' + model_name + '_january_eval.pkl', 'wb') as f:
                #pkl.dump(eval_data, f)

            print('Finished the evaluation on January \n'+
                  'Starting policy evaluation')
            # We will run through a number of combinations for inside temperature,
            # Outside temperature and price. Time and sun will be fixed for this evaluation
            # Values will onlu be saved if decision output by agent is equal to 1
            inside_temperatures_1 = []
            ambient_temperatures_1 = []
            prices_1 = []

            for inside_temp in np.arange(0,30, 1/TEMPERATURE_ROUNDING):
                for ambient_temp in np.arange(-10,15, 1/TEMPERATURE_ROUNDING):
                    for price in range(0,40):
                        state = [inside_temp, ambient_temp, 100, price , 12]
                        state = torch.tensor(state, dtype=torch.float).to(device)
                        state = brain.normalizer.normalize(state).unsqueeze(0)
                        action = brain.select_action(state).type(torch.FloatTensor).item()
                        if action == 1.0:
                            inside_temperatures_1.append(inside_temp)
                            ambient_temperatures_1.append(ambient_temp)
                            prices_1.append(price)

            eval_data = pd.DataFrame()
            eval_data['Inside Temperatures'] = inside_temperatures_1
            eval_data['Ambient Temperatures'] = ambient_temperatures_1
            eval_data['Prices'] = prices_1
            with open(os.getcwd() + '/data/output/' + model_name + 'policy_eval.pkl', 'wb') as f:
                pkl.dump(eval_data, f)

        else:
            print('If no training should be performed, then please choose a model that should be evaluated')

if __name__ == '__main__':
    args = parse_args()
    run(**vars(args))