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
from train_dqn import train_dqn
from train_ddpg import train_ddpg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--model_name", default='')
    parser.add_argument("--dynamic", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--soft", default=False,type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--eval", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--model_type", default='DDPG')
    parser.add_argument("--noisy", default=True, type=lambda x: (str(x).lower() == 'true'))
    return parser.parse_args()


def run(ckpt,model_name,dynamic,soft, eval, model_type, noisy):

    if not eval:
        if model_type == 'DQN':
            train_dqn(ckpt, model_name, dynamic, soft)
        else:
            train_ddpg(ckpt, model_name, dynamic, noisy)
    else:
        if ckpt:
            brain = torch.load(ckpt,map_location=torch.device('cpu'))
            brain.epsilon = 0
            brain.eps_end = 0
            env = Building(dynamic=True, eval=True)
            inside_temperatures = [env.inside_temperature]
            ambient_temperatures = [env.ambient_temperature]
            storage_state = [env.storage]
            prices = [env.price]
            power_from_grid = [env.power_from_grid]
            actions = [[0,0]]
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
                actions.append(action.numpy())
                next_state, reward, done = env.step(action.numpy())
                rewards.append(reward)
                inside_temperatures.append(env.inside_temperature)
                ambient_temperatures.append(env.ambient_temperature)
                storage_state.append(env.storage)
                power_from_grid.append(env.power_from_grid)
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
            eval_data['Storage'] = storage_state
            eval_data['Power'] = power_from_grid
            with open(os.getcwd() + '/data/output/' + model_name + '_eval.pkl', 'wb') as f:
                pkl.dump(eval_data, f)

            print('Finished evaluation on January, evaluating the policy.')

            inside_temperatures = []
            battery_actions = []
            heating_actions = []
            prices = []
            ambient_temperatures = []
            battery_levels = []
            sun_powers = []
            times = []


            for inside_temp in np.arange(19, 21, 1):
                print(inside_temp)
                for ambient_temp in np.arange(-5, 5, 1):
                    for price in range(10, 55):
                        for battery_level in np.arange(1000, 5000, 1000):
                            for sun_power in np.arange(0,0,10):
                                for time in range(0,23):
                                    state = [inside_temp, ambient_temp, sun_power, price, battery_level, time]
                                    state = torch.tensor(state, dtype=torch.float).to(device)
                                    state = brain.normalizer.normalize(state).unsqueeze(0)
                                    action = brain.select_action(state).type(torch.FloatTensor).numpy()
                                    battery_actions.append(action[1])
                                    heating_actions.append(action[0])
                                    inside_temperatures.append(inside_temp)
                                    ambient_temperatures.append(ambient_temp)
                                    prices.append(price)
                                    actions.append(action)
                                    battery_levels.append(battery_level)
                                    sun_powers.append(sun_power)
                                    times.append(time)

            eval_data = pd.DataFrame()
            eval_data['Inside Temperatures'] = inside_temperatures
            eval_data['Ambient Temperatures'] = ambient_temperatures
            eval_data['Battery Level'] = battery_levels
            eval_data['Prices'] = prices
            eval_data['Battery Action'] = battery_actions
            eval_data['Heating Action'] = heating_actions
            eval_data['Sun Power'] = sun_powers
            eval_data['Time'] = times
            with open(os.getcwd() + '/data/output/' + model_name + 'policy_eval.pkl', 'wb') as f:
             pkl.dump(eval_data, f)

        else:
            print('If no training should be performed, then please choose a model that should be evaluated')

if __name__ == '__main__':
    args = parse_args()
    run(**vars(args))