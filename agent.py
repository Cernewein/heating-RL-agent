import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pickle as pkl
import time
from vars import *
from environment import Building
from progress.bar import Bar

class Agent():
    def __init__(self, epsilon = EPSILON, start_q_table=None):
        self.episode_rewards = []
        self.temperature_evolutions = []
        self.epsilon = epsilon
        if start_q_table is None:
            self.q_table = {}
            for t in np.arange(T_BOUND_MIN,T_BOUND_MAX + 1/TEMPERATURE_ROUNDING, 1/TEMPERATURE_ROUNDING):
                self.q_table[np.round(t, decimals = 4)] = np.random.uniform(-100,-50,2)
        else:
            with open(start_q_table, "rb") as f:
                self.q_table = pkl.load(f)
        #print(self.q_table.keys())


    def train(self, number_episodes, number_time_steps):
        #bar = Bar('Training..', max = number_episodes)
        for episode in range(number_episodes):
            start = time.time()
            building = Building()
            episode_reward = 0
            episode_temperatures = []
            for step in range(number_time_steps):
                temperature = building.get_inside_temperature()/1.
                episode_temperatures.append(temperature)
                if np.random.random() > self.epsilon:
                    action = np.argmax(self.q_table[temperature])
                else:
                    action = np.random.randint(0,1)
                #action = 0
                building.action(action)
                reward = building.reward(action)
                new_temperature = building.get_inside_temperature()/1.
                current_q = self.q_table[temperature][action]
                max_future_q = np.max(self.q_table[new_temperature])
                new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT*max_future_q)

                self.q_table[temperature/1.][action] = new_q

                episode_reward += reward
            self.episode_rewards.append(episode_reward)
            self.epsilon *= EPS_DECAY
            if episode %100 == 0:
                self.temperature_evolutions.append(episode_temperatures)
                end = time.time()
                print('Just finished episode {} after {} seconds\n'.format(episode, end - start))
                print('Current reward {}\n'.format(episode_reward))
            #bar.next()
        with open('q_table.pkl', 'wb') as f:
            pkl.dump(self.q_table,f)

        with open('rewards.pkl', 'wb') as f:
            pkl.dump(self.episode_rewards,f)

        with open('temperatures.pkl', 'wb') as f:
            pkl.dump(self.temperature_evolutions,f)
        #bar.finish()

    def basic_controller(self,number_time_steps):
        """

        :param number_time_steps:
        :return:
        """
        building = Building()
        self.temperatures = []
        self.rewards = []
        self.action = 0
        for _ in range(number_time_steps):
            if building.inside_temperature > T_MAX-1/TEMPERATURE_ROUNDING:
                self.action = 0
            elif building.inside_temperature < T_MIN+1/TEMPERATURE_ROUNDING:
                self.action = 1
                #print('taking heating action')
            building.action(self.action)
            self.temperatures.append(building.inside_temperature)
            self.rewards.append(building.reward(self.action))

        with open('rewards_basic.pkl', 'wb') as f:
            pkl.dump(self.rewards,f)

        with open('temperatures_basic.pkl', 'wb') as f:
            pkl.dump(self.temperatures,f)