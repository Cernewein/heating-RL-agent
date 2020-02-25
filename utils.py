import torch
from vars import *
from collections import namedtuple
import random
from environment import Building
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class Normalizer():
    """
    Normalizes the input data by computing an online variance and mean
    """
    def __init__(self, num_inputs):
        self.n = torch.zeros(num_inputs).to(device)
        self.mean = torch.zeros(num_inputs).to(device)
        self.mean_diff = torch.zeros(num_inputs).to(device)
        self.var = torch.zeros(num_inputs).to(device)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

    def normalize(self, inputs):
        obs_std = torch.sqrt(self.var).to(device)
        return (inputs - self.mean)/obs_std


class ReplayMemory(object):
    """
    This class serves as storage capability for the replay memory. It stores the Transition tuple
    (state, action, next_state, reward) that can later be used by a DQN agent for learning based on experience replay.

    :param capacity: The size of the replay memory
    :type capacity: Integer
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition. (the transition tuple)"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Randomly selects batch_size elements from the memory.

        :param batch_size: The wanted batch size
        :type batch_size: Integer
        :return:
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class BasicController():

    def __init__(self, number_time_steps, dynamic):
        self.number_time_steps = number_time_steps
        self.building = Building(dynamic)
        self.temperatures = []
        self.costs = []
        self.action = 0

    def basic_controller(self):
        """
        Represents a very basic control mechanism that is used as baseline for comparision. It heats until T=T_max
        and then turns the heating off until T_min is reached
        :param number_time_steps:
        :return:
        """

        for _ in range(self.number_time_steps):
            if self.building.inside_temperature > T_MAX - 1 / TEMPERATURE_ROUNDING:
                self.action = 0
            elif self.building.inside_temperature < T_MIN + 1 / TEMPERATURE_ROUNDING:
                self.action = 1

            self.building.step(self.action)
            self.temperatures.append(self.building.inside_temperature)
            self.costs.append(action*NOMINAL_HEAT_PUMP_POWER*self.building.price*TIME_STEP_SIZE/3600)

        with open(os.getcwd() + '/data/output/' + 'costs_basic.pkl', 'wb') as f:
            pkl.dump(self.rewards, f)

        with open(os.getcwd() + '/data/output/' + 'temperatures_basic.pkl', 'wb') as f:
            pkl.dump(self.temperatures, f)