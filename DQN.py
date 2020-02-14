import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DeepQNetwork(nn.Module):
    """Deep Q Network for the heating control problem."""
    def __init__(self, lr, input_dims, fc_1_dims, fc_2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc_1_dims = fc_1_dims
        self.fc_2_dims = fc_2_dims
        self.n_actions = n_actions

        self.fc_1 = nn.Linear(self.input_dims, self.fc_1_dims)
        self.fc_2 = nn.Linear(self.fc_1_dims, self.fc_2_dims)
        self.fc_3 = nn.Linear(self.fc_2_dims, self.n_actions)

        self.to(device)

    def forward(self, observation):
        state = observation.clone().detach().to(device)
        x = F.relu(self.fc_1(state))
        x = F.relu(self.fc_2(x))
        actions = self.fc_3(x).type(torch.FloatTensor)
        return actions.to(device)

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size = int(1e6), eps_end = 0.01, eps_dec = 0.996):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_threshold = epsilon
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.mem_counter = 0
        self.action_space = [i for i in range(n_actions)]
        self.max_mem_size = max_mem_size
        self.eps_dec = eps_dec
        self.eps_end = eps_end
        self.policy_net= DeepQNetwork(lr, n_actions=self.n_actions, input_dims = input_dims,
                                   fc_1_dims=256, fc_2_dims=128)
        self.target_net = DeepQNetwork(lr, n_actions=self.n_actions, input_dims=input_dims,
                                       fc_1_dims=256, fc_2_dims=128)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

    def select_action(self,state):
        sample = random.random()
        self.epsilon_threshold = self.epsilon * (self.eps_dec**self.steps_done) if self.epsilon_threshold > self.eps_end else self.eps_end
        self.steps_done += 1
        if sample > self.epsilon_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]],dtype=torch.float).to(device)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)

        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.type(torch.LongTensor))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
