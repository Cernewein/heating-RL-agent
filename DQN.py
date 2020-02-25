import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from vars import *
from utils import Normalizer, ReplayMemory, Transition

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class DeepQNetwork(nn.Module):
    """Deep Q Network for the heating control problem.

    :param lr: The learning rate
    :type lr: Float
    :param input_dims: The number of input dimensions (based on the variables explaining the state)
    :type input_dims: Integer
    :param fc_1_dims: The number of neurons for the first fully-connected layer
    :type fc_1_dims: Integer
    :param fc_2_dims: The number of neurons for the second fully-connected layer
    :type fc_2_dims: Integer
    :param n_actions: The number of actions that can be selected
    :type n_actions: Integer
    """
    def __init__(self, lr, input_dims, fc_1_dims, fc_2_dims, fc_3_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc_1_dims = fc_1_dims
        self.fc_2_dims = fc_2_dims
        self.fc_3_dims = fc_3_dims
        self.n_actions = n_actions

        self.fc_1 = nn.Linear(self.input_dims, self.fc_1_dims)
        self.fc_2 = nn.Linear(self.fc_1_dims, self.fc_2_dims)
        #self.fc_3 = nn.Linear(self.fc_2_dims, self.fc_3_dims)
        self.fc_4 = nn.Linear(self.fc_3_dims, self.n_actions)

        self.to(device)

    def forward(self, observation):
        """
        This method does one forward-pass for the Q-network.

        :param observation: The observation ( or state) for which we want to compute the Q-values
        :return: A Tensor with Q-values associated to each state
        """
        state = observation.clone().detach().to(device)
        x = F.relu(self.fc_1(state))
        x = F.relu(self.fc_2(x))
        #x = F.relu(self.fc_3(x))
        actions = self.fc_4(x).type(torch.FloatTensor)
        return actions.to(device)

class DAgent():
    """
    The agent class that will be controlling the environment.rst.

    :param gamma: The discount factor for the Q-values update
    :param epsilon: The probability for epsilon-greedy approach
    :param lr: The learning rate
    :param input_dims: The number of input dimensions (how many variables characterise the state)
    :param batch_size: The batch size
    :param n_actions: The number of actions that can be performed
    :param mem_size: The number of transitions that should be stored
    :param eps_end: The minimum epsilon that should be achieved
    :param eps_dec: The decay applied to epsilon after each epoch
    """
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 mem_size = int(1e6), momentum=0.95 ,eps_end = 0.1, eps_dec = 0.996,ckpt=None):
        """Constructor method
        """
        self.normalizer = Normalizer(input_dims)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_threshold = epsilon
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.eps_dec = eps_dec
        self.eps_end = eps_end
        self.policy_net= DeepQNetwork(lr, n_actions=self.n_actions, input_dims = input_dims,
                                   fc_1_dims=FC_1_DIMS, fc_2_dims=FC_2_DIMS, fc_3_dims=FC_3_DIMS)
        self.target_net = DeepQNetwork(lr, n_actions=self.n_actions, input_dims=input_dims,
                                       fc_1_dims=FC_1_DIMS, fc_2_dims=FC_2_DIMS, fc_3_dims=FC_3_DIMS)
        if ckpt:
            checkpoint = torch.load(ckpt)
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr, momentum=momentum) #optim.Adam(self.policy_net.parameters(), lr=lr) #
        self.memory = ReplayMemory(mem_size)
        self.steps_done = 0

    def select_action(self,state):
        """
        Selects the next action via an epsilon-greedy approach based on a state

        :param state: The state for which the action should be chosen
        :type state: Tensor
        :return: Returns the selected action
        """
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
        """
        Runs one optimization step if there is enough experience in the replay memory.
        """
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
        state_batch = torch.cat(batch.state).to(device)

        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.type(torch.LongTensor).to(device))

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

    def soft_update(self, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

