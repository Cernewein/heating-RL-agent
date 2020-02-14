import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

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

        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.loss = nn.CrossEntropyLoss()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.device)
        x = F.relu(self.fc_1(state))
        x = F.relu(self.fc_2(x))
        actions = self.fc_3(x)
        return actions

class Agent(object):
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size = int(1e6), eps_end = 0.01, eps_dec = 0.996):
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.mem_counter = 0
        self.action_space = [i for i in range(n_actions)]
        self.max_mem_size = max_mem_size
        self.eps_dec = eps_dec
        self.eps_end = eps_end
        self.Q_eval = DeepQNetwork(lr, n_actions=self.n_actions, input_dims = input_dims,
                                   fc_1_dims=256, fc_2_dims=128)
        self.state_memory = np.zeros((self.max_mem_size, input_dims))
        self.new_state_memory = np.zeros((self.max_mem_size, input_dims))
        self.action_memory = np.zeros((self.max_mem_size, self.n_actions), dtype = np.uint8)
        self.reward_memory = np.zeros(self.max_mem_size)
        self.terminal_memory = np.zeros(self.max_mem_size, dtype = np.uint8)

    def store_transition(self, state, action, reward, state_):
        index = self.mem_counter % self.max_mem_size
        self.state_memory[index] = state
        #One-hot encoding the actions
        actions = np.zeros(self.n_actions)
        actions[action]=1.0
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        #self.terminal_memory = terminal
        self.mem_counter += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            actions = self.Q_eval.forward(observation)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.mem_counter > self.batch_size:
            self.Q_eval.optimizer.zero_grad()
            max_mem = self.mem_counter if self.mem_counter<self.max_mem_size else self.max_mem_size
            batch = np.random.choice(max_mem, self.batch_size)
            state_batch = self.state_memory[batch]
            action_batch = self.action_memory[batch]
            action_values = np.array(self.action_space, dtype=np.uint8)
            action_indices = np.dot(action_batch, action_values)
            reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
            new_state_batch = self.new_state_memory[batch]

            q_eval = self.Q_eval.forward(state_batch).to(self.Q_eval.device)
            q_target = self.Q_eval.forward(state_batch).to(self.Q_eval.device)
            q_next = self.Q_eval.forward(new_state_batch).to(self.Q_eval.device)

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[action_batch] = reward_batch + self.gamma*T.max(q_next,dim=1)[0]
            self.epsilon = self.epsilon*self.eps_dec if self.epsilon > self.eps_end else self.eps_end

            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()

