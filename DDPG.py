from vars import *
from utils import Normalizer, ReplayMemory, Transition, OUNoise
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        num_outputs = action_space.shape[0]

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        #self.ln1 = nn.LayerNorm(hidden_size[0])

        # Layer 2
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        #self.ln2 = nn.LayerNorm(hidden_size[1])

        # Output Layer
        self.mu = nn.Linear(hidden_size[1], num_outputs)

    def forward(self, inputs):
        x = inputs

        # Layer 1
        x = self.linear1(x)
        #x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = self.linear2(x)
        #x = self.ln2(x)
        x = F.relu(x)

        # Output --> Mapped into [0,1] domain
        mu = torch.sigmoid(self.mu(x))
        return mu

class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        num_outputs = action_space.shape[0]

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        #self.ln1 = nn.LayerNorm(hidden_size[0])

        # Layer 2
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])

        #Layer 3 (for the actions)
        self.linear3 = nn.Linear(num_outputs, hidden_size[2])

        # Layer 4 - The combination layer
        # In the fourth layer the actions will be inserted also
        self.linear4 = nn.Linear(hidden_size[1] + hidden_size[2], hidden_size[3])
        #self.ln2 = nn.LayerNorm(hidden_size[1])

        # Output layer (single value)
        self.V = nn.Linear(hidden_size[3], 1)


    def forward(self, inputs, actions):
        x = inputs

        # Layer 1
        x = self.linear1(x)
        #x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = self.linear2(x)
        # x = self.ln1(x)
        #x = F.relu(x)

        # Layer 3
        actions = self.linear3(actions)
        # x = self.ln1(x)
        #x = F.relu(x)

        # Layer 4
        x = torch.cat((x, actions), 1)  # Insert the actions
        x = self.linear4(x)
        #x = self.ln2(x)
        x = F.relu(x)

        # Output
        V = self.V(x)
        return V

class DDPGagent(object):

    def __init__(self, gamma=GAMMA, tau=TAU, hidden_size_actor=[300,600], hidden_size_critic=[300,600,600,600],
                 num_inputs=INPUT_DIMS, action_space=np.array([[0]]), batch_size = BATCH_SIZE, mem_size =int(1e6), epsilon = EPSILON,
                 eps_dec=EPS_DECAY, eps_end = 0.1,lr_actor = LEARNING_RATE_ACTOR, lr_critic = LEARNING_RATE_CRITIC, random_seed = 42,add_noise = True):
        """
        Based on https://arxiv.org/abs/1509.02971 - Continuous control with deep reinforcement learning

        :param gamma: Discount factor
        :param tau: Factor for the soft update of the agent target networks
        :param hidden_size_actor: List for the hidden sizes of the actor. Must be of size 2
        :param hidden_size_critic: List for the hidden sizes of the critic. Must be of size 4
        :param num_inputs: Number of inputs for the layers (number of variables in the state)
        :param action_space: The action space for the used environment.
        """
        self.gamma = gamma
        self.tau = tau
        self.action_space = action_space

        self.epsilon = epsilon
        self.epsilon_threshold = epsilon
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.batch_size = batch_size

        self.normalizer = Normalizer(num_inputs)
        self.memory = ReplayMemory(mem_size)
        self.steps_done = 0

        # Define the actor
        self.actor = Actor(hidden_size_actor, num_inputs, self.action_space).to(device)
        self.actor_target = Actor(hidden_size_actor, num_inputs, self.action_space).to(device)

        # Define the critic
        self.critic = Critic(hidden_size_critic, num_inputs, self.action_space).to(device)
        self.critic_target = Critic(hidden_size_critic, num_inputs, self.action_space).to(device)

        # Define the optimizers for both networks
        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                    lr=lr_actor)  # optimizer for the actor network
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                     lr=lr_critic,
                                     weight_decay=1e-2
                                     )  # optimizer for the critic network

        # Make sure both targets are with the same weight
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        self.actor_target.eval()
        self.critic_target.eval()

        # Noise process
        self.noise = OUNoise(1, random_seed)
        self.add_noise = add_noise

    def hard_update(self, target, source):
        target.load_state_dict(source.state_dict())

    def soft_update(self, target, source):
        """Soft update model parameters.
                θ_target = τ*θ_local + (1 - τ)*θ_target
                Params
                ======
                    local_model (PyTorch model): weights will be copied from
                    target_model (PyTorch model): weights will be copied to
                """
        for target_param, local_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def select_action(self, state):
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action = self.actor(state)
            if self.add_noise:
                action = action.cpu().item()
                action += self.noise.sample()
                action = np.clip(action, 0,1)
                return torch.from_numpy(action).float().to(device)
            else:
                sample = random.random()
                self.epsilon_threshold = self.epsilon * (
                           self.eps_dec ** self.steps_done) if self.epsilon_threshold > self.eps_end else self.eps_end
                self.steps_done += 1
                if sample > self.epsilon_threshold:
                    return action
                else:
                    return torch.tensor([[random.random()]], dtype=torch.float).to(device)


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
        state_batch = torch.cat(batch.state).to(device)

        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)

        # Compute Q(s_{t+1}, a_{t+1}) for all next states and actions.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_action_values = torch.zeros(self.batch_size, device=device)
        next_action_batch = self.actor_target(non_final_next_states)
        next_state_action_values[non_final_mask] = self.critic_target(non_final_next_states,next_action_batch.detach()).squeeze()

        # Compute the expected Q values (or yi in the original paper)
        expected_state_action_values = (next_state_action_values * self.gamma) + reward_batch

        # Update the critic network
        self.critic_optimizer.zero_grad()
        state_action_values = self.critic(state_batch, action_batch)
        # Compute Huber loss
        value_loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        value_loss.backward()
        self.critic_optimizer.step()

        # Update actor network
        predicted_actions = self.actor.forward(state_batch)
        loss_actor = (-self.critic.forward(state_batch, predicted_actions)).mean()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # Soft update for the target networks
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)