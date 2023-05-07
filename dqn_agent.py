"""
DQN Agent implementation for SelfAI project.
This script contains a DQN neural network model and a DQNAgent class for training and action selection.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from collections import deque, namedtuple

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=1e-3, discount_factor=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=64, prioritized_experience_replay_alpha=0.6,
                 prioritized_experience_replay_beta=0.4, prioritized_experience_replay_beta_increment=1e-4):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.prioritized_experience_replay_alpha = prioritized_experience_replay_alpha
        self.prioritized_experience_replay_beta = prioritized_experience_replay_beta
        self.prioritized_experience_replay_beta_increment = prioritized_experience_replay_beta_increment

        # Q-Network and Target Network
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()
        self.update_target_network()

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                return np.argmax(self.q_network(state).numpy())

    def remember(self, state, action, reward, next_state, done):
        max_priority = max(self.memory, default=1, key=lambda transition: transition[5]) if self.memory else 1
        self.memory.append(Transition(state, action, reward, next_state, done, max_priority ** self.prioritized_experience_replay_alpha))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        indices, weights = self.sample_prioritized_transitions()
        transitions = [self.memory[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*[(t.state, t.action, t.reward, t.next_state, t.done) for t in transitions])

        states = torch.FloatTensor(states)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1)

        q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).detach().max
        
        # Double DQN update
        next_actions = self.q_network(next_states).detach().argmax(1).unsqueeze(1)
        next_q_values = self.target_network(next_states).gather(1, next_actions)

        target_q_values = rewards + self.discount_factor * next_q_values * (1 - dones)

        # Compute the TD error and update the priorities
        td_errors = (target_q_values - q_values).abs().detach().numpy().squeeze()
        self.update_priorities(indices, td_errors)

        # Prioritized Experience Replay weight update
        self.prioritized_experience_replay_beta += self.prioritized_experience_replay_beta_increment
        self.prioritized_experience_replay_beta = min(self.prioritized_experience_replay_beta, 1)

        # Compute the loss and perform the optimization step
        loss = (weights * self.loss_function(q_values, target_q_values.detach())).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_end)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def sample_prioritized_transitions(self):
        total_priority = sum(transition.priority for transition in self.memory)
        probabilities = [transition.priority / total_priority for transition in self.memory]

        indices = np.random.choice(range(len(self.memory)), size=self.batch_size, p=probabilities, replace=False)
        weights = [(len(self.memory) * p) ** -self.prioritized_experience_replay_beta for p in probabilities]

        return indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            transition = self.memory[idx]
            updated_priority = (abs(td_error) + 1e-5) ** self.prioritized_experience_replay_alpha
            self.memory[idx] = transition._replace(priority=updated_priority)

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_size)
        )

    def forward(self, x):
        return self.model(x)
