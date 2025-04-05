from collections import defaultdict
import gymnasium as gym
import numpy as np


class QLAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
        self.avg_reward = 0
        self.timestamp = 0

    def get_action(self, obs: int) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environmentse
        self.decay_epsilon()
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: int,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: int,
    ):
        self.timestamp += 1
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)
        self.avg_reward = self.avg_reward + (1/self.timestamp)*(reward - self.avg_reward)
        print(f"Avg reward : {self.avg_reward}\n")
        print(f"Iteration: {self.timestamp}")

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

import torch
from collections import deque
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.fc(x)



import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        buffer_size: int = 2000,
        batch_size: int = 64,
        target_update_freq: int = 100
    ):
        """Initialize a DQN agent with replay buffer and target network.
        
        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
            buffer_size: Size of the replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency for updating target network
        """
        self.env = env
        self.observation_dim = env.observation_space.n
        self.action_dim = env.action_space.n
        
        # Q-Network and target network
        self.policy_net = self._build_network()
        self.target_net = self._build_network()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        
        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.target_update_freq = target_update_freq
        
        # Tracking (maintaining same interface as QLAgent)
        self.training_error = []
        self.avg_reward = 0
        self.timestamp = 0
    
    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.observation_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
    
    def get_action(self, obs: int) -> int:
        """Returns the best action with probability (1 - epsilon), otherwise a random action."""
        self.decay_epsilon()
        
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                obs_onehot = self._one_hot_encode(obs)
                q_values = self.policy_net(obs_onehot)
                return int(torch.argmax(q_values).item())
    
    def update(
        self,
        obs: int,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: int,
    ):
        """Store transition in replay buffer and perform training if enough samples."""
        self.timestamp += 1
        
        # Store transition in replay buffer
        self.replay_buffer.push(obs, action, reward, next_obs, terminated)
        
        # Update average reward (same as QLAgent)
        self.avg_reward = self.avg_reward + (1/self.timestamp)*(reward - self.avg_reward)
        print(f"Avg reward : {self.avg_reward}\n")
        print(f"Iteration: {self.timestamp}")
        
        # Train if enough samples
        if len(self.replay_buffer) >= self.batch_size:
            self._train_network()
    
    def _train_network(self):
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.stack([self._one_hot_encode(s) for s in states])
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.stack([self._one_hot_encode(s) for s in next_states])
        dones = torch.tensor(dones, dtype=torch.float)
        
        # Current Q values for chosen actions
        current_q = self.policy_net(states).gather(1, actions)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.discount_factor * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.training_error.append(loss.item())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        if self.timestamp % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def _one_hot_encode(self, obs: int) -> torch.Tensor:
        one_hot = torch.zeros(self.observation_dim)
        one_hot[obs] = 1
        return one_hot
    
    def decay_epsilon(self):
        """Decay epsilon over time."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)