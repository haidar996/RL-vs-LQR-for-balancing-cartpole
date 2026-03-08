#!usr/bin/env python3 

import rospy 
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=90000000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(
            np.array, zip(*batch)
        )
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(QNetwork, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(state_dim,128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 64),
            
            nn.LeakyReLU(0.01),
            nn.Linear(64, action_dim)
        )
    def forward(self,x):
        return self.net(x)

class DDQNAgent:
    def __init__(self, state_dim=4, action_dim=6, mode="test"):
        self.device = torch.device("cpu")
        self.memory = ReplayBuffer()
        self.batch_size = 128
        self.mode = mode  # "train" or "test"

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        
        if self.mode == "train":
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)
            self.epsilon = 1.0
            self.epsilon_min = 0.0001
            self.epsilon_decay = 0.9985
        else:
            # In test mode, no optimizer or epsilon decay needed
            self.epsilon = 0.0  # Always use the model
            self.optimizer = None

        self.gamma = 0.99
        self.action_dim = action_dim
        
    def load_model(self, model_path="/home/haidar/.ros/interrupted_models/ddqn_model_episode_6157.pth"):
        """Load a trained model"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(checkpoint['q_network_state_dict'])
        if 'target_network_state_dict' in checkpoint:
            self.target_net.load_state_dict(checkpoint['target_network_state_dict'])
        self.q_net.eval()
        self.target_net.eval()
        rospy.loginfo(f"Model loaded from {model_path}")
        
    def select_action(self, state, force_test=False):
        """Select action, with optional force_test parameter"""
        # If in test mode or force_test=True, always use the model
        if self.mode == "test" or force_test:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            # Return only the action, not a tuple
            return q_values.argmax().item()
        
        # Training mode: use epsilon-greedy
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        # Return only the action, not a tuple
        return q_values.argmax().item()
    def select_action_simple(self, state):
        """Simple version: always use model, return only action"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()
    
    def test_prediction(self, state):
        """Test the model with a state and return full info"""
        state_norm = self.normalize_state(state)
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        
        action = q_values.argmax().item()
        q_vals = q_values[0].cpu().numpy()
        
        return {
            'action': action,
            'q_values': q_vals,
            'normalized_state': state_norm,
            'best_q_value': q_vals[action]
        }
    
    def update_epsilon(self):
        if self.mode == "train":
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def normalize_state(self, state):
        normalized = np.copy(state)
        normalized[0] = state[0] / 2.0  # Cart position
        normalized[1] = np.tanh(state[1] / 3.0)  # Cart velocity
        normalized[2] = state[2] / 0.2  # Pole angle
        normalized[3] = np.tanh(state[3] / 3.0)  # Pole angular velocity
        return normalized
    
    def train_step(self):
        if self.mode != "train":
            rospy.logwarn("Cannot train in test mode!")
            return
        
        if len(self.memory) < self.batch_size:
            self.loss = 0
            return
        
        # Sample batch
        state, action, reward, next_state, done = \
            self.memory.sample(self.batch_size)
        state = self.normalize_state(state)
        next_state = self.normalize_state(next_state)
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # Current Q values
        q_values = self.q_net(state).gather(1, action)

        # DDQN target
        with torch.no_grad():
            next_actions = self.q_net(next_state).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_state).gather(1, next_actions)
            target_q = reward + self.gamma * next_q_values * (1 - done)

        # Loss
        self.loss = nn.MSELoss()(q_values, target_q)

        # Optimize
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        if self.mode == "train":
            self.target_net.load_state_dict(self.q_net.state_dict())