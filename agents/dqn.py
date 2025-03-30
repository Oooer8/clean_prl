import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from copy import deepcopy
from core.representations.pointnet import PointNet
from core.representations.mlpnet import MLPNet
from core.policies.mlp import MLPPolicy
from core.values.mlp import MLPValue
from core.buffers.replay import ReplayBuffer
from .base import BaseAgent

class DQNAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(config)
        self.action_dim = action_dim
        
        # Networks
        self.representation = MLPNet(config.representation).to(self.device)
        self.q_net = MLPValue(
            input_dim=config.representation.output_dim + action_dim,
            config=config.value
        ).to(self.device)
        
        self.target_q_net = deepcopy(self.q_net).to(self.device)
        for param in self.target_q_net.parameters():
            param.requires_grad = False
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.representation.parameters()) + list(self.q_net.parameters()),
            lr=config.agent.lr
        )
        
        # Buffer
        self.buffer = ReplayBuffer(state_dim, action_dim, config.buffer)
        
        # Hyperparameters
        self.gamma = config.agent.gamma
        self.tau = config.agent.tau
        self.update_every = config.agent.update_every
        self.steps = 0
        
    def act(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return np.random.uniform(-1, 1, size=self.action_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            representation = self.representation(state)
            
            # DQN specific: find action that maximizes Q-value
            # Here we use a simple approach for continuous actions - sample and pick best
            # For real applications, consider using DDPG or other methods
            actions = torch.rand(10, self.action_dim) * 2 - 1  # Sample 10 random actions
            actions = actions.to(self.device)
            
            # Expand representation to match actions
            representations = representation.repeat(10, 1)
            q_input = torch.cat([representations, actions], dim=1)
            q_values = self.q_net(q_input)
            
            best_action = actions[q_values.argmax()]
            return best_action.cpu().numpy()
    
    def learn(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        representations = self.representation(states)
        q_input = torch.cat([representations, actions], dim=1)
        current_q = self.q_net(q_input)
        
        # Target Q values
        with torch.no_grad():
            next_representations = self.representation(next_states)
            
            # Similar to act() but for batch
            batch_size = states.size(0)
            random_actions = torch.rand(batch_size*10, actions.size(-1)) * 2 - 1
            random_actions = random_actions.to(self.device)
            
            # Expand next_representations
            next_reps_expanded = next_representations.unsqueeze(1).repeat(1, 10, 1).view(-1, next_representations.size(-1))
            q_input_target = torch.cat([next_reps_expanded, random_actions], dim=1)
            next_qs = self.target_q_net(q_input_target).view(batch_size, 10, 1)
            
            max_next_q = next_qs.max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Compute loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.update_every == 0:
            self.soft_update(self.q_net, self.target_q_net, self.tau)
        
        return loss.item()
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def save(self, path):
        torch.save({
            'representation': self.representation.state_dict(),
            'q_net': self.q_net.state_dict(),
            'target_q_net': self.target_q_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.representation.load_state_dict(checkpoint['representation'])
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_q_net.load_state_dict(checkpoint['target_q_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
    
    def add_to_buffer(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)
    
    def sample_from_buffer(self):
        return self.buffer.sample()
