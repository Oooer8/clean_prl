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
        
        self.last_action = np.zeros(action_dim) # 添加用于动作平滑的上一个动作记录
        
        self.exploration_noise = getattr(config.agent, 'exploration_noise', 0.2)    # 添加探索噪声参数
        self.action_smoothing = getattr(config.agent, 'action_smoothing', 0.1)  # 添加动作平滑系数
        self.sampling_size = getattr(config.agent, 'sampling_size', 100)        # 采样数量
        self.use_gaussian = getattr(config.agent, 'use_gaussian', True) # 是否使用高斯分布采样
        self.use_smoothing = getattr(config.agent, 'use_smoothing', True)   # 是否使用动作平滑
        
    def act(self, state, epsilon=0.0):
        if random.random() < epsilon:
            action = np.random.uniform(-1, 1, size=self.action_dim)
            self.last_action = action
            return action
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            representation = self.representation(state)
            
            # 初始采样 - 第一次迭代使用均匀分布
            if not hasattr(self, 'best_action_tensor') or self.steps == 0:
                sampling_size = self.sampling_size
                actions = torch.rand(sampling_size, self.action_dim) * 2 - 1
                actions = actions.to(self.device)
            else:
                # 使用高斯分布围绕上一个最佳动作采样
                if self.use_gaussian:
                    sampling_size = self.sampling_size
                    actions = torch.randn(sampling_size, self.action_dim).to(self.device) * self.exploration_noise
                    actions = actions + self.best_action_tensor.clone()
                    # 裁剪到[-1, 1]范围
                    actions = torch.clamp(actions, -1, 1)
                else:
                    # 仍然使用均匀分布
                    sampling_size = self.sampling_size
                    actions = torch.rand(sampling_size, self.action_dim) * 2 - 1
                    actions = actions.to(self.device)

            # Expand representation to match actions
            representations = representation.repeat(sampling_size, 1)
            q_input = torch.cat([representations, actions], dim=1)
            q_values = self.q_net(q_input)
            
            best_idx = q_values.argmax()
            raw_best_action = actions[best_idx]
            
            # 保存当前最佳动作用于下次采样
            self.best_action_tensor = raw_best_action.clone()
            
            # 应用动作平滑
            if self.use_smoothing:
                last_action_tensor = torch.FloatTensor(self.last_action).to(self.device)
                smooth_factor = self.action_smoothing
                best_action = (1 - smooth_factor) * last_action_tensor + smooth_factor * raw_best_action
            else:
                best_action = raw_best_action
                
            # 更新上一个动作
            best_action_np = best_action.cpu().numpy()
            self.last_action = best_action_np
            
            return best_action_np
    
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
            sampling_size_per_state = 10  # 每个状态采样10个动作
            
            if self.use_gaussian:
                # 对每个状态，围绕零点采样动作
                random_actions = torch.randn(batch_size*sampling_size_per_state, actions.size(-1)) * self.exploration_noise
                random_actions = torch.clamp(random_actions, -1, 1).to(self.device)
            else:
                # 均匀分布采样
                random_actions = torch.rand(batch_size*sampling_size_per_state, actions.size(-1)) * 2 - 1
                random_actions = random_actions.to(self.device)
            
            # Expand next_representations
            next_reps_expanded = next_representations.unsqueeze(1).repeat(1, sampling_size_per_state, 1).view(-1, next_representations.size(-1))
            q_input_target = torch.cat([next_reps_expanded, random_actions], dim=1)
            next_qs = self.target_q_net(q_input_target).view(batch_size, sampling_size_per_state, 1)
            
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
            'steps': self.steps,
            'last_action': self.last_action,
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.representation.load_state_dict(checkpoint['representation'])
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_q_net.load_state_dict(checkpoint['target_q_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'steps' in checkpoint:
            self.steps = checkpoint['steps']
        if 'last_action' in checkpoint:
            self.last_action = checkpoint['last_action']
    
    def add_to_buffer(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)
    
    def sample_from_buffer(self):
        return self.buffer.sample()