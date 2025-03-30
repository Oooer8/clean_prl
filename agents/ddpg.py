import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from copy import deepcopy
from core.representations.mlpnet import MLPNet
from core.policies.mlp import MLPPolicy
from core.values.mlp import MLPValue
from core.buffers.replay import ReplayBuffer
from .base import BaseAgent

class DDPGAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(config)
        self.action_dim = action_dim
        
        # Networks
        self.representation = MLPNet(config.representation).to(self.device)
        
        # Actor (Policy) Network
        rep_output_dim = config.representation.output_dim
        self.actor = MLPPolicy(
            input_dim=rep_output_dim,
            action_dim=action_dim,
            config=config.policy
        ).to(self.device)
        
        # Critic (Value) Network
        self.critic = MLPValue(
            input_dim=rep_output_dim + action_dim,
            config=config.value
        ).to(self.device)
        
        # Target Networks
        self.target_representation = deepcopy(self.representation).to(self.device)
        self.target_actor = deepcopy(self.actor).to(self.device)
        self.target_critic = deepcopy(self.critic).to(self.device)
        
        # Disable gradient for target networks
        for target_net in [self.target_representation, self.target_actor, self.target_critic]:
            for param in target_net.parameters():
                param.requires_grad = False
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            list(self.representation.parameters()) + list(self.actor.parameters()),
            lr=config.agent.actor_lr
        )
        self.critic_optimizer = optim.Adam(
            list(self.critic.parameters()), 
            lr=config.agent.critic_lr
        )
        
        # Buffer
        self.buffer = ReplayBuffer(state_dim, action_dim, config.buffer)
        
        # Hyperparameters
        self.gamma = config.agent.gamma
        self.tau = config.agent.tau
        self.update_every = config.agent.update_every
        self.noise_scale = config.agent.noise_scale
        self.steps = 0
        
        # Action noise for exploration
        self.noise = OUNoise(action_dim, 
                             theta=config.agent.noise_theta, 
                             sigma=config.agent.noise_sigma)
    
    def act(self, state, epsilon=0.0):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            representation = self.representation(state)
            action = self.actor(representation)
            action = action.cpu().numpy().flatten()
            
            # Add noise for exploration if epsilon > 0
            if epsilon > 0:
                noise = self.noise.sample() * epsilon * self.noise_scale
                action = np.clip(action + noise, -1, 1)
            
            return action
    
    def learn(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 获取当前表示
        with torch.no_grad():  # 修改：在计算Critic目标时不需要梯度
            representations = self.representation(states)
        
        # 更新Critic
        with torch.no_grad():
            next_representations = self.target_representation(next_states)
            next_actions = self.target_actor(next_representations)
            target_q_input = torch.cat([next_representations, next_actions], dim=1)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * self.target_critic(target_q_input)  # 修改：确保形状正确
        
        # 当前Q值
        current_representations = self.representation(states)  # 修改：为Critic计算单独的表示
        q_input = torch.cat([current_representations, actions], dim=1)
        current_q = self.critic(q_input)
        
        # 计算Critic损失
        critic_loss = F.mse_loss(current_q, target_q)
        
        # 优化Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor(策略)
        # 为Actor计算新的表示
        actor_representations = self.representation(states)  # 修改：为Actor计算单独的表示
        actor_actions = self.actor(actor_representations)
        actor_q_input = torch.cat([actor_representations, actor_actions], dim=1)
        actor_loss = -self.critic(actor_q_input).mean()  # 最大化Q值
        
        # 优化Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新目标网络
        self.steps += 1
        if self.steps % self.update_every == 0:
            self.soft_update(self.representation, self.target_representation, self.tau)
            self.soft_update(self.actor, self.target_actor, self.tau)
            self.soft_update(self.critic, self.target_critic, self.tau)
        
        return critic_loss.item() + actor_loss.item()
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def save(self, path):
        torch.save({
            'representation': self.representation.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_representation': self.target_representation.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'steps': self.steps,
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.representation.load_state_dict(checkpoint['representation'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_representation.load_state_dict(checkpoint['target_representation'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if 'steps' in checkpoint:
            self.steps = checkpoint['steps']
    
    def add_to_buffer(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)
    
    def sample_from_buffer(self):
        return self.buffer.sample()


class OUNoise:
    """Ornstein-Uhlenbeck process for exploration noise"""
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
        
    def reset(self):
        self.state = np.copy(self.mu)
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state