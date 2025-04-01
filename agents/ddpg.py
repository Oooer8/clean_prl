import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from core.representations.mlpnet import MLPNet
from core.policies.mlp import MLPPolicy
from core.values.mlp import MLPValue
from core.buffers.replay import ReplayBuffer

class DDPGAgent:
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.device = config.device
        self.action_dim = action_dim
        
        # Unified representation network
        self.representation = MLPNet(config.representation, input_dim=state_dim).to(self.device)
        rep_output_dim = config.representation.output_dim
        
        # Actor and Critic networks
        self.actor = MLPPolicy(rep_output_dim, action_dim, config.policy).to(self.device)
        self.critic = MLPValue(rep_output_dim + action_dim, config.value).to(self.device)
        
        # Target networks
        self.target_representation = deepcopy(self.representation).to(self.device)
        self.target_actor = deepcopy(self.actor).to(self.device)
        self.target_critic = deepcopy(self.critic).to(self.device)
        
        # Freeze target networks
        for target in [self.target_representation, self.target_actor, self.target_critic]:
            for param in target.parameters():
                param.requires_grad = False
        
        # Optimizers
        self.rep_optimizer = optim.Adam(self.representation.parameters(), lr=config.agent.rep_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.agent.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.agent.critic_lr)
        
        # Buffer and hyperparameters
        self.buffer = ReplayBuffer(state_dim, action_dim, config.buffer)
        self.gamma = config.agent.gamma
        self.tau = config.agent.tau
        self.noise = OUNoise(action_dim, theta=config.agent.noise_theta, sigma=config.agent.noise_sigma)
        self.steps = 0
        self.max_grad_norm = 1.0  # Gradient clipping

    def act(self, state, epsilon=0.0):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            representation = self.representation(state)
            action = self.actor(representation).cpu().numpy().flatten()
            if epsilon > 0:
                action = np.clip(action + self.noise.sample() * epsilon, -1, 1)
            return action * 2

    def learn(self, batch):
        states, actions, rewards, next_states, dones = [x.to(self.device) for x in batch]
        
        # ------------------- Critic Update ------------------- #
        with torch.no_grad():
            next_repr = self.target_representation(next_states)
            next_actions = self.target_actor(next_repr)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * \
                      self.target_critic(torch.cat([next_repr, next_actions], 1))
        
        current_repr = self.representation(states)
        current_q = self.critic(torch.cat([current_repr, actions], 1))
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        self.rep_optimizer.zero_grad()
        critic_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.representation.parameters(), self.max_grad_norm)
        
        self.critic_optimizer.step()
        self.rep_optimizer.step()

        # ------------------- Actor Update ------------------- #
        # Detach representation to avoid affecting critic update
        current_repr = self.representation(states).detach()
        actor_actions = self.actor(current_repr)
        actor_loss = -self.critic(torch.cat([current_repr, actor_actions], 1)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # ------------------- Target Update ------------------- #
        self.steps += 1
        if self.steps % self.config.agent.update_every == 0:
            for target, source in zip(
                [self.target_representation, self.target_actor, self.target_critic],
                [self.representation, self.actor, self.critic]
            ):
                for target_param, param in zip(target.parameters(), source.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        return abs(actor_loss.item())

    def add_to_buffer(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def sample_from_buffer(self):
        return self.buffer.sample()
    
    def save(self, path):
        save_dict = {
            'representation': self.representation.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_representation': self.target_representation.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'rep_optimizer': self.rep_optimizer.state_dict(),
            'steps': self.steps,
        }
        torch.save(save_dict, path)
    
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
        self.rep_optimizer.load_state_dict(checkpoint['rep_optimizer'])
        self.steps = checkpoint.get('steps', 0)

class OUNoise:
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
        
    def reset(self):
        self.state = np.copy(self.mu)
        
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state