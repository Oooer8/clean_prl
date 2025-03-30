import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BasePolicy

class MLPPolicy(BasePolicy):
    def __init__(self, input_dim, action_dim, config):
        super().__init__(config)
        
        layers = []
        prev_dim = input_dim
        for dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(getattr(nn, config.activation)())
            prev_dim = dim
        
        self.net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, action_dim)
        
        # Output activation
        self.output_activation = getattr(torch, config.output_activation, lambda x: x)
        
    def forward(self, x):
        x = self.net(x)
        return self.output_activation(self.output_layer(x))
    
    def sample(self, x):
        return self.forward(x)
