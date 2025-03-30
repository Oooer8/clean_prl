import torch.nn as nn
import torch.nn.functional as F
from .base import BaseValue

class MLPValue(BaseValue):
    def __init__(self, input_dim, config):
        super().__init__(config)
        
        layers = []
        prev_dim = input_dim
        for dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(getattr(nn, config.activation)())
            prev_dim = dim
        
        self.net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)
        
    def forward(self, x):
        return self.output_layer(self.net(x))
