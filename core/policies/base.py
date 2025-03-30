import torch.nn as nn
from abc import ABC, abstractmethod

class BasePolicy(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, x):
        """Compute action distribution from state representation"""
        pass
    
    @abstractmethod
    def sample(self, x):
        """Sample action from policy"""
        pass
