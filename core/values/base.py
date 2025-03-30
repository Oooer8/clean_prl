import torch.nn as nn
from abc import ABC, abstractmethod

class BaseValue(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, x):
        """Compute value estimate from state representation"""
        pass
