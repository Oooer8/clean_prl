import torch.nn as nn
from abc import ABC, abstractmethod

class BaseRepresentation(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, observations):
        """Process raw observations into latent representations"""
        pass
    
    @property
    @abstractmethod
    def output_dim(self):
        """Return the dimension of the output representation"""
        pass
