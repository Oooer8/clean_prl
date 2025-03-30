from abc import ABC, abstractmethod
import torch

class BaseAgent(ABC):
    def __init__(self, config):
        self.config = config
        self.device = config.device
    
    @abstractmethod
    def act(self, state, epsilon=0.0):
        """Select an action"""
        pass
    
    @abstractmethod
    def learn(self, batch):
        """Update the agent's parameters"""
        pass
    
    @abstractmethod
    def save(self, path):
        """Save the agent's parameters"""
        pass
    
    @abstractmethod
    def load(self, path):
        """Load the agent's parameters"""
        pass
    
    def to(self, device):
        self.device = device
        for attr in vars(self):
            if isinstance(getattr(self, attr), torch.nn.Module):
                getattr(self, attr).to(device)
        return self
