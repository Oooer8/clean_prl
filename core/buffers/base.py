from abc import ABC, abstractmethod
import numpy as np

class BaseBuffer(ABC):
    def __init__(self, config):
        self.config = config
        self.capacity = config.capacity
        self.batch_size = config.batch_size
        self.pos = 0
        self.size = 0
    
    @abstractmethod
    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer"""
        pass
    
    @abstractmethod
    def sample(self):
        """Sample a batch of transitions"""
        pass
    
    def __len__(self):
        return self.size
