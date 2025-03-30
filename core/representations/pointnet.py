import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseRepresentation

class PointNet(BaseRepresentation):
    def __init__(self, config):
        super().__init__(config)
        
        # Shared MLP layers
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, config.output_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(config.output_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        # x shape: (batch, num_points, 3)
        x = x.transpose(1, 2)  # (batch, 3, num_points)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # Max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.config.output_dim)
        
        return self.dropout(x)
    
    @property
    def output_dim(self):
        return self.config.output_dim
