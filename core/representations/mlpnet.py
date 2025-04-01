import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseRepresentation

class MLPNet(BaseRepresentation):
    def __init__(self, config, input_dim=3):  # 默认输入维度为3，适用于Pendulum-v1
        super().__init__(config)
        self.hidden_dims = config.hidden_dims
        self.output_dim = config.output_dim
        self.dropout = config.dropout
        
        # 直接在构造函数中初始化网络
        layers = []
        prev_dim = input_dim
        for dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev_dim = dim
        
        self.feature_extractor = nn.Sequential(*layers) if layers else nn.Identity()
        
        # 创建输出层
        last_hidden = self.hidden_dims[-1] if self.hidden_dims else input_dim
        self.output_layer = nn.Linear(last_hidden, self.output_dim)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.output_layer(features)
        return torch.tanh(output)
    
    
    @property
    def output_dim(self):
        return self._output_dim
    
    @output_dim.setter
    def output_dim(self, value):
        self._output_dim = value
