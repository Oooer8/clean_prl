import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dims = config.hidden_dims
        self.output_dim = config.output_dim
        self.dropout = config.dropout
        
        # We'll initialize the network on first forward pass
        self.input_dim_set = False
        self.feature_extractor = None
        self.output_layer = None
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # If this is the first forward pass, set up the network based on input dimensions
        if not self.input_dim_set:
            input_dim = x.size(1)  # Get the flattened input dimension
            
            # Create the feature extractor
            layers = []
            prev_dim = input_dim
            for dim in self.hidden_dims:
                layers.append(nn.Linear(prev_dim, dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
                prev_dim = dim
            
            self.feature_extractor = nn.Sequential(*layers) if layers else nn.Identity()
            
            # Create the output layer
            last_hidden = self.hidden_dims[-1] if self.hidden_dims else input_dim
            self.output_layer = nn.Linear(last_hidden, self.output_dim)
            self.input_dim_set = True
        
        # Process the input through the network
        features = self.feature_extractor(x)
        output = self.output_layer(features)
        
        return output