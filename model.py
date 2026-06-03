"""
Temporal Fraud GNN
Author: Pratyush Gupta
Copyright (c) 2026 Pratyush Gupta. All Rights Reserved.
"""
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch
import torch.nn as nn

class FraudGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(FraudGNN, self).__init__()
        self.conv1 = GATConv(node_feature_dim + memory_dim, hidden_dim, heads=2, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        h = self.conv3(x, edge_index)
        h = F.relu(h)
        out = self.classifier(h)
        return out, h
    
