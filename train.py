import torch
import torch.nn as nn
import torch.optim as optim
from model import FraudGNN
from sklearn.metrics import f1_score
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.data import Data
from tgn import tgn

data = torch.load('elliptic_graph.pt', weights_only=False)

model = tgn(
    num_nodes=203769,
    node_feature_dim=166,
    memory_dim=64,
    hidden_dim=64,
    out_channels=2
)
weight = torch.tensor([1.0, 9.25])
criterion = nn.CrossEntropyLoss(weight=weight)
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    for epoch in range(1, 51):
        model.memory.reset_memory()
        optimizer.zero_grad()
        total_loss = 0
    
        for t in range(1, 35):
            t_mask = (data.x[:, 0] == t)
            t_edge_mask = data.train_edge_mask & (data.x[data.edge_index[0], 0] == t)
            t_edge_index = data.edge_index[:, t_edge_mask]
            t_timestamps = data.x[t_edge_index[0], 0].float()
        
            if t_edge_index.size(1) == 0:
               continue
            
            out, h = model(data.x, t_edge_index, t_timestamps)
            labeled = data.train_mask & t_mask
            if labeled.sum() == 0:
                continue
            
            loss = criterion(out[labeled], data.y[labeled])
            total_loss += loss
    
        total_loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}, Loss: {total_loss:.4f}')
train()    

model.eval()
with torch.no_grad():
    test_edge_index = data.edge_index[:, data.test_edge_mask]
    test_timestamps = data.x[test_edge_index[0], 0].float()  # ← add this
    out, _ = model(data.x, test_edge_index, test_timestamps) 
    preds = out[data.test_mask].argmax(dim=1)
    true = data.y[data.test_mask]
    test_f1 = f1_score(true.cpu(), preds.cpu(), average='macro')
    print(f'Test F1: {test_f1:.4f}')
torch.save(model.state_dict(), 'fraud_gnn_model.pt')
print('Model saved.')