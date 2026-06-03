"""
Temporal Fraud GNN
Author: Pratyush Gupta
Copyright (c) 2026 Pratyush Gupta. All Rights Reserved.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch_geometric.utils import subgraph
from tgn import tgn

data = torch.load('elliptic_graph.pt', weights_only=False)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = tgn(
    num_nodes=203769,
    node_feature_dim=166,
    memory_dim=64,
    hidden_dim=64,
    out_channels=2
)

model = model.to(device)
weight = torch.tensor([1.0, 9.25]).to(device)
data.x = data.x.to(device)
data.edge_index = data.edge_index.to(device)
data.y = data.y.to(device)
data.train_mask = data.train_mask.to(device)
data.val_mask = data.val_mask.to(device)
data.test_mask = data.test_mask.to(device)
data.train_edge_mask = data.train_edge_mask.to(device)
data.val_edge_mask = data.val_edge_mask.to(device)
data.test_edge_mask = data.test_edge_mask.to(device)

criterion = nn.CrossEntropyLoss(weight=weight)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

def get_subgraph(seed_nodes, edge_index, num_hops=4):
    subset = seed_nodes
    for _ in range(num_hops):
        mask = torch.isin(edge_index[0], subset)
        neighbors = edge_index[1][mask]
        subset = torch.cat([subset, neighbors]).unique()
    sub_edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True)
    return subset, sub_edge_index

def train():
    print(f'Using device: {device}')
    model.train()
    for epoch in range(1, 101):
        model.memory.reset_memory()
        optimizer.zero_grad()
        total_loss = 0

        for t in range(1, 35):
            t_mask = (data.x[:, 0] == t)
            seed_nodes = (data.train_mask & t_mask).nonzero(as_tuple=True)[0]

            if seed_nodes.size(0) == 0:
                continue

            subset, sub_edge_index = get_subgraph(seed_nodes, data.edge_index)
            sub_x = data.x[subset]
            sub_y = data.y[subset]
            sub_train_mask = data.train_mask[subset]
            t_timestamps = sub_x[sub_edge_index[0], 0].float()

            out, h = model(sub_x, sub_edge_index, t_timestamps, n_ids=subset)

            labeled = sub_train_mask
            if labeled.sum() == 0:
                continue

            loss = criterion(out[labeled], sub_y[labeled])
            total_loss += loss

        total_loss.backward()
        optimizer.step()
        scheduler.step()
        print(f'Epoch {epoch}, Loss: {total_loss:.4f}')

train()

model.eval()
with torch.no_grad():
    test_edge_index = data.edge_index[:, data.test_edge_mask]
    test_timestamps = data.x[test_edge_index[0], 0].float()
    out, _ = model(data.x, test_edge_index, test_timestamps, n_ids=None)
    preds = out[data.test_mask].argmax(dim=1)
    true = data.y[data.test_mask]
    test_f1 = f1_score(true.cpu(), preds.cpu(), average='macro')
    print(f'Test F1: {test_f1:.4f}')

torch.save(model.state_dict(), 'fraud_gnn_model.pt')
print('Model saved.')