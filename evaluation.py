import torch
from sklearn.metrics import f1_score
from torch_geometric.utils import subgraph
from tgn import tgn


data = torch.load('elliptic_graph.pt', weights_only=False)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

data.x = data.x.to(device)
data.edge_index = data.edge_index.to(device)
data.y = data.y.to(device)
data.test_mask = data.test_mask.to(device)


model = tgn(
    num_nodes=203769,
    node_feature_dim=166,
    memory_dim=64,
    hidden_dim=64,
    out_channels=2
).to(device)


model.load_state_dict(torch.load('fraud_gnn_model.pt'))
model.eval()

def get_subgraph(seed_nodes, edge_index, num_hops=4):
    subset = seed_nodes
    for _ in range(num_hops):
        mask = torch.isin(edge_index[0], subset)
        neighbors = edge_index[1][mask]
        subset = torch.cat([subset, neighbors]).unique()
    sub_edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True)
    return subset, sub_edge_index

print(f'Starting Inference on device: {device}')




with torch.no_grad():
    print("Fast-forwarding memory (Timesteps 1 to 41)...")
   
    for t in range(35, 42):
        t_mask = (data.x[:, 0] == t)
        
        seed_nodes = t_mask.nonzero(as_tuple=True)[0]
        
        if seed_nodes.size(0) > 0:
            subset, sub_edge_index = get_subgraph(seed_nodes, data.edge_index)
            sub_x = data.x[subset]
            t_timestamps = sub_x[sub_edge_index[0], 0].float()
            
           
            _ = model(sub_x, sub_edge_index, t_timestamps, n_ids=subset)

    print("Evaluating Test Set (Timesteps 42 to 49)...")
    all_preds = []
    all_true = []

    for t in range(42, 50):
        t_mask = (data.x[:, 0] == t)
        
        seed_nodes = (data.test_mask & t_mask).nonzero(as_tuple=True)[0]
        
        if seed_nodes.size(0) > 0:
            subset, sub_edge_index = get_subgraph(seed_nodes, data.edge_index)
            sub_x = data.x[subset]
            sub_y = data.y[subset]
            sub_test_mask = data.test_mask[subset]
            t_timestamps = sub_x[sub_edge_index[0], 0].float()
            
            # Forward pass to get predictions
            out, _ = model(sub_x, sub_edge_index, t_timestamps, n_ids=subset)
            
            labeled = sub_test_mask
            if labeled.sum() > 0:
                preds = out[labeled].argmax(dim=1)
                all_preds.append(preds.cpu())
                all_true.append(sub_y[labeled].cpu())

    if all_preds:
        final_preds = torch.cat(all_preds)
        final_true = torch.cat(all_true)
        test_f1 = f1_score(final_true, final_preds, average='macro')
        print(f'Final Test F1 Score: {test_f1:.4f}')