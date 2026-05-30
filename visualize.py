import torch
from model import FraudGNN
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.data import Data

data = torch.load('elliptic_graph.pt', weights_only=False)

model = FraudGNN(in_channels=166, hidden_channels=64, out_channels=2)
model.load_state_dict(torch.load('fraud_gnn_model.pt'))
model.eval()

with torch.no_grad():
    out, h = model(data.x, data.edge_index)
preds = out.argmax(dim=1)

fraud_nodes = (data.y == 1).nonzero(as_tuple=True)[0]

# find fraud node with most connections

def plot_fraud_ring(node_idx, preds, num_hops=1):
    # manually find neighbors
    src = data.edge_index[0]
    dst = data.edge_index[1]
    
    # find all edges where node_idx is source or destination
    mask = (src == node_idx) | (dst == node_idx)
    relevant_edges = data.edge_index[:, mask]
    
    # get unique nodes
    nodes = relevant_edges.unique()
    print(f'Found {nodes.size(0)} neighboring nodes')
    print(f'Found {mask.sum()} edges')
    
    # build local index mapping
    node_to_local = {n.item(): i for i, n in enumerate(nodes)}
    
    local_src = [node_to_local[s.item()] for s in relevant_edges[0]]
    local_dst = [node_to_local[d.item()] for d in relevant_edges[1]]
    
    sub_edge_index = torch.tensor([local_src, local_dst], dtype=torch.long)
    
    G = nx.DiGraph()
    G.add_nodes_from(range(nodes.size(0)))
    G.add_edges_from(zip(local_src, local_dst))
    
    color_map = []
    for n in nodes:
        global_node = n.item()
        is_unknown = (
            not data.train_mask[global_node].item() and
            not data.val_mask[global_node].item() and
            not data.test_mask[global_node].item()
        )
        if is_unknown:
            color_map.append('grey')
        elif preds[global_node].item() == 1:
            color_map.append('red')
        else:
            color_map.append('green')

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_color=color_map, node_size=100,
            arrows=True, edge_color='black', width=0.5)
    plt.title(f'Fraud Ring around node {node_idx} ({mask.sum()} edges)')
    plt.show()
plot_fraud_ring(148166, preds, num_hops=1)    