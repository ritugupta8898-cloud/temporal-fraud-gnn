import pandas as pd
import torch
from torch_geometric.data import Data

classes = pd.read_csv('/Users/pratyushgupta/Downloads/elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
edges = pd.read_csv('/Users/pratyushgupta/Downloads/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')
features = pd.read_csv('/Users/pratyushgupta/Downloads/elliptic_bitcoin_dataset/elliptic_txs_features.csv', header=None)

# Node index mapping
tx_ids = features[0].values
tx_id_to_idx = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}

# Node features
x = torch.tensor(features.iloc[:, 1:].values, dtype=torch.float)

# Edge index
src = [tx_id_to_idx[i] for i in edges['txId1'].values]
dst = [tx_id_to_idx[i] for i in edges['txId2'].values]
edge_index = torch.tensor([src, dst], dtype=torch.long)

# Labels
label_map = {'1': 1, '2': 0, 'unknown': -1}
classes['label'] = classes['class'].map(label_map)

node_labels = pd.DataFrame({'txId': tx_ids})
node_labels = node_labels.merge(classes, on='txId', how='left')

y = torch.tensor(node_labels['label'].fillna(-1).astype(int).values, dtype=torch.long)

# Split labeled nodes into train/val/test
labeled_indices = (y != -1).nonzero(as_tuple=True)[0]
n = len(labeled_indices)

perm = torch.randperm(n)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

train_idx = labeled_indices[perm[:train_end]]
val_idx = labeled_indices[perm[train_end:val_end]]
test_idx = labeled_indices[perm[val_end:]]

train_mask = torch.zeros(x.shape[0], dtype=torch.bool)
val_mask = torch.zeros(x.shape[0], dtype=torch.bool)
test_mask = torch.zeros(x.shape[0], dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

# Create and save data object
data = Data(
    x=x,
    edge_index=edge_index,
    y=y,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask
)




def create_temporal_splits(data):
    timesteps = data.x[:, 0]
    
    # node masks by time
    train_node_mask = (timesteps <= 34) & (data.y != -1)
    val_node_mask = (timesteps >= 35) & (timesteps <= 41) & (data.y != -1)
    test_node_mask = (timesteps >= 42) & (data.y != -1)
    
    # edge masks
    train_edge_mask = train_node_mask[data.edge_index[0]] & train_node_mask[data.edge_index[1]]
    val_edge_mask = val_node_mask[data.edge_index[0]] & val_node_mask[data.edge_index[1]]
    test_edge_mask = test_node_mask[data.edge_index[0]] & test_node_mask[data.edge_index[1]]
    
    return train_node_mask, val_node_mask, test_node_mask, \
           train_edge_mask, val_edge_mask, test_edge_mask

train_nm, val_nm, test_nm, train_em, val_em, test_em = create_temporal_splits(data)

# update data object with temporal masks
data.train_mask = train_nm
data.val_mask = val_nm
data.test_mask = test_nm
data.train_edge_mask = train_em
data.val_edge_mask = val_em
data.test_edge_mask = test_em

torch.save(data, 'elliptic_graph.pt')
print('Saved with temporal splits')
print(data)


print(f'Train nodes: {train_nm.sum()}')
print(f'Val nodes: {val_nm.sum()}')
print(f'Test nodes: {test_nm.sum()}')
print(f'Train edges: {train_em.sum()}')
print(f'Val edges: {val_em.sum()}')
print(f'Test edges: {test_em.sum()}')