"""
Temporal Fraud GNN
Author: Pratyush Gupta
Copyright (c) 2026 Pratyush Gupta. All Rights Reserved.
"""
import torch
import torch.nn as nn

import  torch.nn.functional  as F
from torch_geometric.nn import SAGEConv
from memory_module import TGNMemory,MessageFunction,MemoryUpdater
class tgn(nn.Module):
    def __init__(self,num_nodes, node_feature_dim, memory_dim, hidden_dim, out_channels):
        super(tgn,self).__init__()
        self.memory_dim = memory_dim
        self.node_feature_dim = node_feature_dim 
        self.memory = TGNMemory(num_nodes,memory_dim)
        self.message_function = MessageFunction(memory_dim)
        self.memory_updater = MemoryUpdater(memory_dim)

        self.conv1 = SAGEConv(node_feature_dim + memory_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim,out_channels)


    def forward(self, x, edge_index, timestamps,n_ids=None):

        src = edge_index[0]
        dst = edge_index[1]
        if n_ids is not None:
           src = n_ids[src]
           dst = n_ids[dst]

        all_nodes = torch.cat([src,dst])


       
           
       

        src_memory = self.memory.get_memory(src)
        dst_memory = self.memory.get_memory(dst)

        time_delta_src = timestamps-self.memory.last_update[src]
        time_delta_dst = timestamps-self.memory.last_update[dst]

        
         
        
        src_message = self.message_function(src_memory, dst_memory, time_delta_src, timestamps)
        dst_message = self.message_function(dst_memory, src_memory, time_delta_dst, timestamps)

        all_messages = torch.cat([src_message,dst_message],dim=0)

        unique_nodes, inverse_indices = torch.unique(all_nodes, return_inverse=True)
        
        batch_aggregated = torch.zeros(
             unique_nodes.size(0),
             all_messages.size(1),
             device=all_messages.device
        )
        
        counts = torch.zeros(
             unique_nodes.size(0),
             device=all_messages.device
        )

        batch_aggregated.index_add_(0, inverse_indices, all_messages)
        counts.index_add_(0, inverse_indices, torch.ones(all_nodes.size(0), device=all_nodes.device))

        batch_aggregated = batch_aggregated / counts.clamp(min=1).unsqueeze(1)
        
        msg_lookup = torch.zeros(
            self.memory.num_nodes, 
            all_messages.size(1), 
            device=all_messages.device
        )
        msg_lookup[unique_nodes] = batch_aggregated
        
        src_message = msg_lookup[src]
        dst_message = msg_lookup[dst]



        new_src_memory = self.memory_updater(src_message, src_memory)
        new_dst_memory = self.memory_updater(dst_message, dst_memory)

        
        self.memory.update_memory(src, new_src_memory.detach(), timestamps)
        self.memory.update_memory(dst, new_dst_memory.detach(), timestamps)


       
        if n_ids is not None:
            all_memory = self.memory.get_memory(n_ids[:x.size(0)])
        else:
            all_memory = self.memory.get_memory(torch.arange(x.size(0)))

        x = torch.cat([x, all_memory], dim=1)

        x = self.conv1(x,edge_index)
        x=F.relu(x)
        x = self.conv2(x,edge_index)
        x=F.relu(x)
        h = self.conv3(x,edge_index)
        h=F.relu(h)
        out = self.classifier(h)

        return out,h
