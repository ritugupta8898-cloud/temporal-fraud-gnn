import torch
import torch.nn as nn
import torch.nn.functional as F

class TGNMemory(nn.Module):
    def __init__(self, num_nodes, memory_dim):
        super(TGNMemory, self).__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.register_buffer('memory', torch.zeros(num_nodes, memory_dim))
        self.register_buffer('last_update', torch.zeros(num_nodes))
    def get_memory(self, node_ids):
        memory_set = self.memory[node_ids]
        return memory_set
    def reset_memory(self):
        self.memory.zero_()
        self.last_update.zero_()
    def update_memory(self, node_ids, new_memory, timestamps):
        self.memory[node_ids] = new_memory
        self.last_update[node_ids] = timestamps

class MessageFunction(nn.Module):
    def __init__(self,memory_dimenstion):
       super(MessageFunction, self).__init__()
       self.linear = nn.Linear(2*memory_dimenstion+2,memory_dimenstion)

    def forward(self, memory1, memory2, time_delta, edge_features):
        time_delta = time_delta.unsqueeze(1)
        edge_features = edge_features.unsqueeze(1)

        raw_message = torch.cat([memory1, memory2, time_delta, edge_features], dim=1)
        out = F.relu(self.linear(raw_message))
        return out
       
        
class MemoryUpdater(nn.Module):
    def __init__(self, memory_dim):
        super(MemoryUpdater, self).__init__()
        self.gru = nn.GRUCell(memory_dim, memory_dim)
    
    def forward(self, message, current_memory):
        return self.gru(message, current_memory)