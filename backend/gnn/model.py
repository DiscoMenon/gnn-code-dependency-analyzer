import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class CodeGNN(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=32):
        super().__init__()
        # Two GCN layers + one GAT layer
        # GCN: aggregates neighbor info
        # GAT: learns which neighbors matter more (attention)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, output_dim, heads=1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 3 — output embeddings
        x = self.conv3(x, edge_index)
        return x  # shape: [num_nodes, output_dim]