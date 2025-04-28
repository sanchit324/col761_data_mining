import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, BatchNorm, global_mean_pool, JumpingKnowledge # type: ignore

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5, task_type='classification'):
        super(GNNModel, self).__init__()
        self.task_type = task_type
        self.num_layers = num_layers
        
        # Input feature normalization
        self.input_norm = BatchNorm(input_dim)
        
        # Multi-scale architecture
        self.convs = nn.ModuleList()
        self.gats = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.gats.append(GATConv(input_dim, hidden_dim // 4, heads=4, concat=True))
        self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.gats.append(GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Final output layer
        self.output = nn.Linear(hidden_dim, output_dim)
        
        # Dropout
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        # Input normalization
        x = self.input_norm(x)
        
        # Input layer
        x1_conv = self.convs[0](x, edge_index)
        x1_gat = self.gats[0](x, edge_index)
        x = x1_conv + x1_gat  # Combine GCN and GAT
        x = self.batch_norms[0](x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers with residual connections
        for i in range(1, self.num_layers - 1):
            identity = x
            x_conv = self.convs[i](x, edge_index)
            x_gat = self.gats[i](x, edge_index)
            x = x_conv + x_gat
            x = self.batch_norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + identity  # Residual connection
        
        # Final convolution
        x = self.convs[-1](x, edge_index)
        x = self.layer_norm(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.output(x)
        
        if self.task_type == 'classification':
            return F.log_softmax(x, dim=1)
        else:
            return x 