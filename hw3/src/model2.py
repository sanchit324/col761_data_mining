import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, BatchNorm, GATv2Conv, TransformerConv

class PersonalityGNN(nn.Module):
    def __init__(self, user_dim, product_dim, hidden_dim, num_personality_dims):
        super(PersonalityGNN, self).__init__()
        
        # Increase hidden dimension for better capacity
        expanded_hidden = hidden_dim * 2
        
        # User feature processing with deeper network
        self.user_norm = BatchNorm(user_dim)
        self.user_linear1 = nn.Linear(user_dim, hidden_dim)
        self.user_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.user_linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.user_batch_norm1 = BatchNorm(hidden_dim)
        self.user_batch_norm2 = BatchNorm(hidden_dim)
        
        # Product feature processing with deeper network
        self.product_norm = BatchNorm(product_dim)
        self.product_linear1 = nn.Linear(product_dim, expanded_hidden)
        self.product_linear2 = nn.Linear(expanded_hidden, hidden_dim)
        self.product_linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.product_batch_norm1 = BatchNorm(expanded_hidden)
        self.product_batch_norm2 = BatchNorm(hidden_dim)
        
        # Advanced graph convolution layers
        self.sage_convs = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim),
            SAGEConv(hidden_dim, hidden_dim),
            SAGEConv(hidden_dim, hidden_dim)
        ])
        
        # More sophisticated attention mechanisms
        self.gat_convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, concat=True),
            TransformerConv(hidden_dim, hidden_dim // 4, heads=4, concat=True),
            GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, concat=True)
        ])
        
        # Layer normalization for each layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LayerNorm(hidden_dim)
        ])
        
        # Enhanced output layers with larger intermediate layers
        self.fc1 = nn.Linear(hidden_dim * 2, expanded_hidden)
        self.fc2 = nn.Linear(expanded_hidden, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_personality_dims)
        
        # Output normalization and batch norm for better training
        self.output_norm = nn.LayerNorm(num_personality_dims)
        self.fc_bn1 = BatchNorm(expanded_hidden)
        self.fc_bn2 = BatchNorm(hidden_dim)
        
        # Self-attention for user features
        self.user_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Improved dropout with different rates
        self.dropout_low = nn.Dropout(0.2)
        self.dropout_high = nn.Dropout(0.4)
        
    def forward(self, x_user, x_product, edge_index):
        # Process user features with improved processing
        x_user = self.user_norm(x_user)
        # First layer with identity
        user_identity = self.user_linear1(x_user)
        x_user = F.gelu(user_identity)  # Using GELU for better gradient flow
        x_user = self.dropout_low(x_user)
        
        # Second layer with skip connection
        x_user = self.user_linear2(x_user)
        x_user = self.user_batch_norm1(x_user)
        x_user = F.gelu(x_user + user_identity)  # Skip connection
        x_user = self.dropout_low(x_user)
        
        # Third layer
        x_user_identity = x_user
        x_user = self.user_linear3(x_user)
        x_user = self.user_batch_norm2(x_user)
        x_user = F.gelu(x_user + x_user_identity)  # Skip connection
        
        # Apply self-attention to user features to capture dependencies
        x_user_reshaped = x_user.unsqueeze(1)  # Add sequence dimension for attention
        x_user_attn, _ = self.user_attention(x_user_reshaped, x_user_reshaped, x_user_reshaped)
        x_user = x_user + x_user_attn.squeeze(1)  # Residual connection
        
        # Process product features with improved processing
        x_product = self.product_norm(x_product)
        # First layer - expand dimension
        x_product = self.product_linear1(x_product)
        x_product = self.product_batch_norm1(x_product)
        x_product = F.gelu(x_product)
        x_product = self.dropout_low(x_product)
        
        # Second layer - compress to hidden_dim
        prod_identity = self.product_linear2(x_product)
        x_product = F.gelu(prod_identity)
        x_product = self.dropout_low(x_product)
        
        # Third layer with skip connection
        x_product = self.product_linear3(x_product)
        x_product = self.product_batch_norm2(x_product)
        x_product = F.gelu(x_product + prod_identity)
        
        # Combine user and product features
        x = torch.cat([x_user, x_product], dim=0)
        
        # Apply graph convolutions with improved residual connections 
        # and combined attention mechanisms
        for i in range(len(self.sage_convs)):
            identity = x
            
            # SAGE convolution
            x_sage = self.sage_convs[i](x, edge_index)
            
            # GAT convolution - using different attention mechanisms
            x_gat = self.gat_convs[i](x, edge_index)
            
            # Combine with weighted sum
            x = x_sage * 0.6 + x_gat * 0.4  # Weighted combination
            
            # Normalization and activation
            x = self.layer_norms[i](x)
            x = F.gelu(x)
            
            # Apply dropout (higher in later layers)
            if i < len(self.sage_convs) - 1:
                x = self.dropout_low(x)
            else:
                x = self.dropout_high(x)
                
            # Residual connection with scaling for stability
            x = x + identity * 0.8
        
        # Extract user embeddings (first m nodes)
        user_embeddings = x[:x_user.size(0)]
        
        # Concatenate with original user features for better prediction
        user_embeddings = torch.cat([user_embeddings, x_user], dim=1)
        
        # Final prediction layers with improved connections
        identity = self.fc1(user_embeddings)
        identity = self.fc_bn1(identity)
        identity = F.gelu(identity)
        identity = self.dropout_low(identity)
        
        out = self.fc2(identity)
        out = self.fc_bn2(out)
        out = F.gelu(out)
        out = self.dropout_high(out)
        
        out = self.fc3(out)
        
        # Normalize output for better stability
        out = self.output_norm(out)
        
        # Sigmoid activation for personality scores
        out = torch.sigmoid(out)
        
        return out 