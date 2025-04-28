import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from model2 import PersonalityGNN
import sys

def load_graph(path):
    # Load user features
    user_features = np.load(os.path.join(path, 'user_features.npy'))
    
    # Load product features
    product_features = np.load(os.path.join(path, 'product_features.npy'))
    
    # Load edges
    edges = np.load(os.path.join(path, 'user_product.npy'))
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    
    # Load labels
    labels = np.load(os.path.join(path, 'label.npy'))
    
    # Convert features to tensors
    x_user = torch.tensor(user_features, dtype=torch.float)
    x_product = torch.tensor(product_features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.float)
    
    return x_user, x_product, edge_index, y

def train(model, x_user, x_product, edge_index, y, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    
    # Move data to device
    x_user = x_user.to(device)
    x_product = x_product.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)
    
    # Forward pass
    out = model(x_user, x_product, edge_index)
    loss = criterion(out, y)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

def main():
    if len(sys.argv) != 3:
        print("Usage: python train2.py <path_to_train_graph> <output_model_file_path>")
        sys.exit(1)
        
    train_path = sys.argv[1]
    model_save_path = sys.argv[2]
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    x_user, x_product, edge_index, y = load_graph(train_path)
    
    # Model parameters
    user_dim = x_user.size(1)
    product_dim = x_product.size(1)
    hidden_dim = 256  # Increased hidden dimension for better capacity
    num_personality_dims = y.size(1)
    
    print(f"Data shapes:")
    print(f"User features: {x_user.shape}")
    print(f"Product features: {x_product.shape}")
    print(f"Edge index: {edge_index.shape}")
    print(f"Labels: {y.shape}")
    
    # Initialize model
    model = PersonalityGNN(
        user_dim=user_dim,
        product_dim=product_dim,
        hidden_dim=hidden_dim,
        num_personality_dims=num_personality_dims
    ).to(device)
    
    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Lower learning rate
    criterion = torch.nn.MSELoss()
    
    # Training loop
    num_epochs = 100  # Reduced epochs due to larger dataset
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        loss = train(model, x_user, x_product, edge_index, y, optimizer, criterion, device)
        
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), model_save_path)
            
        if (epoch + 1) % 5 == 0:  # Print more frequently
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}')

if __name__ == "__main__":
    main() 