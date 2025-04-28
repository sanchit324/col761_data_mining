import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.preprocessing import RobustScaler
import numpy as np
from model1 import GNNModel
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(data_path):
    # Load node features
    node_features = np.load(f"{data_path}/node_feat.npy")
    
    # Robust scaling of features
    scaler = RobustScaler()
    node_features = scaler.fit_transform(node_features)
    node_features = torch.FloatTensor(node_features)
    
    # Load edges
    edges = np.load(f"{data_path}/edges.npy")
    edge_index = torch.LongTensor(edges.T)
    
    # Load labels
    labels = np.load(f"{data_path}/label.npy")
    labels = torch.LongTensor(labels)
    
    # Create train mask (nodes with non-nan labels)
    train_mask = ~np.isnan(labels)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    
    # Compute class weights for balanced training
    valid_labels = labels[train_mask]
    if len(torch.unique(valid_labels)) > 2:  # Multi-class
        class_counts = torch.bincount(valid_labels)
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum()
    else:  # Binary
        pos_weight = (valid_labels == 0).sum().float() / (valid_labels == 1).sum().float()
        class_weights = torch.tensor([1.0, pos_weight])
    
    return node_features, edge_index, labels, train_mask, class_weights

def train(data_path, model_path):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    node_features, edge_index, labels, train_mask, class_weights = load_data(data_path)
    print(f"Node features shape: {node_features.shape}")
    print(f"Edges shape: {edge_index.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of training nodes: {train_mask.sum().item()}")
    
    # Determine task type and output dimension
    num_classes = len(torch.unique(labels))
    task_type = 'classification' if num_classes > 1 else 'regression'
    output_dim = num_classes if task_type == 'classification' else 1
    print(f"Task type: {task_type}")
    print(f"Class weights: {class_weights}")
    
    # Model parameters
    input_dim = node_features.shape[1]
    hidden_dim = 256 if num_classes > 5 else 128
    num_layers = 3
    dropout = 0.5
    
    print(f"Number of features: {input_dim}")
    print(f"Number of classes: {num_classes}")
    
    # Initialize model
    model = GNNModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout,
        task_type=task_type
    ).to(device)
    print(model)
    
    # Training parameters
    num_epochs = 200
    patience = 15
    min_delta = 1e-3
    
    # Initialize optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.05,
        epochs=num_epochs,
        steps_per_epoch=1
    )
    
    # Move data to device
    node_features = node_features.to(device)
    edge_index = edge_index.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    class_weights = class_weights.to(device)
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(node_features, edge_index)
        
        # Compute loss
        if task_type == 'classification':
            loss = F.nll_loss(out[train_mask], labels[train_mask], weight=class_weights)
        else:
            loss = F.mse_loss(out[train_mask].squeeze(), labels[train_mask].float())
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Early stopping
        if loss.item() < best_loss - min_delta:
            best_loss = loss.item()
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'task_type': task_type,
                'num_features': input_dim,
                'num_classes': num_classes,
                'hidden_dim': hidden_dim
            }, model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    print("Training finished!")

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    train(data_path, model_path)