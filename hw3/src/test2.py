import os
import torch
import numpy as np
import pandas as pd
from model2 import PersonalityGNN
import sys
from sklearn.metrics import f1_score

def load_test_graph(path):
    # Load user features
    user_features = np.load(os.path.join(path, 'user_features.npy'))
    
    # Load product features
    product_features = np.load(os.path.join(path, 'product_features.npy'))
    
    # Load edges
    edges = np.load(os.path.join(path, 'user_product.npy'))
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    
    # Load labels if available (for evaluation)
    label_path = os.path.join(path, 'label.npy')
    labels = np.load(label_path) if os.path.exists(label_path) else None
    
    # Convert features to tensors
    x_user = torch.tensor(user_features, dtype=torch.float)
    x_product = torch.tensor(product_features, dtype=torch.float)
    
    return x_user, x_product, edge_index, labels

def main():
    if len(sys.argv) != 4:
        print("Usage: python test2.py <path_to_test_graph> <path_to_model> <output_file_path>")
        sys.exit(1)
        
    test_path = sys.argv[1]
    model_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    x_user, x_product, edge_index, true_labels = load_test_graph(test_path)
    
    print(f"Test data shapes:")
    print(f"User features: {x_user.shape}")
    print(f"Product features: {x_product.shape}")
    print(f"Edge index: {edge_index.shape}")
    
    # Model parameters
    user_dim = x_user.size(1)
    product_dim = x_product.size(1)
    hidden_dim = 256  # Match training hidden dimension
    
    # Load model state to get num_personality_dims
    state_dict = torch.load(model_path)
    num_personality_dims = state_dict['fc3.weight'].size(0)
    print(f"Number of personality dimensions: {num_personality_dims}")
    
    # Initialize model
    model = PersonalityGNN(
        user_dim=user_dim,
        product_dim=product_dim,
        hidden_dim=hidden_dim,
        num_personality_dims=num_personality_dims
    ).to(device)
    
    # Load trained model weights
    model.load_state_dict(state_dict)
    model.eval()
    
    # Move data to device
    x_user = x_user.to(device)
    x_product = x_product.to(device)
    edge_index = edge_index.to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(x_user, x_product, edge_index)
    
    # Convert predictions to numpy
    predictions_np = predictions.cpu().numpy()
    
    # Calculate weighted F1 score if labels are available
    if true_labels is not None:
        # Flatten both predictions and true labels
        pred_flat = predictions_np.reshape(-1)
        true_flat = true_labels.reshape(-1)
        
        # Convert to binary predictions (threshold at 0.5)
        pred_binary = (pred_flat > 0.5).astype(int)
        
        # Calculate weighted F1 score
        f1 = f1_score(true_flat, pred_binary, average='weighted')
        print(f"Weighted F1 Score: {f1:.4f}")
    
    # Save predictions in original shape (mt x ℓ)
    pd.DataFrame(predictions_np).to_csv(output_path, header=False, index=False)
    print(f"Predictions shape: {predictions_np.shape}")
    print(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    main() 