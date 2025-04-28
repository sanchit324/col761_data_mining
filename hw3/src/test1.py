import torch
import numpy as np
import pandas as pd
import sys
import os
from model1 import GNNModel
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler

def load_test_data(graph_path):
    # Load numpy arrays
    node_feat = np.load(os.path.join(graph_path, 'node_feat.npy'))
    edges = np.load(os.path.join(graph_path, 'edges.npy'))
    labels = np.load(os.path.join(graph_path, 'label.npy'))
    
    # Normalize features using RobustScaler for consistency with training
    scaler = RobustScaler()
    node_feat = scaler.fit_transform(node_feat)
    
    # Convert to torch tensors
    x = torch.FloatTensor(node_feat)
    edge_index = torch.LongTensor(edges).t().contiguous()
    
    # Create test mask (nodes with non-nan labels)
    test_mask = ~np.isnan(labels)
    y = labels[test_mask]  # Keep original labels for evaluation
    
    data = Data(x=x, edge_index=edge_index, test_mask=test_mask)
    return data, y

def main():
    if len(sys.argv) != 4:
        print("Usage: python test1.py <path_to_test_graph> <path_to_model> <output_file_path>")
        sys.exit(1)
        
    test_path = sys.argv[1]
    model_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    data, true_labels = load_test_data(test_path)
    print(f"Test data loaded. Features shape: {data.x.shape}")
    
    # Load model checkpoint
    checkpoint = torch.load(model_path)
    
    # Determine task type from model output dimension
    if isinstance(checkpoint, dict) and 'task_type' in checkpoint:
        task_type = checkpoint['task_type']
        num_features = checkpoint['num_features']
        num_classes = checkpoint['num_classes']
        hidden_dim = checkpoint.get('hidden_dim', 256)
    else:
        # If checkpoint is just the state dict, infer parameters
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
        output_dim = state_dict['output.weight'].size(0)
        num_features = state_dict['convs.0.lin.weight'].size(1)
        num_classes = output_dim
        hidden_dim = 256 if num_classes > 5 else 128
        task_type = 'classification'  # Default to classification for this assignment
    
    print(f"Model parameters:")
    print(f"Number of features: {num_features}")
    print(f"Number of classes: {num_classes}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Task type: {task_type}")
    
    # Initialize model
    model = GNNModel(
        input_dim=num_features,
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        num_layers=3,
        task_type=task_type
    ).to(device)
    
    # Load model weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If checkpoint is just the state dict
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Move data to device
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    
    # Get predictions
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        
        if task_type == 'classification':
            # For d1 (ROC-AUC), save probabilities
            if num_classes == 5:  # d1 dataset
                probs = torch.exp(out[data.test_mask])
                predictions = probs.cpu().numpy()
                # Calculate ROC-AUC
                try:
                    roc_auc = roc_auc_score(true_labels, predictions, multi_class='ovr')
                    print(f"ROC-AUC Score (d1): {roc_auc:.4f}")
                except Exception as e:
                    print(f"Error calculating ROC-AUC: {e}")
            else:  # d2 dataset (Accuracy)
                predictions = torch.argmax(out[data.test_mask], dim=1).cpu().numpy()
                accuracy = accuracy_score(true_labels, predictions)
                print(f"Accuracy Score (d2): {accuracy:.4f}")
        else:
            predictions = out[data.test_mask].cpu().numpy()
    
    # Save predictions
    pd.DataFrame(predictions).to_csv(output_path, header=False, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main() 