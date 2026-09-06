"""
04_gnn_semi_supervised.py
=========================
Graph Neural Network for Semi-Supervised Physical Distress Prediction

This script uses GNNs to leverage both labeled (REAC-scored) and unlabeled 
properties by constructing a property graph based on:
1. Spatial proximity (same census tract)
2. Ownership (same owner)
3. Program type similarity
4. Geographic proximity (k-nearest neighbors by lat/lon)

Architecture options:
- GCN (Graph Convolutional Network)
- GraphSAGE
- GAT (Graph Attention Network)

Author: Nischal Subedi
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'input_file': 'data/merged_analysis_data.parquet',
    'output_dir': 'outputs/',
    'random_state': 42,
    
    # Graph construction
    'k_spatial_neighbors': 10,  # k-NN for spatial graph
    'use_tract_edges': True,    # Connect properties in same tract
    'use_owner_edges': True,    # Connect properties with same owner
    'use_knn_edges': True,      # Connect k nearest neighbors by location
    
    # Model architecture
    'hidden_dim': 128,
    'num_layers': 3,
    'dropout': 0.3,
    
    # Training
    'epochs': 200,
    'lr': 0.01,
    'weight_decay': 5e-4,
    'patience': 20,  # Early stopping
    
    # Semi-supervised split
    'val_ratio': 0.1,  # Of labeled data
    'test_ratio': 0.2,  # Of labeled data
}


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_property_graph(df, config):
    """
    Build a graph where nodes are properties and edges connect:
    1. Properties in the same census tract
    2. Properties with the same owner
    3. K-nearest neighbors by geographic coordinates
    """
    print("\n" + "=" * 70)
    print("BUILDING PROPERTY GRAPH")
    print("=" * 70)
    
    n_properties = len(df)
    print(f"Total properties (nodes): {n_properties:,}")
    
    edge_list = []
    
    # --- Edge Type 1: Same Census Tract ---
    if config['use_tract_edges']:
        print("\nBuilding tract-based edges...")
        tract_groups = df.groupby('CensusTract').indices
        
        tract_edges = 0
        for tract, indices in tract_groups.items():
            if len(indices) > 1 and len(indices) < 500:  # Skip very large tracts
                # Connect all properties in same tract
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        edge_list.append((indices[i], indices[j]))
                        tract_edges += 1
        
        print(f"  Tract edges: {tract_edges:,}")
    
    # --- Edge Type 2: Same Owner ---
    if config['use_owner_edges']:
        print("\nBuilding owner-based edges...")
        
        # Clean owner names for matching
        df['Owner_Clean'] = df['Owner'].fillna('UNKNOWN').str.upper().str.strip()
        
        owner_groups = df.groupby('Owner_Clean').indices
        
        owner_edges = 0
        for owner, indices in owner_groups.items():
            if owner != 'UNKNOWN' and len(indices) > 1 and len(indices) < 100:
                # Connect all properties with same owner
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        edge_list.append((indices[i], indices[j]))
                        owner_edges += 1
        
        print(f"  Owner edges: {owner_edges:,}")
    
    # --- Edge Type 3: K-Nearest Neighbors (Spatial) ---
    if config['use_knn_edges']:
        print("\nBuilding spatial k-NN edges...")
        
        # Get coordinates
        coords = df[['Latitude', 'Longitude']].values
        
        # Handle missing coordinates
        valid_coords = ~np.isnan(coords).any(axis=1)
        valid_indices = np.where(valid_coords)[0]
        
        if len(valid_indices) > 0:
            valid_coords_data = coords[valid_indices]
            
            # Fit k-NN
            k = min(config['k_spatial_neighbors'], len(valid_indices) - 1)
            nn = NearestNeighbors(n_neighbors=k + 1, metric='haversine')
            nn.fit(np.radians(valid_coords_data))  # Convert to radians for haversine
            
            distances, neighbors = nn.kneighbors(np.radians(valid_coords_data))
            
            knn_edges = 0
            for i, (dist, neighs) in enumerate(zip(distances, neighbors)):
                source_idx = valid_indices[i]
                for j, neighbor_local_idx in enumerate(neighs[1:]):  # Skip self
                    target_idx = valid_indices[neighbor_local_idx]
                    if dist[j + 1] < 0.1:  # ~6km radius max
                        edge_list.append((source_idx, target_idx))
                        knn_edges += 1
            
            print(f"  Spatial k-NN edges: {knn_edges:,}")
        else:
            print("  Warning: No valid coordinates for k-NN")
    
    # --- Convert to PyG format ---
    print("\nConverting to PyTorch Geometric format...")
    
    if len(edge_list) == 0:
        print("  Warning: No edges created! Creating self-loops only.")
        edge_list = [(i, i) for i in range(n_properties)]
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Make undirected
    edge_index = to_undirected(edge_index)
    
    print(f"  Final edges: {edge_index.shape[1]:,}")
    print(f"  Average degree: {edge_index.shape[1] / n_properties:.1f}")
    
    return edge_index


def prepare_node_features(df, feature_cols):
    """
    Prepare node feature matrix from property/neighborhood attributes.
    """
    print("\nPreparing node features...")
    
    # Select available features
    available_features = [c for c in feature_cols if c in df.columns]
    print(f"  Using {len(available_features)} features")
    
    X = df[available_features].copy()
    
    # Fill missing values
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return torch.tensor(X_scaled, dtype=torch.float32), available_features, scaler


def prepare_labels_and_masks(df, config):
    """
    Prepare labels and train/val/test masks for semi-supervised learning.
    
    Key insight: Only labeled nodes are used for supervised loss,
    but ALL nodes participate in message passing!
    """
    print("\nPreparing labels and masks...")
    
    n_properties = len(df)
    
    # Labels (-1 for unlabeled)
    labels = df['PhysicalDistress'].fillna(-1).values.astype(int)
    
    labeled_mask = labels >= 0
    n_labeled = labeled_mask.sum()
    n_unlabeled = (~labeled_mask).sum()
    
    print(f"  Labeled nodes: {n_labeled:,} ({n_labeled/n_properties*100:.1f}%)")
    print(f"  Unlabeled nodes: {n_unlabeled:,} ({n_unlabeled/n_properties*100:.1f}%)")
    
    # Split labeled nodes into train/val/test
    labeled_indices = np.where(labeled_mask)[0]
    labeled_labels = labels[labeled_indices]
    
    # First split: train vs (val + test)
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        labeled_indices, labeled_labels,
        test_size=config['val_ratio'] + config['test_ratio'],
        random_state=config['random_state'],
        stratify=labeled_labels
    )
    
    # Second split: val vs test
    val_ratio_adjusted = config['val_ratio'] / (config['val_ratio'] + config['test_ratio'])
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - val_ratio_adjusted,
        random_state=config['random_state'],
        stratify=temp_labels
    )
    
    # Create masks
    train_mask = torch.zeros(n_properties, dtype=torch.bool)
    val_mask = torch.zeros(n_properties, dtype=torch.bool)
    test_mask = torch.zeros(n_properties, dtype=torch.bool)
    unlabeled_mask = torch.zeros(n_properties, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    unlabeled_mask[~labeled_mask] = True
    
    print(f"  Train: {train_mask.sum().item():,}")
    print(f"  Val: {val_mask.sum().item():,}")
    print(f"  Test: {test_mask.sum().item():,}")
    print(f"  Unlabeled: {unlabeled_mask.sum().item():,}")
    
    # Convert labels to tensor (use 0 for unlabeled, will be masked anyway)
    labels_tensor = torch.tensor(np.maximum(labels, 0), dtype=torch.long)
    
    return labels_tensor, train_mask, val_mask, test_mask, unlabeled_mask


# =============================================================================
# GNN MODELS
# =============================================================================

class GCN(nn.Module):
    """Graph Convolutional Network"""
    
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout):
        super(GCN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, out_dim))
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


class GraphSAGE(nn.Module):
    """GraphSAGE for inductive learning"""
    
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout):
        super(GraphSAGE, self).__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.convs.append(SAGEConv(hidden_dim, out_dim))
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


class GAT(nn.Module):
    """Graph Attention Network"""
    
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, heads=4):
        super(GAT, self).__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(GATConv(in_dim, hidden_dim // heads, heads=heads))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.convs.append(GATConv(hidden_dim, out_dim, heads=1, concat=False))
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, data, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    """Evaluate model on train/val/test splits."""
    model.eval()
    
    out = model(data.x, data.edge_index)
    probs = F.softmax(out, dim=1)[:, 1]
    preds = out.argmax(dim=1)
    
    results = {}
    
    for split, mask in [('train', data.train_mask), 
                        ('val', data.val_mask), 
                        ('test', data.test_mask)]:
        if mask.sum() > 0:
            y_true = data.y[mask].cpu().numpy()
            y_pred = preds[mask].cpu().numpy()
            y_prob = probs[mask].cpu().numpy()
            
            results[f'{split}_auc'] = roc_auc_score(y_true, y_prob)
            results[f'{split}_f1'] = f1_score(y_true, y_pred)
            results[f'{split}_recall'] = recall_score(y_true, y_pred)
            results[f'{split}_precision'] = precision_score(y_true, y_pred, zero_division=0)
    
    return results


def train_model(model, data, config):
    """Full training loop with early stopping."""
    print("\n" + "=" * 70)
    print(f"TRAINING {model.__class__.__name__}")
    print("=" * 70)
    
    model = model.to(device)
    data = data.to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    # Class weights for imbalanced data
    n_pos = (data.y[data.train_mask] == 1).sum().item()
    n_neg = (data.y[data.train_mask] == 0).sum().item()
    weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    best_val_auc = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(config['epochs']):
        loss = train_epoch(model, data, optimizer, criterion)
        results = evaluate(model, data)
        
        if results['val_auc'] > best_val_auc:
            best_val_auc = results['val_auc']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: Loss={loss:.4f}, "
                  f"Val AUC={results['val_auc']:.4f}, "
                  f"Val Recall={results['val_recall']:.4f}")
        
        if patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    final_results = evaluate(model, data)
    print(f"\nFinal Results:")
    print(f"  Test AUC: {final_results['test_auc']:.4f}")
    print(f"  Test Recall: {final_results['test_recall']:.4f}")
    print(f"  Test Precision: {final_results['test_precision']:.4f}")
    print(f"  Test F1: {final_results['test_f1']:.4f}")
    
    return model, final_results


# =============================================================================
# PREDICTION
# =============================================================================

@torch.no_grad()
def predict_all(model, data):
    """Get predictions for all nodes (including unlabeled)."""
    model.eval()
    
    out = model(data.x.to(device), data.edge_index.to(device))
    probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
    
    return probs


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run GNN semi-supervised learning pipeline."""
    
    print("\n" + "=" * 70)
    print("GNN SEMI-SUPERVISED LEARNING FOR PHYSICAL DISTRESS")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_parquet(CONFIG['input_file'])
    df = df.reset_index(drop=True)  # Ensure clean index
    print(f"Loaded {len(df):,} properties")
    
    # Define features
    feature_cols = [
        # Property features
        'PropertyAge', 'TotalUnits', 'FairMarketRent_2BR',
        'PercentStudioOneBedroom', 'NumActivePrograms',
        'Has_Section8', 'Has_LIHTC', 'Has_Section515',
        'Has_PublicHousing', 'Has_Section202', 'IsMetro',
        'MonthsToExpire',
        # Neighborhood features
        'Poverty_Rate', 'Vacancy_Rate', 'Renter_Share',
        'Elderly_Share', 'Disability_Rate', 'Pre1970_Housing_Share',
        'Pop_Density', 'Median_Household_Income', 'Median_Gross_Rent',
    ]
    
    # Build graph
    edge_index = build_property_graph(df, CONFIG)
    
    # Prepare features
    x, used_features, scaler = prepare_node_features(df, feature_cols)
    
    # Prepare labels and masks
    y, train_mask, val_mask, test_mask, unlabeled_mask = prepare_labels_and_masks(df, CONFIG)
    
    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        unlabeled_mask=unlabeled_mask,
    )
    
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.num_edges:,}")
    print(f"  Features: {data.num_features}")
    
    # Train multiple GNN architectures
    in_dim = data.num_features
    hidden_dim = CONFIG['hidden_dim']
    out_dim = 2  # Binary classification
    num_layers = CONFIG['num_layers']
    dropout = CONFIG['dropout']
    
    models = {
        'GCN': GCN(in_dim, hidden_dim, out_dim, num_layers, dropout),
        'GraphSAGE': GraphSAGE(in_dim, hidden_dim, out_dim, num_layers, dropout),
        'GAT': GAT(in_dim, hidden_dim, out_dim, num_layers, dropout),
    }
    
    all_results = []
    best_model = None
    best_auc = 0
    best_name = None
    
    for name, model in models.items():
        trained_model, results = train_model(model, data, CONFIG)
        
        all_results.append({
            'Model': name,
            'Test_AUC': results['test_auc'],
            'Test_Recall': results['test_recall'],
            'Test_Precision': results['test_precision'],
            'Test_F1': results['test_f1'],
        })
        
        if results['test_auc'] > best_auc:
            best_auc = results['test_auc']
            best_model = trained_model
            best_name = name
    
    # Summary
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))
    results_df.to_csv(f"{CONFIG['output_dir']}/gnn_model_comparison.csv", index=False)
    
    print(f"\nBest Model: {best_name} (AUC={best_auc:.4f})")
    
    # Predict for ALL nodes (including unlabeled)
    print("\n" + "=" * 70)
    print("PREDICTING FOR ALL PROPERTIES")
    print("=" * 70)
    
    all_probs = predict_all(best_model, data)
    
    df['GNN_RiskScore'] = all_probs
    df['GNN_HighRisk'] = (all_probs >= 0.5).astype(int)
    
    # Separate predictions for unlabeled properties
    unlabeled_probs = all_probs[unlabeled_mask.numpy()]
    print(f"\nUnlabeled properties: {len(unlabeled_probs):,}")
    print(f"  High Risk (>=0.5): {(unlabeled_probs >= 0.5).sum():,} ({(unlabeled_probs >= 0.5).mean()*100:.1f}%)")
    print(f"  Mean risk score: {unlabeled_probs.mean():.3f}")
    print(f"  Median risk score: {np.median(unlabeled_probs):.3f}")
    
    # Save predictions
    output_cols = ['NHPDPropertyID', 'PropertyName', 'City', 'State',
                   'TotalUnits', 'PropertyAge', 'PhysicalDistress',
                   'GNN_RiskScore', 'GNN_HighRisk']
    output_cols = [c for c in output_cols if c in df.columns]
    
    df[output_cols].to_csv(f"{CONFIG['output_dir']}/gnn_predictions.csv", index=False)
    
    # Save model
    torch.save(best_model.state_dict(), f"{CONFIG['output_dir']}/best_gnn_model.pt")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    
    return {
        'model': best_model,
        'results': results_df,
        'predictions': df[['NHPDPropertyID', 'GNN_RiskScore', 'GNN_HighRisk']],
    }


if __name__ == "__main__":
    outputs = main()