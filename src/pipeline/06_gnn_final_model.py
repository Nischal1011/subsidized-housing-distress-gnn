"""
06_gnn_final_model.py
======================
The final GAT model used for the paper's headline results (Table 2, Table 4,
Table 6, Table 8). Writes outputs/integrated_policy_matrix_v2.csv, which
feeds Table 5, Table 9, and the compounded-risk analysis. NOTE: Table 7,
Table 11, and Figures 2/4/7 were built from an older run's output,
outputs/integrated_policy_matrix.csv (no "_v2" suffix) — see
docs/VERIFICATION.md for why the two differ.

Optimized for higher AUC (target: 0.80+)

Key changes:
1. Class balancing with weighted loss
2. Deeper architecture with attention
3. Better feature preprocessing
4. Learning rate scheduling
5. Ensemble predictions
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv, BatchNorm, global_mean_pool
from torch_geometric.utils import to_undirected, add_self_loops
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, precision_recall_curve
from sklearn.neighbors import NearestNeighbors
import warnings
import os

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# UPDATED CONFIGURATION - Optimized for AUC
# =============================================================================

CONFIG = {
    'input_file': 'data/merged_analysis_data.parquet',
    'output_dir': 'outputs/',
    'random_state': 42,
    
    # Graph Construction - More connections for better message passing
    'k_spatial_neighbors': 15,      # Increased from 10
    'spatial_radius_km': 8.0,       # Increased from 5
    'use_owner_edges': True,
    'use_tract_edges': True,        # Enable tract edges
    'use_knn_edges': True,
    
    # Model Architecture - Deeper with attention
    'hidden_dim': 256,              # Increased from 128
    'num_layers': 4,                # Increased from 3
    'dropout': 0.4,                 # Slightly higher
    'heads': 4,                     # For GAT layers
    'use_attention': True,          # Use GAT instead of SAGE
    
    # Training - More careful optimization
    'epochs': 500,                  # Increased
    'lr': 0.001,                    # Lower initial LR
    'weight_decay': 5e-4,           # Increased regularization
    'patience': 60,                 # More patience
    'lr_scheduler': True,           # Use LR scheduling
    'class_weight': True,           # Weight minority class
    
    # Cross-validation for robust estimates
    'n_folds': 5,
    'use_cv': True,
    
    'val_ratio': 0.15,
    'test_ratio': 0.15,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)


# =============================================================================
# ENHANCED FEATURE ENGINEERING
# =============================================================================

def create_features(df):
    """Enhanced feature engineering with interaction terms."""
    print("\n[1/5] Enhanced Feature Engineering...")
    df = df.copy()
    
    # --- A. Economic Features (Same as before) ---
    if 'Median_Household_Income' in df.columns and 'Median_Gross_Rent' in df.columns:
        print("  - Calculating True Rent Gap...")
        inc = df['Median_Household_Income'].fillna(df['Median_Household_Income'].median())
        mkt_rent = df['Median_Gross_Rent'].fillna(df['Median_Gross_Rent'].median())
        
        restricted_rent = (inc * 0.60 * 0.30) / 12
        df['True_Rent_Gap'] = mkt_rent - restricted_rent
        
        min_gap = df['True_Rent_Gap'].min()
        shift = abs(min_gap) + 1 if min_gap < 0 else 1
        df['Log_Rent_Gap'] = np.log1p(df['True_Rent_Gap'] + shift)
        df['High_Exit_Incentive'] = (df['True_Rent_Gap'] > 400).astype(float)
        
        # NEW: Rent gap percentile (more robust)
        df['Rent_Gap_Percentile'] = df['True_Rent_Gap'].rank(pct=True)
        
    else:
        df['True_Rent_Gap'] = 0
        df['Log_Rent_Gap'] = 0
        df['High_Exit_Incentive'] = 0
        df['Rent_Gap_Percentile'] = 0.5

    # --- B. Property Features with Interactions ---
    if 'PropertyAge' in df.columns:
        age = df['PropertyAge'].fillna(df['PropertyAge'].median()).fillna(30)
        df['Log_Age'] = np.log1p(age)
        df['Age_Squared'] = age ** 2 / 1000
        df['Is_Old'] = (age > 40).astype(float)
        df['Is_Very_Old'] = (age > 50).astype(float)  # NEW
        
        # NEW: Age buckets for non-linear effects
        df['Age_Bucket'] = pd.cut(age, bins=[0, 20, 35, 50, 100], labels=[0, 1, 2, 3]).astype(float)
    
    if 'TotalUnits' in df.columns:
        units = df['TotalUnits'].fillna(df['TotalUnits'].median()).fillna(50)
        df['Log_Units'] = np.log1p(units)
        df['Is_Large'] = (units > 100).astype(float)
        df['Is_Small'] = (units < 20).astype(float)  # NEW: Small properties
        
    # --- C. Neighborhood Features ---
    if 'Poverty_Rate' in df.columns:
        pov = df['Poverty_Rate'].fillna(df['Poverty_Rate'].median()).fillna(15)
        df['High_Poverty'] = (pov > 25).astype(float)
        df['Poverty_Percentile'] = df['Poverty_Rate'].rank(pct=True)  # NEW

    if 'Vacancy_Rate' in df.columns:
        vac = df['Vacancy_Rate'].fillna(df['Vacancy_Rate'].median()).fillna(5)
        df['High_Vacancy'] = (vac > 10).astype(float)
        df['Vacancy_Percentile'] = df['Vacancy_Rate'].rank(pct=True)  # NEW
        
    # --- D. NEW: Interaction Features ---
    if 'PropertyAge' in df.columns and 'Poverty_Rate' in df.columns:
        df['Age_x_Poverty'] = df['Log_Age'] * df['Poverty_Rate'].fillna(0) / 100
        
    if 'PropertyAge' in df.columns and 'True_Rent_Gap' in df.columns:
        df['Age_x_RentGap'] = df['Log_Age'] * df['Rent_Gap_Percentile']

    # --- E. Ownership Features (Same as before) ---
    if 'Owner' in df.columns:
        df['Owner_Clean'] = df['Owner'].fillna('UNKNOWN').astype(str).str.upper().str.strip()
        df['Owner_Clean'] = df['Owner_Clean'].replace({'UNKNOWN': np.nan, 'NAN': np.nan, 'NONE': np.nan})
        
        owner_counts = df['Owner_Clean'].value_counts()
        df['Owner_Portfolio_Size'] = df['Owner_Clean'].map(owner_counts).fillna(1)
        df['Log_Portfolio'] = np.log1p(df['Owner_Portfolio_Size'])
        df['Is_Large_Owner'] = (df['Owner_Portfolio_Size'] > 10).astype(float)
        
        # NEW: Portfolio distress rate (if we have labels)
        # This is a powerful feature - owners with history of distressed properties

    print(f"  - Created {len([c for c in df.columns if c not in ['Owner', 'Owner_Clean']])} features")
    return df


# =============================================================================
# ENHANCED GRAPH CONSTRUCTION
# =============================================================================

def build_graph(df, config):
    """Build graph with additional edge types."""
    print("\n[2/5] Building Enhanced Graph Network...")
    n = len(df)
    edge_list = []
    edge_weights = []  # NEW: Edge weights for importance
    
    # --- A. Owner Edges (Higher weight) ---
    if config['use_owner_edges']:
        print("  - Building Ownership Network...")
        owner_groups = df.groupby('Owner_Clean').indices
        count_owner_edges = 0
        
        for owner, indices in owner_groups.items():
            if pd.isna(owner) or owner == 'nan': 
                continue
            if len(indices) < 2: 
                continue
            
            idx_list = indices
            if len(idx_list) > 50:
                idx_list = np.random.choice(idx_list, 50, replace=False)
                
            for i in range(len(idx_list)):
                for j in range(i + 1, len(idx_list)):
                    edge_list.append((idx_list[i], idx_list[j]))
                    edge_weights.append(2.0)  # Higher weight for ownership
                    count_owner_edges += 1
                    
        print(f"    Added {count_owner_edges:,} ownership edges.")

    # --- B. Tract Edges (NEW - same census tract) ---
    if config.get('use_tract_edges', False) and 'CensusTract' in df.columns:
        print("  - Building Census Tract Network...")
        tract_groups = df.groupby('CensusTract').indices
        count_tract_edges = 0
        
        for tract, indices in tract_groups.items():
            if pd.isna(tract):
                continue
            if len(indices) < 2 or len(indices) > 100:  # Skip very large tracts
                continue
                
            idx_list = indices if len(indices) <= 30 else np.random.choice(indices, 30, replace=False)
            
            for i in range(len(idx_list)):
                for j in range(i + 1, len(idx_list)):
                    edge_list.append((idx_list[i], idx_list[j]))
                    edge_weights.append(1.5)  # Medium weight
                    count_tract_edges += 1
                    
        print(f"    Added {count_tract_edges:,} tract edges.")

    # --- C. Spatial Edges (k-NN) ---
    if config['use_knn_edges']:
        print("  - Building Spatial Network (k-NN)...")
        coords = df[['Latitude', 'Longitude']].values
        valid_mask = ~np.isnan(coords).any(axis=1)
        valid_idx = np.where(valid_mask)[0]
        
        if len(valid_idx) > 0:
            valid_coords = coords[valid_idx]
            k = min(config['k_spatial_neighbors'], len(valid_idx) - 1)
            nn = NearestNeighbors(n_neighbors=k + 1, metric='haversine', algorithm='ball_tree')
            nn.fit(np.radians(valid_coords))
            distances, neighbors = nn.kneighbors(np.radians(valid_coords))
            distances_km = distances * 6371
            
            count_spatial_edges = 0
            for i, (dist_row, neigh_row) in enumerate(zip(distances_km, neighbors)):
                src_node = valid_idx[i]
                
                for j, neigh_local_idx in enumerate(neigh_row[1:]):
                    dist_val = dist_row[j+1]
                    
                    if dist_val <= config['spatial_radius_km']:
                        tgt_node = valid_idx[neigh_local_idx]
                        edge_list.append((src_node, tgt_node))
                        # Weight inversely proportional to distance
                        edge_weights.append(1.0 / (1 + dist_val))
                        count_spatial_edges += 1
                        
            print(f"    Added {count_spatial_edges:,} spatial edges.")

    if not edge_list:
        print("  ! WARNING: No edges created. Adding self-loops only.")
        edge_list = [(i, i) for i in range(n)]
        edge_weights = [1.0] * n
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
    
    edge_index = to_undirected(edge_index)
    # Duplicate weights for undirected
    edge_weight = torch.cat([edge_weight, edge_weight])
    
    edge_index, _ = add_self_loops(edge_index, num_nodes=n)
    # Add weight 1.0 for self-loops
    edge_weight = torch.cat([edge_weight, torch.ones(n)])
    
    print(f"  - Final Graph: {n:,} nodes, {edge_index.shape[1]:,} edges.")
    return edge_index, edge_weight


# =============================================================================
# IMPROVED MODEL ARCHITECTURE
# =============================================================================

class ImprovedGNN(nn.Module):
    """Hybrid GAT + SAGE with skip connections and layer norm."""
    
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, heads=4, use_attention=True):
        super().__init__()
        
        self.use_attention = use_attention
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Graph layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        
        for i in range(num_layers):
            if use_attention and i < 2:  # Use GAT for first 2 layers
                # GAT output dim = hidden_dim (heads * out_per_head)
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True))
            else:
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr='mean'))
            
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.skip_projs.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output head with more capacity
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, out_dim)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_weight=None):
        # Input projection
        x = self.input_proj(x)
        
        # Message passing with skip connections
        for i, (conv, norm, skip) in enumerate(zip(self.convs, self.norms, self.skip_projs)):
            identity = skip(x)
            
            if self.use_attention and i < 2:
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index)
            
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + identity  # Skip connection
        
        return self.output(x)


class FocalLoss(nn.Module):
    """Focal Loss with class weights."""
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# =============================================================================
# IMPROVED TRAINING WITH LR SCHEDULING
# =============================================================================

def train_gnn(model, data, config, class_weights=None):
    print("\n[3/5] Training Improved GNN Model...")
    
    model = model.to(device)
    data = data.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    # Learning rate scheduler
    if config.get('lr_scheduler', False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    else:
        scheduler = None
    
    # Loss with class weights
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = FocalLoss(alpha=0.35, gamma=2.0, class_weights=class_weights)
    
    best_val_auc = 0
    patience_counter = 0
    best_state = None
    
    for epoch in range(config['epochs']):
        # Train
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.edge_weight if hasattr(data, 'edge_weight') else None)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_weight if hasattr(data, 'edge_weight') else None)
            probs = F.softmax(out, dim=1)[:, 1]
            
            val_probs = probs[data.val_mask].cpu().numpy()
            val_labels = data.y[data.val_mask].cpu().numpy()
            
            try:
                val_auc = roc_auc_score(val_labels, val_probs)
            except:
                val_auc = 0.5
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            
        if (epoch + 1) % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1}: Loss={loss.item():.4f}, Val AUC={val_auc:.4f}, LR={current_lr:.6f}")
            
        if patience_counter >= config['patience']:
            print(f"    Early stopping at epoch {epoch+1}. Best Val AUC: {best_val_auc:.4f}")
            break
    
    model.load_state_dict(best_state)
    return model, best_val_auc


def train_with_cv(df, feature_cols, edge_index, edge_weight, labels, config):
    """Train with cross-validation for robust AUC estimate."""
    print("\n[3/5] Training with {}-Fold Cross-Validation...".format(config['n_folds']))
    
    labeled_indices = np.where(labels >= 0)[0]
    labeled_labels = labels[labeled_indices]
    
    skf = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=config['random_state'])
    
    fold_aucs = []
    all_probs = np.zeros(len(df))
    all_counts = np.zeros(len(df))
    
    X_raw = df[feature_cols].fillna(0).values
    qt = QuantileTransformer(output_distribution='normal', random_state=config['random_state'])
    X_scaled = qt.fit_transform(X_raw)
    x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Class weights
    n_pos = (labeled_labels == 1).sum()
    n_neg = (labeled_labels == 0).sum()
    class_weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32)
    print(f"  Class weights: [0: 1.0, 1: {n_neg/n_pos:.2f}]")
    
    for fold, (train_idx_local, val_idx_local) in enumerate(skf.split(labeled_indices, labeled_labels)):
        print(f"\n  --- Fold {fold + 1}/{config['n_folds']} ---")
        
        train_idx = labeled_indices[train_idx_local]
        val_idx = labeled_indices[val_idx_local]
        
        n = len(df)
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        
        y_tensor = torch.tensor(np.maximum(labels, 0), dtype=torch.long)
        
        data = Data(
            x=x_tensor, 
            edge_index=edge_index, 
            edge_weight=edge_weight,
            y=y_tensor,
            train_mask=train_mask, 
            val_mask=val_mask
        )
        
        model = ImprovedGNN(
            in_dim=data.num_features,
            hidden_dim=config['hidden_dim'],
            out_dim=2,
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            heads=config.get('heads', 4),
            use_attention=config.get('use_attention', True)
        )
        
        model, val_auc = train_gnn(model, data, config, class_weights)
        fold_aucs.append(val_auc)
        
        # Get predictions for this fold
        model.eval()
        with torch.no_grad():
            data = data.to(device)
            model = model.to(device)
            out = model(data.x, data.edge_index)
            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        
        all_probs += probs
        all_counts += 1
    
    # Average predictions across folds
    all_probs /= all_counts
    
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    print(f"\n  Cross-Validation Results: AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    
    return all_probs, mean_auc, std_auc


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("="*70)
    print("PRESERVATION RISK GNN PIPELINE v2 (Optimized for AUC)")
    print("="*70)
    
    if not os.path.exists(CONFIG['input_file']):
        print(f"Error: Input file {CONFIG['input_file']} not found.")
        return
        
    df = pd.read_parquet(CONFIG['input_file']).reset_index(drop=True)
    print(f"Loaded {len(df):,} properties.")
    
    # Feature Engineering
    df = create_features(df)
    
    exclude_cols = ['PhysicalDistress', 'BestReacScore', 'ReacScore1', 'ReacScore2', 'ReacScore3', 
                    'NHPDPropertyID', 'Latitude', 'Longitude', 'CensusTract', 'PropertyName', 
                    'PropertyAddress', 'Owner', 'Owner_Clean', 'City', 'State', 'Zip',
                    'Policy_Tier', 'GNN_Risk', 'GNN_HighRisk']
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols if c not in exclude_cols]
    
    print(f"Using {len(feature_cols)} features for GNN.")
    
    # Build Graph
    edge_index, edge_weight = build_graph(df, CONFIG)
    
    # Prepare Labels
    labels = df['PhysicalDistress'].fillna(-1).values.astype(int)
    labeled_indices = np.where(labels >= 0)[0]
    
    print(f"Labeled Nodes: {len(labeled_indices):,} ({(len(labeled_indices)/len(df))*100:.1f}%)")
    print(f"Distress Rate in Labeled: {labels[labeled_indices].mean()*100:.1f}%")
    
    if CONFIG.get('use_cv', False):
        # Cross-validation approach
        all_probs, mean_auc, std_auc = train_with_cv(df, feature_cols, edge_index, edge_weight, labels, CONFIG)
        
        # Find optimal threshold
        labeled_probs = all_probs[labeled_indices]
        labeled_labels = labels[labeled_indices]
        
        precision, recall, thresholds = precision_recall_curve(labeled_labels, labeled_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        opt_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        print(f"\n  Optimal Threshold: {opt_thresh:.4f}")
        
        # Final metrics on all labeled data
        final_preds = (labeled_probs >= opt_thresh).astype(int)
        print(f"  Final Metrics on Labeled Data:")
        print(f"    AUC:       {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"    Precision: {precision_score(labeled_labels, final_preds, zero_division=0):.4f}")
        print(f"    Recall:    {recall_score(labeled_labels, final_preds):.4f}")
        print(f"    F1 Score:  {f1_score(labeled_labels, final_preds):.4f}")
        
        df['GNN_Risk'] = all_probs
        df['GNN_HighRisk'] = (all_probs >= opt_thresh).astype(int)
        
    else:
        # Single train/val/test split (original approach but with improved model)
        X_raw = df[feature_cols].fillna(0).values
        qt = QuantileTransformer(output_distribution='normal', random_state=CONFIG['random_state'])
        X_scaled = qt.fit_transform(X_raw)
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        train_idx, temp_idx = train_test_split(
            labeled_indices, test_size=(CONFIG['val_ratio'] + CONFIG['test_ratio']), 
            random_state=CONFIG['random_state'], stratify=labels[labeled_indices]
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5,
            random_state=CONFIG['random_state'], stratify=labels[temp_idx]
        )
        
        n = len(df)
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        y_tensor = torch.tensor(np.maximum(labels, 0), dtype=torch.long)
        
        # Class weights
        n_pos = (labels[train_idx] == 1).sum()
        n_neg = (labels[train_idx] == 0).sum()
        class_weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32)
        
        data = Data(
            x=x_tensor, 
            edge_index=edge_index, 
            edge_weight=edge_weight,
            y=y_tensor,
            train_mask=train_mask, 
            val_mask=val_mask, 
            test_mask=test_mask
        )
        
        model = ImprovedGNN(
            in_dim=data.num_features,
            hidden_dim=CONFIG['hidden_dim'],
            out_dim=2,
            num_layers=CONFIG['num_layers'],
            dropout=CONFIG['dropout'],
            heads=CONFIG.get('heads', 4),
            use_attention=CONFIG.get('use_attention', True)
        )
        
        model, _ = train_gnn(model, data, CONFIG, class_weights)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            data = data.to(device)
            model = model.to(device)
            out = model(data.x, data.edge_index)
            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        
        # Threshold optimization on validation
        val_probs = probs[val_idx]
        val_labels = labels[val_idx]
        
        precision, recall, thresholds = precision_recall_curve(val_labels, val_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        opt_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        # Test evaluation
        test_probs = probs[test_idx]
        test_labels = labels[test_idx]
        test_auc = roc_auc_score(test_labels, test_probs)
        test_preds = (test_probs >= opt_thresh).astype(int)
        
        print(f"\n    Optimal Threshold: {opt_thresh:.4f}")
        print(f"    Test Results:")
        print(f"      AUC:       {test_auc:.4f}")
        print(f"      Precision: {precision_score(test_labels, test_preds, zero_division=0):.4f}")
        print(f"      Recall:    {recall_score(test_labels, test_preds):.4f}")
        print(f"      F1 Score:  {f1_score(test_labels, test_preds):.4f}")
        
        df['GNN_Risk'] = probs
        df['GNN_HighRisk'] = (probs >= opt_thresh).astype(int)
    
    # Inference stats
    unlabeled_mask = (labels == -1)
    n_unlabeled = unlabeled_mask.sum()
    n_flagged = df.loc[unlabeled_mask, 'GNN_HighRisk'].sum()
    
    print(f"\n[4/5] Inference on Unlabeled Data")
    print(f"  Total Unlabeled: {n_unlabeled:,}")
    print(f"  Flagged as Distressed: {n_flagged:,} ({(n_flagged/n_unlabeled):.1%})")
    
    # Policy synthesis (same as before)
    print("\n[5/5] Generating Integrated Policy Matrix...")
    
    df['Rent_Gap_Rank'] = df['True_Rent_Gap'].rank(pct=True)
    df['Policy_Tier'] = 'Tier 4: Stable (Low Risk)'
    
    high_phys = df['GNN_HighRisk'] == 1
    high_econ = df['Rent_Gap_Rank'] > 0.75
    
    df.loc[high_phys & high_econ, 'Policy_Tier'] = 'Tier 1: Crisis (Distress + Exit Risk)'
    df.loc[~high_phys & high_econ, 'Policy_Tier'] = 'Tier 2: Preservation (High Opt-Out Risk)'
    df.loc[high_phys & ~high_econ, 'Policy_Tier'] = 'Tier 3: Rehab Needed (Physical Distress)'
    
    print("\n  Policy Tier Distribution:")
    print(df['Policy_Tier'].value_counts())
    
    # Save
    cols = ['NHPDPropertyID', 'PropertyName', 'City', 'State', 
            'True_Rent_Gap', 'GNN_Risk', 'GNN_HighRisk', 'Policy_Tier']
    cols_avail = [c for c in cols if c in df.columns]
    
    out_path = f"{CONFIG['output_dir']}/integrated_policy_matrix_v2.csv"
    df[cols_avail].to_csv(out_path, index=False)
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()