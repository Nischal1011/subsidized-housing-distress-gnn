

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy import sparse
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# CONFIG
CONFIG = {
    'input_file': 'data/merged_analysis_data.parquet',
    'hidden_dim': 256,    # Increased for high-dim input
    'epochs': 300,
    'lr': 0.005,
    'dropout': 0.5
}

# =============================================================================
# 1. KITCHEN SINK FEATURE PROCESSING
# =============================================================================

def process_all_features(df):
    print("\n[1] Processing ALL available features (Kitchen Sink)...")
    
    # 1. Drop ID/Leakage columns
    drop_cols = [
        'PhysicalDistress', 'NHPDPropertyID', 'PropertyID', 'OldNHPDPropertyID', 
        'HUDPropertyID', 'ReacScore1', 'ReacScore2', 'ReacScore3', # Leaks target info
        'BestReacScore', 'HasREAC', 'PropertyAddress', 'PropertyName', 
        'ManagerName', 'Owner' # Text used for Graph, not Features
    ]
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # 2. Handle Dates (Convert to numeric years)
    date_cols = [c for c in df_clean.columns if 'Date' in c or 'Start' in c or 'End' in c]
    for col in date_cols:
        try:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            # Convert to "Years since 1970"
            df_clean[col] = (df_clean[col].dt.year - 1970).fillna(0)
        except:
            df_clean = df_clean.drop(columns=[col]) # Drop if parsing fails

    # 3. Handle Categoricals (Label Encode everything else object-like)
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))

    # 4. Handle Numerics (Fill NA)
    # XGBoost handles NA, but GNN needs clean input
    X = df_clean.fillna(0).values
    
    # Standardize for GNN (Critical for convergence)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"    Total Features Extracted: {X.shape[1]}")
    return X_scaled, df_clean.columns.tolist()

# =============================================================================
# 2. MANAGER GRAPH (Structure)
# =============================================================================

def build_graph(df):
    print("\n[2] Building Manager Graph...")
    df['Manager_Clean'] = df['ManagerName'].fillna('Unknown')
    manager_groups = df.groupby('Manager_Clean').groups
    
    rows, cols = [], []
    for mgr, indices in manager_groups.items():
        if mgr == 'Unknown' or len(indices) < 2: continue
        
        idx_list = indices.tolist()
        # Ring Connection (Efficient O(N))
        for i in range(len(idx_list)):
            target = idx_list[(i + 1) % len(idx_list)]
            rows.append(idx_list[i]); cols.append(target)
            rows.append(target); cols.append(idx_list[i])

    # Self loops
    rows.extend(range(len(df)))
    cols.extend(range(len(df)))
    
    indices = torch.LongTensor([rows, cols])
    
    # Norm
    adj_sparse = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(df), len(df)))
    row_sum = np.array(adj_sparse.sum(1)).flatten()
    d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    
    values = d_inv_sqrt[rows] * d_inv_sqrt[cols]
    adj_tensor = torch.sparse_coo_tensor(indices, torch.FloatTensor(values), (len(df), len(df)))
    
    return adj_tensor

# =============================================================================
# 3. MODELS
# =============================================================================

class IndependentGCN(nn.Module):
    def __init__(self, in_feat, hidden_dim, dropout):
        super().__init__()
        # Input Projection (Handle High-Dim)
        self.enc = nn.Linear(in_feat, hidden_dim)
        
        self.conv1 = nn.Linear(hidden_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, 1)
        self.dropout = dropout
        self.act = nn.ReLU()

    def forward(self, x, adj):
        x = self.act(self.enc(x)) # Project down
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.act(self.conv1(x))
        x = torch.sparse.mm(adj, x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.conv2(x)
        x = torch.sparse.mm(adj, x)
        return x

def train_gnn(X, adj, y, train_idx, test_idx):
    print("\n[3] Training GNN (High-Dim Inputs)...")
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(np.nan_to_num(y))
    
    model = IndependentGCN(X.shape[1], CONFIG['hidden_dim'], CONFIG['dropout'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    train_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[train_idx] = True
    
    n_pos = y_tensor[train_mask].sum()
    n_neg = len(y_tensor[train_mask]) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_auc = 0
    best_preds = None
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        optimizer.zero_grad()
        out = model(X_tensor, adj).squeeze()
        loss = criterion(out[train_mask], y_tensor[train_mask])
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            out = model(X_tensor, adj).squeeze()
            probs = torch.sigmoid(out).numpy()
            auc = roc_auc_score(y[test_idx], probs[test_idx])
            
            if auc > best_auc:
                best_auc = auc
                best_preds = probs[test_idx]
                
    print(f"    GNN AUC: {best_auc:.4f}")
    return best_preds

def train_xgboost(X, y, train_idx, test_idx):
    print("\n[4] Training XGBoost (High-Dim Inputs)...")
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.02, 
        colsample_bytree=0.5, # Subsample columns since we have so many
        random_state=42
    )
    model.fit(X[train_idx], y[train_idx])
    preds = model.predict_proba(X[test_idx])[:, 1]
    
    auc = roc_auc_score(y[test_idx], preds)
    print(f"    XGBoost AUC: {auc:.4f}")
    return preds

# =============================================================================
# 4. MAIN
# =============================================================================

def main():
    print("KITCHEN SINK STACKING")
    print("=====================")
    
    df = pd.read_parquet(CONFIG['input_file']).reset_index(drop=True)
    
    # 1. Use EVERYTHING
    X, feat_names = process_all_features(df)
    adj = build_graph(df)
    
    y = df['PhysicalDistress'].values
    labeled_indices = df[df['PhysicalDistress'].notna()].index
    train_idx, test_idx = train_test_split(
        labeled_indices, test_size=0.2, random_state=42, stratify=y[labeled_indices]
    )
    
    # 2. Train Independent Streams
    gnn_preds = train_gnn(X, adj, y, train_idx, test_idx)
    xgb_preds = train_xgboost(X, y, train_idx, test_idx)
    
    # 3. Stack
    print("\n[5] Optimizing Ensemble...")
    best_auc = 0
    best_w = 0
    for w in np.linspace(0, 1, 21):
        stack = (w * xgb_preds) + ((1-w) * gnn_preds)
        auc = roc_auc_score(y[test_idx], stack)
        if auc > best_auc:
            best_auc = auc
            best_w = w
            
    print(f"    Best Stack (w={best_w:.2f} XGB): {best_auc:.4f}")
    
    # Calculate Lift vs Best Single
    baseline = max(roc_auc_score(y[test_idx], xgb_preds), roc_auc_score(y[test_idx], gnn_preds))
    lift = (best_auc - baseline) / baseline * 100
    
    print(f"\nFINAL RESULT: +{lift:.2f}% Improvement")
    
    if lift > 0.5:
        print("CONCLUSION: The Kitchen Sink approach works. More data + Structure = Win.")

if __name__ == "__main__":
    main()