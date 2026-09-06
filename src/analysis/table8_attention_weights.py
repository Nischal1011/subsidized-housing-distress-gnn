"""
table8_attention_weights.py
============================
Computes "Table 8: Average Attention Weights by Edge Type."
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
import warnings
import os

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

CONFIG = {
    'input_file': 'data/merged_analysis_data.parquet',
    'output_dir': 'outputs/',
    'random_state': 42,

    'k_spatial_neighbors': 12,  
    'spatial_radius_km': 8.0,

    # Architecture
    'hidden_dim': 256,
    'num_layers': 4,
    'dropout': 0.6,    # Feature dropout
    'gat_dropout': 0.0,
    'heads': 4,

    # Training
    'epochs': 200,
    'lr': 0.001,
    'weight_decay': 5e-4,
    'patience': 40,
    'lr_scheduler': True,

    'val_ratio': 0.15,
    'test_ratio': 0.15,

    'seeds': [42],
    'gat_layers_to_check': [0], # Checking only Layer 0
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

EDGE_TYPE_NAMES = {0: 'Ownership', 1: 'Tract', 2: 'Spatial', 3: 'Self-loop'}

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_features(df):
    df = df.copy()

    if 'Median_Household_Income' in df.columns and 'Median_Gross_Rent' in df.columns:
        inc = df['Median_Household_Income'].fillna(df['Median_Household_Income'].median())
        mkt_rent = df['Median_Gross_Rent'].fillna(df['Median_Gross_Rent'].median())
        restricted_rent = (inc * 0.60 * 0.30) / 12
        df['True_Rent_Gap'] = mkt_rent - restricted_rent
        min_gap = df['True_Rent_Gap'].min()
        shift = abs(min_gap) + 1 if min_gap < 0 else 1
        df['Log_Rent_Gap'] = np.log1p(df['True_Rent_Gap'] + shift)
        df['High_Exit_Incentive'] = (df['True_Rent_Gap'] > 400).astype(float)
        df['Rent_Gap_Percentile'] = df['True_Rent_Gap'].rank(pct=True)
    else:
        for col in ['True_Rent_Gap', 'Log_Rent_Gap', 'High_Exit_Incentive']: df[col] = 0
        df['Rent_Gap_Percentile'] = 0.5

    if 'PropertyAge' in df.columns:
        age = df['PropertyAge'].fillna(df['PropertyAge'].median()).fillna(30)
        df['Log_Age'] = np.log1p(age)
        df['Age_Squared'] = age ** 2 / 1000
        df['Is_Old'] = (age > 40).astype(float)
        df['Is_Very_Old'] = (age > 50).astype(float)
        df['Age_Bucket'] = pd.cut(age, bins=[0, 20, 35, 50, 100], labels=[0, 1, 2, 3]).astype(float)

    if 'TotalUnits' in df.columns:
        units = df['TotalUnits'].fillna(df['TotalUnits'].median()).fillna(50)
        df['Log_Units'] = np.log1p(units)
        df['Is_Large'] = (units > 100).astype(float)
        df['Is_Small'] = (units < 20).astype(float)

    if 'Poverty_Rate' in df.columns:
        pov = df['Poverty_Rate'].fillna(df['Poverty_Rate'].median()).fillna(15)
        df['High_Poverty'] = (pov > 25).astype(float)
        df['Poverty_Percentile'] = df['Poverty_Rate'].rank(pct=True)

    if 'Vacancy_Rate' in df.columns:
        vac = df['Vacancy_Rate'].fillna(df['Vacancy_Rate'].median()).fillna(5)
        df['High_Vacancy'] = (vac > 10).astype(float)
        df['Vacancy_Percentile'] = df['Vacancy_Rate'].rank(pct=True)

    if 'PropertyAge' in df.columns and 'Poverty_Rate' in df.columns:
        df['Age_x_Poverty'] = df['Log_Age'] * df['Poverty_Rate'].fillna(0) / 100

    if 'PropertyAge' in df.columns and 'True_Rent_Gap' in df.columns:
        df['Age_x_RentGap'] = df['Log_Age'] * df['Rent_Gap_Percentile']

    if 'Owner' in df.columns:
        df['Owner_Clean'] = df['Owner'].fillna('UNKNOWN').astype(str).str.upper().str.strip()
        df['Owner_Clean'] = df['Owner_Clean'].replace({'UNKNOWN': np.nan, 'NAN': np.nan, 'NONE': np.nan})
        owner_counts = df['Owner_Clean'].value_counts()
        df['Owner_Portfolio_Size'] = df['Owner_Clean'].map(owner_counts).fillna(1)
        df['Log_Portfolio'] = np.log1p(df['Owner_Portfolio_Size'])
        df['Is_Large_Owner'] = (df['Owner_Portfolio_Size'] > 10).astype(float)

    return df


# =============================================================================
# GRAPH CONSTRUCTION (Degree-Capped)
# =============================================================================

def build_graph_with_types(df, config):
    n = len(df)
    edge_list, edge_type_list = [], []

    MAX_OWNER_DEGREE = 2  
    MAX_TRACT_DEGREE = 5

    if 'Owner_Clean' in df.columns:
        owner_groups = df.groupby('Owner_Clean').indices
        for owner, indices in owner_groups.items():
            if pd.isna(owner) or owner == 'nan' or len(indices) < 2:
                continue
            for i in indices:
                candidates = [j for j in indices if j != i]
                if len(candidates) > MAX_OWNER_DEGREE:
                    candidates = np.random.choice(candidates, MAX_OWNER_DEGREE, replace=False)
                for j in candidates:
                    if i < j:
                        edge_list.append((i, j))
                        edge_type_list.append(0)

    if 'CensusTract' in df.columns:
        tract_groups = df.groupby('CensusTract').indices
        for tract, indices in tract_groups.items():
            if pd.isna(tract) or len(indices) < 2 or len(indices) > 100:
                continue
            for i in indices:
                candidates = [j for j in indices if j != i]
                if len(candidates) > MAX_TRACT_DEGREE:
                    candidates = np.random.choice(candidates, MAX_TRACT_DEGREE, replace=False)
                for j in candidates:
                    if i < j:
                        edge_list.append((i, j))
                        edge_type_list.append(1)

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
        
        for i, (dist_row, neigh_row) in enumerate(zip(distances_km, neighbors)):
            src_node = valid_idx[i]
            for j, neigh_local_idx in enumerate(neigh_row[1:]):
                dist_val = dist_row[j + 1]
                if dist_val <= config['spatial_radius_km']:
                    tgt_node = valid_idx[neigh_local_idx]
                    if src_node < tgt_node:
                        edge_list.append((src_node, tgt_node))
                        edge_type_list.append(2)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type_list, dtype=torch.long)

    # Symmetrize
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_type = torch.cat([edge_type, edge_type])

    self_loops = torch.arange(n, dtype=torch.long)
    self_loop_index = torch.stack([self_loops, self_loops])
    edge_index = torch.cat([edge_index, self_loop_index], dim=1)
    edge_type = torch.cat([edge_type, torch.full((n,), 3, dtype=torch.long)])

    return edge_index, edge_type


# =============================================================================
# MODEL (4-Layer Hybrid)
# =============================================================================

class ImprovedGNNProbeable(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout, gat_dropout, heads=4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        for i in range(num_layers):
            if i < 2:
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads,
                                           dropout=gat_dropout, concat=True, add_self_loops=False))
            else:
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr='mean'))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.skip_projs.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 2)
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        for conv, norm, skip in zip(self.convs, self.norms, self.skip_projs):
            identity = skip(x)
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + identity
        return self.output(x)

    def layer_attention(self, x, edge_index, layer_idx):
        h = self.input_proj(x)
        for i in range(layer_idx):
            identity = self.skip_projs[i](h)
            out = self.convs[i](h, edge_index)
            out = self.norms[i](out)
            out = F.relu(out)
            h = out + identity
        _, (ei_out, alpha) = self.convs[layer_idx](h, edge_index, return_attention_weights=True)
        return ei_out, alpha.mean(dim=1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.35, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha, self.gamma, self.class_weights = alpha, gamma, class_weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


# =============================================================================
# TRAINING
# =============================================================================

def train_one_run(data, edge_type, config, seed):
    torch.manual_seed(seed)

    n_pos = (data.y[data.train_mask] == 1).sum().item()
    n_neg = (data.y[data.train_mask] == 0).sum().item()
    class_weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32).to(device)

    model = ImprovedGNNProbeable(
        in_dim=data.num_features, 
        hidden_dim=config['hidden_dim'], 
        num_layers=config['num_layers'],
        dropout=config['dropout'], 
        gat_dropout=config['gat_dropout'],
        heads=config['heads']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2) \
        if config['lr_scheduler'] else None
    
    criterion = FocalLoss(class_weights=class_weights)

    best_val_auc, patience_counter, best_state = 0, 0, None
    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            probs = F.softmax(out, dim=1)[:, 1]
            try:
                val_auc = roc_auc_score(data.y[data.val_mask].cpu().numpy(), probs[data.val_mask].cpu().numpy())
            except ValueError:
                val_auc = 0.5

        if val_auc > best_val_auc:
            best_val_auc, patience_counter = val_auc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}: loss={loss.item():.4f} val_auc={val_auc:.4f}")
        if patience_counter >= config['patience']:
            print(f"    early stop at epoch {epoch+1} (best val AUC {best_val_auc:.4f})")
            break

    model.load_state_dict(best_state)
    model.eval()

    edge_type_np = edge_type.numpy()
    layer_rows = []
    
    for layer_idx in config['gat_layers_to_check']:
        with torch.no_grad():
            ei_out, alpha = model.layer_attention(data.x, data.edge_index, layer_idx)
            
        alpha_np = alpha.cpu().numpy()
        
        for t, name in EDGE_TYPE_NAMES.items():
            mask = edge_type_np == t
            if mask.sum() == 0:
                continue
            layer_rows.append({
                'Layer': layer_idx, 
                'Edge Type': name,
                'Mean Attention': alpha_np[mask].mean(), 
                'Std. Dev.': alpha_np[mask].std(),
                'N Edges': int(mask.sum()),
            })
                
    return layer_rows, best_val_auc


def main():
    print("=" * 70)
    print("TABLE 8: ATTENTION WEIGHTS BY EDGE TYPE")
    print(f"Seeds: {CONFIG['seeds']}   GAT layers checked: {CONFIG['gat_layers_to_check']}")
    print("=" * 70)

    try:
        df = pd.read_parquet(CONFIG['input_file']).reset_index(drop=True)
    except Exception as e:
        print(f"Could not load data: {e}")
        return

    print(f"Loaded {len(df):,} properties.")
    df = create_features(df)

    exclude_cols = ['PhysicalDistress', 'BestReacScore', 'ReacScore1', 'ReacScore2', 'ReacScore3',
                     'NHPDPropertyID', 'Latitude', 'Longitude', 'CensusTract', 'PropertyName',
                     'PropertyAddress', 'Owner', 'Owner_Clean', 'City', 'State', 'Zip',
                     'Policy_Tier', 'GNN_Risk', 'GNN_HighRisk']
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols if c not in exclude_cols]
    print(f"Using {len(feature_cols)} features.")

    X_raw = df[feature_cols].fillna(0).values
    qt = QuantileTransformer(output_distribution='normal', random_state=CONFIG['random_state'])
    x_tensor = torch.tensor(qt.fit_transform(X_raw), dtype=torch.float32)

    print("Building graph...")
    np.random.seed(CONFIG['random_state'])
    edge_index, edge_type = build_graph_with_types(df, CONFIG)
    
    avg_degree = edge_index.shape[1] / len(df)
    print(f"  {edge_index.shape[1]:,} directed edge entries (Avg Degree: {avg_degree:.1f})")
    
    for t, name in EDGE_TYPE_NAMES.items():
        n_edges = (edge_type == t).sum().item()
        print(f"    {name}: {n_edges:,}")

    labels = df['PhysicalDistress'].fillna(-1).values.astype(int)
    labeled_indices = np.where(labels >= 0)[0]

    train_idx, temp_idx = train_test_split(
        labeled_indices, test_size=(CONFIG['val_ratio'] + CONFIG['test_ratio']),
        random_state=CONFIG['random_state'], stratify=labels[labeled_indices]
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=CONFIG['random_state'], stratify=labels[temp_idx]
    )

    n = len(df)
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[train_idx] = True
    val_mask = torch.zeros(n, dtype=torch.bool); val_mask[val_idx] = True
    y_tensor = torch.tensor(np.maximum(labels, 0), dtype=torch.long)

    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor,
                train_mask=train_mask, val_mask=val_mask).to(device)

    all_rows = []
    for seed in CONFIG['seeds']:
        print(f"\n--- Seed {seed} ---")
        layer_rows, best_val_auc = train_one_run(data, edge_type, CONFIG, seed)
        for row in layer_rows:
            row['Seed'] = seed
            row['Val_AUC'] = round(best_val_auc, 4)
            all_rows.append(row)

    result_df = pd.DataFrame(all_rows)[['Seed', 'Layer', 'Edge Type', 'Mean Attention', 'Std. Dev.', 'N Edges', 'Val_AUC']]
    out_path = f"{CONFIG['output_dir']}/attention_weights_by_edge_type.csv"
    result_df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY: mean +/- std per layer (only real edge types)")
    print("=" * 70)
    summary = (
        result_df[result_df['Edge Type'] != 'Self-loop']
        .groupby(['Layer', 'Edge Type'])['Mean Attention']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()