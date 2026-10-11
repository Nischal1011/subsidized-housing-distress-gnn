"""
table4_edge_ablation.py
========================
Computes "Table 4: Ablation Study: Contribution of Edge Types." This trains
the same ImprovedGNN architecture and hyperparameters as
src/pipeline/05_gnn_final_model.py, varying only which edge types are
included in the graph, and reports test-set AUC/Recall/F1 for each
configuration. Every configuration shares one stratified 70/15/15 split and
the decision threshold is chosen on the validation set. This script uses all
numeric non-identifier columns as features and draws its own split, so its
full-graph AUC is not identical to the Table 2 value from
baseline_comparison.py; the comparison of interest is between rows.

The six rows match the paper's Table 4:
  1. Full graph (all edges)
  2. Ownership + Spatial only
  3. Spatial + Tract only
  4. Ownership only
  5. Spatial only
  6. No edges (MLP baseline -- self-loops only, so no real message passing)

Run from the repository root:
    python src/analysis/table4_edge_ablation.py

This trains 6 full models (same size as the paper's final model) and will
take a while on CPU -- set CONFIG['quick_mode'] = True to cut epochs/patience
for a fast sanity-check pass before running the full version.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.utils import to_undirected, add_self_loops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, precision_recall_curve
from sklearn.neighbors import NearestNeighbors
import warnings
import os
import time

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

CONFIG = {
    'input_file': 'data/merged_analysis_data.parquet',
    'output_dir': 'outputs/',
    'random_state': 42,

    'k_spatial_neighbors': 15,
    'spatial_radius_km': 8.0,

    'hidden_dim': 256,
    'num_layers': 4,
    'dropout': 0.4,
    'heads': 4,
    'use_attention': True,

    'epochs': 500,
    'lr': 0.001,
    'weight_decay': 5e-4,
    'patience': 60,
    'lr_scheduler': True,

    'val_ratio': 0.15,
    'test_ratio': 0.15,

    'quick_mode': False,  # set True for a fast sanity-check pass
}

if CONFIG['quick_mode']:
    CONFIG['epochs'] = 60
    CONFIG['patience'] = 15

os.makedirs(CONFIG['output_dir'], exist_ok=True)

EDGE_CONFIGS = [
    ('Full graph (all edges)',       dict(use_owner_edges=True,  use_tract_edges=True,  use_knn_edges=True)),
    ('Ownership + Spatial only',     dict(use_owner_edges=True,  use_tract_edges=False, use_knn_edges=True)),
    ('Spatial + Tract only',         dict(use_owner_edges=False, use_tract_edges=True,  use_knn_edges=True)),
    ('Ownership only',               dict(use_owner_edges=True,  use_tract_edges=False, use_knn_edges=False)),
    ('Spatial only',                 dict(use_owner_edges=False, use_tract_edges=False, use_knn_edges=True)),
    ('No edges (MLP baseline)',      dict(use_owner_edges=False, use_tract_edges=False, use_knn_edges=False)),
]


# =============================================================================
# FEATURE ENGINEERING (identical to 05_gnn_final_model.py)
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
        df['True_Rent_Gap'] = 0
        df['Log_Rent_Gap'] = 0
        df['High_Exit_Incentive'] = 0
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
# GRAPH CONSTRUCTION (identical to 05_gnn_final_model.py's build_graph,
# parameterized by which edge types to include)
# =============================================================================

def build_graph(df, use_owner_edges, use_tract_edges, use_knn_edges, config):
    n = len(df)
    edge_list = []
    edge_weights = []

    if use_owner_edges and 'Owner_Clean' in df.columns:
        owner_groups = df.groupby('Owner_Clean').indices
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
                    edge_weights.append(2.0)

    if use_tract_edges and 'CensusTract' in df.columns:
        tract_groups = df.groupby('CensusTract').indices
        for tract, indices in tract_groups.items():
            if pd.isna(tract):
                continue
            if len(indices) < 2 or len(indices) > 100:
                continue
            idx_list = indices if len(indices) <= 30 else np.random.choice(indices, 30, replace=False)
            for i in range(len(idx_list)):
                for j in range(i + 1, len(idx_list)):
                    edge_list.append((idx_list[i], idx_list[j]))
                    edge_weights.append(1.5)

    if use_knn_edges:
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
                        edge_list.append((src_node, tgt_node))
                        edge_weights.append(1.0 / (1 + dist_val))

    if not edge_list:
        edge_list = [(i, i) for i in range(n)]
        edge_weights = [1.0] * n

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)

    edge_index = to_undirected(edge_index)
    edge_weight = torch.cat([edge_weight, edge_weight])[:edge_index.shape[1]]
    if edge_weight.shape[0] < edge_index.shape[1]:
        edge_weight = F.pad(edge_weight, (0, edge_index.shape[1] - edge_weight.shape[0]), value=1.0)

    edge_index, _ = add_self_loops(edge_index, num_nodes=n)
    edge_weight = torch.cat([edge_weight, torch.ones(n)])

    return edge_index, edge_weight


# =============================================================================
# MODEL (identical to 05_gnn_final_model.py's ImprovedGNN)
# =============================================================================

class ImprovedGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, heads=4, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_projs = nn.ModuleList()

        for i in range(num_layers):
            if use_attention and i < 2:
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True))
            else:
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr='mean'))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.skip_projs.append(nn.Linear(hidden_dim, hidden_dim))

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
        x = self.input_proj(x)
        for conv, norm, skip in zip(self.convs, self.norms, self.skip_projs):
            identity = skip(x)
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + identity
        return self.output(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


def train_and_evaluate(data, config, class_weights, train_idx, val_idx, test_idx, labels):
    model = ImprovedGNN(
        in_dim=data.num_features,
        hidden_dim=config['hidden_dim'],
        out_dim=2,
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        heads=config['heads'],
        use_attention=config['use_attention'],
    ).to(device)
    data = data.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2) \
        if config['lr_scheduler'] else None
    criterion = FocalLoss(alpha=0.35, gamma=2.0, class_weights=class_weights.to(device))

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
            val_probs = probs[data.val_mask].cpu().numpy()
            val_labels = data.y[data.val_mask].cpu().numpy()
            try:
                val_auc = roc_auc_score(val_labels, val_probs)
            except ValueError:
                val_auc = 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0:
            print(f"      epoch {epoch+1}: loss={loss.item():.4f} val_auc={val_auc:.4f}")
        if patience_counter >= config['patience']:
            print(f"      early stop at epoch {epoch+1} (best val AUC {best_val_auc:.4f})")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()

    val_probs, val_labels = probs[val_idx], labels[val_idx]
    precision, recall, thresholds = precision_recall_curve(val_labels, val_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    opt_thresh = thresholds[np.argmax(f1_scores)] if np.argmax(f1_scores) < len(thresholds) else 0.5

    test_probs, test_labels = probs[test_idx], labels[test_idx]
    test_preds = (test_probs >= opt_thresh).astype(int)

    return {
        'AUC': roc_auc_score(test_labels, test_probs),
        'Recall': recall_score(test_labels, test_preds),
        'F1': f1_score(test_labels, test_preds),
        'Precision': precision_score(test_labels, test_preds, zero_division=0),
    }


def main():
    print("=" * 70)
    print("TABLE 4: EDGE-TYPE ABLATION STUDY")
    print("=" * 70)

    df = pd.read_parquet(CONFIG['input_file']).reset_index(drop=True)
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
    X_scaled = qt.fit_transform(X_raw)
    x_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    labels = df['PhysicalDistress'].fillna(-1).values.astype(int)
    labeled_indices = np.where(labels >= 0)[0]

    train_idx, temp_idx = train_test_split(
        labeled_indices, test_size=(CONFIG['val_ratio'] + CONFIG['test_ratio']),
        random_state=CONFIG['random_state'], stratify=labels[labeled_indices]
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5,
        random_state=CONFIG['random_state'], stratify=labels[temp_idx]
    )

    n = len(df)
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[train_idx] = True
    val_mask = torch.zeros(n, dtype=torch.bool); val_mask[val_idx] = True
    test_mask = torch.zeros(n, dtype=torch.bool); test_mask[test_idx] = True
    y_tensor = torch.tensor(np.maximum(labels, 0), dtype=torch.long)

    n_pos = (labels[train_idx] == 1).sum()
    n_neg = (labels[train_idx] == 0).sum()
    class_weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32)

    results = []
    for name, edge_flags in EDGE_CONFIGS:
        print(f"\n--- {name} ---")
        t0 = time.time()
        np.random.seed(CONFIG['random_state'])
        torch.manual_seed(CONFIG['random_state'])

        edge_index, edge_weight = build_graph(
            df, edge_flags['use_owner_edges'], edge_flags['use_tract_edges'], edge_flags['use_knn_edges'], CONFIG
        )
        print(f"    graph: {edge_index.shape[1]:,} directed edge entries (incl. self-loops)")

        data = Data(x=x_tensor, edge_index=edge_index, edge_weight=edge_weight,
                    y=y_tensor, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

        metrics = train_and_evaluate(data, CONFIG, class_weights, train_idx, val_idx, test_idx, labels)
        metrics['Edge Configuration'] = name
        metrics['Seconds'] = round(time.time() - t0, 1)
        results.append(metrics)
        print(f"    AUC={metrics['AUC']:.4f} Recall={metrics['Recall']:.4f} F1={metrics['F1']:.4f} "
              f"({metrics['Seconds']:.0f}s)")

    results_df = pd.DataFrame(results)[['Edge Configuration', 'AUC', 'Precision', 'Recall', 'F1', 'Seconds']]
    full_auc = results_df.loc[results_df['Edge Configuration'] == 'Full graph (all edges)', 'AUC'].iloc[0]
    results_df['Delta_AUC_vs_full'] = (results_df['AUC'] - full_auc).round(4)

    out_path = f"{CONFIG['output_dir']}/edge_ablation_study.csv"
    results_df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print("RESULTS (compare against the paper's Table 4)")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\nSaved to {out_path}")
    print("\nPaper's Table 4, for comparison:")
    print("  Full graph (all edges)     AUC 0.726  Recall 0.587  F1 0.437")
    print("  Ownership + Spatial only   AUC 0.721  Recall 0.573  F1 0.428")
    print("  Spatial + Tract only       AUC 0.708  Recall 0.542  F1 0.401")
    print("  Ownership only             AUC 0.694  Recall 0.521  F1 0.389")
    print("  Spatial only               AUC 0.702  Recall 0.534  F1 0.395")
    print("  No edges (MLP baseline)    AUC 0.685  Recall 0.498  F1 0.371")


if __name__ == "__main__":
    main()
