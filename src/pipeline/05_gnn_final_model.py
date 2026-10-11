"""
05_gnn_final_model.py
======================
The final GAT model described in the paper's Methods (Sections 3.4-3.5):
an input projection, two GAT layers (4 heads), two GraphSAGE layers with
residual connections and layer normalization, trained with class-weighted
focal loss and early stopping on validation AUC.

Scores every property, including the 68,892 without REAC labels, and writes
outputs/integrated_policy_matrix_v2.csv for the screening matrix and the
compounded-risk analysis. The classification threshold is the F1-maximizing
threshold on the validation set.

The same feature engineering, feature list, graph, model, and train/val/test
split are used by src/analysis/baseline_comparison.py (Table 2 model
comparison and Table 3 fixed-budget capture) and
src/analysis/table8_attention_weights.py (Table 8). The edge ablation
(src/analysis/table4_edge_ablation.py, Table 4) uses the same graph rules,
architecture, and hyperparameters; see its docstring for its feature set.

NOTE: several downstream scripts (paper_figures.py, paper_analysis.py,
data_summary_tables.py, geographic_rate.py, 06_validate_hidden_risk.py) read
outputs/integrated_policy_matrix.csv, the output of an earlier run used for
the paper's geographic and screening-category tables and figures. To rerun
them on this model's output, copy integrated_policy_matrix_v2.csv to that name.
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

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# CONFIGURATION (identical to src/analysis/baseline_comparison.py)
# =============================================================================

CONFIG = {
    'input_file': 'data/merged_analysis_data.parquet',
    'output_dir': 'outputs/',
    'random_state': 42,
    'val_ratio': 0.15,
    'test_ratio': 0.15,

    'k_spatial_neighbors': 15,
    'spatial_radius_km': 8.0,

    'gat_hidden_dim': 256,
    'gat_num_layers': 4,
    'gat_dropout': 0.4,
    'gat_heads': 4,
    'gat_epochs': 500,
    'gat_lr': 0.001,
    'gat_weight_decay': 5e-4,
    'gat_patience': 60,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)


# =============================================================================
# FEATURE ENGINEERING (identical to baseline_comparison.py)
# =============================================================================

def create_features(df):
    print("\n[1/4] Feature Engineering...")
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
        df['Rent_Gap_Pctl'] = df['True_Rent_Gap'].rank(pct=True)
    else:
        df['True_Rent_Gap'] = 0
        df['Log_Rent_Gap'] = 0
        df['High_Exit_Incentive'] = 0
        df['Rent_Gap_Pctl'] = 0.5

    if 'PropertyAge' in df.columns:
        age = df['PropertyAge'].fillna(df['PropertyAge'].median()).fillna(30)
        df['Log_Age'] = np.log1p(age)
        df['Age_Squared'] = age ** 2 / 1000
        df['Is_Old'] = (age > 40).astype(float)
        df['Is_Very_Old'] = (age > 50).astype(float)
        df['Age_Pctl'] = df['PropertyAge'].rank(pct=True)

    if 'TotalUnits' in df.columns:
        units = df['TotalUnits'].fillna(df['TotalUnits'].median()).fillna(50)
        df['Log_Units'] = np.log1p(units)
        df['Is_Large'] = (units > 100).astype(float)
        df['Is_Small'] = (units < 20).astype(float)

    if 'Poverty_Rate' in df.columns:
        pov = df['Poverty_Rate'].fillna(df['Poverty_Rate'].median()).fillna(15)
        df['High_Poverty'] = (pov > 25).astype(float)
        df['Very_High_Poverty'] = (pov > 35).astype(float)
        df['Poverty_Pctl'] = df['Poverty_Rate'].rank(pct=True)

    if 'Vacancy_Rate' in df.columns:
        vac = df['Vacancy_Rate'].fillna(df['Vacancy_Rate'].median()).fillna(5)
        df['High_Vacancy'] = (vac > 10).astype(float)
        df['Vacancy_Pctl'] = df['Vacancy_Rate'].rank(pct=True)

    if 'Log_Age' in df.columns:
        df['Age_x_Poverty'] = df['Log_Age'] * df.get('Poverty_Pctl', 0.5)
        df['Age_x_RentGap'] = df['Log_Age'] * df.get('Rent_Gap_Pctl', 0.5)
        df['Age_x_Vacancy'] = df['Log_Age'] * df.get('Vacancy_Pctl', 0.5)

    if 'Owner' in df.columns:
        df['Owner_Clean'] = df['Owner'].fillna('UNKNOWN').astype(str).str.upper().str.strip()
        df['Owner_Clean'] = df['Owner_Clean'].replace({'UNKNOWN': np.nan, 'NAN': np.nan, 'NONE': np.nan})
        owner_counts = df['Owner_Clean'].value_counts()
        df['Owner_Portfolio_Size'] = df['Owner_Clean'].map(owner_counts).fillna(1)
        df['Log_Portfolio'] = np.log1p(df['Owner_Portfolio_Size'])
        df['Is_Large_Owner'] = (df['Owner_Portfolio_Size'] > 10).astype(float)
        df['Is_Single_Owner'] = (df['Owner_Portfolio_Size'] == 1).astype(float)

    print(f"  - Feature engineering complete.")
    return df


# Explicit domain feature list (identical to baseline_comparison.py)
DOMAIN_FEATURES = [
    'Median_Household_Income', 'Median_Gross_Rent', 'True_Rent_Gap', 'Log_Rent_Gap',
    'High_Exit_Incentive', 'Rent_Gap_Pctl', 'PropertyAge', 'Log_Age', 'Age_Squared',
    'Is_Old', 'Is_Very_Old', 'Age_Pctl', 'TotalUnits', 'Log_Units', 'Is_Large',
    'Is_Small', 'Poverty_Rate', 'High_Poverty', 'Very_High_Poverty', 'Poverty_Pctl',
    'Vacancy_Rate', 'High_Vacancy', 'Vacancy_Pctl', 'Age_x_Poverty', 'Age_x_RentGap',
    'Age_x_Vacancy', 'Owner_Portfolio_Size', 'Log_Portfolio', 'Is_Large_Owner',
    'Is_Single_Owner', 'PH_1_AssistedUnits', 'OccupancyRate', 'White_NonHispanic_Share',
    'CountyCode', 'AverageMonthsOfTenancy', 'Gini_Index_Income_Inequality',
    'Disability_Rate', 'OldNHPDPropertyID', 'PercentStudioOneBedroom', 'Total_Population',
    'Pre1970_Housing_Share', 'Elderly_Share', 'Pop_Density', 'Median_Housing_Value',
    'Has_Section8', 'Has_LIHTC', 'Has_PublicHousing', 'Has_HOME'
]


# =============================================================================
# GRAPH CONSTRUCTION (identical to baseline_comparison.py's build_full_graph)
# =============================================================================

def build_graph(df, config):
    n = len(df)
    edge_list = []

    if 'Owner_Clean' in df.columns:
        for owner, indices in df.groupby('Owner_Clean').indices.items():
            if pd.isna(owner) or owner == 'nan' or len(indices) < 2:
                continue
            idx_list = indices if len(indices) <= 50 else np.random.choice(indices, 50, replace=False)
            for i in range(len(idx_list)):
                for j in range(i + 1, len(idx_list)):
                    edge_list.append((idx_list[i], idx_list[j]))

    if 'CensusTract' in df.columns:
        for tract, indices in df.groupby('CensusTract').indices.items():
            if pd.isna(tract) or len(indices) < 2 or len(indices) > 100:
                continue
            idx_list = indices if len(indices) <= 30 else np.random.choice(indices, 30, replace=False)
            for i in range(len(idx_list)):
                for j in range(i + 1, len(idx_list)):
                    edge_list.append((idx_list[i], idx_list[j]))

    coords = df[['Latitude', 'Longitude']].values
    valid_idx = np.where(~np.isnan(coords).any(axis=1))[0]
    if len(valid_idx) > 0:
        valid_coords = coords[valid_idx]
        k = min(config['k_spatial_neighbors'], len(valid_idx) - 1)
        nn_model = NearestNeighbors(n_neighbors=k + 1, metric='haversine', algorithm='ball_tree')
        nn_model.fit(np.radians(valid_coords))
        distances, neighbors = nn_model.kneighbors(np.radians(valid_coords))
        distances_km = distances * 6371
        for i, (dist_row, neigh_row) in enumerate(zip(distances_km, neighbors)):
            src_node = valid_idx[i]
            for j, neigh_local_idx in enumerate(neigh_row[1:]):
                dist_val = dist_row[j + 1]
                if dist_val <= config['spatial_radius_km']:
                    edge_list.append((src_node, valid_idx[neigh_local_idx]))

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=n)
    return edge_index


# =============================================================================
# MODEL (identical to baseline_comparison.py's ImprovedGNN)
# =============================================================================

class ImprovedGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout, heads=4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        for i in range(num_layers):
            if i < 2:
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True))
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.35, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha, self.gamma, self.class_weights = alpha, gamma, class_weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


def find_optimal_threshold(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5


# =============================================================================
# TRAINING (identical to baseline_comparison.py's run_gat_full_graph)
# =============================================================================

def train_gnn(df, X_scaled, labels, train_idx, val_idx, config):
    print("\n[2/4] Training GAT (full graph)...")
    np.random.seed(config['random_state'])
    torch.manual_seed(config['random_state'])

    edge_index = build_graph(df, config)
    print(f"  Full graph: {edge_index.shape[1]:,} directed edge entries (incl. self-loops)")

    n = len(df)
    x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(np.maximum(labels, 0), dtype=torch.long)

    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[train_idx] = True
    val_mask = torch.zeros(n, dtype=torch.bool); val_mask[val_idx] = True

    n_pos = (labels[train_idx] == 1).sum()
    n_neg = (labels[train_idx] == 0).sum()
    class_weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32).to(device)

    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor,
                train_mask=train_mask, val_mask=val_mask).to(device)

    model = ImprovedGNN(
        in_dim=data.num_features, hidden_dim=config['gat_hidden_dim'], num_layers=config['gat_num_layers'],
        dropout=config['gat_dropout'], heads=config['gat_heads']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['gat_lr'], weight_decay=config['gat_weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    criterion = FocalLoss(class_weights=class_weights)

    best_val_auc, patience_counter, best_state = 0, 0, None
    for epoch in range(config['gat_epochs']):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
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

        if (epoch + 1) % 50 == 0:
            print(f"    epoch {epoch+1}: loss={loss.item():.4f} val_auc={val_auc:.4f}")
        if patience_counter >= config['gat_patience']:
            print(f"    early stop at epoch {epoch+1} (best val AUC {best_val_auc:.4f})")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()

    return probs


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("PRESERVATION RISK GNN PIPELINE (06)")
    print("=" * 70)

    if not os.path.exists(CONFIG['input_file']):
        print(f"Error: Input file {CONFIG['input_file']} not found.")
        return

    df = pd.read_parquet(CONFIG['input_file']).reset_index(drop=True)
    print(f"Loaded {len(df):,} properties.")

    df = create_features(df)

    feature_cols = [c for c in DOMAIN_FEATURES if c in df.columns]
    print(f"Using {len(feature_cols)} explicit domain features.")

    labels = df['PhysicalDistress'].fillna(-1).values.astype(int)
    labeled_indices = np.where(labels >= 0)[0]

    print(f"Labeled Nodes: {len(labeled_indices):,} ({(len(labeled_indices)/len(df))*100:.1f}%)")
    print(f"Distress Rate in Labeled: {labels[labeled_indices].mean()*100:.1f}%")

    # Single train/val/test split, identical order to baseline_comparison.py:
    # test carved off first, then val from the remainder.
    train_val_idx, test_idx = train_test_split(
        labeled_indices, test_size=CONFIG['test_ratio'],
        random_state=CONFIG['random_state'], stratify=labels[labeled_indices]
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=CONFIG['val_ratio'] / (1 - CONFIG['test_ratio']),
        random_state=CONFIG['random_state'], stratify=labels[train_val_idx]
    )
    print(f"Split: Train={len(train_idx):,}, Val={len(val_idx):,}, Test={len(test_idx):,}")

    X_raw = df[feature_cols].fillna(0).values
    qt = QuantileTransformer(output_distribution='normal', random_state=CONFIG['random_state'])
    X_scaled = qt.fit_transform(X_raw)

    probs = train_gnn(df, X_scaled, labels, train_idx, val_idx, CONFIG)

    # F1-maximizing threshold chosen on the validation set (same as
    # baseline_comparison.py), then applied unchanged to the test set.
    opt_thresh = find_optimal_threshold(labels[val_idx], probs[val_idx])
    test_probs, test_labels = probs[test_idx], labels[test_idx]
    test_preds = (test_probs >= opt_thresh).astype(int)

    print(f"\n  Optimal Threshold: {opt_thresh:.4f}")
    print(f"  Test Results:")
    print(f"    AUC:       {roc_auc_score(test_labels, test_probs):.4f}")
    print(f"    Precision: {precision_score(test_labels, test_preds, zero_division=0):.4f}")
    print(f"    Recall:    {recall_score(test_labels, test_preds):.4f}")
    print(f"    F1 Score:  {f1_score(test_labels, test_preds):.4f}")

    df['GNN_Risk'] = probs
    df['GNN_HighRisk'] = (probs >= opt_thresh).astype(int)

    # Inference stats
    unlabeled_mask = (labels == -1)
    n_unlabeled = unlabeled_mask.sum()
    n_flagged = df.loc[unlabeled_mask, 'GNN_HighRisk'].sum()

    print(f"\n[3/4] Inference on Unlabeled Data")
    print(f"  Total Unlabeled: {n_unlabeled:,}")
    print(f"  Flagged as Distressed: {n_flagged:,} ({(n_flagged/n_unlabeled):.1%})")

    # Policy synthesis
    print("\n[4/4] Generating Integrated Policy Matrix...")

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
