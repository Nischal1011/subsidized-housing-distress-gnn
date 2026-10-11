"""
table8_attention_weights.py
============================
Computes "Table 8: Average Attention Weights by Edge Type."

Uses exactly the same features, train/val/test split, graph, and model
configuration as src/pipeline/05_gnn_final_model.py (imported from it). The
only addition is that each undirected edge is tagged with the relationship(s)
that created it (ownership, census tract, spatial k-NN). A pair connected by
more than one relationship (e.g. same owner AND spatial neighbors) is counted
under each of its types.

Attention is read from both GAT layers of the trained model (head-averaged)
and averaged over the edges of each type. GATConv is called with
add_self_loops=False because the graph already contains explicit self-loops,
so the coefficients read out are exactly the ones used in the forward pass.

Note: GAT attention is softmax-normalized over each node's neighborhood, so
per-edge weights are smaller for high-degree nodes. Ownership cliques create
the highest-degree nodes in the graph, so per-edge averages should be read
as descriptive, not as a direct measure of an edge type's importance
(see table4_edge_ablation.py for the edge-removal comparison).
"""

import importlib.util
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')

_spec = importlib.util.spec_from_file_location(
    'final_model', os.path.join(os.path.dirname(__file__), '..', 'pipeline', '05_gnn_final_model.py'))
final_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(final_model)

CONFIG = dict(final_model.CONFIG)
CONFIG['seeds'] = [42]
CONFIG['gat_layers_to_check'] = [0, 1]
device = final_model.device

EDGE_TYPES = ['Ownership', 'Tract', 'Spatial', 'Self-loop']


# =============================================================================
# GRAPH CONSTRUCTION (same edges as 05_gnn_final_model.build_graph, plus types)
# =============================================================================

def build_graph_with_types(df, config):
    """Return (edge_index, type_masks). edge_index is identical to
    final_model.build_graph(); type_masks[name] is a boolean mask over its
    columns marking edges created by that relationship."""
    n = len(df)
    pair_types = {}  # (min, max) -> set of type names

    def add(i, j, t):
        a, b = (int(i), int(j)) if i < j else (int(j), int(i))
        pair_types.setdefault((a, b), set()).add(t)

    # Same RNG call order as final_model.build_graph
    if 'Owner_Clean' in df.columns:
        for owner, indices in df.groupby('Owner_Clean').indices.items():
            if pd.isna(owner) or owner == 'nan' or len(indices) < 2:
                continue
            idx_list = indices if len(indices) <= 50 else np.random.choice(indices, 50, replace=False)
            for i in range(len(idx_list)):
                for j in range(i + 1, len(idx_list)):
                    add(idx_list[i], idx_list[j], 'Ownership')

    if 'CensusTract' in df.columns:
        for tract, indices in df.groupby('CensusTract').indices.items():
            if pd.isna(tract) or len(indices) < 2 or len(indices) > 100:
                continue
            idx_list = indices if len(indices) <= 30 else np.random.choice(indices, 30, replace=False)
            for i in range(len(idx_list)):
                for j in range(i + 1, len(idx_list)):
                    add(idx_list[i], idx_list[j], 'Tract')

    coords = df[['Latitude', 'Longitude']].values
    valid_idx = np.where(~np.isnan(coords).any(axis=1))[0]
    if len(valid_idx) > 0:
        k = min(config['k_spatial_neighbors'], len(valid_idx) - 1)
        nn_model = NearestNeighbors(n_neighbors=k + 1, metric='haversine', algorithm='ball_tree')
        nn_model.fit(np.radians(coords[valid_idx]))
        distances, neighbors = nn_model.kneighbors(np.radians(coords[valid_idx]))
        distances_km = distances * 6371
        for i, (dist_row, neigh_row) in enumerate(zip(distances_km, neighbors)):
            for j, neigh_local_idx in enumerate(neigh_row[1:]):
                if dist_row[j + 1] <= config['spatial_radius_km']:
                    if valid_idx[i] != valid_idx[neigh_local_idx]:
                        add(valid_idx[i], valid_idx[neigh_local_idx], 'Spatial')

    pairs = np.array(list(pair_types.keys()), dtype=np.int64)
    types = list(pair_types.values())
    src = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n)])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n)])
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    m = len(pairs)
    type_masks = {}
    for t in ['Ownership', 'Tract', 'Spatial']:
        half = np.array([t in s for s in types])
        type_masks[t] = np.concatenate([half, half, np.zeros(n, dtype=bool)])
    type_masks['Self-loop'] = np.concatenate([np.zeros(2 * m, dtype=bool), np.ones(n, dtype=bool)])
    return edge_index, type_masks


# =============================================================================
# MODEL (final_model.ImprovedGNN, with attention read-out)
# =============================================================================

class ImprovedGNNProbeable(nn.Module):
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
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads,
                                          dropout=dropout, concat=True, add_self_loops=False))
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
        """Head-averaged attention of GAT layer `layer_idx` (eval mode)."""
        h = self.input_proj(x)
        for i in range(layer_idx):
            identity = self.skip_projs[i](h)
            h = F.relu(self.norms[i](self.convs[i](h, edge_index))) + identity
        _, (ei_out, alpha) = self.convs[layer_idx](h, edge_index, return_attention_weights=True)
        assert torch.equal(ei_out.cpu(), edge_index.cpu())
        return alpha.mean(dim=1)


# =============================================================================
# TRAINING (same loop as final_model.train_gnn)
# =============================================================================

def train_one_run(data, config, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_pos = (data.y[data.train_mask] == 1).sum().item()
    n_neg = (data.y[data.train_mask] == 0).sum().item()
    class_weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32).to(device)

    model = ImprovedGNNProbeable(
        in_dim=data.num_features, hidden_dim=config['gat_hidden_dim'], num_layers=config['gat_num_layers'],
        dropout=config['gat_dropout'], heads=config['gat_heads']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['gat_lr'], weight_decay=config['gat_weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    criterion = final_model.FocalLoss(class_weights=class_weights)

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
            probs = F.softmax(model(data.x, data.edge_index), dim=1)[:, 1]
            val_auc = roc_auc_score(data.y[data.val_mask].cpu().numpy(), probs[data.val_mask].cpu().numpy())

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
        probs = F.softmax(model(data.x, data.edge_index), dim=1)[:, 1].cpu().numpy()
    return model, probs, best_val_auc


def main():
    print("=" * 70)
    print("TABLE 8: ATTENTION WEIGHTS BY EDGE TYPE")
    print("=" * 70)

    df = pd.read_parquet(CONFIG['input_file']).reset_index(drop=True)
    print(f"Loaded {len(df):,} properties.")
    df = final_model.create_features(df)

    feature_cols = [c for c in final_model.DOMAIN_FEATURES if c in df.columns]
    print(f"Using {len(feature_cols)} domain features (same list as 05_gnn_final_model.py).")

    labels = df['PhysicalDistress'].fillna(-1).values.astype(int)
    labeled_indices = np.where(labels >= 0)[0]

    # Same split as 05_gnn_final_model.py: test first, then val.
    train_val_idx, test_idx = train_test_split(
        labeled_indices, test_size=CONFIG['test_ratio'],
        random_state=CONFIG['random_state'], stratify=labels[labeled_indices]
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=CONFIG['val_ratio'] / (1 - CONFIG['test_ratio']),
        random_state=CONFIG['random_state'], stratify=labels[train_val_idx]
    )

    X_raw = df[feature_cols].fillna(0).values
    qt = QuantileTransformer(output_distribution='normal', random_state=CONFIG['random_state'])
    x_tensor = torch.tensor(qt.fit_transform(X_raw), dtype=torch.float32)

    print("Building graph...")
    np.random.seed(CONFIG['random_state'])
    edge_index, type_masks = build_graph_with_types(df, CONFIG)
    print(f"  {edge_index.shape[1]:,} directed edge entries (incl. self-loops)")
    for t in EDGE_TYPES:
        print(f"    {t}: {int(type_masks[t].sum()):,}")

    n = len(df)
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[train_idx] = True
    val_mask = torch.zeros(n, dtype=torch.bool); val_mask[val_idx] = True
    y_tensor = torch.tensor(np.maximum(labels, 0), dtype=torch.long)
    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor,
                train_mask=train_mask, val_mask=val_mask).to(device)

    rows = []
    for seed in CONFIG['seeds']:
        print(f"\n--- Seed {seed} ---")
        model, probs, best_val_auc = train_one_run(data, CONFIG, seed)
        test_auc = roc_auc_score(labels[test_idx], probs[test_idx])
        print(f"  Test AUC: {test_auc:.4f}")
        for layer_idx in CONFIG['gat_layers_to_check']:
            with torch.no_grad():
                alpha = model.layer_attention(data.x, data.edge_index, layer_idx).cpu().numpy()
            for t in EDGE_TYPES:
                mask = type_masks[t]
                rows.append({
                    'Seed': seed, 'Layer': layer_idx + 1, 'Edge Type': t,
                    'Mean Attention': alpha[mask].mean(), 'Std. Dev.': alpha[mask].std(),
                    'N Edges': int(mask.sum()), 'Val_AUC': round(best_val_auc, 4),
                    'Test_AUC': round(test_auc, 4),
                })

    result_df = pd.DataFrame(rows)
    out_path = f"{CONFIG['output_dir']}/attention_weights_by_edge_type.csv"
    result_df.to_csv(out_path, index=False)
    print("\n" + result_df.to_string(index=False))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
