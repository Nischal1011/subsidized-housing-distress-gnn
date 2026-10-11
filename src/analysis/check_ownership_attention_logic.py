"""
check_ownership_attention_logic.py
===================================
Cheap correctness check for the ownership-edge attention logic used in
table8_attention_weights.py, run after the full-scale result showed
ownership with the lowest (not highest) mean attention. This does NOT
retrain the full model -- it checks the plumbing directly:

  Part A: a tiny hand-built synthetic graph with known structure, verifying
          GATConv's returned edge_index/attention aligns exactly with the
          input edge_index (order and content, not just length), and that
          each node's incoming attention weights sum to ~1 (the defining
          property of a softmax -- if this fails, something is genuinely
          broken; if it holds, the extraction mechanics are sound).
  Part B: the real graph-construction function, run on a real slice of
          the data, to directly measure whether ownership-heavy ("hub")
          nodes really do have much higher total degree than other nodes --
          the mechanical reason a low PER-EDGE average doesn't necessarily
          mean the model ignores ownership.
  Part C: a quick (small subset, few-epoch) real training, just to confirm
          the softmax-sums-to-1 property holds on real trained attention,
          not only on the synthetic example.

Run from the repository root:
    python src/analysis/check_ownership_attention_logic.py
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

sys.path.insert(0, 'src/analysis')
import table8_attention_weights as t8  # reuse build_graph_with_types and the probeable model
fm = t8.final_model                    # 05_gnn_final_model.py: create_features, DOMAIN_FEATURES, FocalLoss

torch.manual_seed(0)
np.random.seed(0)


def part_a_synthetic_check():
    print("=" * 70)
    print("PART A: synthetic graph, exact edge_index/alignment + softmax check")
    print("=" * 70)

    # Nodes 0,1,2: one owner (fully connected -> 3 ownership edges: 01,02,12)
    # Nodes 3,4:   one tract (1 tract edge: 34)
    # Nodes 5,6,7: a spatial chain (2 spatial edges: 56, 67)
    n = 8
    directed = [(0, 1), (0, 2), (1, 2), (3, 4), (5, 6), (6, 7)]
    types = [0, 0, 0, 1, 2, 2]  # 0=ownership, 1=tract, 2=spatial

    edge_index = torch.tensor(directed, dtype=torch.long).t()
    edge_type = torch.tensor(types, dtype=torch.long)
    # symmetrize and add self-loops, as build_graph_with_types does
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_type = torch.cat([edge_type, edge_type])
    self_loops = torch.arange(n, dtype=torch.long)
    edge_index = torch.cat([edge_index, torch.stack([self_loops, self_loops])], dim=1)
    edge_type = torch.cat([edge_type, torch.full((n,), 3, dtype=torch.long)])

    x = torch.randn(n, 6)
    conv = GATConv(6, 4, heads=2, concat=True, add_self_loops=False)
    conv.eval()
    with torch.no_grad():
        _, (ei_out, alpha) = conv(x, edge_index, return_attention_weights=True)

    print(f"  input edge_index shape:  {tuple(edge_index.shape)}")
    print(f"  output edge_index shape: {tuple(ei_out.shape)}")
    exact_match = torch.equal(ei_out, edge_index)
    print(f"  output edge_index EXACTLY equals input edge_index (order+content): {exact_match}")
    assert exact_match, "FAIL: GATConv reordered/changed edges -- edge_type alignment would be wrong!"

    alpha_mean = alpha.mean(dim=1)  # average over heads, as table8_attention_weights.py does
    print("\n  Per-edge attention (averaged over heads), grouped back by our known structure:")
    names = {0: 'ownership', 1: 'tract', 2: 'spatial', 3: 'self-loop'}
    for t, name in names.items():
        mask = edge_type == t
        vals = alpha_mean[mask]
        print(f"    {name:10s}: {vals.tolist()}")

    print("\n  Softmax sanity check -- for each node, do its INCOMING edges' attention sum to ~1?")
    target_nodes = ei_out[1]
    all_ok = True
    for node in range(n):
        node_mask = target_nodes == node
        total = alpha_mean[node_mask].sum().item()
        ok = abs(total - 1.0) < 1e-4
        all_ok &= ok
        print(f"    node {node}: sum of incoming attention = {total:.4f}  {'OK' if ok else 'FAIL'}")
    print(f"\n  ALL NODES SUM TO ~1.0: {all_ok}  (this is the defining property of softmax attention;"
          f" if False, the extraction logic is broken)")
    print()


def part_b_real_degree_structure():
    print("=" * 70)
    print("PART B: real ownership-edge structure -- do hub nodes really have")
    print("        much higher total degree than other nodes?")
    print("=" * 70)

    df = pd.read_parquet('data/merged_analysis_data.parquet').reset_index(drop=True)
    df = fm.create_features(df)

    np.random.seed(t8.CONFIG['random_state'])
    edge_index, type_masks = t8.build_graph_with_types(df, t8.CONFIG)
    n = len(df)
    print(f"  Full data: {n:,} properties, {edge_index.shape[1]:,} directed edge entries")

    # per-node degree, split by whether the edge is ownership or not
    src = edge_index[0]
    is_ownership = torch.tensor(type_masks['Ownership'])

    total_degree = torch.zeros(n, dtype=torch.long)
    ownership_degree = torch.zeros(n, dtype=torch.long)
    total_degree.scatter_add_(0, src, torch.ones_like(src))
    ownership_degree.scatter_add_(0, src[is_ownership], torch.ones_like(src[is_ownership]))

    has_ownership = ownership_degree > 0
    print(f"\n  Properties with >=1 ownership edge: {has_ownership.sum().item():,} "
          f"({has_ownership.float().mean().item()*100:.1f}% of all properties)")
    print(f"  Mean TOTAL degree, properties WITH ownership edges:    {total_degree[has_ownership].float().mean().item():.1f}")
    print(f"  Mean TOTAL degree, properties WITHOUT ownership edges: {total_degree[~has_ownership].float().mean().item():.1f}")
    print(f"  Max ownership-degree of any single property: {ownership_degree.max().item()}")

    # among properties that DO have ownership edges, how much of their total
    # degree is ownership vs other types?
    own_deg = ownership_degree[has_ownership].float()
    tot_deg = total_degree[has_ownership].float()
    print(f"\n  Among properties WITH ownership edges:")
    print(f"    mean ownership-edge share of their total degree: {(own_deg / tot_deg).mean().item()*100:.1f}%")
    print(f"    (a high total degree at these nodes means each individual edge's softmax\n"
          f"     share is small, by construction -- independent of what the model 'thinks'\n"
          f"     of ownership edges specifically)")
    print()


def part_c_quick_real_softmax_check():
    print("=" * 70)
    print("PART C: quick real (briefly trained) model -- softmax-sums-to-1 check")
    print("        on a real, not synthetic, attention distribution")
    print("=" * 70)

    df = pd.read_parquet('data/merged_analysis_data.parquet').reset_index(drop=True)
    df = df.sample(n=5000, random_state=0).reset_index(drop=True)
    df = fm.create_features(df)
    feature_cols = [c for c in fm.DOMAIN_FEATURES if c in df.columns]

    from sklearn.preprocessing import QuantileTransformer
    X_raw = df[feature_cols].fillna(0).values
    qt = QuantileTransformer(output_distribution='normal', random_state=0, n_quantiles=min(1000, len(df)))
    x_tensor = torch.tensor(qt.fit_transform(X_raw), dtype=torch.float32)

    edge_index, type_masks = t8.build_graph_with_types(df, t8.CONFIG)
    print(f"  {len(df):,}-property subset: {edge_index.shape[1]:,} directed edge entries")

    labels = df['PhysicalDistress'].fillna(-1).values.astype(int)
    labeled_indices = np.where(labels >= 0)[0]
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(labeled_indices, test_size=0.3, random_state=0,
                                           stratify=labels[labeled_indices])

    n = len(df)
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[train_idx] = True
    val_mask = torch.zeros(n, dtype=torch.bool); val_mask[val_idx] = True
    y_tensor = torch.tensor(np.maximum(labels, 0), dtype=torch.long)

    from torch_geometric.data import Data
    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor, train_mask=train_mask, val_mask=val_mask)

    model = t8.ImprovedGNNProbeable(in_dim=data.num_features, hidden_dim=64, num_layers=4, dropout=0.4, heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = fm.FocalLoss()
    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    print(f"  trained 30 epochs, final loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        alpha = model.layer_attention(data.x, data.edge_index, 0)
    alpha = alpha.numpy()
    targets = data.edge_index[1].numpy()

    sums = pd.Series(alpha).groupby(targets).sum()
    print(f"\n  Softmax check across {len(sums):,} real nodes (real trained attention):")
    print(f"    mean of per-node attention sums: {sums.mean():.6f} (should be ~1.0)")
    print(f"    min: {sums.min():.6f}   max: {sums.max():.6f}")
    print(f"    ALL within 1e-3 of 1.0: {bool((sums.sub(1.0).abs() < 1e-3).all())}")

    print("\n  By-type means on this quick/small run (not comparable in scale to the full run,")
    print("  just confirming ownership doesn't come out an obvious outlier/bug here either):")
    for name in t8.EDGE_TYPES:
        mask = type_masks[name]
        if mask.sum() == 0:
            continue
        print(f"    {name:10s}: mean={alpha[mask].mean():.4f}  n={mask.sum():,}")


if __name__ == "__main__":
    part_a_synthetic_check()
    part_b_real_degree_structure()
    part_c_quick_real_softmax_check()
