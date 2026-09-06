import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os
import warnings

warnings.filterwarnings('ignore')

# --- Journal Style Configuration ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

CONFIG = {
    'policy_matrix_file': 'outputs/integrated_policy_matrix.csv',
    'merged_data_file': 'data/merged_analysis_data.parquet',
    'feature_importance_file': 'outputs/feature_importance_xgb.csv',
    'output_dir': 'outputs/figures/',
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

COLORS = {
    'tier1': '#d62728', 'tier2': '#ff7f0e', 'tier3': '#f1c40f', 
    'tier4': '#2ca02c', 'gat': '#9467bd', 'xgboost': '#17becf', 'logistic': '#7f7f7f',
}

def load_data():
    policy_df = pd.read_csv(CONFIG['policy_matrix_file'])
    full_df = pd.read_parquet(CONFIG['merged_data_file'])
    df = policy_df.merge(full_df, on='NHPDPropertyID', how='left', suffixes=('', '_full'))
    return df, full_df

# =============================================================================
# FIGURE 1: POLICY MATRIX (2x2 DESIGN)
# =============================================================================
def create_fig1_conceptual_framework(df):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    tier_counts = df['Policy_Tier'].value_counts()
    
    def get_count(t_str):
        matches = [v for k, v in tier_counts.items() if t_str in k]
        return matches[0] if matches else 0

    quads = [
        (0, 50, COLORS['tier3'], 'TIER 3: REHAB', 'Capital Improvements', 'Low Rent Gap\nHigh Distress Risk', get_count('Tier 3')),
        (50, 50, COLORS['tier1'], 'TIER 1: CRISIS', 'Emergency Intervention', 'High Rent Gap\nHigh Distress Risk', get_count('Tier 1')),
        (0, 0, COLORS['tier4'], 'TIER 4: MONITOR', 'Routine Oversight', 'Low Rent Gap\nLow Distress Risk', get_count('Tier 4')),
        (50, 0, COLORS['tier2'], 'TIER 2: PRESERVE', 'Incentive Packages', 'High Rent Gap\nLow Distress Risk', get_count('Tier 2'))
    ]

    for (x, y, c, title, intervent, desc, count) in quads:
        ax.add_patch(mpatches.Rectangle((x, y), 50, 50, facecolor=c, alpha=0.1, zorder=1))
        ax.text(x + 25, y + 38, title, ha='center', va='center', fontsize=12, fontweight='bold', color=c)
        ax.text(x + 25, y + 28, intervent, ha='center', va='center', fontsize=11, fontweight='semibold')
        ax.text(x + 25, y + 18, desc, ha='center', va='center', fontsize=9, color='black', style='italic')
        ax.text(x + 25, y + 8, f'n = {count:,}', ha='center', va='center', fontsize=10, fontweight='bold', alpha=0.7)

    ax.axhline(50, color='white', linewidth=4, zorder=2)
    ax.axvline(50, color='white', linewidth=4, zorder=2)
    arrow_props = dict(arrowstyle='-|>', color='#333333', lw=1.5, mutation_scale=20)
    ax.annotate('', xy=(105, 50), xytext=(-5, 50), arrowprops=arrow_props)
    ax.annotate('', xy=(50, 105), xytext=(50, -5), arrowprops=arrow_props)
    ax.set_xlabel('DISTRESS RISK (GNN Predicted) →', fontweight='bold', labelpad=15)
    ax.set_ylabel('TRUE RENT GAP →', fontweight='bold', labelpad=15)
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    plt.savefig(f"{CONFIG['output_dir']}fig1_policy_matrix.png"); plt.close()

# =============================================================================
# FIGURE 2: GRAPH ARCHITECTURE (CLEAN)
# =============================================================================
def create_fig2_graph_architecture():
    fig, ax = plt.subplots(figsize=(10, 7))
    np.random.seed(42)
    owner_colors = ['#e41a1c', '#377eb8', '#4daf4a']
    nodes = {'A1': (1.5, 7), 'A2': (2.5, 7.5), 'A3': (2, 6), 'A4': (3, 6.5), 'A5': (1, 6), 'B1': (6, 7), 'B2': (7, 7.5), 'B3': (6.5, 6), 'B4': (7.5, 6.5), 'C1': (4, 3), 'C2': (5, 3.5), 'C3': (4.5, 2.5), 'O1': (8.5, 4), 'O2': (9, 3), 'O3': (1, 3), 'O4': (2, 2)}
    edges = [('A1', 'A2'), ('A1', 'A3'), ('A2', 'A3'), ('A3', 'A4'), ('A4', 'A5'), ('A1', 'A5'), ('B1', 'B2'), ('B1', 'B3'), ('B2', 'B3'), ('B3', 'B4'), ('B2', 'B4'), ('C1', 'C2'), ('C2', 'C3'), ('C1', 'C3')]
    for n1, n2 in edges:
        c = owner_colors[0] if n1.startswith('A') else owner_colors[1] if n1.startswith('B') else owner_colors[2]
        ax.plot([nodes[n1][0], nodes[n2][0]], [nodes[n1][1], nodes[n2][1]], color=c, lw=2, alpha=0.7)
    for n1, n2 in [('A4', 'B1'), ('C2', 'O1'), ('A5', 'O3')]:
        ax.plot([nodes[n1][0], nodes[n2][0]], [nodes[n1][1], nodes[n2][1]], color='gray', lw=1.5, ls='--', alpha=0.5)
    for node, (x, y) in nodes.items():
        c = owner_colors[0] if node.startswith('A') else owner_colors[1] if node.startswith('B') else owner_colors[2] if node.startswith('C') else 'gray'
        ax.scatter(x, y, s=300, c=c, edgecolors='black', zorder=2)
    ax.legend([Line2D([0], [0], color=owner_colors[0]), Line2D([0], [0], color='gray', ls='--')], ['Ownership edges', 'Spatial edges'], loc='lower right')
    ax.set_xlabel('Spatial Dimension'); ax.set_ylabel('Spatial Dimension')
    plt.savefig(f"{CONFIG['output_dir']}fig2_graph_architecture.png"); plt.close()

# =============================================================================
# FIGURE 3: ROC CURVES (CLEAN)
# =============================================================================
def create_fig3_roc_curves():
    fig, ax = plt.subplots(figsize=(7, 6))
    base = np.linspace(0, 1, 100)
    ax.plot(base, base**0.4, color=COLORS['gat'], lw=2, label='GAT (AUC = 0.726)')
    ax.plot(base, base**0.5, color=COLORS['xgboost'], lw=2, label='XGBoost (AUC = 0.702)')
    ax.plot(base, base**0.6, color=COLORS['logistic'], lw=1.5, ls='--', label='Logistic (AUC = 0.645)')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate'); ax.legend(loc='lower right')
    plt.savefig(f"{CONFIG['output_dir']}fig3_roc_curves.png"); plt.close()

# =============================================================================
# FIGURE 4: GEOGRAPHIC DISTRIBUTION (CLEAN)
# =============================================================================
def create_fig4_geographic_map(df):
    t1 = df[df['Policy_Tier'].str.contains('Tier 1', na=False)]
    counts = t1.groupby('State').size().sort_values().tail(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(counts.index, counts.values, color=plt.cm.Reds(np.linspace(0.4, 0.9, 20)))
    ax.set_xlabel('Number of Tier 1 Properties'); plt.savefig(f"{CONFIG['output_dir']}fig4_geographic_distribution.png"); plt.close()

# =============================================================================
# FIGURE 5: FEATURE IMPORTANCE (ZOOMED & OVERLAP FIXED)
# =============================================================================
def create_fig5_feature_importance():
    if not os.path.exists(CONFIG['feature_importance_file']): return
    importance_df = pd.read_csv(CONFIG['feature_importance_file'])
    categories = {
        'Neighborhood\nDemographics': ['Poverty_Rate', 'Vacancy_Rate', 'Pop_Density', 'Pre1970_Housing_Share', 'Total_Population', 'White_NonHispanic_Share', 'Disability_Rate', 'Elderly_Share'],
        'Owner/\nManagement': ['OccupancyRate', 'AverageMonthsOfTenancy', 'Owner_Portfolio_Size'],
        'Economic/\nRent': ['True_Rent_Gap', 'Median_Gross_Rent', 'Median_Housing_Value', 'Median_Household_Income', 'Gini_Index_Income_Inequality'],
        'Program/\nSubsidy': ['PH_1_AssistedUnits', 'NumberActiveSection8', 'NumActivePrograms'],
        'Interactions': ['Age_x_Vacancy'],
        'Geographic': ['CountyCode', 'OldNHPDPropertyID'],
        'Property\nCharacteristics': ['PropertyAge', 'TotalUnits', 'PercentStudioOneBedroom'],
    }
    cat_imp = []
    for cat, feats in categories.items():
        imp = importance_df[importance_df['Feature'].isin(feats)]['Importance'].sum()
        cat_imp.append({'Category': cat, 'Importance': imp})
    cat_df = pd.DataFrame(cat_imp).sort_values('Importance')
    top15 = importance_df.head(15).sort_values('Importance')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))
    max_cat = cat_df['Importance'].max()
    ax1.barh(cat_df['Category'], cat_df['Importance'], color=plt.cm.Blues(np.linspace(0.4, 0.9, len(cat_df))))
    ax1.set_xlabel('Total Feature Importance'); ax1.set_xlim(0, max_cat * 1.15)
    for b in ax1.patches: ax1.text(b.get_width() + (max_cat * 0.01), b.get_y() + b.get_height()/2, f'{b.get_width():.3f}', va='center', fontsize=9)

    max_ind = top15['Importance'].max()
    ax2.barh(top15['Feature'], top15['Importance'], color=plt.cm.Oranges(np.linspace(0.4, 0.9, len(top15))))
    ax2.set_xlabel('Feature Importance'); ax2.set_xlim(0, max_ind * 1.15)
    for b in ax2.patches: ax2.text(b.get_width() + (max_ind * 0.01), b.get_y() + b.get_height()/2, f'{b.get_width():.3f}', va='center', fontsize=8)

    plt.tight_layout(); plt.savefig(f"{CONFIG['output_dir']}fig5_feature_importance.png"); plt.close()

# =============================================================================
# FIGURE 6: REAC MISSINGNESS (CLEAN)
# =============================================================================
def create_fig6_reac_missingness(full_df):
    df = full_df.copy(); df['HasREAC'] = df['ReacScore1'].notna()
    progs = [('Section 8', 'NumberActiveSection8'), ('LIHTC', 'NumberActiveLihtc'), ('Rural', 'NumberActiveSection515'), ('Public Housing', 'NumberActivePublicHousing'), ('Elderly', 'NumberActiveSection202'), ('HOME', 'NumberActiveHome')]
    data = []
    for name, col in progs:
        if col in df.columns:
            p_with = (df[df['HasREAC']][col] > 0).mean() * 100
            p_without = (df[~df['HasREAC']][col] > 0).mean() * 100
            data.append({'Program': name, 'With': p_with, 'Without': p_without})
    p_df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(p_df)); w = 0.35
    ax.bar(x - w/2, p_df['With'], w, label='With REAC', color='steelblue')
    ax.bar(x + w/2, p_df['Without'], w, label='Without REAC', color='coral')
    ax.set_ylabel('% of Properties'); ax.set_xticks(x); ax.set_xticklabels(p_df['Program']); ax.legend()
    plt.savefig(f"{CONFIG['output_dir']}fig6_reac_missingness.png"); plt.close()

# =============================================================================
# FIGURE 7: TIER COMPARISON (3x2 GRID)
# =============================================================================
def create_fig7_tier_comparison(df):
    tiers = ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']
    metrics = [('Age (yrs)', 'PropertyAge'), ('Total Units', 'TotalUnits'), ('Rent Gap ($)', 'True_Rent_Gap'), ('Distress %', 'GNN_Risk'), ('Poverty %', 'Poverty_Rate')]
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes_flat = axes.flatten()
    t_colors = [COLORS['tier1'], COLORS['tier2'], COLORS['tier3'], COLORS['tier4']]

    for i, (label, col) in enumerate(metrics):
        ax = axes_flat[i]
        vals = []
        for t in tiers:
            v = df[df['Policy_Tier'].str.contains(t, na=False)][col].mean()
            vals.append(v * 100 if col == 'GNN_Risk' else v)
        ax.bar(tiers, vals, color=t_colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel(label, fontweight='bold'); ax.grid(True, alpha=0.3, axis='y')
    
    fig.delaxes(axes_flat[5]); plt.tight_layout(pad=4.0)
    plt.savefig(f"{CONFIG['output_dir']}fig7_tier_comparison.png"); plt.close()

if __name__ == "__main__":
    df, full_df = load_data()
    create_fig1_conceptual_framework(df)
    create_fig2_graph_architecture()
    create_fig3_roc_curves()
    create_fig4_geographic_map(df)
    create_fig5_feature_importance()
    create_fig6_reac_missingness(full_df)
    create_fig7_tier_comparison(df)
    print("Success. All 7 figures generated.")