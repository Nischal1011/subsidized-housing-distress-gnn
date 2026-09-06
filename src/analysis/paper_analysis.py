"""
paper_analyses.py
=================
Generate remaining analyses for Data & Policy paper:
1. Example properties per tier (with characteristics)
2. Geographic distribution of Tier 1 crisis properties
3. Feature importance visualization
4. Tier characteristic comparison

Run AFTER the GAT model has generated outputs/integrated_policy_matrix.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

CONFIG = {
    'policy_matrix_file': 'outputs/integrated_policy_matrix.csv',
    'merged_data_file': 'data/merged_analysis_data.parquet',
    'feature_importance_file': 'outputs/feature_importance_xgb.csv',
    'output_dir': 'outputs/',
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)


def load_and_merge_data():
    """Load policy matrix and merge with full data."""
    print("Loading data...")
    
    # Load policy matrix (has GNN predictions and tiers)
    policy_df = pd.read_csv(CONFIG['policy_matrix_file'])
    print(f"  Policy matrix: {len(policy_df):,} properties, {len(policy_df.columns)} columns")
    
    # Load full data (has all property characteristics)
    full_df = pd.read_parquet(CONFIG['merged_data_file'])
    print(f"  Full data: {len(full_df):,} properties, {len(full_df.columns)} columns")
    
    # Merge on NHPDPropertyID
    df = policy_df.merge(
        full_df, 
        on='NHPDPropertyID', 
        how='left',
        suffixes=('', '_full')
    )
    
    # Use columns from full data where policy matrix has duplicates
    for col in ['City', 'State', 'PropertyName']:
        if f'{col}_full' in df.columns:
            df[col] = df[col].fillna(df[f'{col}_full'])
            df = df.drop(columns=[f'{col}_full'])
    
    print(f"  Merged: {len(df):,} properties, {len(df.columns)} columns")
    
    return df


# =============================================================================
# 1. EXAMPLE PROPERTIES PER TIER
# =============================================================================

def generate_tier_examples(df, n_examples=5):
    """Select representative example properties from each policy tier."""
    print("\n" + "="*70)
    print("1. EXAMPLE PROPERTIES BY POLICY TIER")
    print("="*70)
    
    tiers = [
        'Tier 1: Crisis (Distress + Exit Risk)',
        'Tier 2: Preservation (High Opt-Out Risk)',
        'Tier 3: Rehab Needed (Physical Distress)',
        'Tier 4: Stable (Low Risk)'
    ]
    
    all_examples = []
    
    for tier in tiers:
        tier_df = df[df['Policy_Tier'] == tier].copy()
        
        if len(tier_df) == 0:
            print(f"\n{tier}: No properties found")
            continue
            
        print(f"\n{'='*70}")
        print(f"{tier}")
        print(f"Total properties: {len(tier_df):,}")
        print("="*70)
        
        # Sort by GNN risk to get diverse examples
        if 'GNN_Risk' in tier_df.columns:
            tier_df = tier_df.sort_values('GNN_Risk', ascending=False)
        
        # Get examples from different risk levels within tier
        n = min(n_examples, len(tier_df))
        indices = np.linspace(0, len(tier_df)-1, n, dtype=int)
        examples = tier_df.iloc[indices]
        
        # Display
        for idx, (_, row) in enumerate(examples.iterrows(), 1):
            print(f"\nExample {idx}:")
            print(f"  Property: {row.get('PropertyName', 'N/A')}")
            print(f"  Location: {row.get('City', 'N/A')}, {row.get('State', 'N/A')}")
            
            # Units
            units = row.get('TotalUnits')
            print(f"  Units: {int(units)}" if pd.notna(units) else "  Units: N/A")
            
            # Age
            age = row.get('PropertyAge')
            print(f"  Age: {age:.0f} years" if pd.notna(age) else "  Age: N/A")
            
            # Rent Gap
            rent_gap = row.get('True_Rent_Gap')
            print(f"  Rent Gap: ${rent_gap:.0f}/month" if pd.notna(rent_gap) else "  Rent Gap: N/A")
            
            # Distress Risk
            risk = row.get('GNN_Risk')
            print(f"  Distress Risk: {risk*100:.1f}%" if pd.notna(risk) else "  Distress Risk: N/A")
            
            # Programs
            programs = []
            if row.get('Has_Section8') == 1: programs.append('Section 8')
            if row.get('Has_LIHTC') == 1: programs.append('LIHTC')
            if row.get('Has_PublicHousing') == 1: programs.append('Public Housing')
            if row.get('Has_Section515') == 1: programs.append('Section 515')
            if row.get('Has_Section202') == 1: programs.append('Section 202')
            if programs:
                print(f"  Programs: {', '.join(programs)}")
            
            all_examples.append(row)
    
    # Save examples
    examples_df = pd.DataFrame(all_examples)
    examples_df.to_csv(f"{CONFIG['output_dir']}/tier_examples.csv", index=False)
    print(f"\nSaved examples to {CONFIG['output_dir']}/tier_examples.csv")
    
    # Summary statistics by tier
    print("\n" + "="*70)
    print("TIER SUMMARY STATISTICS")
    print("="*70)
    
    summary_stats = []
    for tier in tiers:
        tier_df = df[df['Policy_Tier'] == tier]
        if len(tier_df) == 0:
            continue
            
        stats_row = {
            'Tier': tier.split(':')[0],
            'Count': len(tier_df),
            'Pct': f"{len(tier_df) / len(df) * 100:.1f}%",
            'Mean_Age': f"{tier_df['PropertyAge'].mean():.0f}" if 'PropertyAge' in tier_df.columns else 'N/A',
            'Mean_Units': f"{tier_df['TotalUnits'].mean():.0f}" if 'TotalUnits' in tier_df.columns else 'N/A',
            'Mean_Rent_Gap': f"${tier_df['True_Rent_Gap'].mean():.0f}" if 'True_Rent_Gap' in tier_df.columns else 'N/A',
            'Mean_Risk': f"{tier_df['GNN_Risk'].mean()*100:.0f}%" if 'GNN_Risk' in tier_df.columns else 'N/A',
        }
        summary_stats.append(stats_row)
    
    summary_df = pd.DataFrame(summary_stats)
    print("\n" + summary_df.to_string(index=False))
    summary_df.to_csv(f"{CONFIG['output_dir']}/tier_summary_stats.csv", index=False)
    
    return summary_df


# =============================================================================
# 2. GEOGRAPHIC DISTRIBUTION OF TIER 1
# =============================================================================

def analyze_tier1_geography(df):
    """Analyze geographic distribution of Tier 1 (Crisis) properties."""
    print("\n" + "="*70)
    print("2. GEOGRAPHIC DISTRIBUTION OF TIER 1 (CRISIS) PROPERTIES")
    print("="*70)
    
    tier1 = df[df['Policy_Tier'] == 'Tier 1: Crisis (Distress + Exit Risk)'].copy()
    total_tier1 = len(tier1)
    
    print(f"\nTotal Tier 1 properties: {total_tier1:,}")
    
    if 'State' not in df.columns:
        print("State column not found.")
        return None
    
    # State-level analysis
    state_tier1 = tier1.groupby('State').agg(
        Tier1_Properties=('NHPDPropertyID', 'count'),
        Tier1_Units=('TotalUnits', 'sum'),
        Mean_Risk=('GNN_Risk', 'mean'),
        Mean_Rent_Gap=('True_Rent_Gap', 'mean')
    ).reset_index()
    
    # Total properties per state
    state_total = df.groupby('State')['NHPDPropertyID'].count().reset_index()
    state_total.columns = ['State', 'Total_Properties']
    
    # Merge
    state_counts = state_tier1.merge(state_total, on='State')
    state_counts['Tier1_Pct'] = state_counts['Tier1_Properties'] / state_counts['Total_Properties'] * 100
    state_counts = state_counts.sort_values('Tier1_Properties', ascending=False)
    
    print("\n--- TOP 15 STATES BY TIER 1 COUNT ---")
    top15 = state_counts.head(15).copy()
    top15['Tier1_Units'] = top15['Tier1_Units'].fillna(0).astype(int)
    top15['Mean_Risk'] = (top15['Mean_Risk'] * 100).round(1).astype(str) + '%'
    top15['Mean_Rent_Gap'] = '$' + top15['Mean_Rent_Gap'].round(0).astype(int).astype(str)
    top15['Tier1_Pct'] = top15['Tier1_Pct'].round(1).astype(str) + '%'
    print(top15.to_string(index=False))
    
    print("\n--- TOP 10 STATES BY TIER 1 PERCENTAGE ---")
    top_pct = state_counts.sort_values('Tier1_Pct', ascending=False).head(10).copy()
    top_pct['Tier1_Pct'] = top_pct['Tier1_Pct'].round(1).astype(str) + '%'
    print(top_pct[['State', 'Tier1_Properties', 'Total_Properties', 'Tier1_Pct']].to_string(index=False))
    
    # Regional analysis
    regions = {
        'Northeast': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA'],
        'Midwest': ['IL', 'IN', 'MI', 'OH', 'WI', 'IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD'],
        'South': ['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'DC', 'WV', 'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX'],
        'West': ['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY', 'AK', 'CA', 'HI', 'OR', 'WA']
    }
    
    state_to_region = {state: region for region, states in regions.items() for state in states}
    tier1['Region'] = tier1['State'].map(state_to_region)
    
    print("\n--- TIER 1 BY REGION ---")
    region_counts = tier1.groupby('Region').agg(
        Properties=('NHPDPropertyID', 'count'),
        Units=('TotalUnits', 'sum')
    ).reset_index()
    region_counts['Units'] = region_counts['Units'].fillna(0).astype(int)
    region_counts['Pct'] = (region_counts['Properties'] / total_tier1 * 100).round(1).astype(str) + '%'
    region_counts = region_counts.sort_values('Properties', ascending=False)
    print(region_counts.to_string(index=False))
    
    # Top cities
    print("\n--- TOP 20 CITIES WITH MOST TIER 1 PROPERTIES ---")
    if 'City' in tier1.columns:
        city_counts = tier1.groupby(['City', 'State']).agg(
            Properties=('NHPDPropertyID', 'count'),
            Units=('TotalUnits', 'sum')
        ).reset_index()
        city_counts['Units'] = city_counts['Units'].fillna(0).astype(int)
        city_counts = city_counts.sort_values('Properties', ascending=False)
        print(city_counts.head(20).to_string(index=False))
    
    # Save
    state_counts.to_csv(f"{CONFIG['output_dir']}/tier1_by_state.csv", index=False)
    print(f"\nSaved to {CONFIG['output_dir']}/tier1_by_state.csv")
    
    # Summary for paper
    print("\n" + "="*70)
    print("SUMMARY FOR PAPER")
    print("="*70)
    
    top5 = state_counts.head(5)
    top5_total = top5['Tier1_Properties'].sum()
    
    print(f"""
Tier 1 Crisis properties are concentrated geographically:

Top 5 States:
""")
    for _, row in top5.iterrows():
        units = int(row['Tier1_Units']) if pd.notna(row['Tier1_Units']) else 0
        print(f"  - {row['State']}: {int(row['Tier1_Properties']):,} properties ({units:,} units)")
    
    print(f"""
These five states account for {top5_total / total_tier1 * 100:.0f}% of all Tier 1 properties.

National average: ~{total_tier1 // 50} Tier 1 properties per state
""")
    
    return state_counts


# =============================================================================
# 3. FEATURE IMPORTANCE
# =============================================================================

def analyze_feature_importance(df):
    """Analyze and categorize feature importance from XGBoost."""
    print("\n" + "="*70)
    print("3. FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    if not os.path.exists(CONFIG['feature_importance_file']):
        print(f"Feature importance file not found: {CONFIG['feature_importance_file']}")
        print("Run baseline_comparison.py first.")
        return None
    
    importance_df = pd.read_csv(CONFIG['feature_importance_file'])
    
    print("\n--- TOP 20 MOST IMPORTANT FEATURES ---")
    print(importance_df.head(20).to_string(index=False))
    
    # Categorize features
    categories = {
        'Property Characteristics': [
            'PropertyAge', 'TotalUnits', 'Log_Age', 'Log_Units', 'Is_Old', 
            'Is_Large', 'Age_Squared', 'Is_Very_Old', 'Is_Small', 'Age_Pctl'
        ],
        'Economic/Rent': [
            'True_Rent_Gap', 'Log_Rent_Gap', 'Rent_Gap_Pctl', 'High_Exit_Incentive', 
            'Median_Gross_Rent', 'Median_Housing_Value', 'FairMarketRent_2BR',
            'Gini_Index_Income_Inequality', 'Median_Household_Income'
        ],
        'Neighborhood Demographics': [
            'Poverty_Rate', 'High_Poverty', 'Very_High_Poverty', 'Poverty_Pctl',
            'Vacancy_Rate', 'High_Vacancy', 'Vacancy_Pctl',
            'Elderly_Share', 'Disability_Rate', 'Total_Population', 'Pop_Density',
            'White_NonHispanic_Share', 'Renter_Share', 'Pre1970_Housing_Share',
            'Severe_Rent_Burden_Rate', 'Any_Rent_Burden_Rate'
        ],
        'Program/Subsidy': [
            'PH_1_AssistedUnits', 'S8_1_AssistedUnits', 'LIHTC_1_AssistedUnits',
            'NumberActiveSection8', 'NumberActiveLihtc', 'NumberActivePublicHousing',
            'NumberActiveSection515', 'NumberActiveSection202', 'NumberActiveHome',
            'NumActivePrograms', 'PercentSubsidized', 'Has_Section8', 'Has_LIHTC',
            'Has_PublicHousing', 'Has_Section515', 'Has_Section202', 'Has_HOME'
        ],
        'Owner/Management': [
            'Owner_Portfolio_Size', 'Log_Portfolio', 'Is_Large_Owner', 'Is_Single_Owner',
            'OccupancyRate', 'AverageMonthsOfTenancy'
        ],
        'Geographic': [
            'CountyCode', 'CBSACode', 'IsMetro', 'Latitude', 'Longitude'
        ],
        'Interactions': [
            'Age_x_Poverty', 'Age_x_RentGap', 'Age_x_Vacancy'
        ]
    }
    
    # Aggregate importance by category
    category_importance = []
    for cat_name, cat_features in categories.items():
        cat_df = importance_df[importance_df['Feature'].isin(cat_features)]
        cat_imp = cat_df['Importance'].sum()
        n_features = len(cat_df)
        top_feature = cat_df.iloc[0]['Feature'] if len(cat_df) > 0 else 'N/A'
        category_importance.append({
            'Category': cat_name, 
            'Total_Importance': cat_imp,
            'Num_Features': n_features,
            'Top_Feature': top_feature
        })
    
    cat_df = pd.DataFrame(category_importance).sort_values('Total_Importance', ascending=False)
    
    print("\n--- IMPORTANCE BY FEATURE CATEGORY ---")
    print(cat_df.to_string(index=False))
    
    # Key findings
    top3 = importance_df.head(3)
    
    print(f"""
\n--- KEY FINDINGS FOR PAPER ---

Top 3 Predictors of Physical Distress:
  1. {top3.iloc[0]['Feature']} (importance: {top3.iloc[0]['Importance']:.3f})
  2. {top3.iloc[1]['Feature']} (importance: {top3.iloc[1]['Importance']:.3f})
  3. {top3.iloc[2]['Feature']} (importance: {top3.iloc[2]['Importance']:.3f})

Feature Category Ranking:
""")
    for i, (_, row) in enumerate(cat_df.iterrows(), 1):
        print(f"  {i}. {row['Category']}: {row['Total_Importance']:.3f} ({row['Num_Features']} features)")
    
    cat_df.to_csv(f"{CONFIG['output_dir']}/feature_importance_by_category.csv", index=False)
    
    return importance_df, cat_df


# =============================================================================
# 4. TIER CHARACTERISTIC COMPARISON
# =============================================================================

def analyze_tier_characteristics(df):
    """Detailed comparison of what distinguishes each tier."""
    print("\n" + "="*70)
    print("4. TIER CHARACTERISTIC COMPARISON")
    print("="*70)
    
    tiers = [
        'Tier 1: Crisis (Distress + Exit Risk)',
        'Tier 2: Preservation (High Opt-Out Risk)',
        'Tier 3: Rehab Needed (Physical Distress)',
        'Tier 4: Stable (Low Risk)'
    ]
    
    # Variables to compare
    compare_vars = [
        ('PropertyAge', 'Property Age (years)'),
        ('TotalUnits', 'Total Units'),
        ('True_Rent_Gap', 'Rent Gap ($/month)'),
        ('GNN_Risk', 'Distress Probability'),
        ('Poverty_Rate', 'Poverty Rate (%)'),
        ('Vacancy_Rate', 'Vacancy Rate (%)'),
        ('OccupancyRate', 'Occupancy Rate (%)'),
        ('Owner_Portfolio_Size', 'Owner Portfolio Size'),
    ]
    
    # Build comparison table
    comparison = []
    for tier in tiers:
        tier_df = df[df['Policy_Tier'] == tier]
        if len(tier_df) == 0:
            continue
        
        row = {'Tier': tier.split(':')[0]}
        for var, label in compare_vars:
            if var in tier_df.columns:
                mean_val = tier_df[var].mean()
                if var == 'GNN_Risk':
                    row[label] = f"{mean_val*100:.0f}%"
                elif var in ['Poverty_Rate', 'Vacancy_Rate', 'OccupancyRate']:
                    row[label] = f"{mean_val:.1f}%"
                elif var == 'True_Rent_Gap':
                    row[label] = f"${mean_val:.0f}"
                else:
                    row[label] = f"{mean_val:.1f}"
        comparison.append(row)
    
    comp_df = pd.DataFrame(comparison)
    
    print("\nMean Values by Tier:")
    print(comp_df.to_string(index=False))
    
    # Statistical tests between Tier 1 and Tier 4
    print("\n--- TIER 1 vs TIER 4 COMPARISON ---")
    tier1 = df[df['Policy_Tier'] == tiers[0]]
    tier4 = df[df['Policy_Tier'] == tiers[3]]
    
    if len(tier1) > 0 and len(tier4) > 0:
        for var, label in compare_vars:
            if var in df.columns:
                t1_vals = tier1[var].dropna()
                t4_vals = tier4[var].dropna()
                
                if len(t1_vals) > 0 and len(t4_vals) > 0:
                    t_stat, p_val = stats.ttest_ind(t1_vals, t4_vals)
                    t1_mean = t1_vals.mean()
                    t4_mean = t4_vals.mean()
                    
                    if var == 'GNN_Risk':
                        diff_str = f"{t1_mean*100:.0f}% vs {t4_mean*100:.0f}%"
                    elif var == 'True_Rent_Gap':
                        diff_str = f"${t1_mean:.0f} vs ${t4_mean:.0f}"
                    else:
                        diff_str = f"{t1_mean:.1f} vs {t4_mean:.1f}"
                    
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    print(f"  {label}: {diff_str} (p={p_val:.4f}) {sig}")
    
    # Key insight for paper
    print(f"""
\n--- KEY INSIGHT FOR PAPER ---

Tier 1 (Crisis) properties are characterized by:
  - High distress probability (~{tier1['GNN_Risk'].mean()*100:.0f}%)
  - High rent gap (~${tier1['True_Rent_Gap'].mean():.0f}/month)
  - Older buildings (~{tier1['PropertyAge'].mean():.0f} years)
  - Higher neighborhood poverty rates

This combination represents properties where:
  1. Physical condition is deteriorating (distress signal)
  2. Owners have strong economic incentive to exit (rent gap)
  → Highest priority for preservation intervention
""")
    
    comp_df.to_csv(f"{CONFIG['output_dir']}/tier_comparison.csv", index=False)
    
    return comp_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("PAPER ANALYSES: Results for Data & Policy Submission")
    print("="*70)
    
    # Load and merge data
    df = load_and_merge_data()
    
    # 1. Example properties per tier
    generate_tier_examples(df, n_examples=5)
    
    # 2. Geographic distribution of Tier 1
    analyze_tier1_geography(df)
    
    # 3. Feature importance
    analyze_feature_importance(df)
    
    # 4. Tier characteristics
    analyze_tier_characteristics(df)
    
    print("\n" + "="*70)
    print("ANALYSES COMPLETE")
    print("="*70)
    print(f"""
Generated outputs in {CONFIG['output_dir']}:
  - tier_examples.csv
  - tier_summary_stats.csv
  - tier1_by_state.csv
  - feature_importance_by_category.csv
  - tier_comparison.csv
""")


if __name__ == "__main__":
    main()