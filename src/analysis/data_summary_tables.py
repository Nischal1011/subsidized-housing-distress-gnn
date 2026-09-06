"""
generate_data_tables.py
=======================
Generate summary tables for Data & Policy paper.

Tables:
1. Data Summary (properties, programs, coverage)
2. Variable Definitions
3. Descriptive Statistics by REAC availability
4. Model Comparison (formatted for paper)
5. Policy Tier Summary
"""

import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

CONFIG = {
    'policy_matrix_file': 'outputs/integrated_policy_matrix.csv',
    'merged_data_file': 'data/merged_analysis_data.parquet',
    'output_dir': 'outputs/tables/',
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)


def load_data():
    """Load and merge data."""
    print("Loading data...")
    policy_df = pd.read_csv(CONFIG['policy_matrix_file'])
    full_df = pd.read_parquet(CONFIG['merged_data_file'])
    
    df = policy_df.merge(full_df, on='NHPDPropertyID', how='left', suffixes=('', '_full'))
    
    for col in ['City', 'State', 'PropertyName']:
        if f'{col}_full' in df.columns:
            df[col] = df[col].fillna(df[f'{col}_full'])
            df = df.drop(columns=[f'{col}_full'])
    
    print(f"  Loaded {len(df):,} properties")
    return df, full_df


# =============================================================================
# TABLE 1: DATA SUMMARY
# =============================================================================

def create_table1_data_summary(df):
    """Create comprehensive data summary table."""
    print("\nCreating Table 1: Data Summary...")
    
    # Panel A: Dataset Overview
    panel_a = []
    panel_a.append(('Total properties', f"{len(df):,}"))
    panel_a.append(('Total units', f"{df['TotalUnits'].sum():,.0f}"))
    panel_a.append(('States covered', f"{df['State'].nunique()}"))
    panel_a.append(('Counties covered', f"{df['CountyCode'].nunique():,}"))
    panel_a.append(('Census tracts', f"{df['CensusTract'].nunique():,}"))
    panel_a.append(('Unique owners', f"{df['Owner'].nunique():,}"))
    
    # Panel B: REAC Coverage
    has_reac = df['ReacScore1'].notna().sum()
    missing_reac = df['ReacScore1'].isna().sum()
    
    panel_b = []
    panel_b.append(('Properties with REAC score', f"{has_reac:,} ({has_reac/len(df)*100:.1f}%)"))
    panel_b.append(('Properties without REAC score', f"{missing_reac:,} ({missing_reac/len(df)*100:.1f}%)"))
    
    if 'PhysicalDistress' in df.columns:
        distressed = (df['PhysicalDistress'] == 1).sum()
        labeled = df['PhysicalDistress'].notna().sum()
        panel_b.append(('Labeled as distressed (REAC < 60)', f"{distressed:,} ({distressed/labeled*100:.1f}% of labeled)"))
    
    # Panel C: Program Participation
    programs = [
        ('Section 8 (Project-Based)', 'NumberActiveSection8'),
        ('Low-Income Housing Tax Credit (LIHTC)', 'NumberActiveLihtc'),
        ('Section 515 (Rural Housing)', 'NumberActiveSection515'),
        ('Public Housing', 'NumberActivePublicHousing'),
        ('Section 202 (Elderly Housing)', 'NumberActiveSection202'),
        ('HOME Investment Partnerships', 'NumberActiveHome'),
        ('Section 236', 'NumberActiveSection236'),
    ]
    
    panel_c = []
    for prog_name, prog_col in programs:
        if prog_col in df.columns:
            count = (df[prog_col] > 0).sum()
            pct = count / len(df) * 100
            panel_c.append((prog_name, f"{count:,} ({pct:.1f}%)"))
    
    # Panel D: Property Characteristics
    panel_d = []
    
    age = df['PropertyAge'].dropna()
    panel_d.append(('Property age (years)', f"{age.mean():.1f} ± {age.std():.1f}"))
    panel_d.append(('  Median', f"{age.median():.0f}"))
    panel_d.append(('  Range', f"{age.min():.0f} - {age.max():.0f}"))
    
    units = df['TotalUnits'].dropna()
    panel_d.append(('Total units per property', f"{units.mean():.1f} ± {units.std():.1f}"))
    panel_d.append(('  Median', f"{units.median():.0f}"))
    panel_d.append(('  Range', f"{units.min():.0f} - {units.max():.0f}"))
    
    if 'OccupancyRate' in df.columns:
        occ = df['OccupancyRate'].dropna()
        panel_d.append(('Occupancy rate (%)', f"{occ.mean():.1f} ± {occ.std():.1f}"))
    
    # Panel E: Neighborhood Characteristics (from ACS)
    panel_e = []
    
    if 'Median_Household_Income' in df.columns:
        inc = df['Median_Household_Income'].dropna()
        panel_e.append(('Median household income ($)', f"{inc.mean():,.0f} ± {inc.std():,.0f}"))
    
    if 'Median_Gross_Rent' in df.columns:
        rent = df['Median_Gross_Rent'].dropna()
        panel_e.append(('Median gross rent ($)', f"{rent.mean():.0f} ± {rent.std():.0f}"))
    
    if 'Poverty_Rate' in df.columns:
        pov = df['Poverty_Rate'].dropna()
        panel_e.append(('Poverty rate (%)', f"{pov.mean():.1f} ± {pov.std():.1f}"))
    
    if 'Vacancy_Rate' in df.columns:
        vac = df['Vacancy_Rate'].dropna()
        panel_e.append(('Vacancy rate (%)', f"{vac.mean():.1f} ± {vac.std():.1f}"))
    
    if 'Severe_Rent_Burden_Rate' in df.columns:
        burden = df['Severe_Rent_Burden_Rate'].dropna()
        panel_e.append(('Severe rent burden rate (%)', f"{burden.mean():.1f} ± {burden.std():.1f}"))
    
    # Print formatted table
    print("\n" + "="*70)
    print("TABLE 1: DATA SUMMARY")
    print("="*70)
    
    print("\nPanel A: Dataset Overview")
    print("-"*50)
    for label, value in panel_a:
        print(f"  {label:<35} {value:>15}")
    
    print("\nPanel B: REAC Inspection Coverage")
    print("-"*50)
    for label, value in panel_b:
        print(f"  {label:<35} {value:>20}")
    
    print("\nPanel C: Subsidy Program Participation")
    print("-"*50)
    for label, value in panel_c:
        print(f"  {label:<40} {value:>15}")
    
    print("\nPanel D: Property Characteristics")
    print("-"*50)
    for label, value in panel_d:
        print(f"  {label:<35} {value:>20}")
    
    print("\nPanel E: Neighborhood Characteristics (Census Tract Level)")
    print("-"*50)
    for label, value in panel_e:
        print(f"  {label:<35} {value:>20}")
    
    # Save as CSV
    all_rows = []
    all_rows.append(('Panel A: Dataset Overview', ''))
    all_rows.extend(panel_a)
    all_rows.append(('', ''))
    all_rows.append(('Panel B: REAC Inspection Coverage', ''))
    all_rows.extend(panel_b)
    all_rows.append(('', ''))
    all_rows.append(('Panel C: Subsidy Program Participation', ''))
    all_rows.extend(panel_c)
    all_rows.append(('', ''))
    all_rows.append(('Panel D: Property Characteristics', ''))
    all_rows.extend(panel_d)
    all_rows.append(('', ''))
    all_rows.append(('Panel E: Neighborhood Characteristics', ''))
    all_rows.extend(panel_e)
    
    summary_df = pd.DataFrame(all_rows, columns=['Variable', 'Value'])
    summary_df.to_csv(f"{CONFIG['output_dir']}/table1_data_summary.csv", index=False)
    print(f"\nSaved to {CONFIG['output_dir']}/table1_data_summary.csv")
    
    return summary_df


# =============================================================================
# TABLE 2: VARIABLE DEFINITIONS
# =============================================================================

def create_table2_variable_definitions():
    """Create variable definitions table."""
    print("\nCreating Table 2: Variable Definitions...")
    
    variables = [
        # Outcome
        ('Outcome Variable', '', ''),
        ('Physical Distress', 'Binary', 'REAC inspection score < 60 (scale 0-100)'),
        
        # Key Constructed Variables
        ('', '', ''),
        ('Key Constructed Variables', '', ''),
        ('True Rent Gap', 'Continuous ($)', 'Market Rent - Restricted Rent, where Restricted Rent = (Tract AMI × 0.60 × 0.30) / 12'),
        ('GNN Distress Probability', 'Continuous (0-1)', 'Predicted probability of physical distress from Graph Attention Network'),
        ('Policy Tier', 'Categorical', 'Four-tier classification based on distress risk and rent gap'),
        
        # Graph Features
        ('', '', ''),
        ('Graph Network Features', '', ''),
        ('Ownership Edges', 'Network', 'Properties connected if they share the same owner'),
        ('Spatial Edges', 'Network', 'Properties connected if within 8km (k=15 nearest neighbors)'),
        ('Tract Edges', 'Network', 'Properties connected if in the same census tract'),
        
        # Property Characteristics
        ('', '', ''),
        ('Property Characteristics', '', ''),
        ('Property Age', 'Continuous (years)', 'Years since earliest construction date'),
        ('Total Units', 'Continuous', 'Number of housing units in property'),
        ('Occupancy Rate', 'Continuous (%)', 'Percentage of units currently occupied'),
        ('Owner Portfolio Size', 'Continuous', 'Number of properties owned by same owner'),
        
        # Neighborhood Characteristics (ACS)
        ('', '', ''),
        ('Neighborhood Characteristics (ACS 5-Year)', '', ''),
        ('Median Household Income', 'Continuous ($)', 'Census tract median household income'),
        ('Median Gross Rent', 'Continuous ($)', 'Census tract median gross rent'),
        ('Poverty Rate', 'Continuous (%)', 'Percentage of population below poverty line'),
        ('Vacancy Rate', 'Continuous (%)', 'Percentage of housing units vacant'),
        ('Severe Rent Burden Rate', 'Continuous (%)', 'Percentage of renters paying >50% income on rent'),
        ('Pre-1970 Housing Share', 'Continuous (%)', 'Percentage of housing built before 1970'),
        
        # Program Indicators
        ('', '', ''),
        ('Program Indicators', '', ''),
        ('Has Section 8', 'Binary', 'Property has active Section 8 project-based assistance'),
        ('Has LIHTC', 'Binary', 'Property has active Low-Income Housing Tax Credit'),
        ('Has Public Housing', 'Binary', 'Property is public housing'),
        ('Has Section 515', 'Binary', 'Property has USDA Section 515 rural housing loan'),
    ]
    
    var_df = pd.DataFrame(variables, columns=['Variable', 'Type', 'Description'])
    
    print("\n" + "="*90)
    print("TABLE 2: VARIABLE DEFINITIONS")
    print("="*90)
    print(var_df.to_string(index=False))
    
    var_df.to_csv(f"{CONFIG['output_dir']}/table2_variable_definitions.csv", index=False)
    print(f"\nSaved to {CONFIG['output_dir']}/table2_variable_definitions.csv")
    
    return var_df


# =============================================================================
# TABLE 3: DESCRIPTIVE STATISTICS BY REAC AVAILABILITY
# =============================================================================

def create_table3_reac_comparison(df):
    """Create comparison of properties with/without REAC scores."""
    print("\nCreating Table 3: REAC Availability Comparison...")
    
    df['HasREAC'] = df['ReacScore1'].notna()
    
    with_reac = df[df['HasREAC']]
    without_reac = df[~df['HasREAC']]
    
    variables = [
        ('Property Age (years)', 'PropertyAge'),
        ('Total Units', 'TotalUnits'),
        ('Occupancy Rate (%)', 'OccupancyRate'),
        ('Median Household Income ($)', 'Median_Household_Income'),
        ('Median Gross Rent ($)', 'Median_Gross_Rent'),
        ('Poverty Rate (%)', 'Poverty_Rate'),
        ('Vacancy Rate (%)', 'Vacancy_Rate'),
    ]
    
    rows = []
    for var_name, col in variables:
        if col in df.columns:
            w_mean = with_reac[col].mean()
            w_std = with_reac[col].std()
            wo_mean = without_reac[col].mean()
            wo_std = without_reac[col].std()
            
            # t-test
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(
                with_reac[col].dropna(), 
                without_reac[col].dropna()
            )
            
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            
            rows.append({
                'Variable': var_name,
                'With REAC (Mean)': f"{w_mean:.1f}",
                'With REAC (SD)': f"{w_std:.1f}",
                'Without REAC (Mean)': f"{wo_mean:.1f}",
                'Without REAC (SD)': f"{wo_std:.1f}",
                'p-value': f"{p_val:.4f}{sig}"
            })
    
    # Program participation
    programs = [
        ('Section 8 (%)', 'NumberActiveSection8'),
        ('LIHTC (%)', 'NumberActiveLihtc'),
        ('Public Housing (%)', 'NumberActivePublicHousing'),
        ('Section 515 (%)', 'NumberActiveSection515'),
    ]
    
    for var_name, col in programs:
        if col in df.columns:
            w_pct = (with_reac[col] > 0).mean() * 100
            wo_pct = (without_reac[col] > 0).mean() * 100
            
            # Chi-square test
            contingency = pd.crosstab(df['HasREAC'], df[col] > 0)
            chi2, p_val, _, _ = stats.chi2_contingency(contingency)
            
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            
            rows.append({
                'Variable': var_name,
                'With REAC (Mean)': f"{w_pct:.1f}",
                'With REAC (SD)': '-',
                'Without REAC (Mean)': f"{wo_pct:.1f}",
                'Without REAC (SD)': '-',
                'p-value': f"{p_val:.4f}{sig}"
            })
    
    comp_df = pd.DataFrame(rows)
    
    print("\n" + "="*100)
    print("TABLE 3: COMPARISON OF PROPERTIES BY REAC AVAILABILITY")
    print("="*100)
    print(f"\nN with REAC: {len(with_reac):,}  |  N without REAC: {len(without_reac):,}")
    print("-"*100)
    print(comp_df.to_string(index=False))
    print("\nNote: *** p<0.001, ** p<0.01, * p<0.05")
    
    comp_df.to_csv(f"{CONFIG['output_dir']}/table3_reac_comparison.csv", index=False)
    print(f"\nSaved to {CONFIG['output_dir']}/table3_reac_comparison.csv")
    
    return comp_df


# =============================================================================
# TABLE 4: MODEL COMPARISON
# =============================================================================

def create_table4_model_comparison():
    """Create model comparison table with results from baseline analysis."""
    print("\nCreating Table 4: Model Comparison...")
    
    # Results from your baseline comparison run
    models = [
        {
            'Model': 'XGBoost',
            'AUC': 0.753,
            'Precision': 0.326,
            'Recall': 0.558,
            'F1': 0.412,
            'Lift': 2.25,
            'Uses Graph': 'No'
        },
        {
            'Model': 'GAT (Ours)',
            'AUC': 0.726,
            'Precision': 0.349,
            'Recall': 0.587,
            'F1': 0.437,
            'Lift': 2.41,
            'Uses Graph': 'Yes'
        },
        {
            'Model': 'Logistic Regression',
            'AUC': 0.730,
            'Precision': 0.301,
            'Recall': 0.491,
            'F1': 0.373,
            'Lift': 2.08,
            'Uses Graph': 'No'
        },
    ]
    
    model_df = pd.DataFrame(models)
    
    # Format for display
    display_df = model_df.copy()
    display_df['AUC'] = display_df['AUC'].apply(lambda x: f"{x:.3f}")
    display_df['Precision'] = display_df['Precision'].apply(lambda x: f"{x:.3f}")
    display_df['Recall'] = display_df['Recall'].apply(lambda x: f"{x:.3f}")
    display_df['F1'] = display_df['F1'].apply(lambda x: f"{x:.3f}")
    display_df['Lift'] = display_df['Lift'].apply(lambda x: f"{x:.2f}x")
    
    print("\n" + "="*85)
    print("TABLE 4: MODEL COMPARISON FOR PHYSICAL DISTRESS PREDICTION")
    print("="*85)
    print(display_df.to_string(index=False))
    print("\nNotes:")
    print("- Base rate (distress prevalence): 14.5%")
    print("- All models trained on identical features (n=173) and train/test splits")
    print("- Lift = Precision / Base Rate (improvement over random selection)")
    print("- GAT additionally leverages ownership network and spatial proximity structure")
    print("- Best values in each column: AUC (XGBoost), Recall (GAT), F1 (GAT)")
    
    model_df.to_csv(f"{CONFIG['output_dir']}/table4_model_comparison.csv", index=False)
    print(f"\nSaved to {CONFIG['output_dir']}/table4_model_comparison.csv")
    
    return model_df


# =============================================================================
# TABLE 5: POLICY TIER SUMMARY
# =============================================================================

def create_table5_policy_tiers(df):
    """Create policy tier summary table."""
    print("\nCreating Table 5: Policy Tier Summary...")
    
    tiers = [
        ('Tier 1: Crisis', 'Tier 1: Crisis (Distress + Exit Risk)', 
         'High distress + High rent gap', 'Emergency intervention, acquisition'),
        ('Tier 2: Preservation', 'Tier 2: Preservation (High Opt-Out Risk)',
         'Low distress + High rent gap', 'Contract renewal, nonprofit transfer'),
        ('Tier 3: Rehab', 'Tier 3: Rehab Needed (Physical Distress)',
         'High distress + Low rent gap', 'Capital grants, rehabilitation'),
        ('Tier 4: Stable', 'Tier 4: Stable (Low Risk)',
         'Low distress + Low rent gap', 'Routine monitoring'),
    ]
    
    rows = []
    for tier_short, tier_full, definition, intervention in tiers:
        tier_df = df[df['Policy_Tier'] == tier_full]
        
        if len(tier_df) == 0:
            continue
        
        rows.append({
            'Tier': tier_short,
            'Definition': definition,
            'N Properties': f"{len(tier_df):,}",
            '% of Total': f"{len(tier_df)/len(df)*100:.1f}%",
            'Mean Age': f"{tier_df['PropertyAge'].mean():.0f}" if 'PropertyAge' in tier_df else 'N/A',
            'Mean Units': f"{tier_df['TotalUnits'].mean():.0f}" if 'TotalUnits' in tier_df else 'N/A',
            'Mean Rent Gap': f"${tier_df['True_Rent_Gap'].mean():.0f}",
            'Mean Distress Risk': f"{tier_df['GNN_Risk'].mean()*100:.0f}%",
            'Recommended Intervention': intervention
        })
    
    tier_df = pd.DataFrame(rows)
    
    print("\n" + "="*120)
    print("TABLE 5: POLICY TIER SUMMARY")
    print("="*120)
    print(tier_df.to_string(index=False))
    
    tier_df.to_csv(f"{CONFIG['output_dir']}/table5_policy_tiers.csv", index=False)
    print(f"\nSaved to {CONFIG['output_dir']}/table5_policy_tiers.csv")
    
    return tier_df


# =============================================================================
# TABLE 6: GRAPH NETWORK STATISTICS
# =============================================================================

def create_table6_graph_statistics():
    """Create graph network statistics table."""
    print("\nCreating Table 6: Graph Network Statistics...")
    
    # From your GAT model output
    stats = [
        ('Total nodes (properties)', '81,277'),
        ('Total edges', '1,511,708'),
        ('', ''),
        ('Edge Types:', ''),
        ('  Ownership edges', '160,338'),
        ('  Census tract edges', '127,000'),
        ('  Spatial proximity edges (k=15, r=8km)', '933,906'),
        ('', ''),
        ('Network Density', '0.00023'),
        ('Average degree', '18.6'),
        ('', ''),
        ('Labeled nodes (with REAC)', '12,385 (15.2%)'),
        ('Unlabeled nodes (without REAC)', '68,892 (84.8%)'),
    ]
    
    stats_df = pd.DataFrame(stats, columns=['Metric', 'Value'])
    
    print("\n" + "="*50)
    print("TABLE 6: GRAPH NETWORK STATISTICS")
    print("="*50)
    print(stats_df.to_string(index=False))
    
    stats_df.to_csv(f"{CONFIG['output_dir']}/table6_graph_statistics.csv", index=False)
    print(f"\nSaved to {CONFIG['output_dir']}/table6_graph_statistics.csv")
    
    return stats_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("GENERATING TABLES FOR DATA & POLICY PAPER")
    print("="*70)
    
    # Load data
    df, full_df = load_data()
    
    # Generate all tables
    create_table1_data_summary(df)
    create_table2_variable_definitions()
    create_table3_reac_comparison(df)
    create_table4_model_comparison()
    create_table5_policy_tiers(df)
    create_table6_graph_statistics()
    
    print("\n" + "="*70)
    print("ALL TABLES GENERATED")
    print("="*70)
    print(f"""
Output directory: {CONFIG['output_dir']}

Tables created:
  1. table1_data_summary.csv        - Dataset overview, programs, characteristics
  2. table2_variable_definitions.csv - Variable names, types, descriptions
  3. table3_reac_comparison.csv     - With vs without REAC comparison
  4. table4_model_comparison.csv    - GAT vs baselines performance
  5. table5_policy_tiers.csv        - Tier summary with interventions
  6. table6_graph_statistics.csv    - Network structure statistics
""")


if __name__ == "__main__":
    main()