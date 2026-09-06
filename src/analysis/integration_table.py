"""
compounded_risk_analysis.py
===========================
Generate the integration table showing properties with multiple compounding risk factors.
This is THE KEY ANALYSIS the reviewer requested.

Run AFTER: 06_gnn_final_v2.py has generated the policy matrix
"""

import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

CONFIG = {
    'policy_matrix_file': 'outputs/integrated_policy_matrix_v2.csv',
    'merged_data_file': 'data/merged_analysis_data.parquet',
    'output_dir': 'outputs/',
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)


def load_data():
    """Load and merge data."""
    print("Loading data...")
    
    # Policy matrix (has GNN predictions and initial tiers)
    policy_df = pd.read_csv(CONFIG['policy_matrix_file'])
    print(f"  Policy matrix: {len(policy_df):,} properties")
    
    # Full data (has all characteristics)
    full_df = pd.read_parquet(CONFIG['merged_data_file'])
    print(f"  Full data: {len(full_df):,} properties")
    
    # Merge
    df = policy_df.merge(full_df, on='NHPDPropertyID', how='left', suffixes=('', '_full'))
    
    # Clean up duplicate columns
    for col in ['City', 'State', 'PropertyName']:
        if f'{col}_full' in df.columns:
            df[col] = df[col].fillna(df[f'{col}_full'])
            df = df.drop(columns=[f'{col}_full'])
    
    print(f"  Merged: {len(df):,} properties\n")
    return df


def define_risk_factors(df):
    """
    Define the 4 key risk factors.
    These should be binary (0/1) indicators.
    """
    print("="*80)
    print("DEFINING RISK FACTORS")
    print("="*80)
    
    df = df.copy()
    
    # Calculate medians for thresholds
    distress_median = df['GNN_Risk'].median()
    rent_gap_median = df['True_Rent_Gap'].median()
    age_median = df['PropertyAge'].median() if 'PropertyAge' in df.columns else 45
    
    print(f"\nThresholds:")
    print(f"  Distress Risk: {distress_median:.3f}")
    print(f"  Rent Gap: ${rent_gap_median:.0f}/month")
    print(f"  Property Age: {age_median:.0f} years")
    
    # --- Risk Factor 1: High Physical Distress Risk ---
    df['RF1_HighDistress'] = (df['GNN_Risk'] > distress_median).astype(int)
    n_rf1 = df['RF1_HighDistress'].sum()
    print(f"\nRF1 - High Distress Risk (GNN > {distress_median:.3f}): {n_rf1:,} properties")
    
    # --- Risk Factor 2: High Economic Pressure (Rent Gap) ---
    df['RF2_HighRentGap'] = (df['True_Rent_Gap'] > rent_gap_median).astype(int)
    n_rf2 = df['RF2_HighRentGap'].sum()
    print(f"RF2 - High Rent Gap (>${rent_gap_median:.0f}/mo): {n_rf2:,} properties")
    
    # --- Risk Factor 3: Near-Term Expiration ---
    # Assuming you have expiration data
    if 'LatestEndDate' in df.columns or 'YearsToExpiration' in df.columns:
        if 'YearsToExpiration' in df.columns:
            df['RF3_NearExpiration'] = (df['YearsToExpiration'] <= 5).astype(int)
        else:
            # Calculate from LatestEndDate
            df['LatestEndDate'] = pd.to_datetime(df['LatestEndDate'], errors='coerce')
            cutoff_date = pd.Timestamp('2030-12-31')
            df['RF3_NearExpiration'] = (df['LatestEndDate'] <= cutoff_date).astype(int)
    else:
        # If no expiration data, use a proxy or set to 0
        print("  WARNING: No expiration data found. Setting RF3 to 0.")
        df['RF3_NearExpiration'] = 0
    
    n_rf3 = df['RF3_NearExpiration'].sum()
    print(f"RF3 - Near-Term Expiration (≤5 years): {n_rf3:,} properties")
    
    # --- Risk Factor 4: Advanced Age ---
    if 'PropertyAge' in df.columns:
        df['RF4_OldAge'] = (df['PropertyAge'] > age_median).astype(int)
    else:
        print("  WARNING: PropertyAge not found. Setting RF4 to 0.")
        df['RF4_OldAge'] = 0
    
    n_rf4 = df['RF4_OldAge'].sum()
    print(f"RF4 - Advanced Age (>{age_median:.0f} years): {n_rf4:,} properties")
    
    return df


def create_compounded_risk_table(df):
    """
    Create the integration table showing properties by number of risk factors.
    THIS IS THE KEY TABLE THE REVIEWER WANTS.
    """
    print("\n" + "="*80)
    print("COMPOUNDED RISK ANALYSIS")
    print("="*80)
    
    # Count total risk factors per property
    risk_cols = ['RF1_HighDistress', 'RF2_HighRentGap', 'RF3_NearExpiration', 'RF4_OldAge']
    df['NumRiskFactors'] = df[risk_cols].sum(axis=1)
    
    # Summary by number of risk factors
    summary_rows = []
    
    for n_risks in range(5):  # 0, 1, 2, 3, 4
        subset = df[df['NumRiskFactors'] == n_risks]
        
        if len(subset) == 0:
            continue
        
        # Get top states for this group
        if 'State' in subset.columns:
            top_states = subset['State'].value_counts().head(3)
            top_states_str = ', '.join([f"{state} ({count})" for state, count in top_states.items()])
        else:
            top_states_str = 'N/A'
        
        # Common risk combinations
        if n_risks > 0:
            combinations = []
            for _, row in subset.head(100).iterrows():  # Sample for combinations
                combo = []
                if row['RF1_HighDistress']: combo.append('Distress')
                if row['RF2_HighRentGap']: combo.append('Rent Gap')
                if row['RF3_NearExpiration']: combo.append('Expiring')
                if row['RF4_OldAge']: combo.append('Old')
                if combo:
                    combinations.append('+'.join(combo))
            
            if combinations:
                from collections import Counter
                common_combo = Counter(combinations).most_common(1)[0][0]
            else:
                common_combo = 'N/A'
        else:
            common_combo = 'None'
        
        row_data = {
            'Risk_Factors': f"{n_risks} of 4",
            'Properties': len(subset),
            'Units': int(subset['TotalUnits'].sum()) if 'TotalUnits' in subset.columns else 0,
            'Pct_Total': f"{len(subset) / len(df) * 100:.1f}%",
            'Most_Common_Combination': common_combo,
            'Top_States': top_states_str,
            'Mean_GNN_Risk': f"{subset['GNN_Risk'].mean()*100:.0f}%" if 'GNN_Risk' in subset.columns else 'N/A',
            'Mean_Rent_Gap': f"${subset['True_Rent_Gap'].mean():.0f}" if 'True_Rent_Gap' in subset.columns else 'N/A',
            'Mean_Age': f"{subset['PropertyAge'].mean():.0f}" if 'PropertyAge' in subset.columns else 'N/A',
        }
        
        summary_rows.append(row_data)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('Risk_Factors', ascending=False)
    
    print("\n" + "="*80)
    print("TABLE: PROPERTIES BY NUMBER OF COMPOUNDING RISK FACTORS")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # Save
    summary_df.to_csv(f"{CONFIG['output_dir']}/compounded_risk_table.csv", index=False)
    print(f"\nSaved to {CONFIG['output_dir']}/compounded_risk_table.csv")
    
    return summary_df


def analyze_highest_risk_properties(df):
    """
    Detailed analysis of properties with ALL 4 risk factors.
    This is your TIER 1 CRISIS group.
    """
    print("\n" + "="*80)
    print("PROPERTIES WITH ALL 4 RISK FACTORS (HIGHEST PRIORITY)")
    print("="*80)
    
    all_risks = df[df['NumRiskFactors'] == 4].copy()
    
    print(f"\nTotal properties with all 4 risk factors: {len(all_risks):,}")
    
    if len(all_risks) == 0:
        print("No properties have all 4 risk factors.")
        return None
    
    total_units = all_risks['TotalUnits'].sum() if 'TotalUnits' in all_risks.columns else 0
    print(f"Total units affected: {int(total_units):,}")
    
    # Geographic distribution
    print("\n--- TOP 10 STATES ---")
    if 'State' in all_risks.columns:
        state_counts = all_risks.groupby('State').agg(
            Properties=('NHPDPropertyID', 'count'),
            Units=('TotalUnits', 'sum')
        ).reset_index()
        state_counts['Units'] = state_counts['Units'].fillna(0).astype(int)
        state_counts = state_counts.sort_values('Properties', ascending=False)
        print(state_counts.head(10).to_string(index=False))
        
        top5_props = state_counts.head(5)['Properties'].sum()
        print(f"\nTop 5 states account for: {top5_props / len(all_risks) * 100:.0f}% of all 4-risk properties")
    
    # City distribution
    print("\n--- TOP 10 CITIES ---")
    if 'City' in all_risks.columns and 'State' in all_risks.columns:
        city_counts = all_risks.groupby(['City', 'State']).agg(
            Properties=('NHPDPropertyID', 'count'),
            Units=('TotalUnits', 'sum')
        ).reset_index()
        city_counts['Units'] = city_counts['Units'].fillna(0).astype(int)
        city_counts = city_counts.sort_values('Properties', ascending=False)
        print(city_counts.head(10).to_string(index=False))
    
    # Characteristics
    print("\n--- CHARACTERISTICS OF 4-RISK PROPERTIES ---")
    characteristics = {
        'Mean Property Age': f"{all_risks['PropertyAge'].mean():.0f} years" if 'PropertyAge' in all_risks.columns else 'N/A',
        'Mean Total Units': f"{all_risks['TotalUnits'].mean():.0f}" if 'TotalUnits' in all_risks.columns else 'N/A',
        'Mean Distress Risk': f"{all_risks['GNN_Risk'].mean()*100:.0f}%" if 'GNN_Risk' in all_risks.columns else 'N/A',
        'Mean Rent Gap': f"${all_risks['True_Rent_Gap'].mean():.0f}/month" if 'True_Rent_Gap' in all_risks.columns else 'N/A',
        'Mean Poverty Rate': f"{all_risks['Poverty_Rate'].mean():.1f}%" if 'Poverty_Rate' in all_risks.columns else 'N/A',
        'Mean Occupancy': f"{all_risks['OccupancyRate'].mean():.0f}%" if 'OccupancyRate' in all_risks.columns else 'N/A',
    }
    
    for key, val in characteristics.items():
        print(f"  {key}: {val}")
    
    # Program participation
    print("\n--- PROGRAM PARTICIPATION ---")
    programs = ['Has_Section8', 'Has_LIHTC', 'Has_PublicHousing', 'Has_Section515', 'Has_Section202']
    for prog in programs:
        if prog in all_risks.columns:
            n = all_risks[prog].sum()
            pct = n / len(all_risks) * 100
            print(f"  {prog.replace('Has_', '')}: {n} ({pct:.0f}%)")
    
    # Example properties
    print("\n--- EXAMPLE PROPERTIES (Top 10 by Risk) ---")
    examples = all_risks.nlargest(10, 'GNN_Risk') if 'GNN_Risk' in all_risks.columns else all_risks.head(10)
    
    for i, (_, row) in enumerate(examples.iterrows(), 1):
        print(f"\n{i}. {row.get('PropertyName', 'N/A')}")
        print(f"   Location: {row.get('City', 'N/A')}, {row.get('State', 'N/A')}")
        print(f"   Units: {int(row.get('TotalUnits', 0))}, Age: {row.get('PropertyAge', 'N/A'):.0f} yrs")
        print(f"   Risk: {row.get('GNN_Risk', 0)*100:.0f}%, Rent Gap: ${row.get('True_Rent_Gap', 0):.0f}/mo")
    
    # Save detailed list
    all_risks.to_csv(f"{CONFIG['output_dir']}/all_4_risk_properties.csv", index=False)
    print(f"\nSaved detailed list to {CONFIG['output_dir']}/all_4_risk_properties.csv")
    
    return all_risks


def create_visualization_data(df):
    """
    Create data for visualizations showing risk factor overlap.
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATION DATA")
    print("="*80)
    
    # Risk factor co-occurrence matrix
    risk_cols = ['RF1_HighDistress', 'RF2_HighRentGap', 'RF3_NearExpiration', 'RF4_OldAge']
    
    cooccurrence = pd.DataFrame(index=risk_cols, columns=risk_cols)
    
    for rf1 in risk_cols:
        for rf2 in risk_cols:
            if rf1 == rf2:
                cooccurrence.loc[rf1, rf2] = df[rf1].sum()
            else:
                cooccurrence.loc[rf1, rf2] = ((df[rf1] == 1) & (df[rf2] == 1)).sum()
    
    print("\nRisk Factor Co-occurrence Matrix:")
    print(cooccurrence)
    
    cooccurrence.to_csv(f"{CONFIG['output_dir']}/risk_cooccurrence_matrix.csv")
    
    # Risk progression (for each additional risk factor)
    progression = []
    for n in range(5):
        subset = df[df['NumRiskFactors'] == n]
        if len(subset) > 0:
            progression.append({
                'NumRisks': n,
                'Count': len(subset),
                'Mean_Poverty': subset['Poverty_Rate'].mean() if 'Poverty_Rate' in subset.columns else None,
                'Mean_Vacancy': subset['Vacancy_Rate'].mean() if 'Vacancy_Rate' in subset.columns else None,
                'Mean_Age': subset['PropertyAge'].mean() if 'PropertyAge' in subset.columns else None,
            })
    
    prog_df = pd.DataFrame(progression)
    print("\nRisk Progression:")
    print(prog_df.to_string(index=False))
    
    prog_df.to_csv(f"{CONFIG['output_dir']}/risk_progression.csv", index=False)
    
    print(f"\nVisualization data saved.")


def generate_paper_text(df, summary_df):
    """
    Generate text snippets for the paper.
    """
    print("\n" + "="*80)
    print("TEXT FOR PAPER")
    print("="*80)
    
    # Overall stats
    total = len(df)
    
    # Get counts
    four_risk = len(df[df['NumRiskFactors'] == 4])
    three_risk = len(df[df['NumRiskFactors'] == 3])
    two_risk = len(df[df['NumRiskFactors'] == 2])
    one_risk = len(df[df['NumRiskFactors'] == 1])
    zero_risk = len(df[df['NumRiskFactors'] == 0])
    
    # Units
    four_risk_units = int(df[df['NumRiskFactors'] == 4]['TotalUnits'].sum()) if 'TotalUnits' in df.columns else 0
    
    print(f"""
SECTION 4.X: COMPOUNDED RISK ANALYSIS

Our multi-dimensional framework reveals that preservation risk is not uniformly 
distributed but concentrates in properties facing multiple, compounding threats.
Table X presents properties classified by the number of risk factors present.

[Insert Table X here]

The analysis identifies {four_risk:,} properties ({four_risk/total*100:.1f}% of the 
national inventory, containing approximately {four_risk_units:,} units) exhibiting all 
four risk factors: elevated physical distress probability, high rent gap, near-term 
subsidy expiration, and advanced property age. These properties represent the most 
acute preservation challenge, requiring immediate and intensive intervention.

An additional {three_risk:,} properties ({three_risk/total*100:.1f}%) face three of 
four risk factors, while {two_risk:,} properties ({two_risk/total*100:.1f}%) face two 
factors. Conversely, {zero_risk:,} properties ({zero_risk/total*100:.1f}%) show no 
elevated risk on any dimension, suggesting they can be monitored through routine 
oversight rather than intensive intervention.

Geographic Concentration:
The {four_risk:,} properties with all four risk factors are heavily concentrated 
geographically. [Insert top 5 states data here]. This concentration suggests that 
targeted federal-state partnerships in these high-burden states could reach a majority 
of the highest-priority properties.

Policy Implications:
Properties exhibiting multiple compounding risks require holistic interventions that 
address both physical deterioration (through rehabilitation financing) and economic 
pressure (through subsidy renewal incentives, acquisition by mission-driven entities, 
or enhanced operating support). Single-dimension interventions—for example, providing 
rehabilitation funding to a property with a high rent gap but no mechanism to ensure 
continued affordability—may prove ineffective if owners exit programs shortly after 
receiving capital investment.
""")


def main():
    print("="*80)
    print("COMPOUNDED RISK ANALYSIS")
    print("Integration Table for Data & Policy Paper")
    print("="*80)
    
    # Load data
    df = load_data()
    
    # Define risk factors
    df = define_risk_factors(df)
    
    # Create integration table
    summary_df = create_compounded_risk_table(df)
    
    # Analyze highest-risk properties
    analyze_highest_risk_properties(df)
    
    # Create visualization data
    create_visualization_data(df)
    
    # Generate paper text
    generate_paper_text(df, summary_df)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"""
Generated files in {CONFIG['output_dir']}:
  - compounded_risk_table.csv (THE KEY TABLE)
  - all_4_risk_properties.csv (Detailed list)
  - risk_cooccurrence_matrix.csv
  - risk_progression.csv

Use compounded_risk_table.csv as Table X in your paper.
""")


if __name__ == "__main__":
    main()