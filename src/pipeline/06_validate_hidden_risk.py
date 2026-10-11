"""
06_validate_hidden_risk.py
==========================
Sanity check the "Hidden Risk" narrative.
1. Compare Base Rate (Labeled) vs. Predicted Rate (Unlabeled).
2. Compare characteristics (Age, Rent Gap) to explain WHY the unlabeled rate is higher.
"""
import pandas as pd
import numpy as np
from scipy import stats

# Load data
df = pd.read_parquet('data/merged_analysis_data.parquet')
results = pd.read_csv('outputs/integrated_policy_matrix.csv')

# Merge predictions back if needed, or assume df has 'PhysicalDistress' labels
# We need to distinguish Labeled vs Unlabeled
df['Has_Label'] = df['PhysicalDistress'].notna()
df = df.merge(results[['NHPDPropertyID', 'GNN_HighRisk', 'Policy_Tier']], on='NHPDPropertyID', how='left')

print("="*60)
print("HYPOTHESIS TEST: THE 'HIDDEN RISK' NARRATIVE")
print("="*60)

# 1. Base Rate Analysis
labeled = df[df['Has_Label']]
unlabeled = df[~df['Has_Label']]

base_rate = labeled['PhysicalDistress'].mean()
pred_rate = unlabeled['GNN_HighRisk'].mean()

print(f"\n[1] Distress Rates")
print(f"   Labeled Stock (n={len(labeled):,}):     {base_rate:.1%} (Ground Truth)")
print(f"   Unlabeled Stock (n={len(unlabeled):,}):   {pred_rate:.1%} (Predicted)")
print(f"   Ratio (Predicted / Base):        {pred_rate/base_rate:.1f}x")

if pred_rate > base_rate * 1.5:
    print("   -> RESULT: Significant 'Hidden Risk' detected. Requires feature justification.")
elif pred_rate < base_rate:
    print("   -> RESULT: Unlabeled stock appears safer. Narrative change needed.")
else:
    print("   -> RESULT: Rates are comparable. 'Hidden Risk' is mild.")

# 2. Why is the Unlabeled stock riskier? (Feature Comparison)
# We hypothesize Unlabeled properties are OLDER or have HIGHER RENT GAPS.
print(f"\n[2] Feature Comparison (Why is the rate higher?)")

features_to_check = {
    'PropertyAge': 'Property Age (Years)',
    'True_Rent_Gap': 'True Rent Gap ($)',
    'Poverty_Rate': 'Neighborhood Poverty (%)',
    'TotalUnits': 'Property Size (Units)'
}

print(f"{'Feature':<25} | {'Labeled (Avg)':<15} | {'Unlabeled (Avg)':<15} | {'Diff':<10}")
print("-" * 75)

for col, name in features_to_check.items():
    if col in df.columns:
        val_labeled = labeled[col].median()
        val_unlabeled = unlabeled[col].median()
        diff = ((val_unlabeled - val_labeled) / val_labeled) * 100
        
        print(f"{name:<25} | {val_labeled:<15.1f} | {val_unlabeled:<15.1f} | {diff:+.1f}%")

# 3. Generate the Paper's Calibration Table
print(f"\n[3] Calibration Table for Paper")
print("-" * 60)
print("Subset                | N       | Distress Rate")
print("-" * 60)
print(f"Labeled (Inspected)   | {len(labeled):<7,} | {base_rate:.1%}")
print(f"Unlabeled (Predicted) | {len(unlabeled):<7,} | {pred_rate:.1%} (Flagged)")
print("-" * 60)