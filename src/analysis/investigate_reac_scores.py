"""
investigate_reac_scores.py
==========================
Investigate why we're only getting 6,724 REAC scores when 30,572 should exist.
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("INVESTIGATING REAC SCORE AVAILABILITY")
print("=" * 70)

# Load NHPD
nhpd = pd.read_excel('data/Active and Inconclusive Properties.xlsx')
print(f"\nTotal properties: {len(nhpd):,}")

# Check each REAC score column
print("\n--- REAC Score Column Availability ---")
reac_cols = ['ReacScore1', 'ReacScore2', 'ReacScore3']

for col in reac_cols:
    if col in nhpd.columns:
        # Convert to numeric
        nhpd[col] = pd.to_numeric(nhpd[col], errors='coerce')
        n_valid = nhpd[col].notna().sum()
        pct = n_valid / len(nhpd) * 100
        print(f"{col}: {n_valid:,} valid ({pct:.1f}%)")
        if n_valid > 0:
            print(f"  Mean: {nhpd[col].mean():.1f}, Median: {nhpd[col].median():.1f}")

# Check for properties with ANY REAC score
print("\n--- Properties with ANY REAC Score ---")
has_reac1 = nhpd['ReacScore1'].notna()
has_reac2 = nhpd['ReacScore2'].notna()
has_reac3 = nhpd['ReacScore3'].notna()

has_any_reac = has_reac1 | has_reac2 | has_reac3
print(f"Has ReacScore1: {has_reac1.sum():,}")
print(f"Has ReacScore2: {has_reac2.sum():,}")
print(f"Has ReacScore3: {has_reac3.sum():,}")
print(f"Has ANY REAC score: {has_any_reac.sum():,}")

# Properties with ReacScore2 or 3 but NOT ReacScore1
has_2_not_1 = has_reac2 & ~has_reac1
has_3_not_1 = has_reac3 & ~has_reac1
print(f"\nHas ReacScore2 but NOT ReacScore1: {has_2_not_1.sum():,}")
print(f"Has ReacScore3 but NOT ReacScore1: {has_3_not_1.sum():,}")

# Create "best available" REAC score (most recent first)
print("\n--- Creating Best Available REAC Score ---")
nhpd['BestReacScore'] = nhpd['ReacScore1'].fillna(nhpd['ReacScore2']).fillna(nhpd['ReacScore3'])
n_best = nhpd['BestReacScore'].notna().sum()
print(f"Properties with BestReacScore: {n_best:,} ({n_best/len(nhpd)*100:.1f}%)")

if n_best > 0:
    print(f"  Mean: {nhpd['BestReacScore'].mean():.1f}")
    print(f"  Median: {nhpd['BestReacScore'].median():.1f}")

# Check the REAC date columns to understand recency
print("\n--- REAC Score Dates ---")
date_cols = ['ReacScore1Date', 'ReacScore2Date', 'ReacScore3Date']
for col in date_cols:
    if col in nhpd.columns:
        nhpd[col] = pd.to_datetime(nhpd[col], errors='coerce')
        n_valid = nhpd[col].notna().sum()
        if n_valid > 0:
            print(f"{col}: {n_valid:,} valid")
            print(f"  Range: {nhpd[col].min()} to {nhpd[col].max()}")

# Distress rate with best score
print("\n--- Distress Rate with Best Available Score ---")
threshold = 70
distressed = (nhpd['BestReacScore'] < threshold).sum()
total_with_score = nhpd['BestReacScore'].notna().sum()
print(f"Distressed (REAC < {threshold}): {distressed:,}")
print(f"Distress rate: {distressed/total_with_score*100:.1f}%")

# Distribution of best scores
print("\n--- Distribution of Best REAC Scores ---")
print(nhpd['BestReacScore'].describe())

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print("""
If BestReacScore gives significantly more properties than ReacScore1 alone,
update 01_data_preparation.py to use:

    # Use best available REAC score (most recent available)
    df['ReacScore1'] = pd.to_numeric(df['ReacScore1'], errors='coerce')
    df['ReacScore2'] = pd.to_numeric(df['ReacScore2'], errors='coerce')
    df['ReacScore3'] = pd.to_numeric(df['ReacScore3'], errors='coerce')
    
    df['BestReacScore'] = df['ReacScore1'].fillna(df['ReacScore2']).fillna(df['ReacScore3'])
    
    # Then use BestReacScore for outcome
    df['HasREAC'] = df['BestReacScore'].notna()
""")