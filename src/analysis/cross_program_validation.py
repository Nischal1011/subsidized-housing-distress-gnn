"""
cross_program_validation.py
============================
Address Reviewer Concern #4: Does the model generalize across program types?

The concern: REAC scores are mostly available for HUD programs (Section 8, Public Housing).
Can a model trained on these generalize to state-administered programs (LIHTC, HOME)?

Approach:
1. Train GAT/XGBoost on Section 8 properties with REAC scores only
2. Test on LIHTC properties with REAC scores
3. Compare performance to see if model generalizes

If performance holds → model is robust across program types
If performance drops → acknowledge limitation for LIHTC predictions
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    classification_report, confusion_matrix
)
import warnings
import os

warnings.filterwarnings('ignore')

CONFIG = {
    'input_file': 'data/merged_analysis_data.parquet',
    'output_dir': 'outputs/',
    'random_state': 42,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)


def create_features(df):
    """Feature engineering - same as main pipeline."""
    df = df.copy()
    
    # Economic features
    if 'Median_Household_Income' in df.columns and 'Median_Gross_Rent' in df.columns:
        inc = df['Median_Household_Income'].fillna(df['Median_Household_Income'].median())
        mkt_rent = df['Median_Gross_Rent'].fillna(df['Median_Gross_Rent'].median())
        restricted_rent = (inc * 0.60 * 0.30) / 12
        df['Rent_Gap'] = mkt_rent - restricted_rent
        df['Log_Rent_Gap'] = np.log1p(df['Rent_Gap'] - df['Rent_Gap'].min() + 1)
        df['High_Exit_Incentive'] = (df['Rent_Gap'] > 400).astype(float)

    # Property age
    if 'PropertyAge' in df.columns:
        age = df['PropertyAge'].fillna(df['PropertyAge'].median()).fillna(30)
        df['Log_Age'] = np.log1p(age)
        df['Age_Squared'] = age ** 2 / 1000
        df['Is_Old'] = (age > 40).astype(float)

    # Units
    if 'TotalUnits' in df.columns:
        units = df['TotalUnits'].fillna(df['TotalUnits'].median()).fillna(50)
        df['Log_Units'] = np.log1p(units)
        df['Is_Large'] = (units > 100).astype(float)

    # Neighborhood
    if 'Poverty_Rate' in df.columns:
        pov = df['Poverty_Rate'].fillna(df['Poverty_Rate'].median()).fillna(15)
        df['High_Poverty'] = (pov > 25).astype(float)

    if 'Vacancy_Rate' in df.columns:
        vac = df['Vacancy_Rate'].fillna(df['Vacancy_Rate'].median()).fillna(5)
        df['High_Vacancy'] = (vac > 10).astype(float)

    # Owner portfolio
    if 'Owner' in df.columns:
        df['Owner_Clean'] = df['Owner'].fillna('UNKNOWN').astype(str).str.upper().str.strip()
        owner_counts = df['Owner_Clean'].value_counts()
        df['Owner_Portfolio_Size'] = df['Owner_Clean'].map(owner_counts).fillna(1)
        df['Log_Portfolio'] = np.log1p(df['Owner_Portfolio_Size'])

    return df


def evaluate_model(name, y_true, y_probs, threshold=0.5):
    """Evaluate model performance."""
    y_pred = (y_probs >= threshold).astype(int)
    
    return {
        'Model': name,
        'AUC': roc_auc_score(y_true, y_probs),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'N': len(y_true),
        'Distress_Rate': y_true.mean()
    }


def main():
    print("="*70)
    print("CROSS-PROGRAM VALIDATION")
    print("Training on Section 8, Testing on LIHTC")
    print("="*70)
    
    # Load data
    if not os.path.exists(CONFIG['input_file']):
        print(f"Error: {CONFIG['input_file']} not found.")
        return
    
    df = pd.read_parquet(CONFIG['input_file']).reset_index(drop=True)
    print(f"\nLoaded {len(df):,} properties.")
    
    # Feature engineering
    df = create_features(df)
    
    # Identify program participation
    # Create binary flags if they don't exist
    if 'Has_Section8' not in df.columns:
        df['Has_Section8'] = (df.get('NumberActiveSection8', 0) > 0).astype(int)
    if 'Has_LIHTC' not in df.columns:
        df['Has_LIHTC'] = (df.get('NumberActiveLihtc', 0) > 0).astype(int)
    if 'Has_PublicHousing' not in df.columns:
        df['Has_PublicHousing'] = (df.get('NumberActivePublicHousing', 0) > 0).astype(int)
    if 'Has_Section515' not in df.columns:
        df['Has_Section515'] = (df.get('NumberActiveSection515', 0) > 0).astype(int)
    
    # Get labeled properties (have REAC scores)
    df['HasLabel'] = df['PhysicalDistress'].notna()
    labeled_df = df[df['HasLabel']].copy()
    
    print(f"\nLabeled properties (with REAC): {len(labeled_df):,}")
    
    # =========================================================================
    # ANALYZE PROGRAM OVERLAP IN LABELED DATA
    # =========================================================================
    print("\n" + "="*70)
    print("PROGRAM DISTRIBUTION IN LABELED DATA")
    print("="*70)
    
    programs = ['Has_Section8', 'Has_LIHTC', 'Has_PublicHousing', 'Has_Section515']
    
    for prog in programs:
        if prog in labeled_df.columns:
            n = labeled_df[prog].sum()
            pct = n / len(labeled_df) * 100
            distress_rate = labeled_df[labeled_df[prog] == 1]['PhysicalDistress'].mean() * 100
            print(f"  {prog}: {n:,} ({pct:.1f}%) - Distress rate: {distress_rate:.1f}%")
    
    # Check overlap
    section8_only = labeled_df[(labeled_df['Has_Section8'] == 1) & (labeled_df['Has_LIHTC'] == 0)]
    lihtc_only = labeled_df[(labeled_df['Has_LIHTC'] == 1) & (labeled_df['Has_Section8'] == 0)]
    both = labeled_df[(labeled_df['Has_Section8'] == 1) & (labeled_df['Has_LIHTC'] == 1)]
    
    print(f"\n  Section 8 only: {len(section8_only):,}")
    print(f"  LIHTC only: {len(lihtc_only):,}")
    print(f"  Both Section 8 + LIHTC: {len(both):,}")
    
    # =========================================================================
    # PREPARE FEATURES
    # =========================================================================
    print("\n" + "="*70)
    print("PREPARING FEATURES")
    print("="*70)
    
    # Select features (exclude labels and identifiers)
    exclude_cols = [
        'PhysicalDistress', 'BestReacScore', 'ReacScore1', 'ReacScore2', 'ReacScore3',
        'NHPDPropertyID', 'Latitude', 'Longitude', 'CensusTract', 'PropertyName',
        'PropertyAddress', 'Owner', 'Owner_Clean', 'City', 'State', 'Zip',
        'Policy_Tier', 'GNN_Risk', 'GNN_HighRisk', 'HasLabel',
        'Has_Section8', 'Has_LIHTC', 'Has_PublicHousing', 'Has_Section515', 'Has_Section202', 'Has_HOME'
    ]
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols if c not in exclude_cols]
    print(f"Using {len(feature_cols)} features.")
    
    # =========================================================================
    # EXPERIMENT 1: Train on Section 8 Only, Test on LIHTC Only
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 1: Train on Section 8 Only → Test on LIHTC Only")
    print("="*70)
    
    # Prepare train/test sets
    train_df = section8_only.copy()
    test_df = lihtc_only.copy()
    
    print(f"\nTraining set (Section 8 only): {len(train_df):,} properties")
    print(f"  Distress rate: {train_df['PhysicalDistress'].mean()*100:.1f}%")
    print(f"\nTest set (LIHTC only): {len(test_df):,} properties")
    print(f"  Distress rate: {test_df['PhysicalDistress'].mean()*100:.1f}%")
    
    if len(train_df) < 100 or len(test_df) < 50:
        print("\nWARNING: Insufficient data for reliable cross-program validation.")
        print("This is itself an important finding - LIHTC properties rarely have REAC scores.")
    
    # Prepare features
    X_train = train_df[feature_cols].fillna(0).values
    X_test = test_df[feature_cols].fillna(0).values
    y_train = train_df['PhysicalDistress'].values.astype(int)
    y_test = test_df['PhysicalDistress'].values.astype(int)
    
    # Scale
    qt = QuantileTransformer(output_distribution='normal', random_state=CONFIG['random_state'])
    X_train_scaled = qt.fit_transform(X_train)
    X_test_scaled = qt.transform(X_test)
    
    results = []
    
    # Train Logistic Regression
    print("\n--- Logistic Regression ---")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=CONFIG['random_state'])
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    lr_result = evaluate_model("Logistic Regression", y_test, lr_probs)
    results.append(lr_result)
    print(f"  AUC: {lr_result['AUC']:.4f}")
    print(f"  Recall: {lr_result['Recall']:.4f}")
    print(f"  F1: {lr_result['F1']:.4f}")
    
    # Train XGBoost
    print("\n--- XGBoost ---")
    xgb = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        min_samples_split=20, min_samples_leaf=10,
        random_state=CONFIG['random_state']
    )
    xgb.fit(X_train_scaled, y_train)
    xgb_probs = xgb.predict_proba(X_test_scaled)[:, 1]
    xgb_result = evaluate_model("XGBoost", y_test, xgb_probs)
    results.append(xgb_result)
    print(f"  AUC: {xgb_result['AUC']:.4f}")
    print(f"  Recall: {xgb_result['Recall']:.4f}")
    print(f"  F1: {xgb_result['F1']:.4f}")
    
    # =========================================================================
    # EXPERIMENT 2: Compare to Same-Program Validation
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 2: Same-Program Validation (Section 8 → Section 8)")
    print("="*70)
    
    from sklearn.model_selection import train_test_split
    
    # Split Section 8 only data
    s8_train, s8_test = train_test_split(
        section8_only, test_size=0.2, random_state=CONFIG['random_state'],
        stratify=section8_only['PhysicalDistress']
    )
    
    print(f"\nTraining set (Section 8): {len(s8_train):,}")
    print(f"Test set (Section 8): {len(s8_test):,}")
    
    X_s8_train = qt.fit_transform(s8_train[feature_cols].fillna(0).values)
    X_s8_test = qt.transform(s8_test[feature_cols].fillna(0).values)
    y_s8_train = s8_train['PhysicalDistress'].values.astype(int)
    y_s8_test = s8_test['PhysicalDistress'].values.astype(int)
    
    # XGBoost on same-program
    xgb_same = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=CONFIG['random_state']
    )
    xgb_same.fit(X_s8_train, y_s8_train)
    xgb_same_probs = xgb_same.predict_proba(X_s8_test)[:, 1]
    xgb_same_result = evaluate_model("XGBoost (Same-Program)", y_s8_test, xgb_same_probs)
    
    print(f"\nXGBoost Same-Program (Section 8 → Section 8):")
    print(f"  AUC: {xgb_same_result['AUC']:.4f}")
    print(f"  Recall: {xgb_same_result['Recall']:.4f}")
    print(f"  F1: {xgb_same_result['F1']:.4f}")
    
    # =========================================================================
    # SUMMARY COMPARISON
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY: CROSS-PROGRAM VS SAME-PROGRAM PERFORMANCE")
    print("="*70)
    
    print(f"""
                              | Section 8 → Section 8 | Section 8 → LIHTC | Δ AUC
    --------------------------|----------------------|-------------------|-------
    XGBoost AUC               | {xgb_same_result['AUC']:.3f}                | {xgb_result['AUC']:.3f}             | {xgb_result['AUC'] - xgb_same_result['AUC']:+.3f}
    XGBoost Recall            | {xgb_same_result['Recall']:.3f}                | {xgb_result['Recall']:.3f}             |
    XGBoost F1                | {xgb_same_result['F1']:.3f}                | {xgb_result['F1']:.3f}             |
    """)
    
    # Interpretation
    auc_drop = xgb_same_result['AUC'] - xgb_result['AUC']
    
    print("\n" + "="*70)
    print("INTERPRETATION FOR PAPER")
    print("="*70)
    
    if auc_drop < 0.05:
        print(f"""
FINDING: Model generalizes well across program types (AUC drop: {auc_drop:.3f})

Suggested text for paper:
"To assess generalizability across program types, we trained models exclusively 
on Section 8 properties with REAC inspections and evaluated on LIHTC properties 
with REAC scores. Despite the structural differences between these programs—
Section 8 involves project-based rental assistance while LIHTC provides tax 
credits for affordable housing development—the model maintained comparable 
predictive performance (AUC: {xgb_result['AUC']:.3f} vs. {xgb_same_result['AUC']:.3f} for within-program validation). 
This suggests that the physical distress indicators learned from HUD-inspected 
properties transfer to state-administered LIHTC developments."
        """)
    else:
        print(f"""
FINDING: Model performance drops when generalizing to LIHTC (AUC drop: {auc_drop:.3f})

Suggested text for paper:
"Cross-program validation reveals important limitations. Models trained on 
Section 8 properties showed reduced performance when predicting distress in 
LIHTC properties (AUC: {xgb_result['AUC']:.3f} vs. {xgb_same_result['AUC']:.3f} for within-program validation). 
This {auc_drop:.1%} AUC decline suggests that physical distress patterns differ 
across program types, or that LIHTC properties with REAC scores are not 
representative of the broader LIHTC stock. Predictions for LIHTC properties 
should therefore be interpreted as targeting candidates for inspection rather 
than definitive distress diagnoses."
        """)
    
    # Additional context
    print(f"""
CRITICAL CONTEXT:
- Only {len(lihtc_only):,} LIHTC-only properties have REAC scores ({len(lihtc_only)/len(labeled_df)*100:.1f}% of labeled data)
- This small sample limits the reliability of cross-program validation
- The low LIHTC/REAC overlap itself motivates the imputation approach
    """)
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results_df = pd.DataFrame(results)
    results_df['Experiment'] = 'Section8 → LIHTC'
    
    same_program_df = pd.DataFrame([xgb_same_result])
    same_program_df['Experiment'] = 'Section8 → Section8'
    
    all_results = pd.concat([results_df, same_program_df], ignore_index=True)
    all_results.to_csv(f"{CONFIG['output_dir']}/cross_program_validation.csv", index=False)
    
    print(f"\nResults saved to {CONFIG['output_dir']}/cross_program_validation.csv")
    
    print("\n" + "="*70)
    print("CROSS-PROGRAM VALIDATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()