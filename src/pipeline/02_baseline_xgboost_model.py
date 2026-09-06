"""
02_model_training.py
====================
Train XGBoost model to predict physical distress and interpret with SHAP.

This script:
1. Loads the merged analysis data
2. Prepares features and handles missing values
3. Trains XGBoost with hyperparameter tuning
4. Evaluates on held-out test set
5. Generates SHAP explanations
6. Scores uninspected properties
7. Saves all outputs

Author: Nischal Subedi
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, f1_score, recall_score,
    precision_score, average_precision_score
)
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Input/output paths
    'input_file': 'data/merged_analysis_data.parquet',
    'output_dir': 'outputs/',
    'model_file': 'outputs/xgboost_model.joblib',
    'scaler_file': 'outputs/scaler.joblib',
    
    # Model parameters
    'test_size': 0.3,
    'random_state': 42,
    'cv_folds': 5,
    
    # Feature lists
    'property_features': [
        'PropertyAge',
        'TotalUnits',
        'FairMarketRent_2BR',
        'PercentStudioOneBedroom',
        'NumActivePrograms',
        'Has_Section8',
        'Has_LIHTC',
        'Has_Section515',
        'Has_PublicHousing',
        'Has_Section202',
        'IsMetro',
        'MonthsToExpire',
    ],
    
    'neighborhood_features': [
        'Poverty_Rate',
        'Vacancy_Rate',
        'Renter_Share',
        'Elderly_Share',
        'Disability_Rate',
        'Pre1970_Housing_Share',
        'Pop_Density',
        'Median_Household_Income',
        'Median_Gross_Rent',
    ],
    
    # Owner type will be one-hot encoded
    'categorical_features': ['OwnerType_Clean'],
}

# Create output directory
os.makedirs(CONFIG['output_dir'], exist_ok=True)


# =============================================================================
# STEP 1: LOAD AND PREPARE DATA FOR MODELING
# =============================================================================

def load_and_prepare_features(filepath):
    """
    Load merged data and prepare feature matrix.
    """
    print("=" * 70)
    print("STEP 1: Loading and preparing features")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading: {filepath}")
    df = pd.read_parquet(filepath)
    print(f"Loaded {len(df):,} properties")
    
    # Identify modeling sample (properties with REAC scores)
    has_outcome = df['PhysicalDistress'].notna()
    print(f"\nProperties with outcome (REAC score): {has_outcome.sum():,}")
    
    # Separate modeling and scoring samples
    df_model = df[has_outcome].copy()
    df_score = df[~has_outcome].copy()
    
    print(f"Modeling sample: {len(df_model):,}")
    print(f"Scoring sample (no REAC): {len(df_score):,}")
    
    return df, df_model, df_score


def prepare_feature_matrix(df, feature_config):
    """
    Create feature matrix X and target vector y.
    """
    print("\n" + "-" * 50)
    print("Preparing feature matrix...")
    
    # Combine feature lists
    numeric_features = feature_config['property_features'] + feature_config['neighborhood_features']
    
    # Filter to available features
    available_numeric = [f for f in numeric_features if f in df.columns]
    missing_numeric = [f for f in numeric_features if f not in df.columns]
    
    if missing_numeric:
        print(f"  WARNING: Missing features: {missing_numeric}")
    
    print(f"  Using {len(available_numeric)} numeric features")
    
    # Start with numeric features
    X = df[available_numeric].copy()
    
    # One-hot encode categorical features
    for cat_col in feature_config['categorical_features']:
        if cat_col in df.columns:
            print(f"  One-hot encoding: {cat_col}")
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col.replace('_Clean', ''))
            # Drop reference category to avoid multicollinearity
            if f'{cat_col.replace("_Clean", "")}_Unknown' in dummies.columns:
                dummies = dummies.drop(f'{cat_col.replace("_Clean", "")}_Unknown', axis=1)
            elif len(dummies.columns) > 0:
                dummies = dummies.iloc[:, 1:]  # Drop first column
            X = pd.concat([X, dummies], axis=1)
    
    # Get target
    y = df['PhysicalDistress'].values if 'PhysicalDistress' in df.columns else None
    
    print(f"  Final feature matrix shape: {X.shape}")
    
    # Feature names for later interpretation
    feature_names = X.columns.tolist()
    
    return X, y, feature_names


def handle_missing_values(X_train, X_test, feature_names):
    """
    Handle missing values in features.
    Uses median imputation for numeric features.
    """
    print("\n" + "-" * 50)
    print("Handling missing values...")
    
    # Calculate medians from training data
    medians = {}
    for col in feature_names:
        if X_train[col].isna().any():
            medians[col] = X_train[col].median()
            n_missing_train = X_train[col].isna().sum()
            n_missing_test = X_test[col].isna().sum()
            print(f"  {col}: {n_missing_train} missing in train, {n_missing_test} in test")
    
    # Apply median imputation
    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()
    
    for col, median_val in medians.items():
        X_train_imputed[col] = X_train_imputed[col].fillna(median_val)
        X_test_imputed[col] = X_test_imputed[col].fillna(median_val)
    
    print(f"  Imputed {len(medians)} features")
    
    return X_train_imputed, X_test_imputed, medians


# =============================================================================
# STEP 2: TRAIN XGBOOST MODEL
# =============================================================================

def train_xgboost(X_train, y_train, random_state=42, cv_folds=5):
    """
    Train XGBoost classifier with hyperparameter tuning.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Training XGBoost model")
    print("=" * 70)
    
    # Class balance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos
    
    print(f"\nClass distribution:")
    print(f"  Not Distressed (0): {n_neg:,} ({n_neg/len(y_train)*100:.1f}%)")
    print(f"  Distressed (1): {n_pos:,.0f} ({n_pos/len(y_train)*100:.1f}%)")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Define hyperparameter search space
    param_dist = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [1, 1.5, 2],
    }
    
    # Base model
    base_model = XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='auc',
        n_jobs=-1,
    )
    
    # Randomized search
    print("\nPerforming hyperparameter tuning (this may take a few minutes)...")
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=50,  # Number of random combinations to try
        scoring='roc_auc',
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    
    search.fit(X_train, y_train)
    
    print(f"\nBest cross-validation AUC: {search.best_score_:.4f}")
    print(f"Best parameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    
    # Get best model
    best_model = search.best_estimator_
    
    # Cross-validation scores with best model
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"\nCross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    return best_model


def train_xgboost_simple(X_train, y_train, random_state=42):
    """
    Train XGBoost with reasonable default parameters (faster, for testing).
    """
    print("\n" + "=" * 70)
    print("STEP 2: Training XGBoost model (simple mode)")
    print("=" * 70)
    
    # Class balance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos
    
    print(f"\nClass distribution:")
    print(f"  Not Distressed (0): {n_neg:,} ({n_neg/len(y_train)*100:.1f}%)")
    print(f"  Distressed (1): {n_pos:,.0f} ({n_pos/len(y_train)*100:.1f}%)")
    
    # Model with reasonable defaults
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.01,
        reg_lambda=1.5,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='auc',
        n_jobs=-1,
    )
    
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    return model


# =============================================================================
# STEP 3: EVALUATE MODEL
# =============================================================================

def evaluate_model(model, X_test, y_test, feature_names, output_dir):
    """
    Comprehensive model evaluation on test set.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Model Evaluation")
    print("=" * 70)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Metrics
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n=== TEST SET PERFORMANCE ===")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")
    print(f"Precision (Distressed): {precision:.4f}")
    print(f"Recall (Distressed): {recall:.4f}")
    print(f"F1-Score (Distressed): {f1:.4f}")
    
    # Classification report
    print(f"\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, 
          target_names=['Not Distressed', 'Distressed']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n=== CONFUSION MATRIX ===")
    print(f"                    Predicted")
    print(f"                    Not Dist.  Distressed")
    print(f"Actual Not Dist.    {cm[0,0]:>8,}  {cm[0,1]:>8,}")
    print(f"       Distressed   {cm[1,0]:>8,}  {cm[1,1]:>8,}")
    
    # --- Create Visualizations ---
    
    # 1. ROC Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'XGBoost (AUC = {auc_roc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title('ROC Curve: Physical Distress Prediction', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_dir}/roc_curve.png")
    
    # 2. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    ax.plot(recall_curve, precision_curve, 'b-', linewidth=2, 
            label=f'XGBoost (AUC-PR = {auc_pr:.3f})')
    baseline = y_test.sum() / len(y_test)
    ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.7, 
               label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/precision_recall_curve.png")
    
    # 3. Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=ax,
                xticklabels=['Not Distressed', 'Distressed'],
                yticklabels=['Not Distressed', 'Distressed'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/confusion_matrix.png")
    
    # 4. Predicted Probability Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.5, 
            label='Not Distressed', color='green', density=True)
    ax.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.5, 
            label='Distressed', color='red', density=True)
    ax.axvline(x=0.5, color='black', linestyle='--', label='Default Threshold (0.5)')
    ax.set_xlabel('Predicted Probability of Distress', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Predicted Probabilities by True Class', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/probability_distribution.png")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['AUC-ROC', 'AUC-PR', 'Precision', 'Recall', 'F1-Score'],
        'Value': [auc_roc, auc_pr, precision, recall, f1]
    })
    metrics_df.to_csv(f'{output_dir}/model_metrics.csv', index=False)
    print(f"Saved: {output_dir}/model_metrics.csv")
    
    return {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
        'confusion_matrix': cm,
    }


# =============================================================================
# STEP 4: SHAP ANALYSIS
# =============================================================================

def shap_analysis(model, X_train, X_test, feature_names, output_dir):
    """
    Generate SHAP explanations for model interpretability.
    """
    print("\n" + "=" * 70)
    print("STEP 4: SHAP Analysis")
    print("=" * 70)
    
    print("\nCalculating SHAP values (this may take a few minutes)...")
    
    # Use a sample for faster computation if dataset is large
    if len(X_train) > 5000:
        sample_idx = np.random.choice(len(X_train), 5000, replace=False)
        X_shap = X_train.iloc[sample_idx]
        print(f"  Using sample of 5,000 for SHAP calculation")
    else:
        X_shap = X_train
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    
    # Feature importance (mean |SHAP|)
    importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': importance
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    
    print(f"\n=== TOP 15 FEATURES BY SHAP IMPORTANCE ===")
    print(importance_df.head(15).to_string(index=False))
    
    # Save importance
    importance_df.to_csv(f'{output_dir}/shap_feature_importance.csv', index=False)
    print(f"\nSaved: {output_dir}/shap_feature_importance.csv")
    
    # --- Create SHAP Visualizations ---
    
    # 1. Summary Plot (Beeswarm)
    print("\nCreating SHAP summary plot...")
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(shap_values, X_shap, feature_names=feature_names, 
                      show=False, max_display=20)
    plt.title('SHAP Feature Importance: Drivers of Physical Distress', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/shap_summary_beeswarm.png")
    
    # 2. Bar Plot (Mean |SHAP|)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, feature_names=feature_names, 
                      plot_type='bar', show=False, max_display=20)
    plt.title('Mean |SHAP Value| by Feature', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_summary_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/shap_summary_bar.png")
    
    # 3. Dependence Plots for Top Features
    print("\nCreating SHAP dependence plots for top features...")
    top_features = importance_df.head(6)['Feature'].tolist()
    
    for i, feat in enumerate(top_features):
        if feat in feature_names:
            fig, ax = plt.subplots(figsize=(10, 6))
            feat_idx = feature_names.index(feat)
            shap.dependence_plot(feat_idx, shap_values, X_shap, 
                               feature_names=feature_names, show=False, ax=ax)
            plt.title(f'SHAP Dependence: {feat}', fontsize=14)
            plt.tight_layout()
            safe_feat_name = feat.replace('/', '_').replace(' ', '_')
            plt.savefig(f'{output_dir}/shap_dependence_{safe_feat_name}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Saved dependence plots for top {len(top_features)} features")
    
    return explainer, shap_values, importance_df


# =============================================================================
# STEP 5: SCORE UNINSPECTED PROPERTIES
# =============================================================================

def score_uninspected(model, df_score, feature_names, medians, output_dir):
    """
    Apply model to properties without REAC scores.
    """
    print("\n" + "=" * 70)
    print("STEP 5: Scoring Uninspected Properties")
    print("=" * 70)
    
    print(f"\nProperties without REAC scores: {len(df_score):,}")
    
    if len(df_score) == 0:
        print("No properties to score.")
        return None
    
    # Prepare features (same as for training data)
    X_score = df_score[feature_names].copy()
    
    # Apply same imputation as training
    for col, median_val in medians.items():
        if col in X_score.columns:
            X_score[col] = X_score[col].fillna(median_val)
    
    # Check for remaining missing values
    complete_cases = X_score.notna().all(axis=1)
    n_scorable = complete_cases.sum()
    print(f"Properties with complete features: {n_scorable:,} ({n_scorable/len(df_score)*100:.1f}%)")
    
    if n_scorable == 0:
        print("Cannot score any properties due to missing features.")
        return None
    
    # Score only complete cases
    X_to_score = X_score[complete_cases]
    predictions = model.predict_proba(X_to_score)[:, 1]
    
    # Add predictions to dataframe
    df_scored = df_score.copy()
    df_scored['PredictedDistressProb'] = np.nan
    df_scored.loc[complete_cases.values, 'PredictedDistressProb'] = predictions
    
    # Categorize risk levels
    df_scored['RiskCategory'] = pd.cut(
        df_scored['PredictedDistressProb'],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['Low', 'Moderate', 'Elevated', 'High']
    )
    
    # Summary
    print(f"\n=== RISK DISTRIBUTION ===")
    risk_dist = df_scored['RiskCategory'].value_counts()
    for cat in ['High', 'Elevated', 'Moderate', 'Low']:
        if cat in risk_dist.index:
            n = risk_dist[cat]
            pct = n / n_scorable * 100
            print(f"  {cat}: {n:,} ({pct:.1f}%)")
    
    print(f"\n=== PREDICTED PROBABILITY DISTRIBUTION ===")
    print(f"  Mean: {predictions.mean():.3f}")
    print(f"  Median: {np.median(predictions):.3f}")
    print(f"  Std: {predictions.std():.3f}")
    print(f"  Min: {predictions.min():.3f}")
    print(f"  Max: {predictions.max():.3f}")
    
    # Save scored properties
    output_cols = [
        'NHPDPropertyID', 'PropertyName', 'PropertyAddress', 'City', 'State',
        'CensusTract', 'TotalUnits', 'PropertyAge', 'OwnerType_Clean',
        'Has_Section8', 'Has_LIHTC', 'Has_PublicHousing',
        'Poverty_Rate', 'PredictedDistressProb', 'RiskCategory'
    ]
    output_cols = [c for c in output_cols if c in df_scored.columns]
    
    df_output = df_scored[output_cols].sort_values('PredictedDistressProb', ascending=False)
    df_output.to_csv(f'{output_dir}/uninspected_risk_scores.csv', index=False)
    print(f"\nSaved: {output_dir}/uninspected_risk_scores.csv")
    
    # Also save full dataset
    df_scored.to_parquet(f'{output_dir}/uninspected_scored_full.parquet', index=False)
    print(f"Saved: {output_dir}/uninspected_scored_full.parquet")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(predictions, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='High Risk Threshold')
    ax.axvline(x=0.3, color='orange', linestyle='--', linewidth=2, label='Elevated Risk Threshold')
    ax.set_xlabel('Predicted Probability of Physical Distress', fontsize=12)
    ax.set_ylabel('Number of Properties', fontsize=12)
    ax.set_title(f'Distribution of Predicted Risk Scores\n(n={n_scorable:,} uninspected properties)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/uninspected_risk_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/uninspected_risk_distribution.png")
    
    return df_scored


# =============================================================================
# MAIN
# =============================================================================

def main(use_hyperparameter_tuning=False):
    """
    Run the complete modeling pipeline.
    """
    print("\n" + "=" * 70)
    print("PHYSICAL DISTRESS PREDICTION MODEL")
    print("=" * 70)
    
    # Step 1: Load and prepare data
    df_full, df_model, df_score = load_and_prepare_features(CONFIG['input_file'])
    
    # Prepare feature matrix
    X, y, feature_names = prepare_feature_matrix(df_model, CONFIG)
    
    print(f"\nFeatures: {len(feature_names)}")
    print(f"Samples: {len(X)}")
    print(f"Distress rate: {y.mean()*100:.1f}%")
    
    # Train/test split
    print("\n" + "-" * 50)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=y
    )
    print(f"  Training set: {len(X_train):,}")
    print(f"  Test set: {len(X_test):,}")
    
    # Handle missing values
    X_train, X_test, medians = handle_missing_values(X_train, X_test, feature_names)
    
    # Step 2: Train model
    if use_hyperparameter_tuning:
        model = train_xgboost(X_train, y_train, 
                             CONFIG['random_state'], CONFIG['cv_folds'])
    else:
        model = train_xgboost_simple(X_train, y_train, CONFIG['random_state'])
    
    # Step 3: Evaluate
    results = evaluate_model(model, X_test, y_test, feature_names, CONFIG['output_dir'])
    
    # Step 4: SHAP analysis
    explainer, shap_values, importance_df = shap_analysis(
        model, X_train, X_test, feature_names, CONFIG['output_dir']
    )
    
    # Step 5: Score uninspected properties
    # First prepare their features
    X_score_all, _, _ = prepare_feature_matrix(df_score, CONFIG)
    df_score_with_features = df_score.copy()
    for col in X_score_all.columns:
        df_score_with_features[col] = X_score_all[col].values
    
    df_scored = score_uninspected(model, df_score_with_features, feature_names, 
                                  medians, CONFIG['output_dir'])
    
    # Save model and scaler
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    joblib.dump(model, CONFIG['model_file'])
    print(f"Saved: {CONFIG['model_file']}")
    joblib.dump({'medians': medians, 'feature_names': feature_names}, CONFIG['scaler_file'])
    print(f"Saved: {CONFIG['scaler_file']}")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    
    return {
        'model': model,
        'results': results,
        'shap_values': shap_values,
        'importance_df': importance_df,
        'feature_names': feature_names,
        'df_scored': df_scored,
    }


if __name__ == "__main__":
    # Set to True for full hyperparameter tuning (slower but better results)
    # Set to False for quick training with reasonable defaults
    outputs = main(use_hyperparameter_tuning=False)
