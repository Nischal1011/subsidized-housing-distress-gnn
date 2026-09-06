"""
run_analysis.py
===============
Runs the baseline (pre-GNN) stage of the pipeline: data preparation and the
XGBoost model with SHAP explanations. It does NOT run the GNN/GAT stages
(04-06 in src/pipeline/) or the downstream paper tables/figures in
src/analysis/ — those must be run manually, in order, from the repo root.
See the top-level README for the full pipeline sequence.

Usage (from the repository root):
    python src/pipeline/run_analysis.py

This will:
1. Prepare and merge NHPD + ACS data
2. Train the baseline XGBoost model
3. Generate SHAP explanations
4. Score uninspected properties

Author: Nischal Subedi
"""

import subprocess
import sys
import os
from datetime import datetime

def run_script(script_name):
    """Run a Python script and capture output."""
    print(f"\n{'='*70}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print(f"\nERROR: {script_name} failed with return code {result.returncode}")
        return False
    return True


def main():
    """Run the complete analysis pipeline."""
    
    start_time = datetime.now()
    
    print("\n" + "="*70)
    print("PHYSICAL DISTRESS PREDICTION PIPELINE")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Check for required data files
    print("\nChecking for required data files...")
    
    required_files = [
        ('data/Active and Inconclusive Properties.xlsx', 'NHPD data'),
        ('data/acs5_2023_tract_v1_data_multi_state.parquet', 'ACS data'),
    ]
    
    missing_files = []
    for filepath, description in required_files:
        if os.path.exists(filepath):
            print(f"  ✓ Found: {description}")
        else:
            print(f"  ✗ Missing: {description} ({filepath})")
            missing_files.append(filepath)
    
    if missing_files:
        print("\nERROR: Missing required data files. Please ensure all data files are in the data/ directory.")
        print("See README.md for data download instructions.")
        return
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Step 1: Data Preparation
    if not run_script('src/pipeline/01_data_preparation.py'):
        print("\nPipeline failed at data preparation step.")
        return

    # Check that merged data was created
    if not os.path.exists('data/merged_analysis_data.parquet'):
        print("\nERROR: Merged data file was not created.")
        return

    # Step 2: Baseline XGBoost Model Training
    if not run_script('src/pipeline/02_baseline_xgboost_model.py'):
        print("\nPipeline failed at model training step.")
        return
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\nFinished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    
    print("\n=== OUTPUT FILES ===")
    if os.path.exists('outputs'):
        for f in sorted(os.listdir('outputs')):
            filepath = os.path.join('outputs', f)
            size = os.path.getsize(filepath)
            if size > 1024*1024:
                size_str = f"{size/1024/1024:.1f} MB"
            elif size > 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size} bytes"
            print(f"  {f:<45} {size_str:>10}")
    
    print("\n=== NEXT STEPS ===")
    print("1. Review model metrics in outputs/model_metrics.csv")
    print("2. Check SHAP importance in outputs/shap_feature_importance.csv")
    print("3. Use figures in outputs/ for the paper")
    print("4. Review high-risk properties in outputs/uninspected_risk_scores.csv")


if __name__ == "__main__":
    main()
