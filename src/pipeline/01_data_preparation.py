"""
01_data_preparation.py
======================
Prepare and merge NHPD property data with ACS tract-level data.

This script:
1. Loads NHPD property data
2. Loads ACS census tract data
3. Engineers features from both sources
4. Merges on census tract
5. Creates the outcome variable (Physical Distress)
6. Saves the analysis-ready dataset

Author: Nischal Subedi
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # File paths - UPDATE THESE TO YOUR ACTUAL PATHS
    'nhpd_file': 'data/Active and Inconclusive Properties.xlsx',
    'acs_file': 'data/acs5_2023_tract_physical_distress_data.parquet',  # From census.py
    # Alternative: 'data/acs5_2023_tract_v1_data_multi_state.parquet' (your existing file)
    'output_file': 'data/merged_analysis_data.parquet',
    
    # Analysis parameters
    'baseline_date': datetime(2025, 6, 10),
    'distress_threshold': 70,  # REAC score below this = distressed
}


# =============================================================================
# STEP 1: LOAD AND PREPARE NHPD DATA
# =============================================================================

def load_and_prepare_nhpd(filepath, baseline_date):
    """
    Load NHPD data and engineer property-level features.
    """
    print("=" * 70)
    print("STEP 1: Loading and preparing NHPD data")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading: {filepath}")
    df = pd.read_excel(filepath)
    print(f"Loaded {len(df):,} properties with {len(df.columns)} columns")
    
    # --- Property Age ---
    print("\nCalculating property age...")
    df['EarliestConstructionDate'] = pd.to_datetime(
        df['EarliestConstructionDate'], errors='coerce'
    )
    df['PropertyAge'] = (baseline_date - df['EarliestConstructionDate']).dt.days / 365.25
    
    age_valid = df['PropertyAge'].notna().sum()
    print(f"  Valid construction dates: {age_valid:,} ({age_valid/len(df)*100:.1f}%)")
    
    # --- Months to Expiration ---
    print("\nCalculating months to subsidy expiration...")
    df['LatestEndDate'] = pd.to_datetime(df['LatestEndDate'], errors='coerce')
    df['MonthsToExpire'] = (df['LatestEndDate'] - baseline_date).dt.days / 30.44
    
    # --- Unit Mix ---
    print("\nProcessing unit mix...")
    df['StudioOneBedroomUnits'] = pd.to_numeric(df['StudioOneBedroomUnits'], errors='coerce').fillna(0)
    df['TwoBedroomUnits'] = pd.to_numeric(df['TwoBedroomUnits'], errors='coerce').fillna(0)
    df['ThreePlusBedroomUnits'] = pd.to_numeric(df['ThreePlusBedroomUnits'], errors='coerce').fillna(0)
    
    total_unit_breakdown = df['StudioOneBedroomUnits'] + df['TwoBedroomUnits'] + df['ThreePlusBedroomUnits']
    
    df['PercentStudioOneBedroom'] = np.where(
        total_unit_breakdown > 0,
        df['StudioOneBedroomUnits'] / total_unit_breakdown * 100,
        np.nan
    )
    df['PercentTwoBedroom'] = np.where(
        total_unit_breakdown > 0,
        df['TwoBedroomUnits'] / total_unit_breakdown * 100,
        np.nan
    )
    df['PercentThreePlusBedroom'] = np.where(
        total_unit_breakdown > 0,
        df['ThreePlusBedroomUnits'] / total_unit_breakdown * 100,
        np.nan
    )
    
    # --- Program Type Indicators ---
    print("\nCreating program type indicators...")
    df['Has_Section8'] = (df['NumberActiveSection8'] > 0).astype(int)
    df['Has_LIHTC'] = (df['NumberActiveLihtc'] > 0).astype(int)
    df['Has_Section515'] = (df['NumberActiveSection515'] > 0).astype(int)
    df['Has_PublicHousing'] = (df['NumberActivePublicHousing'] > 0).astype(int)
    df['Has_Section202'] = (df['NumberActiveSection202'] > 0).astype(int)
    df['Has_HOME'] = (df['NumberActiveHome'] > 0).astype(int)
    df['Has_Section236'] = (df['NumberActiveSection236'] > 0).astype(int)
    
    # Count of active subsidy programs
    df['NumActivePrograms'] = (
        df['Has_Section8'] + df['Has_LIHTC'] + df['Has_Section515'] +
        df['Has_PublicHousing'] + df['Has_Section202'] + df['Has_HOME'] +
        df['Has_Section236']
    )
    
    # --- Assisted Units ---
    print("\nCalculating assisted units...")
    # S8_1_AssistedUnits is the primary Section 8 assisted units
    df['S8_1_AssistedUnits'] = pd.to_numeric(df['S8_1_AssistedUnits'], errors='coerce').fillna(0)
    df['LIHTC_1_AssistedUnits'] = pd.to_numeric(df['LIHTC_1_AssistedUnits'], errors='coerce').fillna(0)
    
    # Percent of units that are subsidized
    df['TotalUnits'] = pd.to_numeric(df['TotalUnits'], errors='coerce')
    df['PercentSubsidized'] = np.where(
        df['TotalUnits'] > 0,
        np.minimum((df['S8_1_AssistedUnits'] + df['LIHTC_1_AssistedUnits']) / df['TotalUnits'] * 100, 100),
        np.nan
    )
    
    # --- Owner Type ---
    print("\nCleaning owner type...")
    # Standardize owner type categories
    owner_mapping = {
        'NON PROFIT': 'Non-Profit',
        'Non Profit': 'Non-Profit',
        'Non-Profit': 'Non-Profit',
        'PROFIT MOTIVATED': 'For-Profit',
        'Profit Motivated': 'For-Profit',
        'For Profit': 'For-Profit',
        'LIMITED DIVIDEND': 'Limited Dividend',
        'Limited Dividend': 'Limited Dividend',
        'Limited Partnership': 'Limited Dividend',
        'Limited Profit': 'Limited Dividend',
        'Full Profit': 'For-Profit',
        'Public Entity': 'Public',
        'Multiple': 'Multiple',
    }
    df['OwnerType_Clean'] = df['OwnerType'].map(owner_mapping).fillna('Unknown')
    
    print(f"  Owner type distribution:")
    print(df['OwnerType_Clean'].value_counts())
    
    # --- Fair Market Rent ---
    print("\nProcessing Fair Market Rent...")
    df['FairMarketRent_2BR'] = pd.to_numeric(df['FairMarketRent_2BR'], errors='coerce')
    fmr_valid = df['FairMarketRent_2BR'].notna().sum()
    print(f"  Valid FMR: {fmr_valid:,} ({fmr_valid/len(df)*100:.1f}%)")
    
    # --- CBSA Type (Metro indicator) ---
    print("\nProcessing metro/non-metro...")
    df['IsMetro'] = df['CBSAType'].str.contains('Metro', case=False, na=False).astype(int)
    
    # --- Census Tract ID ---
    print("\nStandardizing census tract IDs...")
    # Census tract should be 11 digits: 2 state + 3 county + 6 tract
    df['CensusTract'] = df['CensusTract'].astype(str).str.strip()
    # Pad to 11 characters if needed
    df['CensusTract'] = df['CensusTract'].str.zfill(11)
    
    # Check for valid tract IDs
    valid_tracts = df['CensusTract'].str.match(r'^\d{11}$')
    print(f"  Valid tract IDs: {valid_tracts.sum():,} ({valid_tracts.mean()*100:.1f}%)")
    
    # --- REAC Scores (Outcome) ---
    print("\nProcessing REAC scores...")
    df['ReacScore1'] = pd.to_numeric(df['ReacScore1'], errors='coerce')
    df['ReacScore2'] = pd.to_numeric(df['ReacScore2'], errors='coerce')
    df['ReacScore3'] = pd.to_numeric(df['ReacScore3'], errors='coerce')
    
    # Use BEST AVAILABLE score (Score1 preferred, then Score2, then Score3)
    # This nearly doubles our training data (6,724 -> 12,385)
    df['BestReacScore'] = df['ReacScore1'].fillna(df['ReacScore2']).fillna(df['ReacScore3'])
    df['HasREAC'] = df['BestReacScore'].notna()
    
    print(f"  ReacScore1 available: {df['ReacScore1'].notna().sum():,}")
    print(f"  ReacScore2 available: {df['ReacScore2'].notna().sum():,}")
    print(f"  ReacScore3 available: {df['ReacScore3'].notna().sum():,}")
    
    reac_valid = df['HasREAC'].sum()
    print(f"  BestReacScore (any): {reac_valid:,} ({reac_valid/len(df)*100:.1f}%)")
    
    if reac_valid > 0:
        print(f"  REAC Score distribution (BestReacScore):")
        print(f"    Mean: {df.loc[df['HasREAC'], 'BestReacScore'].mean():.1f}")
        print(f"    Median: {df.loc[df['HasREAC'], 'BestReacScore'].median():.1f}")
        print(f"    Min: {df.loc[df['HasREAC'], 'BestReacScore'].min():.1f}")
        print(f"    Max: {df.loc[df['HasREAC'], 'BestReacScore'].max():.1f}")
    
    print(f"\nNHPD preparation complete. Shape: {df.shape}")
    return df


# =============================================================================
# STEP 2: LOAD AND PREPARE ACS DATA
# =============================================================================

def load_and_prepare_acs(filepath):
    """
    Load ACS tract-level data and engineer neighborhood features.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Loading and preparing ACS data")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading: {filepath}")
    gdf = gpd.read_parquet(filepath)
    print(f"Loaded {len(gdf):,} census tracts")
    print(f"Columns: {len(gdf.columns)}")
    
    # --- Create Census Tract ID for merging ---
    print("\nCreating tract identifiers...")
    # The tract ID format depends on your ACS data structure
    # Common patterns: GEOID, or STATE+COUNTY+TRACT
    
    if 'GEOID' in gdf.columns:
        gdf['CensusTract'] = gdf['GEOID'].astype(str).str.zfill(11)
    elif 'STATE' in gdf.columns and 'COUNTY' in gdf.columns and 'TRACT' in gdf.columns:
        gdf['CensusTract'] = (
            gdf['STATE'].astype(str).str.zfill(2) +
            gdf['COUNTY'].astype(str).str.zfill(3) +
            gdf['TRACT'].astype(str).str.zfill(6)
        )
    else:
        # Try to find tract identifier
        print("  WARNING: Could not identify tract ID columns")
        print(f"  Available columns: {gdf.columns.tolist()[:20]}...")
        # Check for columns that might contain tract IDs
        for col in ['tract', 'TRACT', 'census_tract', 'geoid', 'GEOID']:
            if col in gdf.columns:
                gdf['CensusTract'] = gdf[col].astype(str).str.zfill(11)
                print(f"  Using column '{col}' as tract identifier")
                break
    
    # --- Poverty Rate ---
    print("\nCalculating derived features...")
    
    if 'Total_Population_Poverty_Status' in gdf.columns and 'Population_Below_Poverty' in gdf.columns:
        gdf['Poverty_Rate'] = np.where(
            gdf['Total_Population_Poverty_Status'] > 0,
            gdf['Population_Below_Poverty'] / gdf['Total_Population_Poverty_Status'] * 100,
            np.nan
        )
        print(f"  Poverty_Rate: mean={gdf['Poverty_Rate'].mean():.1f}%")
    
    # --- Vacancy Rate ---
    if 'Total_Housing_Units' in gdf.columns and 'Vacant_Housing_Units' in gdf.columns:
        gdf['Vacancy_Rate'] = np.where(
            gdf['Total_Housing_Units'] > 0,
            gdf['Vacant_Housing_Units'] / gdf['Total_Housing_Units'] * 100,
            np.nan
        )
        print(f"  Vacancy_Rate: mean={gdf['Vacancy_Rate'].mean():.1f}%")
    
    # --- Renter Share ---
    if 'Total_Occupied_Housing_Units' in gdf.columns and 'Renter_Occupied_Housing_Units' in gdf.columns:
        gdf['Renter_Share'] = np.where(
            gdf['Total_Occupied_Housing_Units'] > 0,
            gdf['Renter_Occupied_Housing_Units'] / gdf['Total_Occupied_Housing_Units'] * 100,
            np.nan
        )
        print(f"  Renter_Share: mean={gdf['Renter_Share'].mean():.1f}%")
    
    # --- Severe Rent Burden (>50% income on rent) ---
    if 'Total_Renter_Occupied_Units_For_Rent_Burden' in gdf.columns and 'Rent_50_Percent_Or_More_Income' in gdf.columns:
        gdf['Severe_Rent_Burden_Rate'] = np.where(
            gdf['Total_Renter_Occupied_Units_For_Rent_Burden'] > 0,
            gdf['Rent_50_Percent_Or_More_Income'] / gdf['Total_Renter_Occupied_Units_For_Rent_Burden'] * 100,
            np.nan
        )
        print(f"  Severe_Rent_Burden_Rate: mean={gdf['Severe_Rent_Burden_Rate'].mean():.1f}%")
    
    # --- Any Rent Burden (>30% income on rent) ---
    rent_burden_cols = ['Rent_30_35_Percent_Income', 'Rent_35_40_Percent_Income', 
                        'Rent_40_50_Percent_Income', 'Rent_50_Percent_Or_More_Income']
    if all(col in gdf.columns for col in rent_burden_cols):
        gdf['Total_Rent_Burdened'] = gdf[rent_burden_cols].sum(axis=1)
        gdf['Any_Rent_Burden_Rate'] = np.where(
            gdf['Total_Renter_Occupied_Units_For_Rent_Burden'] > 0,
            gdf['Total_Rent_Burdened'] / gdf['Total_Renter_Occupied_Units_For_Rent_Burden'] * 100,
            np.nan
        )
        print(f"  Any_Rent_Burden_Rate: mean={gdf['Any_Rent_Burden_Rate'].mean():.1f}%")
    
    # --- Elderly Share (65+) ---
    elderly_cols = [
        'Male_65_66_Years', 'Male_67_69_Years', 'Male_70_74_Years', 'Male_75_Years_And_Over',
        'Female_65_66_Years', 'Female_67_69_Years', 'Female_70_74_Years', 'Female_75_Years_And_Over'
    ]
    if all(col in gdf.columns for col in elderly_cols):
        gdf['Elderly_Population'] = gdf[elderly_cols].sum(axis=1)
        gdf['Elderly_Share'] = np.where(
            gdf['Total_Population'] > 0,
            gdf['Elderly_Population'] / gdf['Total_Population'] * 100,
            np.nan
        )
        print(f"  Elderly_Share: mean={gdf['Elderly_Share'].mean():.1f}%")
    
    # --- Disability Rate ---
    disability_cols = [
        'Male_Under_5_With_Disability', 'Male_5_17_With_Disability',
        'Male_18_34_With_Disability', 'Male_35_64_With_Disability',
        'Male_65_74_With_Disability', 'Male_75_Over_With_Disability',
        'Female_Under_5_With_Disability', 'Female_5_17_With_Disability',
        'Female_18_34_With_Disability', 'Female_35_64_With_Disability',
        'Female_65_74_With_Disability', 'Female_75_Over_With_Disability'
    ]
    if all(col in gdf.columns for col in disability_cols):
        gdf['Total_With_Disability'] = gdf[disability_cols].sum(axis=1)
        gdf['Disability_Rate'] = np.where(
            gdf['Total_Population'] > 0,
            gdf['Total_With_Disability'] / gdf['Total_Population'] * 100,
            np.nan
        )
        print(f"  Disability_Rate: mean={gdf['Disability_Rate'].mean():.1f}%")
    
    # --- Pre-1970 Housing Stock ---
    pre1970_cols = ['Housing_Built_1960_1969', 'Housing_Built_1950_1959',
                    'Housing_Built_1940_1949', 'Housing_Built_1939_or_Earlier']
    if all(col in gdf.columns for col in pre1970_cols):
        gdf['Pre1970_Housing'] = gdf[pre1970_cols].sum(axis=1)
        gdf['Pre1970_Housing_Share'] = np.where(
            gdf['Total_Housing_Units_Year_Built'] > 0,
            gdf['Pre1970_Housing'] / gdf['Total_Housing_Units_Year_Built'] * 100,
            np.nan
        )
        print(f"  Pre1970_Housing_Share: mean={gdf['Pre1970_Housing_Share'].mean():.1f}%")
    
    # --- Low Income Renter Households (for coverage ratio) ---
    low_income_cols = [
        'Renter_HH_Income_Less_5K', 'Renter_HH_Income_5K_10K',
        'Renter_HH_Income_10K_15K', 'Renter_HH_Income_15K_20K',
        'Renter_HH_Income_20K_25K', 'Renter_HH_Income_25K_35K'
    ]
    if all(col in gdf.columns for col in low_income_cols):
        gdf['Low_Income_Renter_HH'] = gdf[low_income_cols].sum(axis=1)
        print(f"  Low_Income_Renter_HH: mean={gdf['Low_Income_Renter_HH'].mean():.1f}")
    
    # --- Population Density ---
    if 'geometry' in gdf.columns and gdf['geometry'].notna().any():
        print("\nCalculating population density...")
        try:
            # Project to equal area for accurate area calculation
            gdf_proj = gdf.to_crs('ESRI:102003')  # USA Contiguous Albers Equal Area
            gdf['Area_SqMi'] = gdf_proj.geometry.area / 2.59e6  # sq meters to sq miles
            gdf['Pop_Density'] = np.where(
                gdf['Area_SqMi'] > 0,
                gdf['Total_Population'] / gdf['Area_SqMi'],
                np.nan
            )
            print(f"  Pop_Density: mean={gdf['Pop_Density'].mean():.1f} per sq mi")
        except Exception as e:
            print(f"  WARNING: Could not calculate area: {e}")
            gdf['Pop_Density'] = np.nan
    
    # --- White Non-Hispanic Share (for segregation analysis) ---
    if 'Total_Population_Race_Ethnicity' in gdf.columns and 'Not_Hispanic_White_Alone' in gdf.columns:
        gdf['White_NonHispanic_Share'] = np.where(
            gdf['Total_Population_Race_Ethnicity'] > 0,
            gdf['Not_Hispanic_White_Alone'] / gdf['Total_Population_Race_Ethnicity'] * 100,
            np.nan
        )
        print(f"  White_NonHispanic_Share: mean={gdf['White_NonHispanic_Share'].mean():.1f}%")
    
    print(f"\nACS preparation complete. Shape: {gdf.shape}")
    return gdf


# =============================================================================
# STEP 3: MERGE NHPD AND ACS DATA
# =============================================================================

def merge_datasets(nhpd_df, acs_gdf):
    """
    Merge NHPD property data with ACS tract-level data.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Merging NHPD with ACS data")
    print("=" * 70)
    
    # Select ACS columns to merge (avoid duplicating geometry, etc.)
    acs_feature_cols = [
        'CensusTract',
        'Total_Population',
        'Median_Household_Income',
        'Median_Gross_Rent',
        'Median_Housing_Value',
        'Poverty_Rate',
        'Vacancy_Rate',
        'Renter_Share',
        'Severe_Rent_Burden_Rate',
        'Any_Rent_Burden_Rate',
        'Elderly_Share',
        'Disability_Rate',
        'Pre1970_Housing_Share',
        'Low_Income_Renter_HH',
        'Pop_Density',
        'White_NonHispanic_Share',
        'Gini_Index_Income_Inequality',
    ]
    
    # Keep only columns that exist
    acs_cols_available = [col for col in acs_feature_cols if col in acs_gdf.columns]
    print(f"\nACS features to merge: {len(acs_cols_available)}")
    
    # Create a non-spatial dataframe for merging
    acs_df = acs_gdf[acs_cols_available].copy()
    if isinstance(acs_df, gpd.GeoDataFrame):
        acs_df = pd.DataFrame(acs_df)
    
    # Merge on CensusTract
    print(f"\nMerging on CensusTract...")
    print(f"  NHPD unique tracts: {nhpd_df['CensusTract'].nunique():,}")
    print(f"  ACS unique tracts: {acs_df['CensusTract'].nunique():,}")
    
    merged = nhpd_df.merge(
        acs_df,
        on='CensusTract',
        how='left'
    )
    
    # Check merge success
    merge_indicator = merged['Poverty_Rate'].notna()
    merge_success = merge_indicator.sum()
    merge_rate = merge_success / len(merged) * 100
    
    print(f"\n  Merge results:")
    print(f"    Total properties: {len(merged):,}")
    print(f"    Successfully merged: {merge_success:,} ({merge_rate:.1f}%)")
    print(f"    Failed to merge: {len(merged) - merge_success:,} ({100-merge_rate:.1f}%)")
    
    return merged


# =============================================================================
# STEP 4: CREATE OUTCOME VARIABLE
# =============================================================================

def create_outcome(df, threshold=70):
    """
    Create the Physical Distress outcome variable.
    Uses BestReacScore (combining Score1, Score2, Score3).
    """
    print("\n" + "=" * 70)
    print("STEP 4: Creating outcome variable")
    print("=" * 70)
    
    print(f"\nDistress threshold: REAC < {threshold}")
    
    # Use BestReacScore (already created in prepare step)
    if 'BestReacScore' not in df.columns:
        # Fallback if not created yet
        df['ReacScore1'] = pd.to_numeric(df['ReacScore1'], errors='coerce')
        df['ReacScore2'] = pd.to_numeric(df['ReacScore2'], errors='coerce')
        df['ReacScore3'] = pd.to_numeric(df['ReacScore3'], errors='coerce')
        df['BestReacScore'] = df['ReacScore1'].fillna(df['ReacScore2']).fillna(df['ReacScore3'])
    
    # Binary outcome based on BestReacScore
    df['PhysicalDistress'] = np.where(
        df['BestReacScore'].notna(),
        (df['BestReacScore'] < threshold).astype(int),
        np.nan
    )
    
    # Summary
    has_reac = df['BestReacScore'].notna()
    n_with_reac = has_reac.sum()
    n_distressed = df.loc[has_reac, 'PhysicalDistress'].sum()
    distress_rate = n_distressed / n_with_reac * 100
    
    print(f"\n  Properties with REAC score (any): {n_with_reac:,}")
    print(f"  Properties with distress (REAC < {threshold}): {n_distressed:,.0f}")
    print(f"  Distress rate: {distress_rate:.1f}%")
    
    # Distribution of REAC scores by distress status
    print(f"\n  REAC score distribution:")
    print(f"    Distressed (REAC < {threshold}):")
    distressed_scores = df.loc[df['PhysicalDistress'] == 1, 'BestReacScore']
    print(f"      Mean: {distressed_scores.mean():.1f}")
    print(f"      Min: {distressed_scores.min():.1f}, Max: {distressed_scores.max():.1f}")
    
    print(f"    Not Distressed (REAC >= {threshold}):")
    not_distressed_scores = df.loc[df['PhysicalDistress'] == 0, 'BestReacScore']
    print(f"      Mean: {not_distressed_scores.mean():.1f}")
    print(f"      Min: {not_distressed_scores.min():.1f}, Max: {not_distressed_scores.max():.1f}")
    
    return df


# =============================================================================
# STEP 5: FINAL DATA SUMMARY
# =============================================================================

def summarize_data(df):
    """
    Print summary of the final merged dataset.
    """
    print("\n" + "=" * 70)
    print("FINAL DATA SUMMARY")
    print("=" * 70)
    
    print(f"\nDataset shape: {df.shape}")
    
    # Key variable availability
    key_vars = [
        ('PropertyAge', 'Property Age'),
        ('TotalUnits', 'Total Units'),
        ('FairMarketRent_2BR', 'Fair Market Rent'),
        ('Poverty_Rate', 'Poverty Rate (ACS)'),
        ('Vacancy_Rate', 'Vacancy Rate (ACS)'),
        ('Median_Household_Income', 'Median Income (ACS)'),
        ('ReacScore1', 'REAC Score'),
        ('PhysicalDistress', 'Physical Distress Outcome'),
    ]
    
    print(f"\nKey variable availability:")
    print(f"{'Variable':<30} {'Non-Missing':>15} {'Percent':>10}")
    print("-" * 60)
    for var, label in key_vars:
        if var in df.columns:
            n_valid = df[var].notna().sum()
            pct = n_valid / len(df) * 100
            print(f"{label:<30} {n_valid:>15,} {pct:>9.1f}%")
    
    # Modeling sample size
    print(f"\n" + "-" * 60)
    has_outcome = df['PhysicalDistress'].notna()
    has_key_features = (
        df['PropertyAge'].notna() & 
        df['TotalUnits'].notna() &
        df['Poverty_Rate'].notna()
    )
    modeling_sample = has_outcome & has_key_features
    
    print(f"Properties with outcome (REAC): {has_outcome.sum():,}")
    print(f"Properties with key features: {has_key_features.sum():,}")
    print(f"Properties for modeling (both): {modeling_sample.sum():,}")
    
    # Properties without REAC (for scoring)
    no_reac = ~df['ReacScore1'].notna()
    scorable = no_reac & has_key_features
    print(f"\nProperties without REAC: {no_reac.sum():,}")
    print(f"Scorable (no REAC + has features): {scorable.sum():,}")
    
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the complete data preparation pipeline."""
    
    print("\n" + "=" * 70)
    print("DATA PREPARATION PIPELINE")
    print("Predicting Physical Distress in Subsidized Housing")
    print("=" * 70)
    
    # Step 1: Load and prepare NHPD
    nhpd_df = load_and_prepare_nhpd(
        CONFIG['nhpd_file'],
        CONFIG['baseline_date']
    )

    # foprmat census tract for nhpd_df
    nhpd_df['CensusTract'] = (
    nhpd_df['CensusTract']
    .astype(str)
    .str.replace('.0', '', regex=False)
)
    nhpd_df['CensusTract'] = nhpd_df['CensusTract'].str.zfill(11)
    
    # Step 2: Load and prepare ACS
    acs_gdf = load_and_prepare_acs(CONFIG['acs_file'])
    
    # Step 3: Merge datasets
    merged_df = merge_datasets(nhpd_df, acs_gdf)
    
    # Step 4: Create outcome variable
    merged_df = create_outcome(merged_df, CONFIG['distress_threshold'])
    
    # Step 5: Summarize
    merged_df = summarize_data(merged_df)
    
    # Save
    print(f"\n" + "=" * 70)
    print("SAVING DATA")
    print("=" * 70)
    
    # Save as parquet (efficient and preserves dtypes)
    output_path = CONFIG['output_file']
    print(f"\nSaving to: {output_path}")
    merged_df.to_parquet(output_path, index=False)
    print("Done!")
    
    return merged_df


if __name__ == "__main__":
    df = main()