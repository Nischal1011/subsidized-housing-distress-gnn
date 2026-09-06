"""
census.py - Download ACS 5-Year Tract Data for Housing Analysis
================================================================

Downloads census tract-level data from ACS 5-Year estimates for use in
predicting physical distress in subsidized housing.

Key variables:
- Income and poverty measures
- Rent burden rates
- Housing stock characteristics (age, tenure, vacancy)
- Demographic composition (elderly, disability)
- Housing market indicators
- Economic distress indicators

IMPORTANT: Creates proper 11-digit GEOID for merging with NHPD CensusTract field

Author: Nischal Subedi
Updated: 2025
"""

import censusdis.data as ced
import geopandas as gpd
import pandas as pd
import numpy as np
from censusdis.datasets import ACS5
from censusdis import states
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import retrying

def download_state_data(state_fips, dataset, vintage, variables, max_retries=3, retry_delay=5):
    """Download data for a single state with retries, returning a GeoDataFrame or None on error."""
    state_name = state_fips
    print(f"--- Downloading data for: {state_name} ---")

    @retrying.retry(
        stop_max_attempt_number=max_retries,
        wait_fixed=retry_delay * 1000,
        retry_on_exception=lambda e: isinstance(e, Exception)
    )
    def attempt_download():
        try:
            start_time = time.time()
            gdf_tracts = ced.download(
                dataset=dataset,
                vintage=vintage,
                download_variables=variables,
                state=state_fips,
                tract='*',
                with_geometry=True,
            )
            print(f"Downloaded {len(gdf_tracts)} tracts for {state_name} in {time.time() - start_time:.2f} seconds.")
            return gdf_tracts
        except Exception as e:
            print(f"Error downloading data for {state_name}: {e}")
            raise

    try:
        return attempt_download()
    except Exception as e:
        print(f"Failed to download data for {state_name} after {max_retries} attempts: {e}")
        return None


def create_derived_variables(gdf):
    """
    Create derived variables from raw ACS data.
    These are the actual features used in the model.
    """
    print("\nCreating derived variables...")
    
    # --- Poverty Rate ---
    gdf['Poverty_Rate'] = np.where(
        gdf['Total_Population_Poverty_Status'] > 0,
        gdf['Population_Below_Poverty'] / gdf['Total_Population_Poverty_Status'] * 100,
        np.nan
    )
    
    # --- Vacancy Rate ---
    gdf['Vacancy_Rate'] = np.where(
        gdf['Total_Housing_Units'] > 0,
        gdf['Vacant_Housing_Units'] / gdf['Total_Housing_Units'] * 100,
        np.nan
    )
    
    # --- Renter Share ---
    gdf['Renter_Share'] = np.where(
        gdf['Total_Occupied_Housing_Units'] > 0,
        gdf['Renter_Occupied_Housing_Units'] / gdf['Total_Occupied_Housing_Units'] * 100,
        np.nan
    )
    
    # --- Severe Rent Burden Rate (>50% income on rent) ---
    gdf['Severe_Rent_Burden_Rate'] = np.where(
        gdf['Total_Renter_Occupied_Units_For_Rent_Burden'] > 0,
        gdf['Rent_50_Percent_Or_More_Income'] / gdf['Total_Renter_Occupied_Units_For_Rent_Burden'] * 100,
        np.nan
    )
    
    # --- Any Rent Burden Rate (>30% income on rent) ---
    gdf['Total_Rent_Burdened'] = (
        gdf['Rent_30_35_Percent_Income'] + 
        gdf['Rent_35_40_Percent_Income'] + 
        gdf['Rent_40_50_Percent_Income'] + 
        gdf['Rent_50_Percent_Or_More_Income']
    )
    gdf['Any_Rent_Burden_Rate'] = np.where(
        gdf['Total_Renter_Occupied_Units_For_Rent_Burden'] > 0,
        gdf['Total_Rent_Burdened'] / gdf['Total_Renter_Occupied_Units_For_Rent_Burden'] * 100,
        np.nan
    )
    
    # --- Elderly Share (65+) ---
    elderly_cols = [
        'Male_65_66_Years', 'Male_67_69_Years', 'Male_70_74_Years', 'Male_75_Years_And_Over',
        'Female_65_66_Years', 'Female_67_69_Years', 'Female_70_74_Years', 'Female_75_Years_And_Over'
    ]
    gdf['Elderly_Population'] = gdf[elderly_cols].sum(axis=1)
    gdf['Elderly_Share'] = np.where(
        gdf['Total_Population'] > 0,
        gdf['Elderly_Population'] / gdf['Total_Population'] * 100,
        np.nan
    )
    
    # --- Disability Rate ---
    disability_cols = [
        'Male_Under_5_With_Disability', 'Male_5_17_With_Disability',
        'Male_18_34_With_Disability', 'Male_35_64_With_Disability',
        'Male_65_74_With_Disability', 'Male_75_Over_With_Disability',
        'Female_Under_5_With_Disability', 'Female_5_17_With_Disability',
        'Female_18_34_With_Disability', 'Female_35_64_With_Disability',
        'Female_65_74_With_Disability', 'Female_75_Over_With_Disability'
    ]
    gdf['Total_With_Disability'] = gdf[disability_cols].sum(axis=1)
    gdf['Disability_Rate'] = np.where(
        gdf['Total_Population'] > 0,
        gdf['Total_With_Disability'] / gdf['Total_Population'] * 100,
        np.nan
    )
    
    # --- Pre-1970 Housing Share (older housing stock) ---
    pre1970_cols = ['Housing_Built_1960_1969', 'Housing_Built_1950_1959',
                   'Housing_Built_1940_1949', 'Housing_Built_1939_or_Earlier']
    gdf['Pre1970_Housing_Units'] = gdf[pre1970_cols].sum(axis=1)
    gdf['Pre1970_Housing_Share'] = np.where(
        gdf['Total_Housing_Units_Year_Built'] > 0,
        gdf['Pre1970_Housing_Units'] / gdf['Total_Housing_Units_Year_Built'] * 100,
        np.nan
    )
    
    # --- 1970-1989 Housing Share (relevant for early LIHTC) ---
    if 'Housing_Built_1970_1979' in gdf.columns and 'Housing_Built_1980_1989' in gdf.columns:
        gdf['Housing_1970_1989_Units'] = gdf['Housing_Built_1970_1979'] + gdf['Housing_Built_1980_1989']
        gdf['Housing_1970_1989_Share'] = np.where(
            gdf['Total_Housing_Units_Year_Built'] > 0,
            gdf['Housing_1970_1989_Units'] / gdf['Total_Housing_Units_Year_Built'] * 100,
            np.nan
        )
    
    # --- Low Income Renter Households (< $35K) ---
    low_income_cols = [
        'Renter_HH_Income_Less_5K', 'Renter_HH_Income_5K_10K',
        'Renter_HH_Income_10K_15K', 'Renter_HH_Income_15K_20K',
        'Renter_HH_Income_20K_25K', 'Renter_HH_Income_25K_35K'
    ]
    gdf['Low_Income_Renter_HH'] = gdf[low_income_cols].sum(axis=1)
    gdf['Low_Income_Renter_Share'] = np.where(
        gdf['Total_Renter_Occupied_HH_Income_Distribution'] > 0,
        gdf['Low_Income_Renter_HH'] / gdf['Total_Renter_Occupied_HH_Income_Distribution'] * 100,
        np.nan
    )
    
    # --- Unemployment Rate (if available) ---
    if 'Labor_Force' in gdf.columns and 'Unemployed' in gdf.columns:
        gdf['Unemployment_Rate'] = np.where(
            gdf['Labor_Force'] > 0,
            gdf['Unemployed'] / gdf['Labor_Force'] * 100,
            np.nan
        )
    
    # --- White Non-Hispanic Share ---
    gdf['White_NonHispanic_Share'] = np.where(
        gdf['Total_Population_Race_Ethnicity'] > 0,
        gdf['Not_Hispanic_White_Alone'] / gdf['Total_Population_Race_Ethnicity'] * 100,
        np.nan
    )
    
    # --- Population Density (per sq mile) ---
    if 'geometry' in gdf.columns and gdf['geometry'].notna().any():
        try:
            # Project to equal area CRS for accurate area calculation
            gdf_proj = gdf.to_crs('ESRI:102003')  # USA Contiguous Albers Equal Area
            gdf['Area_SqMi'] = gdf_proj.geometry.area / 2.59e6  # sq meters to sq miles
            gdf['Pop_Density'] = np.where(
                gdf['Area_SqMi'] > 0,
                gdf['Total_Population'] / gdf['Area_SqMi'],
                np.nan
            )
            print("  Calculated population density")
        except Exception as e:
            print(f"  Warning: Could not calculate area: {e}")
            gdf['Pop_Density'] = np.nan
    
    print("  Derived variables created successfully")
    return gdf


if __name__ == "__main__":
    # --- Configuration ---
    ACS_VINTAGE = 2023
    OUTPUT_FILENAME = f"data/acs5_{ACS_VINTAGE}_tract_physical_distress_data.parquet"

    # All 50 states + DC
    state_fips_list = [
        states.AL, states.AK, states.AZ, states.AR, states.CA, states.CO, states.CT,
        states.DE, states.DC, states.FL, states.GA, states.HI, states.ID, states.IL,
        states.IN, states.IA, states.KS, states.KY, states.LA, states.ME, states.MD,
        states.MA, states.MI, states.MN, states.MS, states.MO, states.MT, states.NE,
        states.NV, states.NH, states.NJ, states.NM, states.NY, states.NC, states.ND,
        states.OH, states.OK, states.OR, states.PA, states.RI, states.SC, states.SD,
        states.TN, states.TX, states.UT, states.VT, states.VA, states.WA, states.WV,
        states.WI, states.WY
    ]

    # ==========================================================================
    # ACS VARIABLES TO DOWNLOAD
    # ==========================================================================
    # 
    # These are the raw variables. Derived rates are calculated after download.
    #
    acs_variables = [
        'NAME',
        
        # --- Population ---
        'B01003_001E',  # Total Population
        
        # --- Income ---
        'B19013_001E',  # Median Household Income
        'B19001_001E',  # Total Households (Income Breakdown)
        'B19001_002E',  # Households Income < $10K
        'B19001_003E',  # Households Income $10K-$15K
        'B19001_004E',  # Households Income $15K-$20K
        'B19001_005E',  # Households Income $20K-$25K
        'B19001_006E',  # Households Income $25K-$30K
        'B19001_007E',  # Households Income $30K-$35K
        'B19001_008E',  # Households Income $35K-$40K
        'B19001_009E',  # Households Income $40K-$45K
        'B19001_010E',  # Households Income $45K-$50K
        'B19001_011E',  # Households Income $50K-$60K
        
        # --- Poverty ---
        'B17001_001E',  # Total Population (Poverty Status Determined)
        'B17001_002E',  # Population Below Poverty Level
        
        # --- Income Inequality ---
        'B19083_001E',  # Gini Index
        
        # --- Rent Burden ---
        'B25070_001E',  # Total Renter-Occupied Units (Rent Burden Calc)
        'B25070_007E',  # Rent 30-35% of Income
        'B25070_008E',  # Rent 35-40% of Income
        'B25070_009E',  # Rent 40-50% of Income
        'B25070_010E',  # Rent 50%+ of Income (Severe Burden)
        
        # --- Rent and Housing Values ---
        'B25064_001E',  # Median Gross Rent
        'B25077_001E',  # Median Housing Value
        
        # --- Housing Stock Age ---
        'B25034_001E',  # Total Housing Units (Year Built)
        'B25034_002E',  # Built 2020 or later
        'B25034_003E',  # Built 2010-2019
        'B25034_004E',  # Built 2000-2009
        'B25034_005E',  # Built 1990-1999
        'B25034_006E',  # Built 1980-1989
        'B25034_007E',  # Built 1970-1979
        'B25034_008E',  # Built 1960-1969
        'B25034_009E',  # Built 1950-1959
        'B25034_010E',  # Built 1940-1949
        'B25034_011E',  # Built 1939 or Earlier
        
        # --- Housing Occupancy ---
        'B25002_001E',  # Total Housing Units
        'B25002_002E',  # Occupied Housing Units
        'B25002_003E',  # Vacant Housing Units
        'B25003_001E',  # Total Occupied Housing Units
        'B25003_002E',  # Owner-Occupied
        'B25003_003E',  # Renter-Occupied
        
        # --- Elderly Population (65+) ---
        'B01001_020E',  # Male 60-61
        'B01001_021E',  # Male 62-64
        'B01001_022E',  # Male 65-66
        'B01001_023E',  # Male 67-69
        'B01001_024E',  # Male 70-74
        'B01001_025E',  # Male 75+
        'B01001_044E',  # Female 60-61
        'B01001_045E',  # Female 62-64
        'B01001_046E',  # Female 65-66
        'B01001_047E',  # Female 67-69
        'B01001_048E',  # Female 70-74
        'B01001_049E',  # Female 75+
        
        # --- Disability ---
        'B18101_001E',  # Total Population (Disability Status)
        'B18101_004E',  # Male Under 5 with Disability
        'B18101_007E',  # Male 5-17 with Disability
        'B18101_010E',  # Male 18-34 with Disability
        'B18101_013E',  # Male 35-64 with Disability
        'B18101_016E',  # Male 65-74 with Disability
        'B18101_019E',  # Male 75+ with Disability
        'B18101_023E',  # Female Under 5 with Disability
        'B18101_026E',  # Female 5-17 with Disability
        'B18101_029E',  # Female 18-34 with Disability
        'B18101_032E',  # Female 35-64 with Disability
        'B18101_035E',  # Female 65-74 with Disability
        'B18101_038E',  # Female 75+ with Disability
        
        # --- Renter Household Income Distribution ---
        'B25118_001E',  # Total Renter-Occupied HH
        'B25118_002E',  # Renter HH Income < $5K
        'B25118_003E',  # Renter HH Income $5K-$10K
        'B25118_004E',  # Renter HH Income $10K-$15K
        'B25118_005E',  # Renter HH Income $15K-$20K
        'B25118_006E',  # Renter HH Income $20K-$25K
        'B25118_007E',  # Renter HH Income $25K-$35K
        'B25118_008E',  # Renter HH Income $35K-$50K
        
        # --- Households ---
        'B11001_001E',  # Total Households
        'B11001_002E',  # Family Households
        
        # --- Race & Ethnicity ---
        'B03002_001E',  # Total Population (Race/Ethnicity)
        'B03002_003E',  # White Alone, Not Hispanic
        'B03002_004E',  # Black Alone
        'B03002_012E',  # Hispanic/Latino
        
        # --- Employment (for unemployment rate) ---
        'B23025_001E',  # Total Population 16+ (Employment Status)
        'B23025_002E',  # In Labor Force
        'B23025_005E',  # Unemployed
        
        # --- SSI (Supplemental Security Income) ---
        'B09010_001E',  # Receipt of SSI
    ]

    # ==========================================================================
    # VARIABLE NAME MAPPING
    # ==========================================================================
    acs_variable_mapping = {
        # --- Population ---
        'B01003_001E': 'Total_Population',

        # --- Income ---
        'B19013_001E': 'Median_Household_Income',
        'B19001_001E': 'Total_Households_Income_Breakdown',
        'B19001_002E': 'Households_Income_Less_10K',
        'B19001_003E': 'Households_Income_10K_15K',
        'B19001_004E': 'Households_Income_15K_20K',
        'B19001_005E': 'Households_Income_20K_25K',
        'B19001_006E': 'Households_Income_25K_30K',
        'B19001_007E': 'Households_Income_30K_35K',
        'B19001_008E': 'Households_Income_35K_40K',
        'B19001_009E': 'Households_Income_40K_45K',
        'B19001_010E': 'Households_Income_45K_50K',
        'B19001_011E': 'Households_Income_50K_60K',

        # --- Poverty ---
        'B17001_001E': 'Total_Population_Poverty_Status',
        'B17001_002E': 'Population_Below_Poverty',

        # --- Income Inequality ---
        'B19083_001E': 'Gini_Index_Income_Inequality',

        # --- Rent Burden ---
        'B25070_001E': 'Total_Renter_Occupied_Units_For_Rent_Burden',
        'B25070_007E': 'Rent_30_35_Percent_Income',
        'B25070_008E': 'Rent_35_40_Percent_Income',
        'B25070_009E': 'Rent_40_50_Percent_Income',
        'B25070_010E': 'Rent_50_Percent_Or_More_Income',

        # --- Rent and Housing Values ---
        'B25064_001E': 'Median_Gross_Rent',
        'B25077_001E': 'Median_Housing_Value',

        # --- Housing Stock Age ---
        'B25034_001E': 'Total_Housing_Units_Year_Built',
        'B25034_002E': 'Housing_Built_2020_or_Later',
        'B25034_003E': 'Housing_Built_2010_2019',
        'B25034_004E': 'Housing_Built_2000_2009',
        'B25034_005E': 'Housing_Built_1990_1999',
        'B25034_006E': 'Housing_Built_1980_1989',
        'B25034_007E': 'Housing_Built_1970_1979',
        'B25034_008E': 'Housing_Built_1960_1969',
        'B25034_009E': 'Housing_Built_1950_1959',
        'B25034_010E': 'Housing_Built_1940_1949',
        'B25034_011E': 'Housing_Built_1939_or_Earlier',

        # --- Housing Occupancy ---
        'B25002_001E': 'Total_Housing_Units',
        'B25002_002E': 'Occupied_Housing_Units',
        'B25002_003E': 'Vacant_Housing_Units',
        'B25003_001E': 'Total_Occupied_Housing_Units',
        'B25003_002E': 'Owner_Occupied_Housing_Units',
        'B25003_003E': 'Renter_Occupied_Housing_Units',

        # --- Elderly Population ---
        'B01001_020E': 'Male_60_61_Years',
        'B01001_021E': 'Male_62_64_Years',
        'B01001_022E': 'Male_65_66_Years',
        'B01001_023E': 'Male_67_69_Years',
        'B01001_024E': 'Male_70_74_Years',
        'B01001_025E': 'Male_75_Years_And_Over',
        'B01001_044E': 'Female_60_61_Years',
        'B01001_045E': 'Female_62_64_Years',
        'B01001_046E': 'Female_65_66_Years',
        'B01001_047E': 'Female_67_69_Years',
        'B01001_048E': 'Female_70_74_Years',
        'B01001_049E': 'Female_75_Years_And_Over',

        # --- Disability ---
        'B18101_001E': 'Total_Population_Disability_Status',
        'B18101_004E': 'Male_Under_5_With_Disability',
        'B18101_007E': 'Male_5_17_With_Disability',
        'B18101_010E': 'Male_18_34_With_Disability',
        'B18101_013E': 'Male_35_64_With_Disability',
        'B18101_016E': 'Male_65_74_With_Disability',
        'B18101_019E': 'Male_75_Over_With_Disability',
        'B18101_023E': 'Female_Under_5_With_Disability',
        'B18101_026E': 'Female_5_17_With_Disability',
        'B18101_029E': 'Female_18_34_With_Disability',
        'B18101_032E': 'Female_35_64_With_Disability',
        'B18101_035E': 'Female_65_74_With_Disability',
        'B18101_038E': 'Female_75_Over_With_Disability',

        # --- Renter Household Income ---
        'B25118_001E': 'Total_Renter_Occupied_HH_Income_Distribution',
        'B25118_002E': 'Renter_HH_Income_Less_5K',
        'B25118_003E': 'Renter_HH_Income_5K_10K',
        'B25118_004E': 'Renter_HH_Income_10K_15K',
        'B25118_005E': 'Renter_HH_Income_15K_20K',
        'B25118_006E': 'Renter_HH_Income_20K_25K',
        'B25118_007E': 'Renter_HH_Income_25K_35K',
        'B25118_008E': 'Renter_HH_Income_35K_50K',

        # --- Households ---
        'B11001_001E': 'Total_Households',
        'B11001_002E': 'Family_Households',

        # --- Race & Ethnicity ---
        'B03002_001E': 'Total_Population_Race_Ethnicity',
        'B03002_003E': 'Not_Hispanic_White_Alone',
        'B03002_004E': 'Black_Alone',
        'B03002_012E': 'Hispanic_Latino',

        # --- Employment ---
        'B23025_001E': 'Total_Population_16_Plus',
        'B23025_002E': 'Labor_Force',
        'B23025_005E': 'Unemployed',

        # --- SSI ---
        'B09010_001E': 'Receipt_of_SSI',
    }

    # ==========================================================================
    # DOWNLOAD DATA
    # ==========================================================================
    all_state_gdfs = []

    print(f"Starting download for {len(state_fips_list)} states/territories...")
    print(f"ACS Vintage: {ACS_VINTAGE}")
    print(f"Variables: {len(acs_variables)}")

    # Parallel download
    max_workers = 4
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_state = {
            executor.submit(download_state_data, state_fips, ACS5, ACS_VINTAGE, acs_variables): state_fips
            for state_fips in state_fips_list
        }
        print(f"Submitted {len(future_to_state)} tasks for processing...")

        for future in as_completed(future_to_state):
            state_fips = future_to_state[future]
            try:
                gdf_tracts = future.result()
                if gdf_tracts is not None:
                    print(f"Successfully processed {state_fips} with {len(gdf_tracts)} tracts.")
                    all_state_gdfs.append(gdf_tracts)
                else:
                    print(f"No data returned for {state_fips}.")
            except Exception as e:
                print(f"Error processing {state_fips}: {e}")

    print("\n--- Download Complete ---")

    # ==========================================================================
    # PROCESS AND SAVE
    # ==========================================================================
    if all_state_gdfs:
        print(f"\nConcatenating data for {len(all_state_gdfs)} states/territories...")
        final_gdf = gpd.GeoDataFrame(
            pd.concat(all_state_gdfs, ignore_index=True), 
            crs=all_state_gdfs[0].crs
        )

        print(f"Final GeoDataFrame shape: {final_gdf.shape}")
        
        # Rename columns
        final_gdf.rename(columns=acs_variable_mapping, inplace=True)
        
        # =======================================================================
        # CRITICAL: CREATE PROPER 11-DIGIT GEOID (CensusTract)
        # =======================================================================
        # This is essential for merging with NHPD data
        # Format: 2-digit state + 3-digit county + 6-digit tract = 11 digits
        #
        print("\nCreating 11-digit CensusTract GEOID...")
        
        # censusdis typically returns STATE, COUNTY, TRACT columns
        if 'STATE' in final_gdf.columns and 'COUNTY' in final_gdf.columns and 'TRACT' in final_gdf.columns:
            final_gdf['CensusTract'] = (
                final_gdf['STATE'].astype(str).str.zfill(2) +
                final_gdf['COUNTY'].astype(str).str.zfill(3) +
                final_gdf['TRACT'].astype(str).str.zfill(6)
            )
            print(f"  Created CensusTract from STATE+COUNTY+TRACT")
        elif 'GEOID' in final_gdf.columns:
            # GEOID might already be the full tract ID
            final_gdf['CensusTract'] = final_gdf['GEOID'].astype(str).str.zfill(11)
            print(f"  Created CensusTract from GEOID")
        else:
            print("  WARNING: Could not identify tract ID columns!")
            print(f"  Available columns: {final_gdf.columns.tolist()[:15]}...")
        
        # Verify format
        if 'CensusTract' in final_gdf.columns:
            sample_tracts = final_gdf['CensusTract'].head(5).tolist()
            print(f"  Sample CensusTract values: {sample_tracts}")
            tract_lengths = final_gdf['CensusTract'].str.len().value_counts()
            print(f"  Tract ID lengths: {tract_lengths.to_dict()}")
        
        # =======================================================================
        # CREATE DERIVED VARIABLES
        # =======================================================================
        final_gdf = create_derived_variables(final_gdf)
        
        # =======================================================================
        # SUMMARY STATISTICS
        # =======================================================================
        print("\n--- Summary Statistics ---")
        key_vars = ['Poverty_Rate', 'Vacancy_Rate', 'Renter_Share', 
                    'Severe_Rent_Burden_Rate', 'Pre1970_Housing_Share']
        for var in key_vars:
            if var in final_gdf.columns:
                print(f"  {var}: mean={final_gdf[var].mean():.2f}, "
                      f"median={final_gdf[var].median():.2f}, "
                      f"missing={final_gdf[var].isna().sum()}")
        
        # =======================================================================
        # SAVE
        # =======================================================================
        print(f"\nSaving to: {OUTPUT_FILENAME}")
        final_gdf.to_parquet(OUTPUT_FILENAME)
        print(f"Data saved successfully!")
        print(f"Final shape: {final_gdf.shape}")
        print(f"Columns: {len(final_gdf.columns)}")
        
    else:
        print("No data was successfully downloaded for any state. No file saved.")

    print("\nScript finished.")