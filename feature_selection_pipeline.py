import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import re
import json
import warnings

# Suppress harmless SciPy ties/variance warnings for clean console output
warnings.filterwarnings('ignore')

def calc_mann_whitney_effect_size(U, n1, n2):
    """Calculates Rank-Biserial Correlation as an effect size (0 to 1)."""
    return abs(1 - (2 * U) / (n1 * n2))

def calc_kruskal_effect_size(H, k, n):
    """Calculates Eta-squared as an effect size for Kruskal-Wallis (0 to 1)."""
    if n <= k: return 0
    return max(0, (H - k + 1) / (n - k))

def detect_confounders(df, test_var, demographic_cols):
    """Mathematically checks if demographic variables significantly affect the test variable."""
    confounders = []
    for demo in demographic_cols:
        if demo not in df.columns:
            continue
            
        clean_df = df.dropna(subset=[test_var, demo])
        if len(clean_df) < 10: 
            continue 

        # Continuous Demographics (e.g., Age)
        if pd.api.types.is_numeric_dtype(df[demo]) and df[demo].nunique() > 5:
            corr, p_val = stats.spearmanr(clean_df[test_var], clean_df[demo])
            if p_val < 0.05 and abs(corr) > 0.2: 
                confounders.append(demo)
                
        # Categorical Demographics (e.g., Gender, Frequency_Tablet_Use)
        else:
            unique_cats = clean_df[demo].dropna().unique()
            groups = [clean_df[clean_df[demo] == cat][test_var].values for cat in unique_cats]
            
            # IMPROVEMENT 4: Enforce a minimum of 5 subjects per group for statistical validity
            groups = [g for g in groups if len(g) >= 5]
            
            if len(groups) == 2:
                try:
                    stat, p_val = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                    effect_size = calc_mann_whitney_effect_size(stat, len(groups[0]), len(groups[1]))
                    # IMPROVEMENT 1: Only flag if p<0.05 AND effect size is medium/large (>0.3)
                    if p_val < 0.05 and effect_size > 0.3:
                        confounders.append(demo)
                except ValueError:
                    # Catches zero-variance errors
                    pass
                    
            elif len(groups) > 2:
                try:
                    stat, p_val = stats.kruskal(*groups)
                    n_total = sum(len(g) for g in groups)
                    effect_size = calc_kruskal_effect_size(stat, len(groups), n_total)
                    # IMPROVEMENT 1: Only flag if p<0.05 AND effect size is medium (>0.06)
                    if p_val < 0.05 and effect_size > 0.06:
                        confounders.append(demo)
                except ValueError:
                    pass
                    
    return confounders

def run_feature_selection_pipeline(file_path, demographic_cols, corr_threshold=0.85):
    dataset_name = os.path.basename(file_path).split('_')[0]
    print(f"\n{'='*60}")
    print(f"Running Automated Feature Selection for: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}.")
        return

    if 'status' not in df.columns:
        print("Error: 'status' column not found.")
        return

    test_cols = [c for c in df.columns if re.match(r'^T\d+_', c)]
    numeric_test_cols = df[test_cols].select_dtypes(include=['float64', 'int64']).columns.tolist()
    print(f"Total test variables found: {len(numeric_test_cols)}")

    results = {
        "dataset": dataset_name,
        "significant_vars": [],
        "insignificant_vars_to_drop": [],
        "skewed_vars_for_log_transform": [],
        "confounders_map": {},
        "highly_correlated_pairs": [],
        "redundant_vars_to_drop": []
    }

    # Dictionary to map feature to its p-value for the smart redundancy check
    feature_p_values = {}

    # ==========================================
    # STEP 1 & 2 & 3: Stats, Skewness & Confounders
    # ==========================================
    for col in numeric_test_cols:
        clean_df = df.dropna(subset=[col, 'status'])
        current_statuses = clean_df['status'].unique()
        
        groups = [clean_df[clean_df['status'] == s][col].values for s in current_statuses]
        
        # IMPROVEMENT 4: Minimum 5 subjects per clinical status group
        groups = [g for g in groups if len(g) >= 5]
        
        if len(groups) < 2:
            results["insignificant_vars_to_drop"].append(col)
            continue
            
        all_values = pd.concat([pd.Series(g) for g in groups])
        if all_values.nunique() <= 1:
            results["insignificant_vars_to_drop"].append(col)
            continue

        try:
            if len(groups) == 2:
                stat, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            else:
                stat, p_value = stats.kruskal(*groups)
        except ValueError:
            # Safely catch cases where groups have identical values (zero variance)
            results["insignificant_vars_to_drop"].append(col)
            continue

        if p_value < 0.05:
            results["significant_vars"].append(col)
            feature_p_values[col] = p_value  # Store p-value for tie-breaking later
            
            # IMPROVEMENT 3: Check skewness only within the largest group (often the Control group)
            largest_group = max(groups, key=len)
            skew_val = pd.Series(largest_group).skew()
            if abs(skew_val) > 1.0: 
                results["skewed_vars_for_log_transform"].append(col)
                
            confounders = detect_confounders(clean_df, col, demographic_cols)
            if confounders:
                results["confounders_map"][col] = confounders
        else:
            results["insignificant_vars_to_drop"].append(col)

    # ==========================================
    # STEP 4: Test-to-Test Correlation (Smart Redundancy Check)
    # ==========================================
    if len(results["significant_vars"]) > 1:
        sig_df = df[results["significant_vars"]]
        corr_matrix = sig_df.corr(method='spearman').abs()
        
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        redundant_set = set()
        
        for column in upper_triangle.columns:
            if column in redundant_set: 
                continue # Already tagged to drop
                
            correlated_with = upper_triangle.index[upper_triangle[column] > corr_threshold].tolist()
            
            for match in correlated_with:
                if match in redundant_set: 
                    continue
                    
                correlation_value = round(upper_triangle.loc[match, column], 3)
                results["highly_correlated_pairs"].append(f"{match} & {column} (r={correlation_value})")
                
                # IMPROVEMENT 2: Smart Drop Logic
                # Compare p-values. The variable with the LOWER p-value is kept because 
                # it is more statistically significant to the clinical status.
                p_val_col = feature_p_values.get(column, 1.0)
                p_val_match = feature_p_values.get(match, 1.0)
                
                if p_val_col > p_val_match:
                    redundant_set.add(column)
                    break # Column is dropped, move to the next 'column' in the outer loop
                else:
                    redundant_set.add(match)
                
        results["redundant_vars_to_drop"] = list(redundant_set)

    # ==========================================
    # STEP 5: Export Results
    # ==========================================
    print(f"  -> {len(results['significant_vars'])} Significant variables initially kept.")
    print(f"  -> {len(results['redundant_vars_to_drop'])} Redundant variables tagged for dropping (Keeping the most predictive ones).")
    print(f"  -> {len(results['significant_vars']) - len(results['redundant_vars_to_drop'])} Final features remain for modeling.")
    print(f"  -> {len(results['insignificant_vars_to_drop'])} Insignificant variables tagged for dropping.")
    print(f"  -> {len(results['skewed_vars_for_log_transform'])} Flagged for Log Transformation.")
    print(f"  -> {len(results['confounders_map'])} Variables have detected demographic confounders.")

    output_filename = f"{dataset_name}_feature_metadata.json"
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSUCCESS: All lists saved to '{output_filename}'")

if __name__ == "__main__":
    my_demographics = ['Age', 'Gender', 'Education', 'Frequency_Tablet_Use']
    
    run_feature_selection_pipeline('Children_Final_ML_Ready.csv', my_demographics, corr_threshold=0.85)
    run_feature_selection_pipeline('Eldery_Final_ML_Ready.csv', my_demographics, corr_threshold=0.85)