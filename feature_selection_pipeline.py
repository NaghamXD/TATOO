import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import re
import json

def detect_confounders(df, test_var, demographic_cols):
    """Mathematically checks if demographic variables significantly affect the test variable."""
    confounders = []
    for demo in demographic_cols:
        if demo not in df.columns:
            continue
            
        clean_df = df.dropna(subset=[test_var, demo])
        if len(clean_df) < 10: 
            continue 

        if pd.api.types.is_numeric_dtype(df[demo]) and df[demo].nunique() > 5:
            corr, p_val = stats.spearmanr(clean_df[test_var], clean_df[demo])
            if p_val < 0.05 and abs(corr) > 0.2: 
                confounders.append(demo)
        else:
            unique_cats = clean_df[demo].unique()
            groups = [clean_df[clean_df[demo] == cat][test_var].values for cat in unique_cats]
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) >= 2:
                if len(groups) == 2:
                    stat, p_val = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                else:
                    stat, p_val = stats.kruskal(*groups)
                if p_val < 0.05:
                    confounders.append(demo)
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

    # ==========================================
    # STEP 1 & 2 & 3: Stats, Skewness & Confounders
    # ==========================================
    for col in numeric_test_cols:
        clean_df = df.dropna(subset=[col, 'status'])
        current_statuses = clean_df['status'].unique()
        
        groups = [clean_df[clean_df['status'] == s][col].values for s in current_statuses]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) < 2:
            results["insignificant_vars_to_drop"].append(col)
            continue
            
        all_values = pd.concat([pd.Series(g) for g in groups])
        if all_values.nunique() <= 1:
            results["insignificant_vars_to_drop"].append(col)
            continue

        if len(groups) == 2:
            stat, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
        else:
            stat, p_value = stats.kruskal(*groups)

        if p_value < 0.05:
            results["significant_vars"].append(col)
            
            skew_val = clean_df[col].skew()
            if abs(skew_val) > 1.0: 
                results["skewed_vars_for_log_transform"].append(col)
                
            confounders = detect_confounders(clean_df, col, demographic_cols)
            if confounders:
                results["confounders_map"][col] = confounders
        else:
            results["insignificant_vars_to_drop"].append(col)

    # ==========================================
    # STEP 4: Test-to-Test Correlation (Redundancy Check)
    # ==========================================
    if len(results["significant_vars"]) > 1:
        # We use Spearman correlation because physiological data is often skewed
        sig_df = df[results["significant_vars"]]
        corr_matrix = sig_df.corr(method='spearman').abs()
        
        # Look only at the upper triangle of the matrix to avoid duplicate pairs
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        redundant_set = set()
        
        for column in upper_triangle.columns:
            # Find any variable highly correlated with this column
            correlated_with = upper_triangle.index[upper_triangle[column] > corr_threshold].tolist()
            if correlated_with:
                for match in correlated_with:
                    correlation_value = round(upper_triangle.loc[match, column], 3)
                    # Save the pair so you can see exactly who is correlated with who
                    results["highly_correlated_pairs"].append(f"{match} & {column} (r={correlation_value})")
                
                # Tag this column to be dropped since it's redundant
                redundant_set.add(column)
                
        results["redundant_vars_to_drop"] = list(redundant_set)

    # ==========================================
    # STEP 5: Export Results
    # ==========================================
    print(f"  -> {len(results['significant_vars'])} Significant variables kept.")
    print(f"  -> {len(results['insignificant_vars_to_drop'])} Insignificant variables tagged for dropping.")
    print(f"  -> {len(results['skewed_vars_for_log_transform'])} Flagged for Log Transformation.")
    print(f"  -> {len(results['confounders_map'])} Variables have detected demographic confounders.")
    print(f"  -> {len(results['redundant_vars_to_drop'])} Redundant variables tagged for dropping (Correlation > {corr_threshold}).")

    output_filename = f"{dataset_name}_feature_metadata.json"
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSUCCESS: All lists saved to '{output_filename}'")

if __name__ == "__main__":
    my_demographics = ['Age', 'Gender', 'Education', 'Frequency_Tablet_Use']
    
    # You can change corr_threshold to 0.90 if you want to be more strict about dropping variables
    run_feature_selection_pipeline('Children_Final_ML_Ready.csv', my_demographics, corr_threshold=0.85)
    run_feature_selection_pipeline('Eldery_Final_ML_Ready.csv', my_demographics, corr_threshold=0.85)
