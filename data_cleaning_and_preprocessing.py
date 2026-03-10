import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. FUNCTION DEFINITIONS
# ==========================================

def get_test_prefixes(columns):
    """Identifies test prefixes (T1, T2, etc.) from columns."""
    test_cols = [col for col in columns if re.match(r'^T\d+_', col)]
    prefixes = sorted(
        list(set([re.match(r'^(T\d+)_', col).group(1) for col in test_cols])), 
        key=lambda x: int(x[1:])
    )
    return prefixes

def generate_summary(datasets):
    """Analyzes groups, statuses, and tests, then exports a summary CSV."""
    print("\n--- Step 1: Analyzing groups, statuses, and tests ---")
    export_data = []

    for dataset_name, df in datasets.items():
        current_tests = get_test_prefixes(df.columns)
        
        if 'Group' in df.columns and 'status' in df.columns:
            valid_subjects = df.dropna(subset=['Group', 'status'])
            group_status_combos = valid_subjects[['Group', 'status']].drop_duplicates().sort_values(by='Group')
            
            for index, row in group_status_combos.iterrows():
                current_group = row['Group']
                current_status = row['status']
                
                subgroup_df = valid_subjects[(valid_subjects['Group'] == current_group) & 
                                             (valid_subjects['status'] == current_status)]
                total_subjects = len(subgroup_df)
                
                for test in current_tests:
                    cols_for_test = [c for c in df.columns if c.startswith(f"{test}_")]
                    skipped_test_mask = subgroup_df[cols_for_test].isna().all(axis=1)
                    num_skipped = skipped_test_mask.sum()
                    num_played = total_subjects - num_skipped
                    
                    export_data.append({
                        'Dataset': dataset_name,
                        'Group': current_group,
                        'Status': current_status,
                        'Test': test,
                        'Total_Subjects': total_subjects,
                        'Played': num_played,
                        'Skipped_Entirely': num_skipped
                    })

    output_df = pd.DataFrame(export_data)
    output_file_name = 'combined_datasets_summary.csv'
    output_df.to_csv(output_file_name, index=False)
    print(f"Analysis complete! Saved '{output_file_name}' ({len(output_df)} rows).")

def generate_heatmaps():
    """Generates heatmaps from the combined summary data."""
    print("\n--- Step 2: Generating Heatmaps ---")
    df = pd.read_csv('combined_datasets_summary.csv')
    df['Participation_%'] = (df['Played'] / df['Total_Subjects']) * 100
    df['Test_Num'] = df['Test'].str.extract(r'(\d+)').astype(int)

    datasets = ['Elderly', 'Children']
    fig, axes = plt.subplots(len(datasets), 1, figsize=(14, 12))

    for i, dataset_name in enumerate(datasets):
        subset = df[df['Dataset'] == dataset_name].copy() 
        
        if not subset.empty:
            subset['Group_Status'] = "Grp " + subset['Group'].astype(str) + " - " + subset['Status']
            pivot = subset.pivot_table(index='Group_Status', columns='Test', values='Participation_%', aggfunc='mean')
            
            test_cols = pivot.columns.tolist()
            test_cols_sorted = sorted(test_cols, key=lambda x: int(x[1:]))
            pivot = pivot[test_cols_sorted]
            
            sns.heatmap(pivot, ax=axes[i], annot=True, fmt=".0f", cmap="Blues", 
                        vmin=0, vmax=100, linewidths=.5, cbar_kws={'label': '% of Subjects Played'})
            
            axes[i].set_title(f'{dataset_name} Dataset: Test Participation by Group and Status', fontsize=14, pad=15)
            axes[i].set_ylabel('Group & Status', fontsize=12)
            axes[i].set_xlabel('Tests', fontsize=12)
            axes[i].tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig('test_participation_heatmap.png', dpi=300, bbox_inches='tight')
    print("Heatmap saved successfully as test_participation_heatmap.png")

def clean_tablet_data(input_file, output_file):
    """Cleans tablet data, consolidates dominant hands, and applies imputation."""
    print(f"\n--- Step 3: Processing and Cleaning {input_file} ---")
    
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file, na_values=['NA', 'N/A', ' ', ''])
    else:
        df = pd.read_excel(input_file, na_values=['NA', 'N/A', ' ', ''])

    # Quick fix for dirty string numbers (e.g., " 4.8 9" -> 4.89)
    # We apply this only to columns that start with T to avoid messing up IDs
    test_cols_all = [c for c in df.columns if re.match(r'^T\d+_', c)]
    for col in test_cols_all:
        if df[col].dtype == 'object':
            # Remove all spaces from the string and convert to float
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(' ', ''), errors='coerce')

    test_cols = [c for c in df.columns if re.match(r'^T\d+_', c)]
    r_cols = [c for c in test_cols if '_R_' in c]
    l_cols = [c for c in test_cols if '_L_' in c]

    common_features = set([c.replace('_R_', '_DOM_') for c in r_cols]).intersection(
                      set([c.replace('_L_', '_DOM_') for c in l_cols]))
    print(f"Consolidating {len(common_features)} paired R/L features...")

    # Create all new DOM columns at once to avoid FragmentationWarning
    new_dom_data = {col: np.nan for col in common_features}
    new_dom_df = pd.DataFrame(new_dom_data, index=df.index)
    df = pd.concat([df, new_dom_df], axis=1)

    for dom_col in common_features:
        r_col = dom_col.replace('_DOM_', '_R_')
        l_col = dom_col.replace('_DOM_', '_L_')
        
        for idx, row in df.iterrows():
            val_r = row[r_col] if r_col in df.columns else np.nan
            val_l = row[l_col] if l_col in df.columns else np.nan
            dom_hand = str(row.get('Dominant_Hand', '1')).strip() 
            
            if pd.notna(val_r) and pd.isna(val_l):
                df.at[idx, dom_col] = val_r
            elif pd.isna(val_r) and pd.notna(val_l):
                df.at[idx, dom_col] = val_l
            else:
                if dom_hand == '2' or dom_hand == '2.0':
                    df.at[idx, dom_col] = val_l
                else:
                    df.at[idx, dom_col] = val_r

    df.drop(columns=r_cols + l_cols, inplace=True, errors='ignore')

    print("Applying smart group-mean imputation...")
    new_test_cols = [c for c in df.columns if re.match(r'^T\d+_', c)]
    test_prefixes = set([re.match(r'^(T\d+)_', c).group(1) for c in new_test_cols])
    
    if 'Group' in df.columns:
        groups = df['Group'].dropna().unique()
        for group in groups:
            group_idx = df[df['Group'] == group].index
            for test in test_prefixes:
                cols_for_test = [c for c in new_test_cols if c.startswith(f"{test}_")]
                if not cols_for_test: continue
                
                skipped_mask = df.loc[group_idx, cols_for_test].isna().all(axis=1)
                played_idx = group_idx[~skipped_mask]
                
                if len(played_idx) > 0:
                    group_means = df.loc[played_idx, cols_for_test].mean(numeric_only=True)
                    for col in cols_for_test:
                        if col in group_means and pd.notna(group_means[col]):
                            df.loc[played_idx, col] = df.loc[played_idx, col].fillna(group_means[col])

    df.to_csv(output_file, index=False)
    print(f"Saved cleaned data to: {output_file}")


def process_demographics(input_file, is_children=False):
    print(f"\n{'='*50}")
    print(f"--- Processing Demographics: {input_file} ---")
    print(f"{'='*50}")
    
    df = pd.read_csv(input_file)
    
    cols_to_drop = ['Language', 'Internal_ID', 'Internal_Group', 'Dominant_Hand', 'Tablet_Use']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    if is_children:
        if 'SMA_Type' in df.columns:
            df = df.drop(columns=['SMA_Type'])
            
        text_cols = [c for c in df.columns if 'TextEntry' in str(c)]
        if text_cols:
            df = df.drop(columns=text_cols)
            
        # SPECIFIC REQUEST: Drop Group 3 from Children Data
        if 'Group' in df.columns:
            initial_len = len(df)
            # Filter out Group 3 (handling both float and int types just in case)
            df = df[~df['Group'].isin([3, 3.0, '3', '3.0'])]
            print(f"Dropped Group 3 from Children data. Removed {initial_len - len(df)} subjects.")
    
    # NEW LOGIC: Drop the Group column entirely, keep 'status' as the sole target
    if 'Group' in df.columns:
        df = df.drop(columns=['Group'])
        print("Dropped 'Group' column. 'status' is now the sole target variable.")
        
    print(f"\n--- Identifying and Dropping 100% NA Variables in Played Tests ---")
    test_cols = [c for c in df.columns if re.match(r'^T\d+_', c)]
    test_prefixes = sorted(
        list(set([re.match(r'^(T\d+)_', c).group(1) for c in test_cols])), 
        key=lambda x: int(x[1:])
    )
    
    vars_to_drop = set() 
    
    # Iterate over pure 'status' to delete the variables that has NA for some of the status groups in the played tests
    if 'status' in df.columns:
        for stat in sorted(df['status'].dropna().unique()):
            stat_df = df[df['status'] == stat]
            print(f"\n[Status: {stat}] (Total Subjects: {len(stat_df)})")
            
            for test in test_prefixes:
                cols_for_test = [c for c in test_cols if c.startswith(f"{test}_")]
                if not cols_for_test:
                    continue
                
                skipped_test_mask = stat_df[cols_for_test].isna().all(axis=1)
                
                if not skipped_test_mask.all():
                    na_vars = [col for col in cols_for_test if stat_df[col].isna().all()]
                    
                    if na_vars:
                        for var in na_vars:
                            vars_to_drop.add(var)
    
    if vars_to_drop:
        print(f"\n➔ Dropping {len(vars_to_drop)} poisoned variables from the entire dataset:")
        for v in vars_to_drop:
            print(f"      - {v}")
        df = df.drop(columns=list(vars_to_drop))
    else:
        print("\n➔ No completely missing variables found in played tests.")

    if 'ID' in df.columns:
        df = df.set_index('ID')
        
    if 'Education' in df.columns:
        if is_children:
            df = df.drop(columns=['Education'])
        else:
            median_edu = df['Education'].median()
            df['Education'] = df['Education'].fillna(median_edu)
            
    for col in ['Frequency_Tablet_Use']:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
            
    return df

# ==========================================
# 2. MAIN EXECUTION PIPELINE
# ==========================================

if __name__ == "__main__":
    print("Starting EDA Pipeline...")
    
    # Load Datasets
    try:
        print("Loading initial datasets...")
        df_elderly = pd.read_excel('data/Eldery_data.xlsx', na_values=['NA', 'N/A', ' ', ''])
        df_children = pd.read_excel('data/Children_data.xlsx', na_values=['NA', 'N/A', ' ', '']) 
        
        datasets_dict = {
            "Elderly": df_elderly,
            "Children": df_children
        }
        
        # Run Sequence
        generate_summary(datasets_dict)
        generate_heatmaps()
        
    except Exception as e:
        print(f"Error loading initial files for summary/heatmap: {e}")
        print("Skipping Steps 1 & 2. Please ensure data files are in the 'data/' folder.")

    # Process Cleaned Data files (independent file loading inside the function)
    try:
        clean_tablet_data('data/Eldery_data.xlsx', 'data/Eldery_data_Cleaned_DominantOnly.csv') 
        clean_tablet_data('data/Children_data.xlsx', 'data/Children_data_Cleaned_DominantOnly.csv')
    except Exception as e:
        print(f"Error during data cleaning step: {e}")
        
    print("\nEDA Pipeline Finished.")


    # Apply the processing function to both files
    df_elderly_final = process_demographics('data/Eldery_data_Cleaned_DominantOnly.csv', is_children=False)
    df_children_final = process_demographics('data/Children_data_Cleaned_DominantOnly.csv', is_children=True)

    # Save the final, ML-Ready DataFrames
    elderly_output = 'data/Eldery_Final_ML_Ready.csv'
    children_output = 'data/Children_Final_ML_Ready.csv'

    df_elderly_final.to_csv(elderly_output)
    df_children_final.to_csv(children_output)

    print("\n--- FINAL RESULTS ---")
    print(f"Elderly Dataset saved to '{elderly_output}'. Shape: {df_elderly_final.shape}")
    print(f"Children Dataset saved to '{children_output}'. Shape: {df_children_final.shape}")