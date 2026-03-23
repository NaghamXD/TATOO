import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')

def coerce_numeric(df):
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['status', 'Group_Status', 'ID']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(' ', ''), errors='coerce')
    return df

def plot_imputed_demographic_tsne(csv_file, json_file, tests_to_include, demographics, dataset_name, exclude_groups=None):
    print(f"\n--- Generating Imputed Demographic t-SNE for {dataset_name} ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(csv_file)
        with open(json_file, 'r') as f:
            meta = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find CSV or JSON file.")
        return

    df = coerce_numeric(df)
    target_col = 'status' if 'status' in df.columns else 'Group_Status'
    
    # 2. Exclude specified groups (if requested)
    if exclude_groups:
        initial_len = len(df)
        df = df[~df[target_col].isin(exclude_groups)]
        print(f"Excluded groups: {exclude_groups}. Dropped {initial_len - len(df)} subjects.")

    # 3. FIX THE DATA LEAK: Neutralize the -1 in Frequency_Tablet_Use
    if 'Frequency_Tablet_Use' in df.columns:
        df['Frequency_Tablet_Use'] = df['Frequency_Tablet_Use'].replace(-1, np.nan)
        df['Frequency_Tablet_Use'] = df['Frequency_Tablet_Use'].fillna(df['Frequency_Tablet_Use'].median())

    # 4. Filter features based on JSON
    sig_vars = set(meta['significant_vars'])
    redundant_vars = set(meta['redundant_vars_to_drop'])
    skewed_vars = set(meta['skewed_vars_for_log_transform'])
    
    valid_features = list(sig_vars - redundant_vars)
    test_prefixes = tuple([f"{t}_" for t in tests_to_include])
    final_scenario_features = [f for f in valid_features if f.startswith(test_prefixes)]
    
    if not final_scenario_features:
        print("Error: No valid features found for these specific tests!")
        return

    print(f"Tests included: {tests_to_include}")
    print(f"Using {len(final_scenario_features)} total features.")

    valid_demographics = [d for d in demographics if d in df.columns]

    # 5. Drop NA strictly for the target column and demographics
    df = df.dropna(subset=[target_col] + valid_demographics)
    
    # 6. IMPUTATION: Fill missing test data with the median
    print("Imputing missing tablet test data with the median...")
    for col in final_scenario_features:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    X = df[final_scenario_features].copy()
    y = df[target_col].copy()
    
    # 7. Log-Transform (with np.clip to fix negative sensor glitches)
    for col in skewed_vars:
        if col in X.columns:
            X[col] = np.log1p(np.clip(X[col], 0, None))
            
    # 8. Scale Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 9. Run t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_viz = tsne.fit_transform(X_scaled)
    
    # 10. Create Plotting DataFrame
    plot_df = pd.DataFrame({
        'Dim_1': X_viz[:, 0],
        'Dim_2': X_viz[:, 1],
        'Status': y.values
    })
    
    # Add Demographics to plot dataframe
    for demo in valid_demographics:
        plot_df[demo] = df[demo].values

    # 11. Plotting Grid
    n_plots = len(valid_demographics) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
    
    # Plot A: Ground Truth Clinical Status
    sns.scatterplot(data=plot_df, x='Dim_1', y='Dim_2', hue='Status', palette='tab10', ax=axes[0], s=60, alpha=0.8)
    axes[0].set_title(f"Clinical Status\n({dataset_name})")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    # Plots B-E: Demographics
    for i, demo in enumerate(valid_demographics):
        ax = axes[i+1]
        
        # Continuous (Age, Education) -> Gradient color map
        if plot_df[demo].nunique() > 5:
            scatter = ax.scatter(plot_df['Dim_1'], plot_df['Dim_2'], c=plot_df[demo], cmap='coolwarm', s=60, alpha=0.8)
            plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        # Categorical (Gender) -> Discrete color map
        else:
            plot_df[demo] = plot_df[demo].astype(str)
            sns.scatterplot(data=plot_df, x='Dim_1', y='Dim_2', hue=demo, palette='Set2', ax=ax, s=60, alpha=0.8)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            
        ax.set_title(f"Colored by {demo}")
        
    plt.tight_layout()
    filename = f"{dataset_name.replace(' ', '_')}_Imputed_Demographic_tSNE.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}\n")


if __name__ == "__main__":
    
    # We test the entire 6-Test Battery you want to analyze
    target_tests = ['T1', 'T2', 'T5', 'T12', 'T13', 'T20']
    
    # 1. Elderly Data 
    elderly_demographics = ['Age', 'Gender', 'Education', 'Frequency_Tablet_Use']
    plot_imputed_demographic_tsne(
        csv_file='data/Eldery_Final_ML_Ready.csv', 
        json_file='data/Eldery_feature_metadata.json',
        tests_to_include=target_tests, 
        demographics=elderly_demographics, 
        dataset_name="Elderly - All 6 Tests",
        exclude_groups=None
    )
    
    # 2. Children Data
    children_demographics = ['Age', 'Gender', 'Frequency_Tablet_Use']
    plot_imputed_demographic_tsne(
        csv_file='data/Children_Final_ML_Ready.csv', 
        json_file='data/Children_feature_metadata.json',
        tests_to_include=target_tests, 
        demographics=children_demographics, 
        dataset_name="Children - All 6 Tests",
        exclude_groups=None
    )