import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

def run_targeted_clustering(csv_file, json_file, tests_to_include, exclude_groups=None, target_clusters=4, scenario_name=""):
    print(f"\n{'='*60}")
    print(f"--- Running Targeted Clustering: {scenario_name} ---")
    
    # 1. Load Data
    df = pd.read_csv(csv_file)
    with open(json_file, 'r') as f:
        meta = json.load(f)
        
    target_col = 'status' if 'status' in df.columns else 'Group_Status'
    
    # 2. Exclude specified groups (e.g., dropping Parkinson's)
    if exclude_groups:
        initial_len = len(df)
        df = df[~df[target_col].isin(exclude_groups)]
        print(f"Excluded groups: {exclude_groups}. Dropped {initial_len - len(df)} subjects.")

    # 3. Filter features: Must be significant, not redundant, AND belong to the requested tests
    sig_vars = set(meta['significant_vars'])
    redundant_vars = set(meta['redundant_vars_to_drop'])
    skewed_vars = set(meta['skewed_vars_for_log_transform'])
    
    valid_features = list(sig_vars - redundant_vars)
    
    # Format test prefixes safely (e.g., 'T1' -> 'T1_') so 'T1' doesn't accidentally grab 'T12'
    test_prefixes = tuple([f"{t}_" for t in tests_to_include])
    
    # Keep only the features that belong to the tests we want for this specific scenario
    final_scenario_features = [f for f in valid_features if f.startswith(test_prefixes)]
    
    if not final_scenario_features:
        print("Error: No valid features found for these specific tests after JSON filtering!")
        return

    print(f"Tests included: {tests_to_include}")
    print(f"Using {len(final_scenario_features)} features for this scenario.")
    
    # 4. NOW we safely drop NA, knowing it's only looking at our targeted tests
    initial_len = len(df)
    df = df.dropna(subset=final_scenario_features + [target_col])
    print(f"Dropped {initial_len - len(df)} subjects due to missing data in THESE specific tests. Remaining subjects: {len(df)}")
    
    if len(df) < target_clusters:
        print("Not enough data left to cluster! Skipping...")
        return

    X = df[final_scenario_features].copy()
    y = df[target_col].copy()
    
    # 5. Log-Transform
    for col in skewed_vars:
        if col in X.columns:
            # Clip at 0 to prevent negative sensor glitches from causing NaN in log1p
            X[col] = np.log1p(np.clip(X[col], 0, None))
            
    # 6. Scale & Cluster
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=target_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # 7. PCA for Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plot_df = pd.DataFrame({
        'PCA_1': X_pca[:, 0],
        'PCA_2': X_pca[:, 1],
        'True_Status': y.values,
        'KMeans_Cluster': [f"Cluster {c}" for c in cluster_labels]
    })
    
    # 8. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.scatterplot(data=plot_df, x='PCA_1', y='PCA_2', hue='True_Status', palette='tab10', ax=axes[0], alpha=0.8, s=60)
    axes[0].set_title(f'Actual Clinical Status (Ground Truth)\n{scenario_name}')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    sns.scatterplot(data=plot_df, x='PCA_1', y='PCA_2', hue='KMeans_Cluster', palette='Set2', ax=axes[1], alpha=0.8, s=60)
    axes[1].set_title(f'K-Means Blind Clustering ({target_clusters} Clusters)\n{scenario_name}')
    
    plt.tight_layout()
    filename = scenario_name.replace(" ", "_").replace(",", "") + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Success! Saved visualization to '{filename}'\n")


if __name__ == "__main__":
    # ==========================================
    # ELDERLY SCENARIOS
    # ==========================================
    try:
        # Scenario 1: The "Universal" Elderly Clustering (All groups, fewer tests)
        run_targeted_clustering(
            csv_file='Eldery_Final_ML_Ready.csv', 
            json_file='Eldery_feature_metadata.json',
            tests_to_include=['T5', 'T13', 'T20'], 
            exclude_groups=None, 
            target_clusters=5, 
            scenario_name="Elderly - Universal Battery (T5, T13, T20)"
        )
        
        # Scenario 2: The "Deep Dive" Elderly Clustering (Exclude Parkinson's, more tests)
        run_targeted_clustering(
            csv_file='Eldery_Final_ML_Ready.csv', 
            json_file='Eldery_feature_metadata.json',
            tests_to_include=['T1', 'T2', 'T5', 'T12', 'T13', 'T20'], 
            exclude_groups=['Parkinson'], 
            target_clusters=4, 
            scenario_name="Elderly - Deep Dive (No Parkinson's)"
        )
    except FileNotFoundError:
        print("Elderly files not found.")

    # ==========================================
    # CHILDREN SCENARIOS
    # ==========================================
    try:
        # Scenario 1: Base Children Clustering
        run_targeted_clustering(
            csv_file='Children_Final_ML_Ready.csv', 
            json_file='Children_feature_metadata.json',
            tests_to_include=['T1', 'T2', 'T5', 'T12', 'T13', 'T20'], 
            exclude_groups=None, 
            target_clusters=4, 
            scenario_name="Children - Core Battery"
        )
        
        '''# Scenario 2: Children Deep Dive (Adding T1)
        run_targeted_clustering(
            csv_file='Children_Final_ML_Ready.csv', 
            json_file='Children_feature_metadata.json',
            tests_to_include=['T1', 'T2', 'T5', 'T12', 'T13', 'T20'], 
            exclude_groups=None, 
            target_clusters=3, 
            scenario_name="Children - Extended Battery (With T1)"
        )'''
    except FileNotFoundError:
        print("Children files not found.")