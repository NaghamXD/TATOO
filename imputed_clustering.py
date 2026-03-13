import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

def run_imputed_clustering(csv_file, json_file, tests_to_include, target_clusters=5, scenario_name=""):
    print(f"\n{'='*60}")
    print(f"--- Running Imputed Clustering: {scenario_name} ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(csv_file)
        with open(json_file, 'r') as f:
            meta = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find CSV or JSON file.")
        return
        
    target_col = 'status' if 'status' in df.columns else 'Group_Status'

    # 2. Filter features based on JSON
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
    print(f"Using {len(final_scenario_features)} features.")
    
    # 3. Safely drop NA strictly for the target column
    df = df.dropna(subset=[target_col])
    
    # 4. IMPUTATION: Fill missing test data with the median
    print("Imputing missing tablet test data with the median...")
    for col in final_scenario_features:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    X = df[final_scenario_features].copy()
    y = df[target_col].copy()
    
    # 5. Log-Transform (with np.clip to fix negative sensor glitches)
    for col in skewed_vars:
        if col in X.columns:
            X[col] = np.log1p(np.clip(X[col], 0, None))
            
    # 6. Scale Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 7. Gaussian Mixture Model (GMM) Soft Clustering
    print("Running Gaussian Mixture Model (Agglomerative)...")
    gmm = AgglomerativeClustering(n_clusters=target_clusters, linkage='ward')
    cluster_labels = gmm.fit_predict(X_scaled)
    
    
    # 8. t-SNE for Visualization
    print("Running t-SNE to generate the map...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_viz = tsne.fit_transform(X_scaled)
    
    plot_df = pd.DataFrame({
        'Dim_1': X_viz[:, 0],
        'Dim_2': X_viz[:, 1],
        'True_Status': y.values,
        'Cluster': [f"Cluster {c}" for c in cluster_labels]
    })
    
    # 9. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.scatterplot(data=plot_df, x='Dim_1', y='Dim_2', hue='True_Status', palette='tab10', ax=axes[0], alpha=0.8, s=60)
    axes[0].set_title(f'Actual Clinical Status (t-SNE)\n{scenario_name}')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    sns.scatterplot(data=plot_df, x='Dim_1', y='Dim_2', hue='Cluster', palette='Set2', ax=axes[1], alpha=0.8, s=60)
    axes[1].set_title(f'Agglomerative Clustering ({target_clusters} Clusters)\n{scenario_name}')
    
    plt.tight_layout()
    filename = scenario_name.replace(" ", "_").replace(",", "").replace("'", "") + "_Imputed_Agglomerative.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Success! Saved visualization to '{filename}'\n")

def run_imputed_pca_agglomerative(csv_file, json_file, tests_to_include, target_clusters=4, scenario_name=""):
    print(f"\n{'='*60}")
    print(f"--- Running Imputed Clustering (PCA + Agglomerative): {scenario_name} ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(csv_file)
        with open(json_file, 'r') as f:
            meta = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find CSV or JSON file.")
        return
        
    target_col = 'status' if 'status' in df.columns else 'Group_Status'

    # 2. Filter features
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
    print(f"Using {len(final_scenario_features)} features.")
    
    # 3. Drop NA for target
    df = df.dropna(subset=[target_col])
    
    # 4. IMPUTATION: Median
    print("Imputing missing tablet test data with the median...")
    for col in final_scenario_features:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    X = df[final_scenario_features].copy()
    y = df[target_col].copy()
    
    # 5. Log-Transform
    for col in skewed_vars:
        if col in X.columns:
            X[col] = np.log1p(np.clip(X[col], 0, None))
            
    # 6. Scale Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 7. Agglomerative Clustering
    print("Running Agglomerative Clustering...")
    agg_cluster = AgglomerativeClustering(n_clusters=target_clusters, linkage='ward')
    cluster_labels = agg_cluster.fit_predict(X_scaled)
    
    # 8. PCA Visualization
    print("Running PCA to generate the map...")
    pca = PCA(n_components=2, random_state=42)
    X_viz = pca.fit_transform(X_scaled)
    
    plot_df = pd.DataFrame({
        'PCA_1': X_viz[:, 0],
        'PCA_2': X_viz[:, 1],
        'True_Status': y.values,
        'Cluster': [f"Cluster {c}" for c in cluster_labels]
    })
    
    # 9. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(data=plot_df, x='PCA_1', y='PCA_2', hue='True_Status', palette='tab10', ax=axes[0], alpha=0.8, s=60)
    axes[0].set_title(f'Actual Clinical Status (PCA)\n{scenario_name}')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    sns.scatterplot(data=plot_df, x='PCA_1', y='PCA_2', hue='Cluster', palette='Set2', ax=axes[1], alpha=0.8, s=60)
    axes[1].set_title(f'Agglomerative Clustering ({target_clusters} Clusters)\n{scenario_name}')
    
    plt.tight_layout()
    filename = scenario_name.replace(" ", "_").replace(",", "").replace("'", "") + "_Imputed_PCA_Agg.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Success! Saved visualization to '{filename}'\n")

if __name__ == "__main__":
    # ==========================================================
    # 1. Elderly Data (GMM + t-SNE for Parkinson's check)
    # ==========================================================
    elderly_target_tests = ['T1', 'T2', 'T5', 'T12', 'T13', 'T20']
    run_imputed_clustering(
        csv_file='data/Eldery_Final_ML_Ready.csv', 
        json_file='data/Eldery_feature_metadata.json',
        tests_to_include=elderly_target_tests, 
        target_clusters=5, 
        scenario_name="Elderly - All 6 Tests (Imputed)"
    )

    # ==========================================================
    # 2. Children Data (PCA + Agglomerative on T1, T12, T20)
    # ==========================================================
    children_target_tests = ['T1', 'T12', 'T20']
    
    run_imputed_pca_agglomerative(
        csv_file='data/Children_Final_ML_Ready.csv', 
        json_file='data/Children_feature_metadata.json',
        tests_to_include=children_target_tests, 
        target_clusters=3, # Set to 3 because Children usually have 3 classes: Healthy, SMA, COM. Adjust if needed!
        scenario_name="Children - T1, T12, T20"
    )
