import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import hdbscan
import warnings

warnings.filterwarnings('ignore')

def run_hdbscan_clustering(csv_file, json_file, tests_to_include, scenario_name=""):
    print(f"\n{'='*60}")
    print(f"--- Running HDBSCAN Clustering: {scenario_name} ---")
    
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
    final_features = [f for f in valid_features if f.startswith(test_prefixes)]
    
    # 3. Drop NA for target
    df = df.dropna(subset=[target_col])
    
    # 4. IMPUTATION: Median (for missing Parkinson's tests)
    print("Imputing missing tablet test data with the median...")
    for col in final_features:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    X = df[final_features].copy()
    y = df[target_col].copy()
    
    # 5. Log-Transform & Scale
    for col in skewed_vars:
        if col in X.columns:
            X[col] = np.log1p(np.clip(X[col], 0, None))
            
    X_scaled = StandardScaler().fit_transform(X)
    
    # 6. t-SNE Visualization (We do this first to map the space)
    print("Running t-SNE to generate the map...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_viz = tsne.fit_transform(X_scaled)
    
    # 7. HDBSCAN Clustering
    print("Running HDBSCAN...")
    # min_cluster_size dictates how small a cluster can be to be considered its own group
    # We set it to 10 so it doesn't create a million tiny clusters
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric='euclidean')
    
    # We cluster directly on the t-SNE map so the colors match the visual islands perfectly!
    cluster_labels = clusterer.fit_predict(X_viz)
    
    plot_df = pd.DataFrame({
        'Dim_1': X_viz[:, 0],
        'Dim_2': X_viz[:, 1],
        'True_Status': y.values,
        'Cluster': [f"Cluster {c}" if c != -1 else "Noise (Overlap)" for c in cluster_labels]
    })
    
    # 8. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot A: Ground Truth
    sns.scatterplot(data=plot_df, x='Dim_1', y='Dim_2', hue='True_Status', palette='tab10', ax=axes[0], alpha=0.8, s=60)
    axes[0].set_title(f'Actual Clinical Status\n{scenario_name}')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    # Plot B: HDBSCAN
    # Create a custom palette: Grey for Noise (-1), distinct colors for actual clusters
    unique_clusters = plot_df['Cluster'].unique()
    palette = sns.color_palette("Set2", len(unique_clusters))
    color_dict = {cluster: color for cluster, color in zip(unique_clusters, palette)}
    if "Noise (Overlap)" in color_dict:
        color_dict["Noise (Overlap)"] = (0.7, 0.7, 0.7) # Set noise to grey
        
    sns.scatterplot(data=plot_df, x='Dim_1', y='Dim_2', hue='Cluster', palette=color_dict, ax=axes[1], alpha=0.9, s=60)
    axes[1].set_title(f'HDBSCAN Density Clustering\n(Grey = Overlapping/Ambiguous Patients)')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    plt.tight_layout()
    filename = scenario_name.replace(" ", "_").replace(",", "").replace("'", "") + "_HDBSCAN.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Success! Saved visualization to '{filename}'\n")

if __name__ == "__main__":
    elderly_target_tests = ['T1', 'T2', 'T5', 'T12', 'T13', 'T20']
    
    """    run_hdbscan_clustering(
        csv_file='data/Eldery_Final_ML_Ready.csv', 
        json_file='data/Eldery_feature_metadata.json',
        tests_to_include=elderly_target_tests, 
        scenario_name="Elderly - Imputed 6 Tests"
    )"""

    run_hdbscan_clustering(
    csv_file='data/Children_Final_ML_Ready.csv', 
    json_file='data/Children_feature_metadata.json',
    tests_to_include=elderly_target_tests, 
    scenario_name="Children - Core Battery"
)