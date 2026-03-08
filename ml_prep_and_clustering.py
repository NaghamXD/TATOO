import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

def load_and_prep_data(csv_file, json_file):
    print(f"\n{'='*50}\nPreparing ML Data for: {csv_file}\n{'='*50}")
    
    # 1. Load Data and Metadata
    df = pd.read_csv(csv_file)
    with open(json_file, 'r') as f:
        meta = json.load(f)
        
    target_col = 'status' if 'status' in df.columns else 'Group_Status'
    
    # 2. Filter Features based on JSON pipeline results
    sig_vars = set(meta['significant_vars'])
    redundant_vars = set(meta['redundant_vars_to_drop'])
    skewed_vars = set(meta['skewed_vars_for_log_transform'])
    
    # Keep only significant features that are NOT redundant
    final_features = list(sig_vars - redundant_vars)
    print(f"Using {len(final_features)} highly predictive, non-redundant features.")
    
    # Drop rows with missing values in our final features or target
    df = df.dropna(subset=final_features + [target_col])
    
    X = df[final_features].copy()
    y = df[target_col].copy()
    
    # 3. Log-Transform skewed variables
    for col in skewed_vars:
        if col in X.columns:
            # log1p handles 0 values safely by doing log(1+x)
            X[col] = np.log1p(X[col])
            
    # 4. Train/Test Split (Stratified ensures equal class distribution)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Scale the Data (Fit ONLY on training data to prevent leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to dataframes for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    return X, y, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def run_unsupervised_clustering(X, y, dataset_name):
    print(f"\n--- Running K-Means Clustering for {dataset_name} ---")
    
    # We scale the full dataset for unsupervised visualization
    scaler = StandardScaler()
    X_scaled_full = scaler.fit_transform(X)
    
    # Figure out how many unique clinical statuses we actually have
    n_clusters = len(y.unique())
    print(f"Looking for {n_clusters} natural clusters...")
    
    # Run K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled_full)
    
    # Let's compress our 100+ features into 2 dimensions using PCA so we can draw a 2D map
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled_full)
    
    # Create a plotting dataframe
    plot_df = pd.DataFrame({
        'PCA_1': X_pca[:, 0],
        'PCA_2': X_pca[:, 1],
        'True_Status': y.values,
        'KMeans_Cluster': [f"Cluster {c}" for c in cluster_labels]
    })
    
    # Plot True Status vs KMeans Clusters side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Ground Truth
    sns.scatterplot(data=plot_df, x='PCA_1', y='PCA_2', hue='True_Status', palette='tab10', ax=axes[0], alpha=0.8)
    axes[0].set_title(f'Actual Clinical Status (Ground Truth)\n{dataset_name}')
    
    # Plot 2: What K-Means found blind
    sns.scatterplot(data=plot_df, x='PCA_1', y='PCA_2', hue='KMeans_Cluster', palette='Set2', ax=axes[1], alpha=0.8)
    axes[1].set_title(f'K-Means Blind Clustering\n{dataset_name}')
    
    plt.tight_layout()
    image_name = f'{dataset_name}_Clustering_PCA.png'
    plt.savefig(image_name, dpi=300)
    print(f"Saved clustering visualization to '{image_name}'")

if __name__ == "__main__":
    # --- Process Elderly Dataset ---
    # Ensure you run your feature selection pipeline first to generate the JSON!
    try:
        X_eld, y_eld, X_train_e, X_test_e, y_train_e, y_test_e, scaler_e = load_and_prep_data(
            'Eldery_Final_ML_Ready.csv', 'Eldery_feature_metadata.json'
        )
        run_unsupervised_clustering(X_eld, y_eld, "Elderly")
    except FileNotFoundError:
        print("Could not find the Elderly JSON file. Run the feature selection pipeline first!")

    # --- Process Children Dataset ---
    try:
        X_chi, y_chi, X_train_c, X_test_c, y_train_c, y_test_c, scaler_c = load_and_prep_data(
            'Children_Final_ML_Ready.csv', 'Children_feature_metadata.json'
        )
        run_unsupervised_clustering(X_chi, y_chi, "Children")
    except FileNotFoundError:
        print("Could not find the Children JSON file. Run the feature selection pipeline first!")