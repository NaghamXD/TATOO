import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')

def coerce_numeric(df):
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['status', 'Group_Status', 'ID']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(' ', ''), errors='coerce')
    return df

def run_top_features_clustering(csv_file, top_features, target_clusters, scenario_name=""):
    print(f"\n{'='*60}")
    print(f"--- Clustering strictly on Top Features: {scenario_name} ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        return

    df = coerce_numeric(df)
    target_col = 'status' if 'status' in df.columns else 'Group_Status'
    
    # 2. Verify features exist in the dataset
    valid_features = [f for f in top_features if f in df.columns]
    if len(valid_features) != len(top_features):
        missing = set(top_features) - set(valid_features)
        print(f"Warning: The following features were not found in the dataset: {missing}")
        if not valid_features:
            print("No valid features found. Exiting.")
            return
            
    print(f"Using exactly {len(valid_features)} features: {valid_features}")

    # 3. Safely drop NA strictly for these 4 features
    initial_len = len(df)
    df = df.dropna(subset=valid_features + [target_col])
    print(f"Remaining subjects after NA drop: {len(df)} (Dropped {initial_len - len(df)})")
    
    if len(df) < target_clusters:
        print("Not enough data left to cluster! Skipping...")
        return

    X = df[valid_features].copy()
    y = df[target_col].copy()
    
    # 4. Handle Negative Glitches & Scale
    # We apply the clip and log1p just in case any of the top 4 are highly skewed time variables
    for col in X.columns:
        X[col] = np.log1p(np.clip(X[col], 0, None))
            
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5. Agglomerative Clustering
    print("Running Agglomerative Clustering...")
    agg_cluster = AgglomerativeClustering(n_clusters=target_clusters, linkage='ward')
    cluster_labels = agg_cluster.fit_predict(X_scaled)
    
    # 6. t-SNE for Visualization
    print("Running t-SNE...")
    # Lower perplexity is sometimes better for very few features. We'll use 20.
    tsne = TSNE(n_components=2, random_state=42, perplexity=20, max_iter=1500)
    X_viz = tsne.fit_transform(X_scaled)
    
    plot_df = pd.DataFrame({
        'Dim_1': X_viz[:, 0],
        'Dim_2': X_viz[:, 1],
        'True_Status': y.values,
        'Cluster': [f"Cluster {c}" for c in cluster_labels]
    })
    
    # 7. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.scatterplot(data=plot_df, x='Dim_1', y='Dim_2', hue='True_Status', palette='tab10', ax=axes[0], alpha=0.8, s=70)
    axes[0].set_title(f'Ground Truth Clinical Status (t-SNE)\nBased ONLY on Top {len(valid_features)} Features')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    sns.scatterplot(data=plot_df, x='Dim_1', y='Dim_2', hue='Cluster', palette='Set2', ax=axes[1], alpha=0.8, s=70)
    axes[1].set_title(f'Blind Clustering ({target_clusters} Clusters)\nBased ONLY on Top {len(valid_features)} Features')
    
    plt.tight_layout()
    filename = scenario_name.replace(" ", "_").replace(",", "") + "_TopFeatures.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Success! Saved visualization to '{filename}'\n")

if __name__ == "__main__":
    
    # =====================================================================
    # 🛑 PASTE YOUR 4 FEATURES FOR ELDERLY HERE:
    # =====================================================================
    elderly_top_4_1 = [
        "T20_DOM_Pinch_Not_Completed",      
        "T5_DOM_high_pressure",  
        "T5_DOM_Flight_Time",     
        "T5_DOM_Touch_outside"   
    ]
    
    run_top_features_clustering(
        csv_file='data/Eldery_Final_ML_Ready.csv', 
        top_features=elderly_top_4_1, 
        target_clusters=5, 
        scenario_name="Elderly - Top 4 Features"
    )

    # =====================================================================
    # 🛑 PASTE YOUR 4 FEATURES FOR ELDERLY HERE:
    # =====================================================================
    elderly_top_4_2 = [
        "T2_DOM_Touch_Time",      
        "T5_DOM_Flight_Time",  
        "T1_DOM_low_pressure",     
        "T1_DOM_Flight_Time"   
    ]
    
    run_top_features_clustering(
        csv_file='data/Eldery_Final_ML_Ready.csv', 
        top_features=elderly_top_4_2, 
        target_clusters=4, 
        scenario_name="Elderly - Top 4 Features - Parkinson's Excluded"
    )

    elderly_top_12 = [
        "T20_DOM_Pinch_Not_Completed",      
        "T5_DOM_high_pressure",
        "T5_DOM_Flight_Time",
        "T5_DOM_Touch_outside",
        "T5_DOM_Number_Taps",
        "T5_DOM_Duration",
        "T5_DOM_Correct_Attempts",
        "T13_DOM_Test_Duration",
        "T13_DOM_Reactione_Time",
        "T5_DOM_Touch_Time",
        "T13_DOM_touch_time",
        "T13_DOM_Total_Drag_Attempts"]
    
    run_top_features_clustering(
        csv_file='data/Eldery_Final_ML_Ready.csv', 
        top_features=elderly_top_12, 
        target_clusters=5, 
        scenario_name="Elderly - Top 12 Features"
    )

    # =====================================================================
    # 🛑 PASTE YOUR 4 FEATURES FOR CHILDREN HERE:
    # =====================================================================
    children_top_4 = [
        "T1_DOM_high_pressure",      
        "T5_DOM_Test_Duration",      
        "T20_DOM_Medium_Pressure",       
        "T2_DOM_low_pressure" 
    ]
    
    run_top_features_clustering(
        csv_file='data/Children_Final_ML_Ready.csv', 
        top_features=children_top_4, 
        target_clusters=4, 
        scenario_name="Children - Top 4 Features"
    )