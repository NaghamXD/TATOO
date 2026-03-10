import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')

def coerce_numeric(df):
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['status', 'Group_Status', 'ID']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(' ', ''), errors='coerce')
    return df

def plot_demographic_tsne(csv_file, top_features, demographics, dataset_name, exclude_groups=None):
    print(f"\n--- Generating Demographic t-SNE for {dataset_name} ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        return

    df = coerce_numeric(df)
    target_col = 'status' if 'status' in df.columns else 'Group_Status'
    
    # 2. Exclude specified groups (e.g., Parkinson's)
    if exclude_groups:
        initial_len = len(df)
        df = df[~df[target_col].isin(exclude_groups)]
        print(f"Excluded groups: {exclude_groups}. Dropped {initial_len - len(df)} subjects.")

    # 3. Filter dataset safely
    valid_features = [f for f in top_features if f in df.columns]
    
    if not valid_features:
        print("Error: No valid features found in dataset!")
        return
        
    df_clean = df.dropna(subset=valid_features + [target_col] + demographics).copy()
    print(f"Remaining subjects after NA drop: {len(df_clean)}")
    
    X = df_clean[valid_features].copy()
    
    # Handle glitches & scale
    for col in X.columns:
        X[col] = np.log1p(np.clip(X[col], 0, None))
    X_scaled = StandardScaler().fit_transform(X)
    
    # 4. Run t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_viz = tsne.fit_transform(X_scaled)
    
    # 5. Create Plotting DataFrame
    plot_df = pd.DataFrame({
        'Dim_1': X_viz[:, 0],
        'Dim_2': X_viz[:, 1],
        'Status': df_clean[target_col].values
    })
    
    # Add Demographics to plot dataframe
    for demo in demographics:
        plot_df[demo] = df_clean[demo].values

    # 6. Plotting Grid
    n_plots = len(demographics) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
    
    # Plot A: Ground Truth Clinical Status
    sns.scatterplot(data=plot_df, x='Dim_1', y='Dim_2', hue='Status', palette='tab10', ax=axes[0], s=60, alpha=0.8)
    axes[0].set_title(f"Clinical Status\n({dataset_name})")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    # Plots B-E: Demographics
    for i, demo in enumerate(demographics):
        ax = axes[i+1]
        
        # If the demographic is continuous (like Age or Education), use a color gradient
        if plot_df[demo].nunique() > 5:
            scatter = ax.scatter(plot_df['Dim_1'], plot_df['Dim_2'], c=plot_df[demo], cmap='coolwarm', s=60, alpha=0.8)
            plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        # If it is categorical (like Gender), use distinct colors
        else:
            plot_df[demo] = plot_df[demo].astype(str) # Force categorical
            sns.scatterplot(data=plot_df, x='Dim_1', y='Dim_2', hue=demo, palette='Set2', ax=ax, s=60, alpha=0.8)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            
        ax.set_title(f"Colored by {demo}")
        
    plt.tight_layout()
    filename = f"{dataset_name.replace(' ', '_')}_Demographic_tSNE.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")

if __name__ == "__main__":
    
    # =================================================================
    # SCENARIO 1: Elderly (Universal Battery)
    # =================================================================
    elderly_features = ["T5_DOM_high_pressure", "T5_DOM_Flight_Time", "T20_DOM_Pinch_Not_Completed", "T5_DOM_Touch_outside"]
    
    plot_demographic_tsne(
        csv_file='data/Eldery_Final_ML_Ready.csv', 
        top_features=elderly_features, 
        demographics=['Age', 'Gender', 'Education', 'Frequency_Tablet_Use'], 
        dataset_name="Elderly - Universal Battery",
        exclude_groups=None
    )
    
    # =================================================================
    # SCENARIO 2: Elderly (Parkinson's Excluded with New Features)
    # =================================================================
    elderly_top_4_2 = [
        "T2_DOM_Touch_Time",      
        "T5_DOM_Flight_Time",  
        "T1_DOM_low_pressure",     
        "T1_DOM_Flight_Time"   
    ]
    
    plot_demographic_tsne(
        csv_file='data/Eldery_Final_ML_Ready.csv', 
        top_features=elderly_top_4_2, 
        demographics=['Age', 'Gender', 'Education', 'Frequency_Tablet_Use'], 
        dataset_name="Elderly - Top 4 Features (No Parkinson's)",
        exclude_groups=['Parkinson']
    )

    # =================================================================
    # SCENARIO 3: Children 
    # =================================================================
    children_features = ["T1_DOM_high_pressure", "T5_DOM_Test_Duration", "T20_DOM_Medium_Pressure", "T2_DOM_low_pressure"]

    plot_demographic_tsne(
        csv_file='data/Children_Final_ML_Ready.csv', 
        top_features=children_features, 
        demographics=['Age', 'Gender', 'Frequency_Tablet_Use'], 
        dataset_name="Children - Core Battery",
        exclude_groups=None
    )
