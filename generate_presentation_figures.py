import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import json
import os
import numpy as np

def generate_presentation_figures(csv_file, json_file):
    dataset_name = os.path.basename(csv_file).split('_')[0]
    print(f"\nGenerating presentation figures for: {dataset_name}")
    
    # 1. Load Data and Metadata
    try:
        df = pd.read_csv(csv_file)
        with open(json_file, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    sig_vars = metadata.get("significant_vars", [])
    if not sig_vars:
        print("No significant variables found in JSON.")
        return

    output_dir = f"{dataset_name}_Presentation_Figures"
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # FIGURE 1: Correlation Heatmap (Feature Redundancy)
    # ==========================================
    print("Plotting Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    
    # Calculate Spearman correlation for the significant variables
    corr_matrix = df[sig_vars].corr(method='spearman')
    
    # Create a mask to hide the upper triangle (makes it look cleaner for presentations)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Draw the heatmap
    sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", vmin=-1, vmax=1, 
                annot=False, square=True, linewidths=.5, cbar_kws={"shrink": .75})
    
    plt.title(f"{dataset_name}: Correlation of Significant Features", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_Correlation_Heatmap.png"), dpi=300)
    plt.close()

    # ==========================================
    # FIGURE 2: Top 4 Differentiating Features Grid
    # ==========================================
    print("Finding the top 4 features and plotting the grid...")
    
    # Re-calculate p-values quickly to find the absolute lowest ones
    p_values = {}
    for col in sig_vars:
        clean_df = df.dropna(subset=[col, 'status'])
        groups = [clean_df[clean_df['status'] == s][col].values for s in clean_df['status'].unique()]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) == 2:
            _, p_val = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
        elif len(groups) > 2:
            _, p_val = stats.kruskal(*groups)
        else:
            continue
        p_values[col] = p_val
        
    # Sort and grab the top 4 variables with the smallest p-values
    top_4_vars = sorted(p_values, key=p_values.get)[:4]

    # Create a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, col in enumerate(top_4_vars):
        clean_df = df.dropna(subset=[col, 'status'])
        pval = p_values[col]
        
        # Format the p-value nicely for the title
        pval_str = "< 0.0001" if pval < 0.0001 else f"= {pval:.4f}"
        
        # Boxplot with a swarmplot overlay for a very professional look
        sns.boxplot(x='status', y=col, data=clean_df, ax=axes[i], palette="Pastel1", 
                    showmeans=True, meanprops={"marker":"D", "markerfacecolor":"white", "markeredgecolor":"black"})
        sns.swarmplot(x='status', y=col, data=clean_df, ax=axes[i], color=".25", alpha=0.5, size=4)
        
        axes[i].set_title(f"{col.replace('_', ' ')}\n(p {pval_str})", fontsize=14)
        axes[i].set_xlabel("") # Remove x-label to reduce clutter
        axes[i].set_ylabel("Score / Time", fontsize=12)

    plt.suptitle(f"{dataset_name}: Top 4 Most Discriminative Hand Function Tests", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_Top4_Features_Grid.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Success! High-quality figures saved in '{output_dir}'.")

if __name__ == "__main__":
    # Generate presentation figures for both datasets
    generate_presentation_figures('Children_Final_ML_Ready.csv', 'Children_feature_metadata.json')
    generate_presentation_figures('Eldery_Final_ML_Ready.csv', 'Eldery_feature_metadata.json')