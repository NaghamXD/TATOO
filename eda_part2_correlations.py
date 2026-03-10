import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Ignore harmless pandas/seaborn warnings for clean output
warnings.filterwarnings('ignore')

def coerce_numeric(df):
    """Ensures all game columns are cleanly formatted as numbers."""
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['status', 'Group_Status', 'ID']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(' ', ''), errors='coerce')
    return df

def get_target_col(df):
    """Finds the correct target column depending on the dataset version."""
    if 'status' in df.columns: return 'status'
    elif 'Group_Status' in df.columns: return 'Group_Status'
    return None

def generate_status_correlation_heatmap(input_file, output_image_name, dataset_name):
    print(f"\n--- Generating Correlation Heatmap for {dataset_name} Dataset ---")
    
    # 1. Load and prepare the data
    df = pd.read_csv(input_file)
    df = coerce_numeric(df)
    target_col = get_target_col(df)
    
    if not target_col:
        print(f"Error: No target column found in {input_file}")
        return
        
    # Drop completely empty columns and IDs to prevent math errors
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    df = df.dropna(axis=1, how='all')
    
    # 2. One-Hot Encode the Status column
    # This creates columns like 'status_Parkinson', 'status_healthy', etc.
    df_encoded = pd.get_dummies(df, columns=[target_col])
    
    # Get lists of our new status columns and our numeric feature columns
    status_cols = [c for c in df_encoded.columns if c.startswith(f"{target_col}_")]
    feature_cols = [c for c in df_encoded.columns if c not in status_cols]
    
    # 3. Calculate the Correlation Matrix
    print("Calculating correlations across all variables. This takes a second...")
    corr_matrix = df_encoded.corr()
    
    # Isolate the intersection: Features (Rows) vs Statuses (Columns)
    feat_vs_status_corr = corr_matrix.loc[feature_cols, status_cols]
    
    # 4. Filter for the "Top 25 Most Important" Features
    # We find the maximum absolute correlation for each feature across ANY status
    feat_vs_status_corr['Max_Abs_Corr'] = feat_vs_status_corr.abs().max(axis=1)
    
    # Sort and pick the top 25 most highly correlated features
    top_features = feat_vs_status_corr.sort_values(by='Max_Abs_Corr', ascending=False).head(25)
    
    # Drop our sorting column so it doesn't plot
    top_features = top_features.drop(columns=['Max_Abs_Corr'])
    
    # Clean up the column names for the plot (remove 'status_' prefix)
    clean_status_names = [col.replace(f"{target_col}_", "") for col in top_features.columns]
    top_features.columns = clean_status_names
    
    # 5. Plot the Heatmap
    plt.figure(figsize=(12, 10))
    
    # Center the colormap at 0 so positive is Red, negative is Blue, and zero is White
    sns.heatmap(top_features, 
                annot=True,          # Show the actual correlation numbers
                fmt=".2f",           # 2 decimal places
                cmap="coolwarm",     # Blue (negative) to Red (positive)
                center=0,            # 0 correlation is pure white
                linewidths=0.5,
                cbar_kws={'label': 'Pearson Correlation Coefficient'})
    
    plt.title(f'Top 25 Clinical Correlates in {dataset_name} Data\n(Red = Positive Link, Blue = Negative Link)', 
              fontsize=16, pad=20)
    plt.ylabel('Tablet OT Metrics (Features)', fontsize=12)
    plt.xlabel('Clinical Status', fontsize=12)
    
    # Rotate x-axis labels so status names are readable
    plt.xticks(rotation=45, ha='right')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_image_name, dpi=300)
    print(f"Success! Heatmap saved as {output_image_name}")

# Run for both finalized datasets
if __name__ == "__main__":
    generate_status_correlation_heatmap('data/Eldery_Final_ML_Ready.csv', 'Elderly_Correlation_Heatmap.png', 'Elderly')
    generate_status_correlation_heatmap('data/Children_Final_ML_Ready.csv', 'Children_Correlation_Heatmap.png', 'Children')