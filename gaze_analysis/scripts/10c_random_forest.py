#matrix of feature importance using random forest, but with the new logic of dropping features based on correlation and relevance before calculating importance. This will give us a cleaner importance matrix that focuses on truly unique and relevant features for each game.

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import matplotlib.patches as mpatches
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
PROCESSED_DIR = "gaze_analysis/data/processed/"
RESULTS_DIR = "gaze_analysis/results/"
MAPPING_PATH = os.path.join(RESULTS_DIR, "Feature_Game_Mapping.csv")
TARGET_COLUMN = 'Target_Age_Old'

COLS_TO_DROP = [
    'Patient_ID', 'Age', 'Target_Age_Old', 'Target_Visual_Impaired',
    'Correct_Taps', 'Outside_Taps', 
    'Flight_Duration_sec', 'Touch_Duration_sec',
    'Gaze_Pct_Within_Screen', 'Gaze_Pct_Outside_Screen',
    'Pressure_Med_Pct', 'Game'
]

def analyze_game_features(df, game_name, mapping_df):
    """
    Identifies if a feature is Participated, Redundant, Low-Predictive, or Not-In-Game.
    """
    game_df = df[df['Game'] == game_name].dropna(subset=[TARGET_COLUMN])
    
    # 1. Base Participation (from Mapping CSV)
    participating_in_design = mapping_df.index[mapping_df[game_name] == True].tolist()
    # Filter out requested drop columns
    active_features = [f for f in participating_in_design if f not in COLS_TO_DROP and f in game_df.columns]
    
    status_map = {} # {feature: {'val': float, 'label': str, 'color_code': int}}
    
    if not active_features or len(game_df) < 10:
        return status_map

    X_raw = game_df[active_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = game_df[TARGET_COLUMN]

    # --- FILTERING LOGIC ---
    # Stage 1: Inter-Feature Correlation (Redundancy)
    corr_matrix = X_raw.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_high_corr = [column for column in upper.columns if any(upper[column] > 0.85)]
    X_no_redundancy = X_raw.drop(columns=to_drop_high_corr)

    # Stage 2: Feature-Target Correlation (Relevance)
    correlations = X_no_redundancy.apply(lambda x: x.corr(y))
    to_drop_low_target_corr = correlations[correlations.abs() < 0.25].index
    X_final = X_no_redundancy.drop(columns=to_drop_low_target_corr)

    # Calculate Importance for the survivors
    importances = pd.Series()
    if not X_final.empty:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_final, y)
        importances = pd.Series(rf.feature_importances_, index=X_final.columns)

    # Map results for every feature in the global mapping
    for f in mapping_df.index:
        if f not in participating_in_design:
            status_map[f] = {'val': np.nan, 'label': "", 'type': 'not_participated'}
        elif f in to_drop_high_corr:
            status_map[f] = {'val': np.nan, 'label': "High Corr", 'type': 'high_corr'}
        elif f in to_drop_low_target_corr:
            status_map[f] = {'val': np.nan, 'label': "Low Corr", 'type': 'low_corr'}
        elif f in importances.index:
            val = importances[f]
            status_map[f] = {'val': val, 'label': f"{val:.2f}", 'type': 'importance'}
        else:
            status_map[f] = {'val': 0.0, 'label': "0.00", 'type': 'importance'}
            
    return status_map

if __name__ == "__main__":
    if not os.path.exists(MAPPING_PATH):
        print(f"❌ Error: Mapping file not found.")
    else:
        mapping_df = pd.read_csv(MAPPING_PATH).set_index('Feature_Name')
        game_cols = ['TouchIt', 'CornerIt', 'DoubleTapIt', 'PinchIt', 'SlideIt', 'DragIt']
        
        # Initial cleanup of the mapping
        mapping_df = mapping_df[~mapping_df.index.isin(COLS_TO_DROP)]
        mapping_df = mapping_df[mapping_df[game_cols].any(axis=1)]

        # Load Master Data
        all_dfs = [pd.read_csv(os.path.join(PROCESSED_DIR, f"MASTER_{g}.csv")).assign(Game=g) 
                   for g in game_cols if os.path.exists(os.path.join(PROCESSED_DIR, f"MASTER_{g}.csv"))]
        master_df = pd.concat(all_dfs, ignore_index=True)

        # Matrices for plotting
        importance_matrix = pd.DataFrame(index=mapping_df.index, columns=game_cols)
        label_matrix = pd.DataFrame(index=mapping_df.index, columns=game_cols)
        type_matrix = pd.DataFrame(index=mapping_df.index, columns=game_cols)

        for game in game_cols:
            res = analyze_game_features(master_df, game, mapping_df)
            for feat, data in res.items():
                importance_matrix.loc[feat, game] = data['val']
                label_matrix.loc[feat, game] = data['label']
                type_matrix.loc[feat, game] = data['type']

        # --- VISUALIZATION ---
        plt.figure(figsize=(18, 14))
        
        # 1. Background: Gray for everything (Not Participated)
        ax = sns.heatmap(importance_matrix.astype(float), mask=True, cbar=False, cmap=['#e0e0e0'])
        
        # 2. Layer 1: Importance Scores (The Heatmap with the 0-1 scale)
        sns.heatmap(
            importance_matrix.astype(float), 
            annot=label_matrix, 
            fmt="", 
            cmap="YlGnBu", 
            linewidths=.5,
            cbar_kws={'label': 'Importance Score (0.0 to 1.0)'},
            ax=ax
        )

        # 3. Layer 2: Overlaying specific colors for "High Corr" and "Low Corr"
        # We manually color these cells so they don't affect the colorbar scale
        for y_idx, feature in enumerate(mapping_df.index):
            for x_idx, game in enumerate(game_cols):
                cell_type = type_matrix.loc[feature, game]
                if cell_type == 'high_corr':
                    ax.add_patch(plt.Rectangle((x_idx, y_idx), 1, 1, fill=True, color='#ffcc80', lw=0)) # Light Orange
                    ax.text(x_idx + 0.5, y_idx + 0.5, "High Corr", ha='center', va='center', fontsize=9, fontweight='bold')
                elif cell_type == 'low_corr':
                    ax.add_patch(plt.Rectangle((x_idx, y_idx), 1, 1, fill=True, color="#f7d1d1", lw=0)) # Light Red
                    ax.text(x_idx + 0.5, y_idx + 0.5, "Low Corr", ha='center', va='center', fontsize=9, fontweight='bold')

        # 4. Legend for Exclusions
        orange_patch = mpatches.Patch(color='#ffcc80', label='Dropped: High Correlation (>85%)')
        red_patch = mpatches.Patch(color="#f7d1d1", label='Dropped: Low Relevance (<25%)')
        gray_patch = mpatches.Patch(color='#e0e0e0', label='Not Participated in Game Design')
        plt.legend(handles=[orange_patch, red_patch, gray_patch], bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.title("Biomarker Diagnostic Matrix: Importance & Exclusion Audit\n(Scale represents only predictive survivors)", 
                  fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        out_path = os.path.join(RESULTS_DIR, "Final_Audit_Importance_Matrix.png")
        plt.savefig(out_path, dpi=300)
        print(f"✅ Success! Matrix saved to: {out_path}")