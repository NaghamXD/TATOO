import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
PROCESSED_DIR = "gaze_analysis/data/processed/"
RESULTS_DIR = "gaze_analysis/results/"
MAPPING_PATH = os.path.join(RESULTS_DIR, "Feature_Game_Mapping.csv")
TARGET_COLUMN = 'Target_Age_Old'  # 1 = Old (65+), 0 = Young (<65)

COLS_TO_DROP = [
    'Patient_ID', 'Age', 'Target_Age_Old', 'Target_Visual_Impaired',
    'Correct_Taps', 'Outside_Taps', 
    'Flight_Duration_sec', 'Touch_Duration_sec',
    'Gaze_Pct_Within_Screen', 'Gaze_Pct_Outside_Screen',
    'Pressure_Med_Pct', 'Game'
]

def prepare_game_data(df, game_name, mapping_df):
    game_df = df[df['Game'] == game_name].dropna(subset=[TARGET_COLUMN])
    
    # 1. Participation Filter
    participating = mapping_df.index[mapping_df[game_name] == True].tolist()
    valid_features = [f for f in participating if f in game_df.columns and f not in COLS_TO_DROP]
    
    if len(game_df) < 10 or not valid_features:
        return None, None, None

    X = game_df[valid_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = game_df[TARGET_COLUMN]
    ids = game_df['Patient_ID'].values

    # 2. Redundancy Filter (> 85% Correlation)
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    X = X.drop(columns=[c for c in upper.columns if any(upper[c] > 0.85)])
    
    # 3. Relevance Filter (> 25% Correlation with Age)
    age_corr = X.apply(lambda x: x.corr(y))
    X = X.drop(columns=age_corr[age_corr.abs() < 0.25].index)

    if X.shape[1] < 2:
        return None, None, None

    # 4. Standardization
    X_scaled = StandardScaler().fit_transform(X)
    
    return X_scaled, ids, y

if __name__ == "__main__":
    if not os.path.exists(MAPPING_PATH):
        print(f"❌ Mapping file not found at: {MAPPING_PATH}")
    else:
        mapping_df = pd.read_csv(MAPPING_PATH).set_index('Feature_Name')
        game_cols = ['TouchIt', 'CornerIt', 'DoubleTapIt', 'PinchIt', 'SlideIt', 'DragIt']
        perplexities = [2, 5, 10, 20]

        all_dfs = []
        for g in game_cols:
            path = os.path.join(PROCESSED_DIR, f"MASTER_{g}.csv")
            if os.path.exists(path):
                all_dfs.append(pd.read_csv(path).assign(Game=g))
        
        if not all_dfs:
            print("❌ No MASTER files found.")
        else:
            master_df = pd.concat(all_dfs, ignore_index=True)

            for game in game_cols:
                print(f"🧬 Analyzing {game} with multiple perplexities...")
                X_scaled, ids, y_age = prepare_game_data(master_df, game, mapping_df)
                
                if X_scaled is not None:
                    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
                    fig.suptitle(f"t-SNE Age Clustering: {game}\nBlue = Young (<65), Red = Old (65+)", fontsize=20, fontweight='bold')
                    
                    # Convert binary labels to readable strings for the legend
                    age_labels = y_age.map({1: 'Old (65+)', 0: 'Young (<65)'}).values
                    
                    for i, perp in enumerate(perplexities):
                        ax = axes[i // 2, i % 2]
                        actual_perp = min(perp, len(ids) - 1)
                        tsne = TSNE(n_components=2, perplexity=actual_perp, random_state=42, init='pca', learning_rate='auto')
                        results = tsne.fit_transform(X_scaled)
                        
                        plot_df = pd.DataFrame(results, columns=['x', 'y'])
                        plot_df['Patient_ID'] = ids
                        plot_df['Age_Category'] = age_labels
                        
                        # Use a specific palette to differentiate Age groups clearly
                        sns.scatterplot(
                            data=plot_df, x='x', y='y', 
                            hue='Age_Category', 
                            palette={'Old (65+)': '#e74c3c', 'Young (<65)': '#3498db'}, 
                            s=120, ax=ax, alpha=0.8
                        )
                        
                        # Add Participant ID labels to each point
                        for j, patient in enumerate(ids):
                            ax.text(results[j, 0]+0.3, results[j, 1]+0.3, patient, fontsize=8)
                        
                        ax.set_title(f"Perplexity = {perp}", fontsize=14)
                        ax.legend(loc='best', fontsize=10)

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    save_path = os.path.join(RESULTS_DIR, f"tSNE_Grid_{game}_Age_Categorized.png")
                    plt.savefig(save_path, dpi=300)
                    plt.close()
                    print(f"✅ Saved comparison grid to: {save_path}")