# correlations for each game, and a heatmap to visualize the relationships between features.
# It also checks for missing data and zero variance features that could harm model performance.
# Finally, it compiles a master dataset combining all games to explore cross-game feature distributions.

import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Suppress harmless SciPy ties/variance warnings for clean console output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
PROCESSED_DIR = "gaze_analysis/data/processed/"
RESULTS_DIR = "gaze_analysis/results/"
DATA_EXPLORATION_DIR = os.path.join(RESULTS_DIR, "data_exploration")

TARGET_GAMES = ['TouchIt', 'CornerIt', 'DoubleTapIt', 'PinchIt', 'SlideIt', 'DragIt']

if not os.path.exists(DATA_EXPLORATION_DIR):
    os.makedirs(DATA_EXPLORATION_DIR)

def explore_demographics():
    """
    Reads one of the master files to plot the distributions of Age and Target labels.
    """
    print(f"\n{'='*60}")
    print(f"📊 EXPLORING DEMOGRAPHICS AND LABELS")
    print(f"{'='*60}")
    
    # We only need one master file since the patients are the same across games
    master_files = glob.glob(os.path.join(PROCESSED_DIR, "MASTER_*.csv"))
    if not master_files:
        print("❌ No MASTER files found to extract demographics.")
        return
        
    df = pd.read_csv(master_files[0])
    
    # Ensure Age is numeric
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. Age Histogram
    sns.histplot(data=df, x='Age', bins=15, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Patient Age Distribution')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Count')
    
    # 2. Target Age Old Label
    if 'Target_Age_Old' in df.columns:
        sns.countplot(data=df, x='Target_Age_Old', ax=axes[1], palette='Set2')
        axes[1].set_title('Label: Age Target (0=Young, 1=Old)')
        axes[1].set_xlabel('Class')
        
    # 3. Target Visual Impaired Label
    if 'Target_Visual_Impaired' in df.columns:
        sns.countplot(data=df, x='Target_Visual_Impaired', ax=axes[2], palette='Set2')
        axes[2].set_title('Label: Visual Impairment (0=No, 1=Yes)')
        axes[2].set_xlabel('Class')
        
    plt.tight_layout()
    plot_path = os.path.join(DATA_EXPLORATION_DIR, "Demographics_Distributions.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"🖼️ Saved Demographics Distribution Plot to: {plot_path}")

def map_features_to_games():
    """
    Scans all MASTER game files and creates a CSV matrix showing exactly 
    which features exist in which games.
    """
    print(f"\n{'='*60}")
    print(f"🗺️ MAPPING FEATURES ACROSS GAMES")
    print(f"{'='*60}")

    feature_map = {}
    
    for game in TARGET_GAMES:
        file_path = os.path.join(PROCESSED_DIR, f"MASTER_{game}.csv")
        if os.path.exists(file_path):
            # We only need the column names, reading nrows=0 is instantaneous
            df = pd.read_csv(file_path, nrows=0) 
            columns = df.columns.tolist()
            
            for col in columns:
                if col not in feature_map:
                    # Initialize the feature with False for all games
                    feature_map[col] = {g: False for g in TARGET_GAMES}
                # Mark True for the current game
                feature_map[col][game] = True
                
    if not feature_map:
        print("❌ No MASTER files found to map features.")
        return

    # Convert our dictionary into a beautiful Pandas DataFrame
    map_df = pd.DataFrame.from_dict(feature_map, orient='index')
    map_df.index.name = 'Feature_Name'
    map_df.reset_index(inplace=True)
    
    # Calculate the total number of games each feature appears in
    map_df['Total_Games'] = map_df[TARGET_GAMES].sum(axis=1)
    
    # Sort so the universal features are at the top, and game-specific ones at the bottom
    map_df = map_df.sort_values(by=['Total_Games', 'Feature_Name'], ascending=[False, True])
    
    out_path = os.path.join(DATA_EXPLORATION_DIR, "Feature_Game_Mapping.csv")
    map_df.to_csv(out_path, index=False)
    
    print(f"✅ Mapped {len(map_df)} unique features across all datasets.")
    print(f"💾 Saved Feature Inventory to: {out_path}")


def explore_cross_game_features():
    """
    Combines data from all 6 games to plot how universal features overlap.
    """
    print(f"\n{'='*60}")
    print(f"📈 EXPLORING CROSS-GAME FEATURE OVERLAPS")
    print(f"{'='*60}")
    
    all_data = []
    for game in TARGET_GAMES:
        file_path = os.path.join(PROCESSED_DIR, f"MASTER_{game}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Game_Name'] = game  # Add a column to identify the game
            all_data.append(df)
            
    if not all_data:
        print("❌ No MASTER files found.")
        return
        
    # Combine all 6 games into one massive dataset
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Define which universal features you want to compare across games
    universal_features = [
        'First_Reaction_Time_sec',
        'Flight_Touch_Ratio', 
        'Game_Duration_sec', 
        'Mean_Eye_Hand_Distance_cm',
        'Pressure_Low_Pct',
        'Pressure_High_Pct',
        'Gaze_In_Out_Ratio'
    ]
    
    for feature in universal_features:
        if feature in combined_df.columns:
            plt.figure(figsize=(10, 6))
            
            # Violin plots perfectly show data distribution, overlap, and median
            sns.violinplot(data=combined_df, x='Game_Name', y=feature, palette='pastel', inner='quartile')
            
            # Add a strip plot to see the individual patient dots on top of the violin
            sns.stripplot(data=combined_df, x='Game_Name', y=feature, color='black', alpha=0.4, size=4)
            
            plt.title(f'Cross-Game Distribution: {feature}', fontsize=14)
            plt.xticks(rotation=30)
            plt.ylabel(feature)
            plt.xlabel("Game")
            plt.tight_layout()
            
            plot_path = os.path.join(DATA_EXPLORATION_DIR, f"CrossGame_{feature}_Distribution.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"🖼️ Saved Overlap Plot for '{feature}' to: {plot_path}")


def explore_game_data(game_name, correlation_threshold=0.85, missing_threshold=0.30):
    """
    Checks for high correlation, missing data, and zero variance within a specific game.
    """
    file_path = os.path.join(PROCESSED_DIR, f"MASTER_{game_name}.csv")
    if not os.path.exists(file_path):
        print(f"❌ Could not find {file_path}")
        return
        
    df = pd.read_csv(file_path)
    print(f"\n{'='*60}")
    print(f"🔍 CORRELATION & INTEGRITY ANALYSIS: {game_name}")
    print(f"Dataset Shape: {df.shape[0]} Patients | {df.shape[1]} Columns")
    print(f"{'='*60}")

    cols_to_exclude = [
        'Patient_ID', 'Age', 'Target_Age_Old', 'Target_Visual_Impaired',
        'Correct_Taps', 'Outside_Taps', 
        'Flight_Duration_sec', 'Touch_Duration_sec',
        'Gaze_Pct_Within_Screen', 'Gaze_Pct_Outside_Screen',
        'Pressure_Med_Pct', 'Game'
    ]
    feature_df = df.drop(columns=[col for col in cols_to_exclude if col in df.columns])
    feature_df = feature_df.apply(pd.to_numeric, errors='coerce')

    # --- 1. FIND HIGH MISSINGNESS ---
    missing_pct = feature_df.isnull().mean()
    high_missing = missing_pct[missing_pct > missing_threshold]
    
    print("\n👻 FEATURES WITH HIGH MISSING DATA (>30%):")
    if high_missing.empty:
        print("   ✅ None. Data is highly complete.")
    else:
        for col, pct in high_missing.items():
            print(f"   ⚠️ {col}: {pct*100:.1f}% missing")

    # --- 2. FIND ZERO VARIANCE ---
    variances = feature_df.var()
    zero_var = variances[variances == 0]
    
    print("\n🗿 ZERO VARIANCE FEATURES (Every patient has the same score):")
    if zero_var.empty:
        print("   ✅ None. All features have variance.")
    else:
        for col in zero_var.index:
            print(f"   🗑️ Drop candidate: {col}")

    # --- 3. FIND HIGHLY CORRELATED PAIRS ---
    corr_matrix = feature_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    high_corr_pairs = []
    for col in upper_tri.columns:
        for row in upper_tri.index:
            val = upper_tri.loc[row, col]
            if pd.notna(val) and val > correlation_threshold:
                high_corr_pairs.append((row, col, val))
                
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"\n👯 HIGHLY CORRELATED PAIRS (>{correlation_threshold}):")
    if not high_corr_pairs:
        print("   ✅ None. Features are beautifully independent.")
    else:
        for pair in high_corr_pairs:
            print(f"   🔗 {pair[2]:.2f} | {pair[0]}  <-->  {pair[1]}")

    # --- 4. CORRELATION HEATMAP ---
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(feature_df.corr(), dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(feature_df.corr(), mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, 
                center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.title(f"Biomarker Correlation Matrix - {game_name}", fontsize=16)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "correlation_heatmaps", f"{game_name}_Correlation_Heatmap.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\n🖼️ Saved Heatmap to: {plot_path}\n")


if __name__ == "__main__":
    # 1. Feature Mapping (NEW)
    map_features_to_games()

    # 1. Run Demographics Check First
    explore_demographics()
    
    # 2. Run Cross-Game Feature Comparison
    explore_cross_game_features()
    
    # 3. Check Correlation for the specific games
    explore_game_data('PinchIt')
    explore_game_data('CornerIt')
    explore_game_data('DoubleTapIt')
    explore_game_data('TouchIt')
    explore_game_data('DragIt')
    explore_game_data('SlideIt')  # Uncomment to check others