"""
10_explore_data.py — New App
EDA: feature mapping, cross-game distributions, correlation heatmaps.
No demographics/labels available yet — demographic plots are skipped.
"""
import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_new_app import PROCESSED_DIR, RESULTS_DIR, ALL_GAMES

processed_dir = str(PROCESSED_DIR)
results_dir   = str(RESULTS_DIR)
heatmap_dir   = os.path.join(results_dir, "correlation_heatmaps")
os.makedirs(results_dir,  exist_ok=True)
os.makedirs(heatmap_dir,  exist_ok=True)

NON_FEATURE_COLS = [
    'Patient_ID', 'Age', 'Target_Age_Old', 'Target_Visual_Impaired',
    'Correct_Taps', 'Outside_Taps',
    'Flight_Duration_sec', 'Touch_Duration_sec',
    'Gaze_Pct_Within_Screen', 'Gaze_Pct_Outside_Screen',
    'Pressure_Med_Pct', 'Game',
]


def map_features_to_games():
    print(f"\n{'='*60}\nFEATURE → GAME MAPPING\n{'='*60}")
    feature_map = {}
    for game in ALL_GAMES:
        fp = os.path.join(processed_dir, f"MASTER_{game}.csv")
        if os.path.exists(fp):
            for col in pd.read_csv(fp, nrows=0).columns:
                if col not in feature_map:
                    feature_map[col] = {g: False for g in ALL_GAMES}
                feature_map[col][game] = True

    if not feature_map:
        print("  No MASTER files found.")
        return

    map_df = pd.DataFrame.from_dict(feature_map, orient='index')
    map_df.index.name = 'Feature_Name'
    map_df.reset_index(inplace=True)
    map_df['Total_Games'] = map_df[ALL_GAMES].sum(axis=1)
    map_df = map_df.sort_values(['Total_Games', 'Feature_Name'], ascending=[False, True])

    out = os.path.join(results_dir, "Feature_Game_Mapping.csv")
    map_df.to_csv(out, index=False)
    print(f"  Mapped {len(map_df)} features  ->  {out}")


def explore_cross_game_features():
    print(f"\n{'='*60}\nCROSS-GAME FEATURE DISTRIBUTIONS\n{'='*60}")
    all_data = []
    for game in ALL_GAMES:
        fp = os.path.join(processed_dir, f"MASTER_{game}.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            df['Game_Name'] = game
            all_data.append(df)

    if not all_data:
        print("  No MASTER files found.")
        return

    combined = pd.concat(all_data, ignore_index=True)
    universal = [
        'First_Reaction_Time_sec', 'Flight_Touch_Ratio',
        'Game_Duration_sec', 'Mean_Eye_Hand_Distance_cm',
        'Pressure_Low_Pct', 'Pressure_High_Pct', 'Gaze_In_Out_Ratio',
    ]
    for feature in universal:
        if feature not in combined.columns:
            continue
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=combined, x='Game_Name', y=feature, palette='pastel', inner='quartile')
        sns.stripplot(data=combined, x='Game_Name', y=feature, color='black', alpha=0.4, size=4)
        plt.title(f'Cross-Game Distribution: {feature}', fontsize=14)
        plt.xticks(rotation=30)
        plt.tight_layout()
        out = os.path.join(results_dir, f"CrossGame_{feature}_Distribution.png")
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"  Saved: CrossGame_{feature}_Distribution.png")


def explore_game_data(game_name, corr_thresh=0.85, miss_thresh=0.30):
    fp = os.path.join(processed_dir, f"MASTER_{game_name}.csv")
    if not os.path.exists(fp):
        return

    df = pd.read_csv(fp)
    print(f"\n{'='*60}\nCORRELATION & INTEGRITY: {game_name}  ({df.shape[0]} participants, {df.shape[1]} cols)\n{'='*60}")

    feat_df = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])
    feat_df = feat_df.apply(pd.to_numeric, errors='coerce')

    # Missing data
    high_miss = feat_df.isnull().mean()
    high_miss = high_miss[high_miss > miss_thresh]
    print("\nHigh missing (>30%):")
    print("  None" if high_miss.empty else "\n".join(f"  {c}: {v*100:.1f}%" for c, v in high_miss.items()))

    # Zero variance
    zv = feat_df.var()
    zv = zv[zv == 0]
    print("\nZero variance features:")
    print("  None" if zv.empty else "\n".join(f"  {c}" for c in zv.index))

    # High correlation pairs
    corr = feat_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = [(r, c, upper.loc[r, c]) for c in upper.columns for r in upper.index
             if pd.notna(upper.loc[r, c]) and upper.loc[r, c] > corr_thresh]
    pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"\nHighly correlated pairs (>{corr_thresh}):")
    if pairs:
        for a, b, v in pairs:
            print(f"  {v:.2f}  {a} <-> {b}")
    else:
        print("  None")

    # Heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(feat_df.corr(), dtype=bool))
    sns.heatmap(feat_df.corr(), mask=mask,
                cmap=sns.diverging_palette(230, 20, as_cmap=True),
                vmax=1.0, vmin=-1.0, center=0, square=True,
                linewidths=.5, cbar_kws={"shrink": .5})
    plt.title(f"Biomarker Correlation — {game_name}", fontsize=16)
    plt.tight_layout()
    out = os.path.join(heatmap_dir, f"{game_name}_Correlation_Heatmap.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  Heatmap saved: {out}")


if __name__ == "__main__":
    map_features_to_games()
    explore_cross_game_features()
    for game in ALL_GAMES:
        explore_game_data(game)
