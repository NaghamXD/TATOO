"""
10b_explore_data.py — New App
Violin plots of feature distributions across all 8 games.
"""
import os
import sys
import glob
import warnings
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
plots_dir     = os.path.join(str(RESULTS_DIR), "violin_plots")
os.makedirs(plots_dir, exist_ok=True)

TARGET_FEATURES = [
    'Fixation_Count',
    'Mean_Fixation_Duration_sec',
    'Mean_Saccadic_Amplitude_cm',
    'Peak_Saccadic_Velocity_cm_s',
    'Saccade_Frequency_Hz',
    'Directional_Velocity_Alignment',
    'Mean_Action_Duration_sec',
    'Mean_Hold_Duration_sec',
    'Mean_Pressure_Overall',
    'Pressure_Jitter_Stationary',
    'Spatial_Jerk_Magnitude',
    'True_Eye_Hand_Latency_sec',
    'Accuracy_Ratio',
    'Number_of_Taps',
    'First_Reaction_Time_sec',
    'Flight_Touch_Ratio',
    'Gaze_In_Out_Ratio',
    'Mean_Eye_Hand_Distance_cm',
]


def compile_master_df():
    all_data = []
    for game in ALL_GAMES:
        for suffix in ('ML_Features', 'Gaze_Features', 'Sync_Features'):
            fp = os.path.join(processed_dir, f"{game}_{suffix}.csv")
            if os.path.exists(fp):
                df = pd.read_csv(fp)
                df['Game'] = game
                all_data.append(df)
    if not all_data:
        return pd.DataFrame()

    combined = all_data[0]
    for df in all_data[1:]:
        new_cols = df.columns.difference(combined.columns).tolist() + ['Patient_ID', 'Game']
        combined = pd.merge(combined, df[new_cols], on=['Patient_ID', 'Game'], how='outer')
    return combined


def generate_violin_plots(df):
    sns.set_theme(style="whitegrid", palette="muted")
    for feature in TARGET_FEATURES:
        if feature not in df.columns:
            print(f"  Skipping '{feature}': not found.")
            continue
        plot_data = df.dropna(subset=[feature])
        if plot_data.empty:
            continue

        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(data=plot_data, x='Game', y=feature,
                            cut=0, scale='width', inner='quartile')
        sns.swarmplot(data=plot_data, x='Game', y=feature,
                      color='black', alpha=0.5, size=4)
        plt.title(f'Distribution of {feature} Across Games', fontsize=14, fontweight='bold')
        plt.xlabel('Game')
        plt.ylabel(feature.replace('_', ' '))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        out = os.path.join(plots_dir, f"Violin_{feature.replace('/', '_')}.png")
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"  Saved: Violin_{feature}.png")


if __name__ == "__main__":
    print(f"\n{'='*55}\nVIOLIN PLOTS — New App\n{'='*55}")
    master_df = compile_master_df()
    if master_df.empty:
        print("No feature CSVs found.")
    else:
        print(f"  Compiled {len(master_df)} rows, {len(master_df.columns)} columns.")
        generate_violin_plots(master_df)
        print(f"\nAll plots saved to: {plots_dir}")
