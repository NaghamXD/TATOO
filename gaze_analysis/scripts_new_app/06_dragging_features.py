"""
06_dragging_features.py — New App
Extracts dragging/kinematics features for PinchIt, SlideIt, DragIt, PinchItOut.

Changes vs old pipeline:
- PinchItOut added to DRAGGING_GAMES
- Pressure_Med_Pct imported for all games (aligned with other games, not used in ML)
- Accuracy_Ratio is NOT extracted here (only valid for tapping-target games)
"""
import os
import glob
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_new_app import PROCESSED_DIR, DRAGGING_GAMES

FPS = 30.0
DT  = 1.0 / FPS


def get_kinematics_and_duration(df_synced):
    touch_x_col = 'touch_x_raw' if 'touch_x_raw' in df_synced.columns else 'touch_x_cm'
    touch_y_col = 'touch_y_raw' if 'touch_y_raw' in df_synced.columns else 'touch_y_cm'

    mask         = df_synced[touch_x_col].notna()
    valid_indices = df_synced.index[mask].tolist()
    if not valid_indices:
        return 0.0, 0.0

    strokes = np.split(valid_indices, np.where(np.diff(valid_indices) != 1)[0] + 1)
    durations       = []
    jerk_magnitudes = []

    for stroke in strokes:
        if len(stroke) >= 2:
            durations.append(len(stroke) * DT)

        if len(stroke) >= 5:
            x = df_synced.loc[stroke, touch_x_col] * 0.01
            y = df_synced.loc[stroke, touch_y_col] * 0.01
            x = x.rolling(window=2, min_periods=1).mean().values
            y = y.rolling(window=2, min_periods=1).mean().values

            vx = np.gradient(x) / DT
            vy = np.gradient(y) / DT
            ax = np.gradient(vx) / DT
            ay = np.gradient(vy) / DT
            jx = np.gradient(ax) / DT
            jy = np.gradient(ay) / DT
            jerk_magnitudes.append(np.mean(np.sqrt(jx**2 + jy**2)))

    return (np.mean(durations) if durations else 0.0,
            np.mean(jerk_magnitudes) if jerk_magnitudes else 0.0)


def extract_dragging_features(patient_folder, game_keyword):
    patient_id = os.path.basename(patient_folder)

    summary_files = glob.glob(os.path.join(patient_folder, f'*{game_keyword}*.mp4.csv'))
    synced_files  = glob.glob(os.path.join(patient_folder, f'*{game_keyword}*synced_timeseries.csv'))

    if not summary_files or not synced_files:
        return None

    df_summary = pd.read_csv(summary_files[0])
    df_synced  = pd.read_csv(synced_files[0])

    features = {'Patient_ID': patient_id, 'Game': game_keyword}
    df_summary['Metric'] = df_summary['Metric'].astype(str).str.strip()

    try:
        features['Game_Duration_sec']       = float(df_summary.loc[df_summary['Metric'] == 'Game Duration (sec)',       'Value'].values[0])
        features['First_Reaction_Time_sec'] = float(df_summary.loc[df_summary['Metric'] == 'First reaction time (sec)', 'Value'].values[0])

        flight_dur = float(df_summary.loc[df_summary['Metric'] == 'Flight Duration (sec)', 'Value'].values[0])
        touch_dur  = float(df_summary.loc[df_summary['Metric'] == 'Touch Duration (sec)',  'Value'].values[0])
        features['Flight_Duration_sec'] = flight_dur
        features['Touch_Duration_sec']  = touch_dur
        features['Flight_Touch_Ratio']  = flight_dur / touch_dur if touch_dur > 0 else flight_dur

        # Pressure — Med_Pct imported for alignment but not used in ML model
        features['Pressure_Low_Pct']  = float(df_summary.loc[df_summary['Metric'] == 'Low pressure',    'Value'].values[0])
        features['Pressure_Med_Pct']  = float(df_summary.loc[df_summary['Metric'] == 'Medium pressure', 'Value'].values[0])
        features['Pressure_High_Pct'] = float(df_summary.loc[df_summary['Metric'] == 'High pressure',   'Value'].values[0])

    except IndexError:
        print(f"  Error: missing macro metric for {patient_id} ({game_keyword}).")
        return None

    try:
        mean_dur, mean_jerk = get_kinematics_and_duration(df_synced)
        features['Mean_Action_Duration_sec'] = mean_dur
        features['Spatial_Jerk_Magnitude']   = mean_jerk
    except Exception as e:
        print(f"  Error computing kinematics for {patient_id} ({game_keyword}): {e}")
        return None

    return features


if __name__ == "__main__":
    processed_dir   = str(PROCESSED_DIR)
    patient_folders = sorted(f.path for f in os.scandir(processed_dir) if f.is_dir())

    for game in DRAGGING_GAMES:
        print(f"\n{'='*50}")
        print(f"DRAGGING FEATURES: {game}")
        print(f"{'='*50}")

        game_features = []
        for folder in patient_folders:
            feats = extract_dragging_features(folder, game)
            if feats:
                game_features.append(feats)
                print(f"  OK: {feats['Patient_ID']} | Duration: {feats['Mean_Action_Duration_sec']:.2f}s | Jerk: {feats['Spatial_Jerk_Magnitude']:.2f}")

        if game_features:
            out_path = os.path.join(processed_dir, f"{game}_ML_Features.csv")
            pd.DataFrame(game_features).to_csv(out_path, index=False)
            print(f"  Saved: {out_path}")
