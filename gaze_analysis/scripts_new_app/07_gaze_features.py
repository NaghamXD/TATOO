"""
07_gaze_features.py — New App
Extracts I-VT fixation/saccade features for all 8 games.
PinchItOut added to SACCADE_GAMES vs old pipeline.
HoldIt added to FIXATION_GAMES.
"""
import os
import glob
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_new_app import (
    PROCESSED_DIR, ALL_GAMES, FIXATION_GAMES, SACCADE_GAMES,
    VELOCITY_THRESHOLD_CM_S, MIN_FIXATION_DURATION_SEC,
)


def process_ivt_gaze_kinematics(df_synced):
    gaze_df = df_synced[['timestamp', 'gaze_x_cm', 'gaze_y_cm']].dropna().copy()
    gaze_df['gaze_x_cm'] = gaze_df['gaze_x_cm'] * 0.01
    gaze_df['gaze_y_cm'] = gaze_df['gaze_y_cm'] * 0.01

    if gaze_df.empty or len(gaze_df) < 5:
        return {}

    gaze_df['x_smooth'] = gaze_df['gaze_x_cm'].rolling(window=3, center=True, min_periods=1).mean()
    gaze_df['y_smooth'] = gaze_df['gaze_y_cm'].rolling(window=3, center=True, min_periods=1).mean()

    dt = gaze_df['timestamp'].diff().fillna(1 / 30.0).replace(0, 1 / 30.0)
    dx = gaze_df['x_smooth'].diff().fillna(0)
    dy = gaze_df['y_smooth'].diff().fillna(0)
    gaze_df['velocity'] = np.sqrt(dx**2 + dy**2) / dt
    gaze_df['is_saccade'] = gaze_df['velocity'] > VELOCITY_THRESHOLD_CM_S
    gaze_df['block'] = (gaze_df['is_saccade'] != gaze_df['is_saccade'].shift()).cumsum()

    # Fixations
    fix_durations = []
    for _, block in gaze_df[~gaze_df['is_saccade']].groupby('block'):
        dur = len(block) * (1 / 30.0)
        if dur >= MIN_FIXATION_DURATION_SEC:
            fix_durations.append(dur)

    # Saccades
    peak_velocities, amplitudes = [], []
    for _, block in gaze_df[gaze_df['is_saccade']].groupby('block'):
        if len(block) >= 2:
            peak_velocities.append(block['velocity'].max())
            sx, sy = block['x_smooth'].iloc[0], block['y_smooth'].iloc[0]
            ex, ey = block['x_smooth'].iloc[-1], block['y_smooth'].iloc[-1]
            amplitudes.append(np.sqrt((ex - sx)**2 + (ey - sy)**2))

    total_time   = gaze_df['timestamp'].max() - gaze_df['timestamp'].min()
    sacc_count   = len(peak_velocities)

    return {
        'Fixation_Count':             len(fix_durations),
        'Mean_Fixation_Duration_sec': np.mean(fix_durations) if fix_durations else 0.0,
        'Saccade_Frequency_Hz':       sacc_count / total_time if total_time > 0 else 0.0,
        'Mean_Saccadic_Amplitude_cm': np.mean(amplitudes) if amplitudes else 0.0,
        'Peak_Saccadic_Velocity_cm_s': np.mean(peak_velocities) if peak_velocities else 0.0,
    }


def extract_gaze_features(patient_folder, game_keyword):
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
        gaze_in  = float(df_summary.loc[df_summary['Metric'] == 'Gaze detected within tablet view',  'Value'].values[0])
        gaze_out = float(df_summary.loc[df_summary['Metric'] == 'Gaze detected outside tablet view', 'Value'].values[0])
        features['Gaze_In_Out_Ratio']        = gaze_in / gaze_out if gaze_out > 0 else gaze_in
        features['Gaze_Pct_Within_Screen']   = gaze_in
        features['Gaze_Pct_Outside_Screen']  = gaze_out
    except IndexError:
        print(f"  Error: missing gaze ratio for {patient_id} ({game_keyword}).")
        return None

    ivt = process_ivt_gaze_kinematics(df_synced)
    if not ivt:
        return None

    if game_keyword in FIXATION_GAMES:
        features['Fixation_Count']            = ivt['Fixation_Count']
        features['Mean_Fixation_Duration_sec'] = ivt['Mean_Fixation_Duration_sec']

    if game_keyword in SACCADE_GAMES:
        features['Saccade_Frequency_Hz']         = ivt['Saccade_Frequency_Hz']
        features['Mean_Saccadic_Amplitude_cm']   = ivt['Mean_Saccadic_Amplitude_cm']
        features['Peak_Saccadic_Velocity_cm_s']  = ivt['Peak_Saccadic_Velocity_cm_s']

    return features


if __name__ == "__main__":
    processed_dir   = str(PROCESSED_DIR)
    patient_folders = sorted(f.path for f in os.scandir(processed_dir) if f.is_dir())

    # Run on all games except HoldIt (handled by script 15)
    target_games = [g for g in ALL_GAMES if g != 'HoldIt']

    for game in target_games:
        print(f"\n{'='*50}")
        print(f"GAZE FEATURES: {game}")
        print(f"{'='*50}")

        game_features = []
        for folder in patient_folders:
            feats = extract_gaze_features(folder, game)
            if feats:
                game_features.append(feats)
                print(f"  OK: {feats['Patient_ID']} | Focus: {feats['Gaze_Pct_Within_Screen']:.1f}%")

        if game_features:
            out_path = os.path.join(processed_dir, f"{game}_Gaze_Features.csv")
            pd.DataFrame(game_features).to_csv(out_path, index=False)
            print(f"  Saved: {out_path}")
