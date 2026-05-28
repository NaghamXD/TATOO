"""
05_tapping_features.py — New App
Extracts tapping features for TouchIt, CornerIt, DoubleTapIt.
Logic identical to old pipeline; only PROCESSED_DIR and game list change.
Note: Accuracy_Ratio only for CornerIt and DoubleTapIt (not TouchIt).
"""
import os
import glob
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_new_app import PROCESSED_DIR, TAPPING_GAMES


def get_pressure_jitter(raw_events_df):
    is_stationary = raw_events_df['Phase'] == 'Stationary'
    block_ids     = (~is_stationary).cumsum()
    stationary    = raw_events_df[is_stationary]

    if stationary.empty:
        return 0.0

    blocks = stationary.groupby(block_ids.loc[is_stationary.index])
    valid_jitters = []
    for _, block in blocks:
        if len(block) > 2:
            j = block['Pressure'].std()
            valid_jitters.append(0.0 if pd.isna(j) else j)

    return np.mean(valid_jitters) if valid_jitters else 0.0


def extract_tapping_features(patient_folder, game_keyword):
    patient_id = os.path.basename(patient_folder)

    summary_files   = glob.glob(os.path.join(patient_folder, f'*{game_keyword}*.mp4.csv'))
    all_csvs        = glob.glob(os.path.join(patient_folder, f'*{game_keyword}*.csv'))
    raw_event_files = [f for f in all_csvs if 'mp4' not in f and 'synced' not in f]

    if not summary_files:
        return None

    df_summary = pd.read_csv(summary_files[0])
    df_raw     = pd.read_csv(raw_event_files[0]) if raw_event_files else None

    features = {'Patient_ID': patient_id, 'Game': game_keyword}
    df_summary['Metric'] = df_summary['Metric'].astype(str).str.strip()

    try:
        if game_keyword == 'TouchIt':
            features['Number_of_Taps'] = float(
                df_summary.loc[df_summary['Metric'] == 'Number of taps', 'Value'].values[0])
        else:
            # CornerIt and DoubleTapIt only
            features['Game_Duration_sec'] = float(
                df_summary.loc[df_summary['Metric'] == 'Game Duration (sec)', 'Value'].values[0])
            correct = float(
                df_summary.loc[df_summary['Metric'] == 'Number of correct taps', 'Value'].values[0])
            outside = float(
                df_summary.loc[df_summary['Metric'] == 'Number of taps outside target', 'Value'].values[0])
            features['Correct_Taps']   = correct
            features['Outside_Taps']   = outside
            features['Accuracy_Ratio'] = correct / outside if outside > 0 else correct

        features['First_Reaction_Time_sec'] = float(
            df_summary.loc[df_summary['Metric'] == 'First reaction time (sec)', 'Value'].values[0])

        flight_dur = float(df_summary.loc[df_summary['Metric'] == 'Flight Duration (sec)', 'Value'].values[0])
        touch_dur  = float(df_summary.loc[df_summary['Metric'] == 'Touch Duration (sec)', 'Value'].values[0])
        features['Flight_Duration_sec'] = flight_dur
        features['Touch_Duration_sec']  = touch_dur
        features['Flight_Touch_Ratio']  = flight_dur / touch_dur if touch_dur > 0 else flight_dur

        features['Pressure_Low_Pct']  = float(df_summary.loc[df_summary['Metric'] == 'Low pressure',    'Value'].values[0])
        features['Pressure_Med_Pct']  = float(df_summary.loc[df_summary['Metric'] == 'Medium pressure', 'Value'].values[0])
        features['Pressure_High_Pct'] = float(df_summary.loc[df_summary['Metric'] == 'High pressure',   'Value'].values[0])

    except IndexError:
        print(f"  Error: missing metric for {patient_id} ({game_keyword}).")
        return None

    if df_raw is not None:
        try:
            features['Pressure_Jitter_Stationary'] = get_pressure_jitter(df_raw)
            features['Mean_Pressure_Overall']       = df_raw['Pressure'].mean()
        except Exception as e:
            print(f"  Warning: could not parse raw events for {patient_id} ({game_keyword}): {e}")
            features['Pressure_Jitter_Stationary'] = np.nan
            features['Mean_Pressure_Overall']       = np.nan
    else:
        # New app does not produce raw event files — pressure micro-features unavailable
        features['Pressure_Jitter_Stationary'] = np.nan
        features['Mean_Pressure_Overall']       = np.nan

    return features


if __name__ == "__main__":
    processed_dir  = str(PROCESSED_DIR)
    patient_folders = sorted(f.path for f in os.scandir(processed_dir) if f.is_dir())

    for game in TAPPING_GAMES:
        print(f"\n{'='*50}")
        print(f"TAPPING FEATURES: {game}")
        print(f"{'='*50}")

        game_features = []
        for folder in patient_folders:
            feats = extract_tapping_features(folder, game)
            if feats:
                game_features.append(feats)
                if game == 'TouchIt':
                    print(f"  OK: {feats['Patient_ID']} | Taps: {feats['Number_of_Taps']}")
                else:
                    print(f"  OK: {feats['Patient_ID']} | Accuracy: {feats['Accuracy_Ratio']:.2f}")

        if game_features:
            out_path = os.path.join(processed_dir, f"{game}_ML_Features.csv")
            pd.DataFrame(game_features).to_csv(out_path, index=False)
            print(f"  Saved: {out_path}")
