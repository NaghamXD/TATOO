"""
15_holdit_features.py — New App (Brand New Script)
Extracts all features for the HoldIt game: sustained press-and-hold with precision timing.

Outputs three CSVs (matching the pattern of scripts 05-08):
  HoldIt_ML_Features.csv    — touch/motor features
  HoldIt_Gaze_Features.csv  — I-VT fixation features
  HoldIt_Sync_Features.csv  — spatial eye-hand distance
"""
import os
import glob
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_new_app import (
    PROCESSED_DIR,
    VELOCITY_THRESHOLD_CM_S, MIN_FIXATION_DURATION_SEC,
)

GAME = 'HoldIt'
FPS  = 30.0
DT   = 1.0 / FPS
MIN_HOLD_FRAMES = 5   # minimum contiguous touch frames to count as a hold


# ─────────────────────────────────────────────────────────────────────────────
# Touch / motor features
# ─────────────────────────────────────────────────────────────────────────────

def get_hold_segments(df_synced):
    """
    Split the synced timeseries into contiguous touch segments (holds).
    Returns a list of DataFrames, one per hold.
    """
    touch_x_col = 'touch_x_raw' if 'touch_x_raw' in df_synced.columns else 'touch_x_cm'
    mask          = df_synced[touch_x_col].notna()
    valid_indices = df_synced.index[mask].tolist()

    if not valid_indices:
        return []

    groups = np.split(valid_indices, np.where(np.diff(valid_indices) != 1)[0] + 1)
    return [df_synced.loc[g] for g in groups if len(g) >= MIN_HOLD_FRAMES]


def get_pressure_jitter_hold(df_synced):
    """
    Std of pressure across all hold segments — tremor index during sustained press.
    Uses the raw pressure column from the synced timeseries if available,
    otherwise falls back to the touch column variance as a proxy.
    """
    if 'Pressure' not in df_synced.columns:
        return np.nan
    touch_x_col = 'touch_x_raw' if 'touch_x_raw' in df_synced.columns else 'touch_x_cm'
    hold_mask   = df_synced[touch_x_col].notna()
    pressures   = df_synced.loc[hold_mask, 'Pressure'].dropna()
    return pressures.std() if len(pressures) > 2 else 0.0


def extract_holdit_ml_features(patient_folder):
    patient_id = os.path.basename(patient_folder)

    summary_files = glob.glob(os.path.join(patient_folder, f'*{GAME}*.mp4.csv'))
    synced_files  = glob.glob(os.path.join(patient_folder, f'*{GAME}*synced_timeseries.csv'))

    if not summary_files or not synced_files:
        return None

    df_summary = pd.read_csv(summary_files[0])
    df_synced  = pd.read_csv(synced_files[0])

    features = {'Patient_ID': patient_id, 'Game': GAME}
    df_summary['Metric'] = df_summary['Metric'].astype(str).str.strip()

    try:
        features['First_Reaction_Time_sec'] = float(
            df_summary.loc[df_summary['Metric'] == 'First reaction time (sec)', 'Value'].values[0])
        features['Game_Duration_sec'] = float(
            df_summary.loc[df_summary['Metric'] == 'Game Duration (sec)', 'Value'].values[0])

        flight_dur = float(df_summary.loc[df_summary['Metric'] == 'Flight Duration (sec)', 'Value'].values[0])
        touch_dur  = float(df_summary.loc[df_summary['Metric'] == 'Touch Duration (sec)',  'Value'].values[0])
        features['Flight_Duration_sec'] = flight_dur
        features['Touch_Duration_sec']  = touch_dur
        features['Flight_Touch_Ratio']  = flight_dur / touch_dur if touch_dur > 0 else flight_dur

        # Pressure percentages — Med_Pct imported for alignment, not used in ML
        features['Pressure_Low_Pct']  = float(df_summary.loc[df_summary['Metric'] == 'Low pressure',    'Value'].values[0])
        features['Pressure_Med_Pct']  = float(df_summary.loc[df_summary['Metric'] == 'Medium pressure', 'Value'].values[0])
        features['Pressure_High_Pct'] = float(df_summary.loc[df_summary['Metric'] == 'High pressure',   'Value'].values[0])

        # Hold attempt counts
        hold_count = float(df_summary.loc[df_summary['Metric'] == 'Number of taps',              'Value'].values[0])
        hold_ok    = float(df_summary.loc[df_summary['Metric'] == 'Number of correct taps',      'Value'].values[0])
        features['Hold_Count']        = hold_count
        features['Hold_Success_Rate'] = hold_ok / hold_count if hold_count > 0 else 0.0

    except IndexError:
        print(f"  Error: missing metric for {patient_id} ({GAME}).")
        return None

    # Timeseries-based hold features
    hold_segments = get_hold_segments(df_synced)
    if hold_segments:
        durations = [len(s) * DT for s in hold_segments]
        features['Mean_Hold_Duration_sec']      = np.mean(durations)
        features['Hold_Duration_Std_sec']       = np.std(durations) if len(durations) > 1 else 0.0

        # Inter-hold intervals (flight segments between holds)
        inter_intervals = []
        for i in range(1, len(hold_segments)):
            gap_end   = hold_segments[i].index[0]
            gap_start = hold_segments[i - 1].index[-1]
            inter_intervals.append((gap_end - gap_start) * DT)
        features['Mean_Inter_Hold_Interval_sec'] = np.mean(inter_intervals) if inter_intervals else 0.0
    else:
        features['Mean_Hold_Duration_sec']       = 0.0
        features['Hold_Duration_Std_sec']        = 0.0
        features['Mean_Inter_Hold_Interval_sec'] = 0.0

    # Pressure during hold (Mean + Jitter)
    features['Pressure_Jitter_Stationary'] = get_pressure_jitter_hold(df_synced)
    if 'Pressure' in df_synced.columns:
        touch_x_col = 'touch_x_raw' if 'touch_x_raw' in df_synced.columns else 'touch_x_cm'
        hold_mask   = df_synced[touch_x_col].notna()
        features['Mean_Pressure_Overall'] = df_synced.loc[hold_mask, 'Pressure'].mean() \
            if hold_mask.any() else np.nan
    else:
        features['Mean_Pressure_Overall'] = np.nan

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Gaze features (I-VT — fixation only, HoldIt is fixation-dominant)
# ─────────────────────────────────────────────────────────────────────────────

def extract_holdit_gaze_features(patient_folder):
    patient_id = os.path.basename(patient_folder)

    summary_files = glob.glob(os.path.join(patient_folder, f'*{GAME}*.mp4.csv'))
    synced_files  = glob.glob(os.path.join(patient_folder, f'*{GAME}*synced_timeseries.csv'))
    if not summary_files or not synced_files:
        return None

    df_summary = pd.read_csv(summary_files[0])
    df_synced  = pd.read_csv(synced_files[0])
    features   = {'Patient_ID': patient_id, 'Game': GAME}
    df_summary['Metric'] = df_summary['Metric'].astype(str).str.strip()

    try:
        gaze_in  = float(df_summary.loc[df_summary['Metric'] == 'Gaze detected within tablet view',  'Value'].values[0])
        gaze_out = float(df_summary.loc[df_summary['Metric'] == 'Gaze detected outside tablet view', 'Value'].values[0])
        features['Gaze_In_Out_Ratio']       = gaze_in / gaze_out if gaze_out > 0 else gaze_in
        features['Gaze_Pct_Within_Screen']  = gaze_in
        features['Gaze_Pct_Outside_Screen'] = gaze_out
    except IndexError:
        return None

    # I-VT fixations
    gaze_df = df_synced[['timestamp', 'gaze_x_cm', 'gaze_y_cm']].dropna().copy()
    gaze_df['gaze_x_cm'] = gaze_df['gaze_x_cm'] * 0.01
    gaze_df['gaze_y_cm'] = gaze_df['gaze_y_cm'] * 0.01

    if len(gaze_df) >= 5:
        gaze_df['x_s'] = gaze_df['gaze_x_cm'].rolling(window=3, center=True, min_periods=1).mean()
        gaze_df['y_s'] = gaze_df['gaze_y_cm'].rolling(window=3, center=True, min_periods=1).mean()
        dt = gaze_df['timestamp'].diff().fillna(1 / 30.0).replace(0, 1 / 30.0)
        dx = gaze_df['x_s'].diff().fillna(0)
        dy = gaze_df['y_s'].diff().fillna(0)
        velocity = np.sqrt(dx**2 + dy**2) / dt
        is_fix   = velocity <= VELOCITY_THRESHOLD_CM_S
        blocks   = (is_fix != is_fix.shift()).cumsum()

        fix_durs = []
        for _, block in gaze_df[is_fix].groupby(blocks[is_fix]):
            dur = len(block) * (1 / 30.0)
            if dur >= MIN_FIXATION_DURATION_SEC:
                fix_durs.append(dur)

        features['Fixation_Count']            = len(fix_durs)
        features['Mean_Fixation_Duration_sec'] = np.mean(fix_durs) if fix_durs else 0.0
    else:
        features['Fixation_Count']            = 0
        features['Mean_Fixation_Duration_sec'] = 0.0

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Sync features (spatial eye-hand distance only — no motion to correlate)
# ─────────────────────────────────────────────────────────────────────────────

def extract_holdit_sync_features(patient_folder):
    patient_id = os.path.basename(patient_folder)

    synced_files = glob.glob(os.path.join(patient_folder, f'*{GAME}*synced_timeseries.csv'))
    if not synced_files:
        return None

    df_synced   = pd.read_csv(synced_files[0])
    features    = {'Patient_ID': patient_id, 'Game': GAME}
    touch_x_col = 'touch_x_raw' if 'touch_x_raw' in df_synced.columns else 'touch_x_cm'
    touch_y_col = 'touch_y_raw' if 'touch_y_raw' in df_synced.columns else 'touch_y_cm'

    df_valid = df_synced.dropna(subset=['gaze_x_cm', 'gaze_y_cm', touch_x_col, touch_y_col]).copy()
    if df_valid.empty:
        return None

    df_valid['gaze_x_cm']  = df_valid['gaze_x_cm'] * 0.01
    df_valid['gaze_y_cm']  = df_valid['gaze_y_cm'] * 0.01
    df_valid['touch_x_cm'] = df_valid[touch_x_col] * 0.01
    df_valid['touch_y_cm'] = df_valid[touch_y_col] * 0.01

    dx = df_valid['gaze_x_cm'] - df_valid['touch_x_cm']
    dy = df_valid['gaze_y_cm'] - df_valid['touch_y_cm']
    features['Mean_Eye_Hand_Distance_cm'] = np.sqrt(dx**2 + dy**2).mean()

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    processed_dir   = str(PROCESSED_DIR)
    patient_folders = sorted(f.path for f in os.scandir(processed_dir) if f.is_dir())

    print(f"\n{'='*55}")
    print(f"HOLDIT FEATURE EXTRACTION")
    print(f"{'='*55}")

    ml_feats, gaze_feats, sync_feats = [], [], []

    for folder in patient_folders:
        ml   = extract_holdit_ml_features(folder)
        gaze = extract_holdit_gaze_features(folder)
        sync = extract_holdit_sync_features(folder)

        if ml:
            ml_feats.append(ml)
            print(f"  ML   OK: {ml['Patient_ID']} | Holds: {ml['Hold_Count']} | Success: {ml['Hold_Success_Rate']:.2f}")
        if gaze:
            gaze_feats.append(gaze)
        if sync:
            sync_feats.append(sync)

    for feats, suffix in [(ml_feats, 'ML_Features'), (gaze_feats, 'Gaze_Features'), (sync_feats, 'Sync_Features')]:
        if feats:
            out = os.path.join(processed_dir, f"{GAME}_{suffix}.csv")
            pd.DataFrame(feats).to_csv(out, index=False)
            print(f"  Saved: {out}")
