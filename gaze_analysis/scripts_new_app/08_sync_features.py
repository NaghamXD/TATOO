"""
08_sync_features.py — New App
Extracts eye-hand coordination features for all 8 games.
PinchItOut added to dragging_games (gets latency + directional alignment).
HoldIt handled by script 15 (only Mean_Eye_Hand_Distance_cm applies).
"""
import os
import glob
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_new_app import PROCESSED_DIR, ALL_GAMES, TAPPING_GAMES, DRAGGING_GAMES

FPS = 30.0
DT  = 1.0 / FPS


def calculate_continuous_sync_metrics(df_valid):
    valid_indices = df_valid.index.tolist()
    if not valid_indices:
        return np.nan, np.nan

    strokes = np.split(valid_indices, np.where(np.diff(valid_indices) != 1)[0] + 1)
    latencies, alignments = [], []

    for stroke in strokes:
        if len(stroke) < 35:
            continue

        s = df_valid.loc[stroke].copy()
        gaze_dx  = s['gaze_x_cm'].diff().fillna(0)
        gaze_dy  = s['gaze_y_cm'].diff().fillna(0)
        touch_dx = s['touch_x_cm'].diff().fillna(0)
        touch_dy = s['touch_y_cm'].diff().fillna(0)
        gaze_speed  = np.sqrt(gaze_dx**2  + gaze_dy**2)
        touch_speed = np.sqrt(touch_dx**2 + touch_dy**2)

        # Directional alignment
        dot   = gaze_dx * touch_dx + gaze_dy * touch_dy
        mag   = gaze_speed * touch_speed
        valid = mag > 0
        if valid.any():
            alignments.append((dot[valid] / mag[valid]).mean())

        # Cross-correlation latency
        best_lag, max_corr = 0, -1.0
        if touch_speed.std() > 0 and gaze_speed.std() > 0:
            for shift in range(-15, 16):
                c = touch_speed.corr(gaze_speed.shift(shift))
                if pd.notna(c) and c > max_corr:
                    max_corr, best_lag = c, shift
        latencies.append(best_lag * DT)

    return (np.mean(latencies)  if latencies  else np.nan,
            np.mean(alignments) if alignments else np.nan)


def extract_sync_features(patient_folder, game_keyword):
    patient_id = os.path.basename(patient_folder)

    synced_files = glob.glob(os.path.join(patient_folder, f'*{game_keyword}*synced_timeseries.csv'))
    if not synced_files:
        return None

    df_synced = pd.read_csv(synced_files[0])
    features  = {'Patient_ID': patient_id, 'Game': game_keyword}

    touch_x_col = 'touch_x_raw' if 'touch_x_raw' in df_synced.columns else 'touch_x_cm'
    touch_y_col = 'touch_y_raw' if 'touch_y_raw' in df_synced.columns else 'touch_y_cm'

    df_valid = df_synced.dropna(subset=['gaze_x_cm', 'gaze_y_cm', touch_x_col, touch_y_col]).copy()
    if df_valid.empty:
        return None

    df_valid['gaze_x_cm']  = df_valid['gaze_x_cm'] * 0.01
    df_valid['gaze_y_cm']  = df_valid['gaze_y_cm'] * 0.01
    df_valid['touch_x_cm'] = df_valid[touch_x_col]  * 0.01
    df_valid['touch_y_cm'] = df_valid[touch_y_col]  * 0.01

    # Multi-touch centroid smoothing for PinchIt and PinchItOut
    if game_keyword in ('PinchIt', 'PinchItOut'):
        df_valid['touch_x_cm'] = df_valid['touch_x_cm'].rolling(window=2, min_periods=1).mean()
        df_valid['touch_y_cm'] = df_valid['touch_y_cm'].rolling(window=2, min_periods=1).mean()

    # Universal: spatial distance
    dx = df_valid['gaze_x_cm'] - df_valid['touch_x_cm']
    dy = df_valid['gaze_y_cm'] - df_valid['touch_y_cm']
    features['Mean_Eye_Hand_Distance_cm'] = np.sqrt(dx**2 + dy**2).mean()

    # Dragging games only: latency + directional alignment
    if game_keyword in DRAGGING_GAMES:
        lat, align = calculate_continuous_sync_metrics(df_valid)
        features['True_Eye_Hand_Latency_sec']     = lat
        features['Directional_Velocity_Alignment'] = align

    return features


if __name__ == "__main__":
    processed_dir   = str(PROCESSED_DIR)
    patient_folders = sorted(f.path for f in os.scandir(processed_dir) if f.is_dir())

    # All games except HoldIt (handled by script 15)
    target_games = [g for g in ALL_GAMES if g != 'HoldIt']

    for game in target_games:
        print(f"\n{'='*50}")
        print(f"SYNC FEATURES: {game}")
        print(f"{'='*50}")

        game_features = []
        for folder in patient_folders:
            feats = extract_sync_features(folder, game)
            if feats:
                game_features.append(feats)
                dist = feats.get('Mean_Eye_Hand_Distance_cm', 0)
                if game in DRAGGING_GAMES:
                    lag   = feats.get('True_Eye_Hand_Latency_sec', 0)
                    align = feats.get('Directional_Velocity_Alignment', 0)
                    print(f"  OK: {feats['Patient_ID']} | Gap: {dist:.2f}cm | Lag: {lag:.2f}s | Align: {align:.2f}")
                else:
                    print(f"  OK: {feats['Patient_ID']} | Gap: {dist:.2f}cm")

        if game_features:
            out_path = os.path.join(processed_dir, f"{game}_Sync_Features.csv")
            pd.DataFrame(game_features).to_csv(out_path, index=False)
            print(f"  Saved: {out_path}")
