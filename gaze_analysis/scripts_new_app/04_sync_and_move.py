"""
04_sync_and_move.py — New App
Extracts NPZ gaze+touch arrays, syncs them at 30 Hz, saves synced timeseries CSVs.
Moves summary CSVs to processed_new_app/.

New filename format: {participant}_{timestamp}_{Game}.mp4_res.mp4multiple_arrays.npz
Game name is the last underscore-delimited token before .mp4
"""
import os
import re
import glob
import shutil
import sys
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_new_app import RAW_DIR, PROCESSED_DIR, SYNC_TOLERANCE_MS

SYNC_TOLERANCE_SEC = SYNC_TOLERANCE_MS / 1000.0

# Regex to extract game name from new filename
GAME_RE = re.compile(r"_(\w+)\.mp4_res\.mp4multiple_arrays\.npz$")


def extract_game_name(npz_filename):
    m = GAME_RE.search(npz_filename)
    return m.group(1) if m else "UnknownGame"


def extract_and_sync_npz(npz_path):
    with np.load(npz_path) as data:
        gaze_x_arr = data['first']
        gaze_y_arr = data['second']
        touch_x_arr = data['third']
        touch_y_arr = data['fourth']

    df_gaze = pd.DataFrame({
        'timestamp': gaze_x_arr[:, 0],
        'gaze_x_cm': gaze_x_arr[:, 1],
        'gaze_y_cm': gaze_y_arr[:, 1],
    }).sort_values('timestamp')

    df_touch = pd.DataFrame({
        'timestamp': touch_x_arr[:, 0],
        'touch_x_raw': touch_x_arr[:, 1],
        'touch_y_raw': touch_y_arr[:, 1],
    }).sort_values('timestamp')

    df_synced = pd.merge_asof(
        df_gaze, df_touch,
        on='timestamp',
        direction='nearest',
        tolerance=SYNC_TOLERANCE_SEC,
    )
    return df_synced


def run_extraction_pipeline(raw_dir, processed_dir):
    raw_dir       = str(raw_dir)
    processed_dir = str(processed_dir)

    print(f"\n{'='*60}")
    print(f"DATA EXTRACTION & SYNC PIPELINE — New App")
    print(f"{'='*60}")

    os.makedirs(processed_dir, exist_ok=True)

    patient_folders = sorted(f.path for f in os.scandir(raw_dir) if f.is_dir())
    if not patient_folders:
        print("No participant subfolders found. Run script 01 first.")
        return

    for folder in patient_folders:
        patient_id    = os.path.basename(folder)
        patient_out   = os.path.join(processed_dir, patient_id)
        os.makedirs(patient_out, exist_ok=True)
        print(f"\nProcessing: {patient_id}")

        # Phase 1: move summary CSVs
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        for csv_path in csv_files:
            dest = os.path.join(patient_out, os.path.basename(csv_path))
            shutil.copy2(csv_path, dest)
        print(f"  Copied {len(csv_files)} CSV files.")

        # Phase 2: sync NPZ files
        npz_files = glob.glob(os.path.join(folder, "*.npz"))
        if not npz_files:
            print("  No NPZ files found.")
            continue

        for npz_path in npz_files:
            npz_filename = os.path.basename(npz_path)
            game_name    = extract_game_name(npz_filename)
            print(f"  Syncing: {game_name}")
            try:
                df_synced = extract_and_sync_npz(npz_path)
                out_path  = os.path.join(patient_out, f"{game_name}_synced_timeseries.csv")
                df_synced.to_csv(out_path, index=False)
            except Exception as e:
                print(f"  Error syncing {npz_filename}: {e}")

        print(f"  Done: {patient_id}")

    print(f"\n{'='*60}")
    print("ALL FOLDERS PROCESSED")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_extraction_pipeline(RAW_DIR, PROCESSED_DIR)
