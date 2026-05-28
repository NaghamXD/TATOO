"""
09_merge_matrices.py — New App
Merges ML, Gaze, and Sync feature CSVs into one MASTER_{game}.csv per game.

No demographics are attached (not available for new-app participants yet).
MASTER files contain features + Patient_ID only — no Target columns.
When demographics become available, run a separate attach_labels.py script.
"""
import os
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_new_app import PROCESSED_DIR, ALL_GAMES


def merge_game(processed_dir, game):
    touch_file = os.path.join(processed_dir, f"{game}_ML_Features.csv")
    gaze_file  = os.path.join(processed_dir, f"{game}_Gaze_Features.csv")
    sync_file  = os.path.join(processed_dir, f"{game}_Sync_Features.csv")

    if not os.path.exists(touch_file):
        print(f"  Skipping {game} — ML_Features not found.")
        return

    df_touch = pd.read_csv(touch_file)
    df_gaze  = pd.read_csv(gaze_file) if os.path.exists(gaze_file) else pd.DataFrame()
    df_sync  = pd.read_csv(sync_file) if os.path.exists(sync_file) else pd.DataFrame()

    df_master = df_touch.copy()

    if not df_gaze.empty:
        df_gaze = df_gaze.drop(columns=['Game'], errors='ignore')
        df_master = pd.merge(df_master, df_gaze, on='Patient_ID', how='outer')

    if not df_sync.empty:
        df_sync = df_sync.drop(columns=['Game'], errors='ignore')
        df_master = pd.merge(df_master, df_sync, on='Patient_ID', how='outer')

    df_master['Patient_ID'] = df_master['Patient_ID'].astype(str).str.strip()

    out_path = os.path.join(processed_dir, f"MASTER_{game}.csv")
    df_master.to_csv(out_path, index=False)
    print(f"  {game}: {df_master.shape[0]} participants, {df_master.shape[1]} features  -> MASTER_{game}.csv")


if __name__ == "__main__":
    processed_dir = str(PROCESSED_DIR)

    print(f"\n{'='*55}")
    print(f"MERGING MASTER DATASETS — New App (no demographics)")
    print(f"{'='*55}")

    for game in ALL_GAMES:
        merge_game(processed_dir, game)

    print(f"\nDone. MASTER files saved to: {processed_dir}")
    print("Note: Target labels will be attached when demographics become available.")
