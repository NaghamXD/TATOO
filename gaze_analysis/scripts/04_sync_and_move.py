import os
import glob
import shutil
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def extract_and_sync_npz(npz_file_path):
    """
    Extracts the 'first', 'second', 'third', and 'fourth' arrays from the NPZ,
    builds DataFrames, and syncs them by timestamp at 30Hz.
    """
    with np.load(npz_file_path) as data:
        # Pulling explicitly by the array names you provided
        gaze_x_arr = data['first']
        gaze_y_arr = data['second']
        touch_x_arr = data['third']
        touch_y_arr = data['fourth']

    # Build Gaze DataFrame
    df_gaze = pd.DataFrame({
        'timestamp': gaze_x_arr[:, 0],
        'gaze_x_cm': gaze_x_arr[:, 1],  # Based on your "Vision X Smoothed cm" note
        'gaze_y_cm': gaze_y_arr[:, 1]
    }).sort_values('timestamp')

    # Build Touch DataFrame
    df_touch = pd.DataFrame({
        'timestamp': touch_x_arr[:, 0],
        'touch_x_raw': touch_x_arr[:, 1], # Based on your "Touch X Raw" note
        'touch_y_raw': touch_y_arr[:, 1]
    }).sort_values('timestamp')

    # Synchronize using the 34ms tolerance
    df_synced = pd.merge_asof(
        df_gaze, 
        df_touch, 
        on='timestamp', 
        direction='nearest',
        tolerance=0.034 
    )
    
    return df_synced

def run_extraction_pipeline(raw_dir, processed_dir):
    print(f"\n{'='*60}")
    print(f"🚀 INITIATING DATA EXTRACTION & SYNC PIPELINE")
    print(f"{'='*60}")
    
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    patient_folders = [f.path for f in os.scandir(raw_dir) if f.is_dir()]

    for folder in sorted(patient_folders):
        patient_id = os.path.basename(folder)
        print(f"\n⚙️ Processing Patient: {patient_id}")
        
        # Create the destination folder in the processed directory
        patient_out_dir = os.path.join(processed_dir, patient_id)
        if not os.path.exists(patient_out_dir):
            os.makedirs(patient_out_dir)

        # ---------------------------------------------------------
        # PHASE 1: Move the CSV Files
        # ---------------------------------------------------------
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        
        if not csv_files:
            print("  ⚠️ No CSV files found to move.")
        else:
            for csv_path in csv_files:
                csv_filename = os.path.basename(csv_path)
                dest_path = os.path.join(patient_out_dir, csv_filename)
                
                # Move the file from raw to processed
                shutil.move(csv_path, dest_path)
            print(f"  -> Moved {len(csv_files)} CSV files to processed folder.")

        # ---------------------------------------------------------
        # PHASE 2: Sync the NPZ Matrices
        # ---------------------------------------------------------
        npz_files = glob.glob(os.path.join(folder, "*.npz"))
        
        if not npz_files:
            print("  ⚠️ No NPZ files found to sync.")
            continue
            
        for npz_path in npz_files:
            npz_filename = os.path.basename(npz_path)
            
            # Extract the core game name for the output file
            try:
                game_name = npz_filename.split('_VID_eyedata_')[0]
            except IndexError:
                game_name = "UnknownGame"
                
            print(f"  -> Syncing matrices for: {game_name}")
            
            try:
                synced_ts_df = extract_and_sync_npz(npz_path)
                
                # Save the synced time-series next to the moved CSVs
                out_ts_path = os.path.join(patient_out_dir, f"{game_name}_synced_timeseries.csv")
                synced_ts_df.to_csv(out_ts_path, index=False)
            except Exception as e:
                print(f"  ❌ Error syncing {npz_filename}: {e}")

        print(f"  ✅ Completed {patient_id}.")

    print(f"\n{'='*60}")
    print("✅ ALL FOLDERS PROCESSED SUCCESSFULLY")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Update these paths to match your directories
    RAW_DATA_DIRECTORY = "gaze_analysis/data/raw/"
    PROCESSED_DATA_DIRECTORY = "gaze_analysis/data/processed/"
    
    run_extraction_pipeline(RAW_DATA_DIRECTORY, PROCESSED_DATA_DIRECTORY)