import os
import glob
import numpy as np
import pandas as pd

# I-VT Algorithm Constants
VELOCITY_THRESHOLD_CM_S = 20.0  # Speeds > 20 cm/s are Saccades. Speeds < 20 cm/s are Fixations
MIN_FIXATION_DURATION_SEC = 0.100 # A true fixation must last at least 100 milliseconds

def process_ivt_gaze_kinematics(df_synced):
    """
    Applies the I-VT (Velocity-Threshold) algorithm to raw gaze data to extract 
    Fixations and Saccades, filtering out hardware noise.
    """
    # 1. Clean the data (Drop rows where the eye-tracker lost the eyes)
    gaze_df = df_synced[['timestamp', 'gaze_x_cm', 'gaze_y_cm']].dropna().copy()
    
    # FIX: The column says '_cm' but the raw data is x10 mm. 
    # Multiply by 0.01 to convert back to true Centimeters.
    gaze_df['gaze_x_cm'] = gaze_df['gaze_x_cm'] * 0.01
    gaze_df['gaze_y_cm'] = gaze_df['gaze_y_cm'] * 0.01
    
    if gaze_df.empty or len(gaze_df) < 5:
        return {} # Return empty if no gaze data exists
        
    # 2. Smooth the coordinates to remove micro-jitter (Rolling mean over 3 frames)
    gaze_df['x_smooth'] = gaze_df['gaze_x_cm'].rolling(window=3, center=True, min_periods=1).mean()
    gaze_df['y_smooth'] = gaze_df['gaze_y_cm'].rolling(window=3, center=True, min_periods=1).mean()
    
    # 3. Calculate instantaneous velocity (cm/sec)
    dt = gaze_df['timestamp'].diff().fillna(1/30.0)
    dt = dt.replace(0, 1/30.0) # Prevent division by zero
    
    dx = gaze_df['x_smooth'].diff().fillna(0)
    dy = gaze_df['y_smooth'].diff().fillna(0)
    
    gaze_df['velocity'] = np.sqrt(dx**2 + dy**2) / dt
    
    # 4. Classify every frame as Saccade (True) or Fixation (False)
    gaze_df['is_saccade'] = gaze_df['velocity'] > VELOCITY_THRESHOLD_CM_S
    
    # Group contiguous blocks of movement/fixation
    gaze_df['block'] = (gaze_df['is_saccade'] != gaze_df['is_saccade'].shift()).cumsum()
    
    # --- CALCULATE FIXATIONS ---
    fixation_blocks = gaze_df[~gaze_df['is_saccade']].groupby('block')
    valid_fix_durations = []
    
    for _, block in fixation_blocks:
        duration = len(block) * (1/30.0)
        # Clinical standard: only count if they hold still for > 100ms
        if duration >= MIN_FIXATION_DURATION_SEC:
            valid_fix_durations.append(duration)
            
    fix_count = len(valid_fix_durations)
    mean_fix_dur = np.mean(valid_fix_durations) if valid_fix_durations else 0.0
    
    # --- CALCULATE SACCADES ---
    saccade_blocks = gaze_df[gaze_df['is_saccade']].groupby('block')
    peak_velocities = []
    amplitudes = []
    
    for _, block in saccade_blocks:
        if len(block) >= 2: # Ignore 1-frame spikes
            peak_v = block['velocity'].max()
            
            # Amplitude = straight-line distance from start of jump to end of jump
            start_x, start_y = block['x_smooth'].iloc[0], block['y_smooth'].iloc[0]
            end_x, end_y = block['x_smooth'].iloc[-1], block['y_smooth'].iloc[-1]
            amp = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            
            peak_velocities.append(peak_v)
            amplitudes.append(amp)
            
    total_gaze_time = gaze_df['timestamp'].max() - gaze_df['timestamp'].min()
    saccade_count = len(peak_velocities)
    sacc_freq = saccade_count / total_gaze_time if total_gaze_time > 0 else 0.0
    mean_amp = np.mean(amplitudes) if amplitudes else 0.0
    mean_peak_vel = np.mean(peak_velocities) if peak_velocities else 0.0

    return {
        'Fixation_Count': fix_count,
        'Mean_Fixation_Duration_sec': mean_fix_dur,
        'Saccade_Frequency_Hz': sacc_freq,
        'Mean_Saccadic_Amplitude_cm': mean_amp,
        'Peak_Saccadic_Velocity_cm_s': mean_peak_vel
    }

def extract_gaze_features(patient_folder, game_keyword):
    patient_id = os.path.basename(patient_folder)
    
    summary_files = glob.glob(os.path.join(patient_folder, f'*{game_keyword}*.mp4.csv'))
    synced_files = glob.glob(os.path.join(patient_folder, f'*{game_keyword}*synced_timeseries.csv'))
    
    if not summary_files or not synced_files:
        return None

    df_summary = pd.read_csv(summary_files[0])
    df_synced = pd.read_csv(synced_files[0])
    
    features = {'Patient_ID': patient_id, 'Game': game_keyword}

    # --- 1. EXTRACT UNIVERSAL GAZE RATIO (From mp4.csv) ---
    df_summary['Metric'] = df_summary['Metric'].astype(str).str.strip()
    
    try:
        gaze_in = float(df_summary.loc[df_summary['Metric'] == 'Gaze detected within tablet view', 'Value'].values[0])
        gaze_out = float(df_summary.loc[df_summary['Metric'] == 'Gaze detected outside tablet view', 'Value'].values[0])
        
        # Calculate Ratio (Safeguard against divide-by-zero if they never looked away)
        features['Gaze_In_Out_Ratio'] = gaze_in / gaze_out if gaze_out > 0 else gaze_in
        
        # Save raw percentages just in case
        features['Gaze_Pct_Within_Screen'] = gaze_in
        features['Gaze_Pct_Outside_Screen'] = gaze_out
        
    except IndexError:
        print(f"❌ Error: Missing Gaze Ratio metrics for {patient_id} ({game_keyword}).")
        return None

    # --- 2. EXTRACT DYNAMIC KINEMATICS (From synced_timeseries) ---
    ivt_metrics = process_ivt_gaze_kinematics(df_synced)
    if not ivt_metrics:
        return None # Skip if no usable gaze data was found
        
    # Game Logic Arrays
    fixation_games = ['TouchIt', 'CornerIt', 'DoubleTapIt', 'DragIt']
    saccade_games = ['CornerIt', 'DoubleTapIt', 'PinchIt', 'SlideIt']
    
    if game_keyword in fixation_games:
        features['Fixation_Count'] = ivt_metrics['Fixation_Count']
        features['Mean_Fixation_Duration_sec'] = ivt_metrics['Mean_Fixation_Duration_sec']
        
    if game_keyword in saccade_games:
        features['Saccade_Frequency_Hz'] = ivt_metrics['Saccade_Frequency_Hz']
        features['Mean_Saccadic_Amplitude_cm'] = ivt_metrics['Mean_Saccadic_Amplitude_cm']
        features['Peak_Saccadic_Velocity_cm_s'] = ivt_metrics['Peak_Saccadic_Velocity_cm_s']

    return features

if __name__ == "__main__":
    PROCESSED_DIR = "/Users/naghamdaood/Projects/TATOO/gaze_analysis/data/processed/"
    
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    patient_folders = [f.path for f in os.scandir(PROCESSED_DIR) if f.is_dir()]
    
    # We run this universally on ALL 6 games!
    TARGET_GAMES = ['TouchIt', 'CornerIt', 'DoubleTapIt', 'PinchIt', 'SlideIt', 'DragIt']
    
    for game in TARGET_GAMES:
        print(f"\n{'='*50}")
        print(f"👁️  EXTRACTING GAZE FEATURES FOR: {game}")
        print(f"{'='*50}\n")
        
        game_features = []
        
        for folder in sorted(patient_folders):
            feats = extract_gaze_features(folder, game)
            if feats:
                game_features.append(feats)
                print(f"✅ Success: {feats['Patient_ID']} | Screen Focus: {feats['Gaze_Pct_Within_Screen']:.1f}%")

        if game_features:
            final_df = pd.DataFrame(game_features)
            out_path = os.path.join(PROCESSED_DIR, f"{game}_Gaze_Features.csv")
            final_df.to_csv(out_path, index=False)
            print(f"\n💾 Saved {game} Gaze Matrix to: {out_path}")