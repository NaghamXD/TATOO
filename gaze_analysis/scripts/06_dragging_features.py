import os
import glob
import numpy as np
import pandas as pd

# Tablet sample rate constant
FPS = 30.0
DT = 1.0 / FPS

def get_kinematics_and_duration(df_synced):
    """
    Finds continuous touch strokes in the synced CSV.
    Calculates the Mean Action Duration (Pinch/Slide/Drag) and the Spatial Jerk.
    """
    # Check column names dynamically to match your exact CSV headers
    touch_x_col = 'touch_x_raw' if 'touch_x_raw' in df_synced.columns else 'touch_x_cm'
    touch_y_col = 'touch_y_raw' if 'touch_y_raw' in df_synced.columns else 'touch_y_cm'
    
    # Find indices where touch data exists (finger is on the screen, not ',,')
    mask = df_synced[touch_x_col].notna()
    valid_indices = df_synced.index[mask].tolist()
    
    if not valid_indices:
        return 0.0, 0.0
        
    # Split the indices into contiguous blocks (each block is one continuous pinch/drag)
    strokes = np.split(valid_indices, np.where(np.diff(valid_indices) != 1)[0] + 1)
    
    durations = []
    jerk_magnitudes = []
    
    for stroke in strokes:
        # 1. Calculate Duration of this specific action
        if len(stroke) >= 2:
            duration_sec = len(stroke) * DT
            durations.append(duration_sec)
            
        # 2. Calculate Spatial Jerk (requires at least 5 points for 3rd derivative)
        if len(stroke) >= 5:
            # Convert to Centimeters
            x_series = df_synced.loc[stroke, touch_x_col] * 0.01
            y_series = df_synced.loc[stroke, touch_y_col] * 0.01
            
            # Apply Pinch Centroid smoothing (You can safely apply this rolling mean 
            # to all dragging games to remove hardware micro-jitter)
            x = x_series.rolling(window=2, min_periods=1).mean().values
            y = y_series.rolling(window=2, min_periods=1).mean().values
            
            vx = np.gradient(x) / DT
            vy = np.gradient(y) / DT
            ax = np.gradient(vx) / DT
            ay = np.gradient(vy) / DT
            jx = np.gradient(ax) / DT
            jy = np.gradient(ay) / DT
            
            # Get the mean magnitude of jerk for this single stroke
            stroke_jerk = np.mean(np.sqrt(jx**2 + jy**2))
            jerk_magnitudes.append(stroke_jerk)
            
    mean_duration = np.mean(durations) if durations else 0.0
    mean_jerk = np.mean(jerk_magnitudes) if jerk_magnitudes else 0.0
    
    return mean_duration, mean_jerk

def extract_dragging_features(patient_folder, game_keyword):
    """
    Locates the files for a dragging game, extracts Macro summaries, 
    and computes advanced continuous kinematics.
    """
    patient_id = os.path.basename(patient_folder)
    
    # --- 1. FILE DISCOVERY ---
    summary_files = glob.glob(os.path.join(patient_folder, f'*{game_keyword}*.mp4.csv'))
    synced_files = glob.glob(os.path.join(patient_folder, f'*{game_keyword}*synced_timeseries.csv'))
    
    if not summary_files or not synced_files:
        return None

    df_summary = pd.read_csv(summary_files[0])
    df_synced = pd.read_csv(synced_files[0])
    
    features = {'Patient_ID': patient_id, 'Game': game_keyword}

    # --- 2. EXTRACT MACRO FEATURES (Summary CSV) ---
    df_summary['Metric'] = df_summary['Metric'].astype(str).str.strip()
    
    try:
        # Core universally relevant metrics
        features['Game_Duration_sec'] = float(df_summary.loc[df_summary['Metric'] == 'Game Duration (sec)', 'Value'].values[0])
        features['First_Reaction_Time_sec'] = float(df_summary.loc[df_summary['Metric'] == 'First reaction time (sec)', 'Value'].values[0])
        
        flight_dur = float(df_summary.loc[df_summary['Metric'] == 'Flight Duration (sec)', 'Value'].values[0])
        touch_dur = float(df_summary.loc[df_summary['Metric'] == 'Touch Duration (sec)', 'Value'].values[0])
        features['Flight_Duration_sec'] = flight_dur
        features['Touch_Duration_sec'] = touch_dur
        features['Flight_Touch_Ratio'] = flight_dur / touch_dur if touch_dur > 0 else flight_dur
        
        # Pressure Distributions
        features['Pressure_Low_Pct'] = float(df_summary.loc[df_summary['Metric'] == 'Low pressure', 'Value'].values[0])
        features['Pressure_Med_Pct'] = float(df_summary.loc[df_summary['Metric'] == 'Medium pressure', 'Value'].values[0])
        features['Pressure_High_Pct'] = float(df_summary.loc[df_summary['Metric'] == 'High pressure', 'Value'].values[0])
        
    except IndexError:
        print(f"❌ Error: Missing macro metric in summary for {patient_id} ({game_keyword}).")
        return None

    # --- 3. EXTRACT MICRO FEATURES (Synced Path Kinematics) ---
    try:
        mean_dur, mean_jerk = get_kinematics_and_duration(df_synced)
        
        if game_keyword == 'PinchIt':
            features['Mean_Action_Duration_sec'] = mean_dur
        elif game_keyword == 'SlideIt':
            features['Mean_Action_Duration_sec'] = mean_dur
        elif game_keyword == 'DragIt':
            features['Mean_Action_Duration_sec'] = mean_dur
            
        features['Spatial_Jerk_Magnitude'] = mean_jerk
        
    except Exception as e:
        print(f"❌ Error computing spatial kinematics for {patient_id} ({game_keyword}): {e}")
        return None

    return features


if __name__ == "__main__":
    PROCESSED_DIR = "gaze_analysis/data/processed/"
    
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    patient_folders = [f.path for f in os.scandir(PROCESSED_DIR) if f.is_dir()]
    
    TARGET_GAMES = ['PinchIt', 'SlideIt', 'DragIt']
    
    for game in TARGET_GAMES:
        print(f"\n{'='*50}")
        print(f"🚀 EXTRACTING CONTINUOUS KINEMATICS FOR: {game}")
        print(f"{'='*50}\n")
        
        game_features = []
        
        for folder in sorted(patient_folders):
            feats = extract_dragging_features(folder, game)
            if feats:
                game_features.append(feats)
                
                # Dynamic Print Output based on the specific game action
                action_name = "Pinch" if game == "PinchIt" else "Slide" if game == "SlideIt" else "Drag"
                action_dur = feats.get(f'Mean_{action_name}_Duration_sec', 0.0)
                print(f"✅ Success: {feats['Patient_ID']} | Mean {action_name}: {action_dur:.2f}s | Jerk: {feats['Spatial_Jerk_Magnitude']:.2f}")

        if game_features:
            final_df = pd.DataFrame(game_features)
            out_path = os.path.join(PROCESSED_DIR, f"{game}_ML_Features.csv")
            final_df.to_csv(out_path, index=False)
            print(f"\n💾 Saved {game} Flat ML Matrix to: {out_path}")