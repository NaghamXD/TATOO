import os
import glob
import numpy as np
import pandas as pd

def get_pressure_jitter(raw_events_df):
    """
    Calculates the Pressure Jitter (Tremor Index) by finding continuous blocks 
    of 'Stationary' phases that have > 2 samples.
    """
    is_stationary = raw_events_df['Phase'] == 'Stationary'
    block_ids = (~is_stationary).cumsum()
    stationary_only = raw_events_df[is_stationary]
    
    if stationary_only.empty:
        return 0.0

    blocks = stationary_only.groupby(block_ids.loc[is_stationary.index])
    valid_jitters = []
    
    for _, block in blocks:
        if len(block) > 2:
            jitter = block['Pressure'].std()
            if pd.isna(jitter):
                jitter = 0.0
            valid_jitters.append(jitter)
            
    return np.mean(valid_jitters) if valid_jitters else 0.0

def extract_tapping_features(patient_folder, game_keyword):
    """
    Locates the files for a SPECIFIC game, and extracts Macro and Micro features.
    """
    patient_id = os.path.basename(patient_folder)
    
    # --- 1. FILE DISCOVERY USING GAME KEYWORD ---
    summary_files = glob.glob(os.path.join(patient_folder, f'*{game_keyword}*.mp4.csv'))
    
    all_csvs = glob.glob(os.path.join(patient_folder, f'*{game_keyword}*.csv'))
    raw_event_files = [f for f in all_csvs if 'mp4' not in f and 'synced' not in f]
    
    if not summary_files or not raw_event_files:
        return None

    df_summary = pd.read_csv(summary_files[0])
    df_raw = pd.read_csv(raw_event_files[0])
    
    features = {'Patient_ID': patient_id, 'Game': game_keyword}

    # --- 2. EXTRACT MACRO FEATURES (Summary CSV) ---
    df_summary['Metric'] = df_summary['Metric'].astype(str).str.strip()
    
    try:
        # GAME-SPECIFIC LOGIC
        if game_keyword == 'TouchIt':
            # TouchIt relies on raw speed
            features['Number_of_Taps'] = float(df_summary.loc[df_summary['Metric'] == 'Number of taps', 'Value'].values[0])
        else:
            # CornerIt & DoubleTap rely on duration and precision
            features['Game_Duration_sec'] = float(df_summary.loc[df_summary['Metric'] == 'Game Duration (sec)', 'Value'].values[0])
            
            correct_taps = float(df_summary.loc[df_summary['Metric'] == 'Number of correct taps', 'Value'].values[0])
            outside_taps = float(df_summary.loc[df_summary['Metric'] == 'Number of taps outside target', 'Value'].values[0])
            
            # Save the raw numbers just in case we need them later
            features['Correct_Taps'] = correct_taps
            features['Outside_Taps'] = outside_taps
            
            # Calculate the Accuracy Ratio (Safe against dividing by zero)
            features['Accuracy_Ratio'] = correct_taps / outside_taps if outside_taps > 0 else correct_taps
        
        # UNIVERSAL METRICS (Apply to all 3 tapping games)
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
        
    except IndexError as e:
        print(f"❌ Error: Missing metric in summary for {patient_id} ({game_keyword}).")
        return None

    # --- 3. EXTRACT MICRO FEATURES (Raw Unity CSV) ---
    try:
        features['Pressure_Jitter_Stationary'] = get_pressure_jitter(df_raw)
        features['Mean_Pressure_Overall'] = df_raw['Pressure'].mean()
    except Exception as e:
        print(f"❌ Error parsing raw events for {patient_id} ({game_keyword}): {e}")
        return None

    return features

if __name__ == "__main__":
    PROCESSED_DIR = "gaze_analysis/data/processed/"
    patient_folders = [f.path for f in os.scandir(PROCESSED_DIR) if f.is_dir()]
    
    # You can safely put all 3 games here now!
    TARGET_GAMES = ['TouchIt', 'CornerIt', 'DoubleTapIt']
    
    for game in TARGET_GAMES:
        print(f"\n{'='*50}")
        print(f"🎯 EXTRACTING FEATURES FOR: {game}")
        print(f"{'='*50}\n")
        
        game_features = []
        
        for folder in sorted(patient_folders):
            feats = extract_tapping_features(folder, game)
            if feats:
                game_features.append(feats)
                
                # DYNAMIC PRINT STATEMENT:
                if game == 'TouchIt':
                    print(f"✅ Success: {feats['Patient_ID']} | Taps: {feats['Number_of_Taps']}")
                else:
                    print(f"✅ Success: {feats['Patient_ID']} | Duration: {feats['Game_Duration_sec']}s | Accuracy: {feats['Accuracy_Ratio']:.2f}")

        # Save a dedicated ML matrix for this specific game
        if game_features:
            final_df = pd.DataFrame(game_features)
            out_path = os.path.join(PROCESSED_DIR, f"{game}_ML_Features.csv")
            final_df.to_csv(out_path, index=False)
            print(f"\n💾 Saved {game} Flat ML Matrix to: {out_path}")