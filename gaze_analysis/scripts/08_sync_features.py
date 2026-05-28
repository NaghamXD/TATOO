import os
import glob
import numpy as np
import pandas as pd

FPS = 30.0
DT = 1.0 / FPS

def calculate_continuous_sync_metrics(df_valid):
    """
    Calculates Cross-Correlation Latency and Directional Alignment for dragging games.
    Processes data in continuous 'strokes' to avoid teleportation bugs during flight phases.
    """
    # Find contiguous blocks of time where the finger never left the screen
    valid_indices = df_valid.index.tolist()
    if not valid_indices:
        return np.nan, np.nan
        
    strokes = np.split(valid_indices, np.where(np.diff(valid_indices) != 1)[0] + 1)
    
    stroke_latencies = []
    stroke_alignments = []
    
    for stroke in strokes:
        if len(stroke) < 35: 
            continue # Skip tiny accidental taps (less than 1.1 seconds)
            
        stroke_df = df_valid.loc[stroke].copy()
        
        # 1. Calculate Velocities (cm/sec) WITHIN this specific stroke
        gaze_dx = stroke_df['gaze_x_cm'].diff().fillna(0)
        gaze_dy = stroke_df['gaze_y_cm'].diff().fillna(0)
        touch_dx = stroke_df['touch_x_cm'].diff().fillna(0)
        touch_dy = stroke_df['touch_y_cm'].diff().fillna(0)
        
        gaze_speed = np.sqrt(gaze_dx**2 + gaze_dy**2)
        touch_speed = np.sqrt(touch_dx**2 + touch_dy**2)
        
        # --- DIRECTIONAL VELOCITY CORRELATION ---
        dot_product = (gaze_dx * touch_dx) + (gaze_dy * touch_dy)
        magnitude_product = gaze_speed * touch_speed
        
        valid_vectors = magnitude_product > 0
        cos_theta = dot_product[valid_vectors] / magnitude_product[valid_vectors]
        
        if not cos_theta.empty:
            stroke_alignments.append(cos_theta.mean())
        
        # --- TRUE EYE-HAND LATENCY (Cross-Correlation) ---
        best_lag_frames = 0
        max_corr = -1.0
        
        if touch_speed.std() > 0 and gaze_speed.std() > 0:
            for shift in range(-15, 16):
                shifted_gaze = gaze_speed.shift(shift)
                corr = touch_speed.corr(shifted_gaze) 
                
                if pd.notna(corr) and corr > max_corr:
                    max_corr = corr
                    best_lag_frames = shift
                    
        stroke_latencies.append(best_lag_frames * DT)

    # Average the metrics across all valid balloons/drags the patient performed
    final_latency = np.mean(stroke_latencies) if stroke_latencies else np.nan
    final_alignment = np.mean(stroke_alignments) if stroke_alignments else np.nan

    return final_latency, final_alignment

def extract_sync_features(patient_folder, game_keyword):
    patient_id = os.path.basename(patient_folder)
    
    synced_files = glob.glob(os.path.join(patient_folder, f'*{game_keyword}*synced_timeseries.csv'))
    if not synced_files:
        return None

    df_synced = pd.read_csv(synced_files[0])
    features = {'Patient_ID': patient_id, 'Game': game_keyword}

    # 1. Dynamically find columns and convert to true cm
    touch_x_col = 'touch_x_raw' if 'touch_x_raw' in df_synced.columns else 'touch_x_cm'
    touch_y_col = 'touch_y_raw' if 'touch_y_raw' in df_synced.columns else 'touch_y_cm'
    
    df_valid = df_synced.dropna(subset=['gaze_x_cm', 'gaze_y_cm', touch_x_col, touch_y_col]).copy()
    
    if df_valid.empty:
        return None
        
    df_valid['gaze_x_cm'] = df_valid['gaze_x_cm'] * 0.01
    df_valid['gaze_y_cm'] = df_valid['gaze_y_cm'] * 0.01
    df_valid['touch_x_cm'] = df_valid[touch_x_col] * 0.01
    df_valid['touch_y_cm'] = df_valid[touch_y_col] * 0.01

    # --- CRITICAL MULTI-TOUCH FIX ---
    # Merge the alternating Thumb and Index coordinates into a single 'Pinch Centroid'
    if game_keyword == 'PinchIt':
        df_valid['touch_x_cm'] = df_valid['touch_x_cm'].rolling(window=2, min_periods=1).mean()
        df_valid['touch_y_cm'] = df_valid['touch_y_cm'].rolling(window=2, min_periods=1).mean()
        
    # --- ROUTING BASED ON GAME TYPE ---
    # --- ROUTING BASED ON GAME TYPE ---
    tapping_games = ['TouchIt', 'CornerIt', 'DoubleTapIt']
    dragging_games = ['PinchIt', 'SlideIt', 'DragIt']
    
    # ALL GAMES: Get Euclidean Spatial Error
    dx = df_valid['gaze_x_cm'] - df_valid['touch_x_cm']
    dy = df_valid['gaze_y_cm'] - df_valid['touch_y_cm']
    spatial_error_cm = np.sqrt(dx**2 + dy**2)
    features['Mean_Eye_Hand_Distance_cm'] = spatial_error_cm.mean()
    
    # ONLY DRAGGING GAMES: Get Latency and Directional Alignment
    if game_keyword in dragging_games:
        latency, dir_align = calculate_continuous_sync_metrics(df_valid)
        features['True_Eye_Hand_Latency_sec'] = latency
        features['Directional_Velocity_Alignment'] = dir_align

    return features


if __name__ == "__main__":
    PROCESSED_DIR = "gaze_analysis/data/processed/"
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    patient_folders = [f.path for f in os.scandir(PROCESSED_DIR) if f.is_dir()]
    TARGET_GAMES = ['TouchIt', 'CornerIt', 'DoubleTapIt', 'PinchIt', 'SlideIt', 'DragIt']
    
    for game in TARGET_GAMES:
        print(f"\n{'='*50}")
        print(f"🔗 EXTRACTING SYNC (VMI) FEATURES FOR: {game}")
        print(f"{'='*50}\n")
        
        game_features = []
        for folder in sorted(patient_folders):
            feats = extract_sync_features(folder, game)
            if feats:
                game_features.append(feats)
                
                # Dynamic terminal output
                mean_dist = feats.get('Mean_Eye_Hand_Distance_cm', 0)
                if game in ['PinchIt', 'SlideIt', 'DragIt']:
                    latency = feats.get('True_Eye_Hand_Latency_sec', 0)
                    align = feats.get('Directional_Velocity_Alignment', 0)
                    print(f"✅ {feats['Patient_ID']} | Gap: {mean_dist:.2f}cm | Lag: {latency:.2f}s | Align: {align:.2f}")
                else:
                    print(f"✅ {feats['Patient_ID']} | Mean Spatial Gap: {mean_dist:.2f}cm")

        if game_features:
            final_df = pd.DataFrame(game_features)
            out_path = os.path.join(PROCESSED_DIR, f"{game}_Sync_Features.csv")
            final_df.to_csv(out_path, index=False)