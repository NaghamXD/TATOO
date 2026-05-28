import os
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
# Define the age that separates "Young" from "Old" in your specific study
AGE_THRESHOLD = 60  

PROCESSED_DIR = "gaze_analysis/data/processed/"
DEMO_FILE = os.path.join(PROCESSED_DIR, "demographicsv2.csv")

def prepare_ground_truth():
    """Loads demographics and cleans the target columns for Machine Learning."""
    if not os.path.exists(DEMO_FILE):
        print("❌ Error: demographics.csv not found in the processed folder!")
        return None
        
    df_demo = pd.read_csv(DEMO_FILE)
    
    # Clean column names (strip extra spaces)
    df_demo.columns = df_demo.columns.str.strip()
    
    # 1. Create Binary Age Label (0 = Young, 1 = Old)
    # Ensure Age is numeric, forcing errors to NaN
    df_demo['Age'] = pd.to_numeric(df_demo['Age'], errors='coerce') 
    df_demo['Target_Age_Old'] = (df_demo['Age'] >= AGE_THRESHOLD).astype(int)
    
    # 2. Create Binary Visual Impairment Label (0 = No, 1 = Yes)
    vis_col = 'Do you suffer from a visual impairment?'
    if vis_col in df_demo.columns:
        # Convert Yes/No strings into 1s and 0s
        df_demo['Target_Visual_Impaired'] = df_demo[vis_col].astype(str).str.lower().map({'yes': 1, 'no': 0})
    else:
        print(f"⚠️ Warning: Could not find column '{vis_col}'")
        df_demo['Target_Visual_Impaired'] = np.nan

    # Keep only the ID and our new clean ML targets
    # Assuming the column linking to patient folders is called 'ID' or 'Code'
    link_col = 'Code' if 'Code' in df_demo.columns else 'ID'
    df_labels = df_demo[[link_col, 'Age', 'Target_Age_Old', 'Target_Visual_Impaired']].copy()
    
    # Standardize the ID column name to match our feature matrices
    df_labels.rename(columns={link_col: 'Patient_ID'}, inplace=True)
    # Strip accidental spaces from IDs
    df_labels['Patient_ID'] = df_labels['Patient_ID'].astype(str).str.strip()
    
    return df_labels


if __name__ == "__main__":
    TARGET_GAMES = ['TouchIt', 'CornerIt', 'DoubleTapIt', 'PinchIt', 'SlideIt', 'DragIt']
    
    df_labels = prepare_ground_truth()
    
    if df_labels is not None:
        for game in TARGET_GAMES:
            print(f"\n{'='*50}")
            print(f"🧬 MERGING MASTER DATASET FOR: {game}")
            
            # File paths
            touch_file = os.path.join(PROCESSED_DIR, f"{game}_ML_Features.csv")
            gaze_file = os.path.join(PROCESSED_DIR, f"{game}_Gaze_Features.csv")
            sync_file = os.path.join(PROCESSED_DIR, f"{game}_Sync_Features.csv")
            
            # Load files if they exist
            df_touch = pd.read_csv(touch_file) if os.path.exists(touch_file) else pd.DataFrame()
            df_gaze = pd.read_csv(gaze_file) if os.path.exists(gaze_file) else pd.DataFrame()
            df_sync = pd.read_csv(sync_file) if os.path.exists(sync_file) else pd.DataFrame()
            
            if df_touch.empty:
                print(f"⏭️ Skipping {game} (Touch features not found).")
                continue
                
            # 1. Merge the 3 feature sets together on Patient_ID
            df_master = df_touch.copy()
            
            if not df_gaze.empty:
                # Drop 'Game' column before merge to avoid duplicates like Game_x, Game_y
                df_gaze = df_gaze.drop(columns=['Game'], errors='ignore')
                df_master = pd.merge(df_master, df_gaze, on='Patient_ID', how='outer')
                
            if not df_sync.empty:
                df_sync = df_sync.drop(columns=['Game'], errors='ignore')
                df_master = pd.merge(df_master, df_sync, on='Patient_ID', how='outer')
                
            # 2. Attach the Demographic Ground Truth Labels
            df_master['Patient_ID'] = df_master['Patient_ID'].astype(str).str.strip()
            df_final = pd.merge(df_master, df_labels, on='Patient_ID', how='left')
            
            # Save the final Master Matrix
            out_path = os.path.join(PROCESSED_DIR, f"MASTER_{game}.csv")
            df_final.to_csv(out_path, index=False)
            
            print(f"✅ Generated {df_final.shape[1]} total features for {df_final.shape[0]} patients.")
            print(f"💾 Saved to: MASTER_{game}.csv")