import os
import glob

def audit_tatoo_data(base_data_folder):
    """
    Scans patient folders to ensure both CSV and NPZ files are present.
    Flags any missing files or empty folders.
    """
    print(f"--- Starting Data Audit in: {base_data_folder} ---\n")
    
    # Get all subdirectories (patient folders like 'janan-13', 'salma-7')
    if not os.path.exists(base_data_folder):
        print(f"Error: The folder '{base_data_folder}' does not exist.")
        return

    patient_folders = [f.path for f in os.scandir(base_data_folder) if f.is_dir()]
    
    if not patient_folders:
        print("No patient folders found.")
        return

    issues_found = 0

    for folder in sorted(patient_folders):
        folder_name = os.path.basename(folder)
        
        # Count the specific file types in this patient's folder
        csv_files = glob.glob(os.path.join(folder, '*.csv'))
        npz_files = glob.glob(os.path.join(folder, '*.npz'))
        
        # Check for anomalies
        if len(csv_files) == 0 and len(npz_files) == 0:
            print(f"⚠️  [EMPTY FOLDER] {folder_name}: No data found at all.")
            issues_found += 1
            
        elif len(csv_files) > 0 and len(npz_files) == 0:
            print(f"❌ [MISSING NPZ]  {folder_name}: Found {len(csv_files)} CSV(s), but missing raw .npz gaze data.")
            issues_found += 1
            
        elif len(npz_files) > 0 and len(csv_files) == 0:
            print(f"❌ [MISSING CSV]  {folder_name}: Found {len(npz_files)} NPZ(s), but missing .csv reports.")
            issues_found += 1
            
        # Optional: Add a strict check if you know exactly how many games should be played
        # elif len(csv_files) < 6:
        #     print(f"⚠️  [INCOMPLETE]   {folder_name}: Only found {len(csv_files)} CSVs. Expected 6 games.")
        #     issues_found += 1

    print("\n--- Audit Complete ---")
    if issues_found == 0:
        print("✅ All patient folders have both CSV and NPZ data. You are good to process!")
    else:
        print(f"Total issues flagged: {issues_found}. Please review the missing data above before running ML models.")

if __name__ == "__main__":
    # Change this path to wherever you are storing the new incoming patient folders
    TARGET_DIRECTORY = "../data/raw/" 
    audit_tatoo_data(TARGET_DIRECTORY)