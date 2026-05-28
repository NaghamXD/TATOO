import os
import shutil
import re

def organize_participant_data(source_dir):
    print(f"\n{'='*50}")
    print(f"📂 INITIATING PARTICIPANT DATA SORTING")
    print(f"{'='*50}")

    # Regex Pattern Explanation:
    # 1. Looks for "_VID_eyedata_"
    # 2. (.+?) captures the participant name (e.g., 'salma', 'janan 11')
    # 3. _(\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}) finds the exact start of the date/time string
    pattern = re.compile(r"_VID_eyedata_(.+?)_(\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})")

    if not os.path.exists(source_dir):
        print(f"Error: The directory '{source_dir}' does not exist.")
        print("Please ensure this script is in the same folder as 'totl_results', or provide the full path.")
        return

    moved_count = 0
    skipped_count = 0
    created_folders = set()

    # Iterate through all files in the directory
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)

        # Only process files, ignore existing directories
        if os.path.isfile(file_path):
            match = pattern.search(filename)
            
            if match:
                # Extract the raw participant identifier
                participant_name = match.group(1).strip()
                
                # --- OPTIONAL: Name Cleaning ---
                # If you want 'janan 11', 'janan51', and 'janan1' to ALL go into a single 'janan' folder,
                # uncomment the line below. Otherwise, it safely creates separate folders for each ID.
                # participant_name = ''.join([i for i in participant_name if not i.isdigit()]).strip()
                
                # Create the participant's folder path
                participant_folder = os.path.join(source_dir, participant_name)
                
                # Create the folder if we haven't already
                if not os.path.exists(participant_folder):
                    os.makedirs(participant_folder)
                    created_folders.add(participant_name)
                
                # Move the file into the participant's folder
                dest_path = os.path.join(participant_folder, filename)
                shutil.move(file_path, dest_path)
                
                moved_count += 1
                print(f"Moved: {filename[:30]}... -> {participant_name}/")
            else:
                # If a file doesn't match the expected naming convention, skip it safely
                skipped_count += 1
                print(f"Skipped (Unrecognized format): {filename}")

    print(f"\n{'='*50}")
    print("✅ SORTING COMPLETE")
    print(f"Folders Created: {len(created_folders)} {list(created_folders)}")
    print(f"Files Organized: {moved_count}")
    print(f"Files Skipped:   {skipped_count}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    # Update this string if your folder is located elsewhere
    TARGET_DIRECTORY = "/Users/naghamdaood/Projects/TATOO_Second_Version/total_results" 
    organize_participant_data(TARGET_DIRECTORY)