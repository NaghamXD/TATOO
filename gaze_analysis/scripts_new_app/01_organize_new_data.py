"""
01_organize_new_data.py — New App
Organizes flat raw files in raw_new_app/ into per-participant subfolders.

New filename format: {participant}_{timestamp}_{Game}.mp4_res.mp4
Old format was:      {Game}_VID_eyedata_{participant}_{timestamp}.mp4_res.mp4
"""
import os
import re
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_new_app import RAW_DIR

# New pattern: participant_timestamp_GameName.mp4...
PATTERN = re.compile(r"^(.+?)_(\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_(\w+)\.mp4")

def organize_participant_data(source_dir):
    print(f"\n{'='*55}")
    print(f"INITIATING PARTICIPANT DATA SORTING (New App)")
    print(f"Source: {source_dir}")
    print(f"{'='*55}")

    source_dir = str(source_dir)
    if not os.path.exists(source_dir):
        print(f"Error: '{source_dir}' does not exist.")
        return

    moved_count   = 0
    skipped_count = 0
    created_folders = set()

    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        if not os.path.isfile(file_path):
            continue

        match = PATTERN.match(filename)
        if match:
            participant_name = match.group(1).strip()

            participant_folder = os.path.join(source_dir, participant_name)
            if not os.path.exists(participant_folder):
                os.makedirs(participant_folder)
                created_folders.add(participant_name)

            dest_path = os.path.join(participant_folder, filename)
            shutil.move(file_path, dest_path)
            moved_count += 1
            print(f"  Moved: {filename[:50]} -> {participant_name}/")
        else:
            skipped_count += 1
            print(f"  Skipped (unrecognized format): {filename}")

    print(f"\n{'='*55}")
    print(f"SORTING COMPLETE")
    print(f"  Folders created : {len(created_folders)} {sorted(created_folders)}")
    print(f"  Files organized : {moved_count}")
    print(f"  Files skipped   : {skipped_count}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    organize_participant_data(RAW_DIR)
