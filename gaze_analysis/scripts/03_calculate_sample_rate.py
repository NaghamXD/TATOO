import numpy as np
import os

def analyze_sample_rates(npz_file_path):
    print(f"\n{'='*50}")
    print(f"⏱️ TIMELINE & SAMPLE RATE ANALYSIS")
    print(f"File: {os.path.basename(npz_file_path)}")
    print(f"{'='*50}")
    
    with np.load(npz_file_path) as data:
        keys = data.files
        gaze_timestamps = data[keys[0]][:, 0]
        touch_timestamps = data[keys[2]][:, 0]

    # --- 1. Determine the Time Unit ---
    # If the timestamps are in milliseconds, a 7-second game will end around 7000.
    # If they are in seconds, it will end around 7.0.
    total_time_raw = gaze_timestamps[-1] - gaze_timestamps[0]
    
    if total_time_raw > 1000:
        print("🕒 Time Unit Detected: Milliseconds (ms)")
        unit_converter = 1000.0 # To convert to seconds for Hz calculation
    else:
        print("🕒 Time Unit Detected: Seconds (s)")
        unit_converter = 1.0

    total_duration_sec = total_time_raw / unit_converter
    print(f"Total Recording Duration: {total_duration_sec:.2f} seconds\n")

    # --- 2. Calculate Gaze Sample Rate (Hz) ---
    # np.diff calculates the time difference between consecutive frames
    gaze_diffs = np.diff(gaze_timestamps) / unit_converter 
    
    gaze_median_dt = np.median(gaze_diffs)
    gaze_hz = 1.0 / gaze_median_dt
    
    print("👁️ GAZE DATA (CAMERA)")
    print(f"  - Total Data Points: {len(gaze_timestamps)}")
    print(f"  - Median time between frames: {gaze_median_dt:.4f} seconds")
    print(f"  - Calculated Sample Rate: {gaze_hz:.1f} Hz (Frames per second)")
    print(f"  - Max Lag Spike (longest gap): {np.max(gaze_diffs):.4f} seconds")

    # --- 3. Calculate Touch Sample Rate (Hz) ---
    touch_diffs = np.diff(touch_timestamps) / unit_converter
    
    touch_median_dt = np.median(touch_diffs)
    touch_hz = 1.0 / touch_median_dt
    
    print("\n👆 TOUCH DATA (TABLET SCREEN)")
    print(f"  - Total Data Points: {len(touch_timestamps)}")
    print(f"  - Median time between frames: {touch_median_dt:.4f} seconds")
    print(f"  - Calculated Sample Rate: {touch_hz:.1f} Hz (Frames per second)")
    print(f"  - Max gap between touches: {np.max(touch_diffs):.4f} seconds")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    # Update this to a valid path from your raw data folder
    TEST_FILE = "/Users/naghamdaood/Projects/TATOO_Second_Version/total_results/janan23/TouchIt_VID_eyedata_janan23_14-01-26-11-57-32_2026-01-14-11-57-32-332.mp4_res.mp4multiple_arrays.npz" 
    
    if os.path.exists(TEST_FILE):
        analyze_sample_rates(TEST_FILE)
    else:
        print(f"Update the TEST_FILE path! Could not find: {TEST_FILE}")