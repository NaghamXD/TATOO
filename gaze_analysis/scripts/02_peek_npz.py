import numpy as np
import os

def peek_inside_npz(file_path):
    print(f"\n{'='*50}")
    print(f"🔍 PEEKING INSIDE: {os.path.basename(file_path)}")
    print(f"{'='*50}")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return

    # Open the NPZ file
    with np.load(file_path) as data:
        arrays = data.files
        print(f"Total arrays found: {len(arrays)}\n")
        
        # Loop through each array and print its stats and first 5 rows
        for i, array_name in enumerate(arrays):
            arr = data[array_name]
            
            # Add context based on what you told me about the structure
            if i == 0: context = "(Expected: Vision X Smoothed cm)"
            elif i == 1: context = "(Expected: Vision Y Smoothed cm)"
            elif i == 2: context = "(Expected: Touch X Raw 30Hz)"
            elif i == 3: context = "(Expected: Touch Y Raw 30Hz)"
            else: context = "(Unknown)"

            print(f"--- Array {i+1}: '{array_name}' {context} ---")
            print(f"Shape: {arr.shape} | Type: {arr.dtype}")
            
            # Print the first 5 rows safely
            num_rows_to_print = min(5, arr.shape[0])
            print(f"First {num_rows_to_print} rows [Timestamp, Value]:")
            for row in arr[:num_rows_to_print]:
                print(f"  {row}")
            print("\n")

    print(f"{'='*50}\n")

if __name__ == "__main__":
    # --- UPDATE THIS PATH TO ONE OF YOUR NEW NPZ FILES ---
    TEST_FILE = "/Users/naghamdaood/Projects/TATOO/gaze_analysis/data/raw/janan/CornerIt_VID_eyedata_janan_02-02-26-10-45-09_2026-02-02-10-45-09-398.mp4_res.mp4multiple_arrays.npz" 
    
    peek_inside_npz(TEST_FILE)