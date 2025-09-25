import os
import numpy as np
from tqdm import tqdm
from badminton_utils import extract_3d_landmarks_from_video
from config import CROP_CONFIG # <-- IMPORT THE CONFIG

if __name__ == "__main__":
    
    DATA_PATH = "/home/smayan/Desktop/IPD/Data" 

    print("--- MODE: Preprocessing videos to .npy landmark files ---")
    print(f"--- Using Crop Settings: {CROP_CONFIG} ---")
    
    shot_types = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    
    for shot_type in shot_types:
        shot_folder = os.path.join(DATA_PATH, shot_type)
        print(f"\nProcessing shot type: {shot_type}")
        
        video_files = [f for f in os.listdir(shot_folder) if f.lower().endswith((".mp4", ".mov", ".avi"))]
        
        for video_file in tqdm(video_files, desc=f"Extracting landmarks for {shot_type}"):
            video_file_path = os.path.join(shot_folder, video_file)
            output_file_path = os.path.join(shot_folder, f"{os.path.splitext(video_file)[0]}.npy")
            
            if os.path.exists(output_file_path):
                continue

            # Pass the imported config to the function
            landmarks = extract_3d_landmarks_from_video(video_file_path, crop_config=CROP_CONFIG)
            
            if landmarks is not None and len(landmarks) > 0:
                np.save(output_file_path, landmarks)
            else:
                print(f"Warning: No valid landmarks detected in {video_file}. Skipping.")
                
    print("\nPreprocessing complete.")
