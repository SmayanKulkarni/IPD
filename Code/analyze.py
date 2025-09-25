import numpy as np
import os
from badminton_utils import extract_3d_landmarks_from_video, normalize_landmarks, calculate_ksi
from config import CROP_CONFIG

if __name__ == "__main__":
    
    EXPERT_VIDEO_PATH = "/home/smayan/Desktop/IPD/Data/forehand_clear/004.mp4"
    USER_VIDEO_PATH = "/home/smayan/Desktop/IPD/Data/backhand_drive/backhand_drive (1).mp4"
    
    print("--- MODE: Analyzing a user's shot against an expert ---")
    
    expert_filename = os.path.basename(EXPERT_VIDEO_PATH)
    expert_filename_without_ext = os.path.splitext(expert_filename)[0]
    expert_crop_config = CROP_CONFIG if expert_filename_without_ext.isdigit() else None
    print(f"Analyzing EXPERT video '{expert_filename}' with {'cropping' if expert_crop_config else 'full frame'}.")
    
    expert_landmarks_raw = extract_3d_landmarks_from_video(
        EXPERT_VIDEO_PATH,
        crop_config=expert_crop_config,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2
    )
    
    user_filename = os.path.basename(USER_VIDEO_PATH)
    user_filename_without_ext = os.path.splitext(user_filename)[0]
    user_crop_config = CROP_CONFIG if user_filename_without_ext.isdigit() else None
    print(f"Analyzing USER video '{user_filename}' with {'cropping' if user_crop_config else 'full frame'}.")

    user_landmarks_raw = extract_3d_landmarks_from_video(
        USER_VIDEO_PATH,
        crop_config=user_crop_config,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2
    )

    if expert_landmarks_raw is None or user_landmarks_raw is None:
        print("Could not detect poses in one or both videos. Analysis aborted.")
    else:
        print("Normalizing landmarks...")
        expert_landmarks = normalize_landmarks(expert_landmarks_raw)
        user_landmarks = normalize_landmarks(user_landmarks_raw)

        RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST = 12, 14, 16
        expert_arm_seq = expert_landmarks[:, [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST], :]
        user_arm_seq = user_landmarks[:, [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST], :]

        print("\nCalculating Kinetic Similarity Index (KSI)...")
        ksi_results = calculate_ksi(expert_arm_seq, user_arm_seq)
        
        print("\n--- ✅ Analysis Results ---")
        print(f"Overall KSI Score: {ksi_results['ksi_total']:.2f}")
        print(f"  - Postural Similarity: {ksi_results['pose_similarity']:.2f}")
        print(f"  - Velocity Coherence:  {ksi_results['velocity_coherence']:.2f}")
        print(f"  - Acceleration Profile: {ksi_results['acceleration_profile']:.2f}")
        
        print("\n--- 💡 Corrective Feedback ---")
        if ksi_results['pose_similarity'] < 0.80:
            print("- Elbow Movement is inaccurate: Your arm's shape and angle through the swing differ from the expert's.")
        if ksi_results['velocity_coherence'] < 0.70:
            print("- Swing Timing is off: Your swing's speed variation doesn't match the expert.")
        if ksi_results['acceleration_profile'] < 0.65:
            print("- Power Generation needs work: You're not accelerating the racket as explosively into the shuttle.")

