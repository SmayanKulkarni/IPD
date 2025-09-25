import numpy as np
from badminton_utils import extract_3d_landmarks_from_video, normalize_landmarks, calculate_ksi
from config import CROP_CONFIG # <-- IMPORT THE CONFIG

if __name__ == "__main__":
    
    EXPERT_VIDEO_PATH = "expert_forehand_drive.mp4"
    USER_VIDEO_PATH = "user_forehand_drive.mp4"

    print("--- MODE: Analyzing a user's shot against an expert ---")
    print(f"--- Using Crop Settings: {CROP_CONFIG} ---")
    
    print(f"Loading expert landmarks from {EXPERT_VIDEO_PATH}...")
    expert_landmarks_raw = extract_3d_landmarks_from_video(EXPERT_VIDEO_PATH, crop_config=CROP_CONFIG)
    
    print(f"Loading user landmarks from {USER_VIDEO_PATH}...")
    user_landmarks_raw = extract_3d_landmarks_from_video(USER_VIDEO_PATH, crop_config=CROP_CONFIG)

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
        feedback_provided = False
        if ksi_results['pose_similarity'] < 0.80:
            print("- Elbow Movement is inaccurate: Your arm's shape and angle differ from the expert's.")
            feedback_provided = True
        if ksi_results['velocity_coherence'] < 0.70:
            print("- Swing Timing is off: Your swing's speed variation doesn't match the expert.")
            feedback_provided = True
        if ksi_results['acceleration_profile'] < 0.65:
            print("- Power Generation needs work: You're not accelerating the racket as explosively.")
            feedback_provided = True
        if not feedback_provided:
            print("Great form! Your motion is very similar to the expert's.")
    print("\nAnalysis complete.")