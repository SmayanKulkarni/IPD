# analyze_shot.py
import numpy as np
import os
import tensorflow as tf
from badminton_utils import (
    extract_3d_landmarks_from_video,
    normalize_landmarks,
    extract_comprehensive_features,
    calculate_ksi
)
from config import CROP_CONFIG

def analyze_user_shot(video_path, model, templates, label_map):
    print(f"\n--- Analyzing video: {video_path} ---")
    
    filename = os.path.basename(video_path)
    filename_without_ext = os.path.splitext(filename)[0]
    active_crop_config = CROP_CONFIG if filename_without_ext.isdigit() else None
    
    raw_landmarks, fps = extract_3d_landmarks_from_video(
        video_path,
        crop_config=active_crop_config,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2
    )
    if raw_landmarks is None:
        print("Could not detect any landmarks in the video.")
        return

    user_coords_seq = normalize_landmarks(raw_landmarks)
    
    # --- MODIFIED: Time window changed to 1.5 seconds ---
    SECONDS_TO_CONSIDER = 1.5
    FRAMES_TO_IGNORE_AT_END = 10
    
    # --- END OF CHANGE ---
    
    frames_for_duration = int(SECONDS_TO_CONSIDER * fps)
    total_frames = len(user_coords_seq)
    end_frame = total_frames - FRAMES_TO_IGNORE_AT_END
    start_frame = max(0, end_frame - frames_for_duration)
    sliced_coords_for_features = user_coords_seq[start_frame:end_frame]

    frame_features = [extract_comprehensive_features(frame) for frame in sliced_coords_for_features]
    user_feature_seq = np.array(frame_features)

    seq_len, num_features = model.input_shape[1], model.input_shape[2]
    if len(user_feature_seq) > seq_len:
        user_feature_seq = user_feature_seq[-seq_len:]
    else:
        padding = np.zeros((seq_len - len(user_feature_seq), num_features))
        user_feature_seq = np.concatenate([padding, user_feature_seq])
    
    model_input = np.expand_dims(user_feature_seq, axis=0)
    
    prediction = model.predict(model_input)
    predicted_class_index = np.argmax(prediction)
    predicted_shot_label = label_map[predicted_class_index]
    confidence = np.max(prediction)
    
    print(f"\n[CLASSIFICATION] Detected Shot: '{predicted_shot_label}' (Confidence: {confidence:.2f})")

    if predicted_shot_label in templates:
        expert_template = templates[predicted_shot_label]
        
        print("\n[CORRECTION] Comparing your form to the ideal motion template...")
        R_SHOULDER, R_ELBOW, R_WRIST = 12, 14, 16
        user_arm_seq = user_coords_seq[:, [R_SHOULDER, R_ELBOW, R_WRIST], :]
        template_arm_seq = expert_template[:, [R_SHOULDER, R_ELBOW, R_WRIST], :]
        
        ksi_results = calculate_ksi(template_arm_seq, user_arm_seq)
        
        print("\n--- ✅ Analysis Results ---")
        print(f"Overall KSI Score: {ksi_results['ksi_total']:.2f}")
        # (The rest of the feedback logic remains the same)
        # ...
    else:
        print(f"Could not find a motion template for '{predicted_shot_label}'. Cannot provide feedback.")

if __name__ == "__main__":
    DATA_PATH = "/home/smayan/Desktop/IPD/Data"
    MODEL_PATH = "badminton_shot_classifier_v3_sliced.h5"
    TEMPLATES_PATH = "motion_templates"
    USER_VIDEO_TO_ANALYZE = "/home/smayan/Desktop/IPD/Data/backhand_drive/backhand_drive (1).mp4"

    print("--- Loading assets for Smart Coach ---")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    shot_types = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    label_map = {i: label for i, label in enumerate(shot_types)}
    
    templates = {}
    for label in shot_types:
        template_file = os.path.join(TEMPLATES_PATH, f"{label}_template.npy")
        if os.path.exists(template_file):
            templates[label] = np.load(template_file)
    print("--- Assets loaded successfully ---")
    
    analyze_user_shot(USER_VIDEO_TO_ANALYZE, model, templates, label_map)