import numpy as np
import tensorflow as tf
from collections import Counter
import cv2
import mediapipe as mp
import os
import re
import argparse # Import argparse for command-line arguments

# --- Define a base directory to make all paths relative and portable ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Directly import the required function from your utility file ---
try:
    from badminton_utils2 import normalize_sequence
except ImportError:
    print("FATAL ERROR: badminton_utils2.py not found.")
    print("Please ensure this script is in the same directory as your utility file.")
    exit()

def get_project_config():
    """
    Sets up configuration by constructing paths relative to this script's location.
    """
    config = {}
    print("--- Configuring Paths ---")

    # --- FIX: Construct paths directly and robustly ---
    # Assumes the script is in 'IPD/Code' and data is in 'IPD/Data_Normalized'
    project_root = os.path.dirname(BASE_DIR)
    
    config['DATA_PATH'] = os.path.join(project_root, "Data_Normalized")
    config['MODEL_NAME'] = os.path.join(BASE_DIR, "badminton_shot_classifier_normalized.h5")
    config['SEQUENCE_LENGTH'] = 40 # This should match your training configuration
    config['CROP_CONFIG'] = { "top": 0.10, "bottom": 0.45, "left": 0.25, "right": 0.25 }

    print(f"  -> Data path set to: {config['DATA_PATH']}")
    print(f"  -> Model path set to: {config['MODEL_NAME']}")

    # --- Automatically generate LABEL_MAP from the data folder structure ---
    try:
        if not os.path.isdir(config['DATA_PATH']):
            raise FileNotFoundError(f"The directory '{config['DATA_PATH']}' does not exist.")
            
        shot_types = sorted([d for d in os.listdir(config['DATA_PATH']) if os.path.isdir(os.path.join(config['DATA_PATH'], d))])
        if not shot_types:
            raise FileNotFoundError(f"No shot type subdirectories found in {config['DATA_PATH']}.")
            
        config['LABEL_MAP'] = {num: label for num, label in enumerate(shot_types)}
        print(f"  -> Successfully generated LABEL_MAP: {config['LABEL_MAP']}")
        
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        print("Please ensure the 'Data_Normalized' directory is in the correct location and contains subfolders for each shot type.")
        exit()

    return config

def extract_3d_landmarks_from_video(video_path, crop_config):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    all_landmarks = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        h, w, _ = frame.shape
        start_row, end_row = int(h * crop_config["top"]), h - int(h * crop_config["bottom"])
        start_col, end_col = int(w * crop_config["left"]), w - int(w * crop_config["right"])
        frame_cropped = frame[start_row:end_row, start_col:end_col]
        if frame_cropped.size == 0: continue
        results = pose.process(cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB))
        if results.pose_world_landmarks:
            all_landmarks.append([[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark])
    cap.release()
    pose.close()
    return np.array(all_landmarks) if all_landmarks else None

def predict_shot_from_video(video_path, model, config):
    print(f"\nProcessing video: {video_path}...")
    landmarks = extract_3d_landmarks_from_video(video_path, config['CROP_CONFIG'])
    if landmarks is None or len(landmarks) < config['SEQUENCE_LENGTH']:
        return "Not enough data to classify"
    
    video_sequences = []
    for i in range(0, len(landmarks) - config['SEQUENCE_LENGTH'] + 1, 10):
        video_sequences.append(landmarks[i:i+config['SEQUENCE_LENGTH']])
    
    if not video_sequences: return "Could not generate sequences"

    normalized_sequences = np.array([normalize_sequence(seq) for seq in video_sequences])
    if normalized_sequences.size == 0: return "Could not normalize landmarks"
    
    sequences_for_model = np.array([seq.reshape(seq.shape[0], -1) for seq in normalized_sequences])
    
    all_predictions = model.predict(sequences_for_model, verbose=0)
    avg_probabilities = np.mean(all_predictions, axis=0)
    final_prediction_index = np.argmax(avg_probabilities)

    return config['LABEL_MAP'].get(final_prediction_index, "Unknown Shot")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify a badminton shot from a video file.")
    # --- FIX: Completed the argument definition ---
    parser.add_argument("video", type=str, help="Path to the video file to classify.")
    args = parser.parse_args()

    config = get_project_config()
    
    try:
        model = tf.keras.models.load_model(config['MODEL_NAME'])
        final_prediction = predict_shot_from_video(args.video, model, config)
        
        print("\n" + "="*30)
        print(f"🚀 Final Predicted Shot: {final_prediction}")
        print("="*30)
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")