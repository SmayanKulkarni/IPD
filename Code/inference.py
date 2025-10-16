import numpy as np
import tensorflow as tf
from collections import Counter
import cv2
import mediapipe as mp
import os
import re

# --- Directly import the required function from your utility file ---
try:
    from badminton_utils2 import normalize_landmarks
except ImportError:
    print("FATAL ERROR: badminton_utils.py not found.")
    print("Please ensure this script is in the same directory as your utility file.")
    exit()

def get_project_config():
    """
    Reads configuration from project files with robust error checking.
    """
    config = {}
    print("--- Automatically Configuring from Project Files ---")

    def find_variable(file_path, var_name, pattern):
        """Helper to find a variable in a file and handle errors."""
        try:
            with open(file_path, "r") as f:
                content = f.read()
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    return match.group(1)
                else:
                    print(f"  -> WARNING: Could not find '{var_name}' in {file_path}.")
                    return None
        except FileNotFoundError:
            print(f"FATAL ERROR: The file '{file_path}' was not found.")
            exit()

    # --- Read from proprocessing.py ---
    proc_file = "proprocessing.py"
    data_path_str = find_variable(proc_file, 'DATA_PATH', r"DATA_PATH\s*=\s*[\"'](.*?)[\"']")
    seq_len_str = find_variable(proc_file, 'SEQUENCE_LENGTH', r"SEQUENCE_LENGTH\s*=\s*(\d+)")
    crop_config_str = find_variable(proc_file, 'CROP_CONFIG', r"CROP_CONFIG\s*=\s*(\{.*?\})")

    # --- Read from train.py ---
    train_file = "train.py"
    model_name_str = find_variable(train_file, 'MODEL_NAME', r"MODEL_NAME\s*=\s*[\"'](.*?)[\"']")
    
    # --- Validate and build config dictionary ---
    if not all([data_path_str, seq_len_str, crop_config_str, model_name_str]):
        print("\nFATAL ERROR: One or more configuration variables could not be found. Please check your files.")
        exit()

    config['DATA_PATH'] = data_path_str
    config['SEQUENCE_LENGTH'] = int(seq_len_str)
    config['MODEL_NAME'] = "/home/smayan/Desktop/IPD/Code/2_bigger_window_reduced_data_badminton_shot_classifier_v5.h5"
    try:
        config['CROP_CONFIG'] = eval(crop_config_str) # Safely evaluate the dict string
    except:
        print(f"FATAL ERROR: Could not parse CROP_CONFIG dictionary: {crop_config_str}")
        exit()

    print(f"  -> Config loaded successfully.")

    # --- Automatically generate LABEL_MAP from data folder structure ---
    try:
        shot_types = sorted([d for d in os.listdir(config['DATA_PATH']) if os.path.isdir(os.path.join(config['DATA_PATH'], d))])
        if not shot_types:
            raise FileNotFoundError(f"No subdirectories found in {config['DATA_PATH']}.")
        config['LABEL_MAP'] = {num: label for num, label in enumerate(shot_types)}
        print(f"  -> Successfully generated LABEL_MAP: {config['LABEL_MAP']}")
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}")
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
    
    normalized_landmarks = normalize_landmarks(landmarks)
    if normalized_landmarks is None: return "Could not normalize landmarks"
    
    video_sequences = []
    for i in range(0, len(normalized_landmarks) - config['SEQUENCE_LENGTH'] + 1, 10):
        video_sequences.append(normalized_landmarks[i:i+config['SEQUENCE_LENGTH']])
    
    if not video_sequences: return "Could not generate sequences"
    
    sequences_for_model = np.array([seq.reshape(seq.shape[0], -1) for seq in video_sequences])
    
    # Average the prediction probabilities across all windows
    all_predictions = model.predict(sequences_for_model, verbose=0)
    avg_probabilities = np.mean(all_predictions, axis=0)
    final_prediction_index = np.argmax(avg_probabilities)

    return config['LABEL_MAP'].get(final_prediction_index, "Unknown Shot")

if __name__ == '__main__':
    VIDEO_TO_CLASSIFY = "/home/smayan/Desktop/IPD/Data/forehand_net_shot/052.mp4"
    
    config = get_project_config()
    
    try:
        model = tf.keras.models.load_model(config['MODEL_NAME'])
        final_prediction = predict_shot_from_video(VIDEO_TO_CLASSIFY, model, config)
        
        print("\n" + "="*30)
        print(f"🚀 Final Predicted Shot: {final_prediction}")
        print("="*30)
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")