import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque, Counter
import os
import argparse
import sys # Import sys for exiting gracefully

# --- Import the necessary functions from your utilities ---
try:
    from badminton_utils2 import normalize_sequence
except ImportError:
    print("FATAL ERROR: badminton_utils2.py not found.")
    print("Please ensure this script is in the same directory as your utility file.")
    sys.exit()

# --- DYNAMIC PATH CONFIGURATION ---
# Get the absolute path to the directory containing this script (e.g., /path/to/IPD/Code)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (e.g., /path/to/IPD), which is our project root
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Construct absolute paths. This is the correct way to do it.
MODEL_PATH = os.path.join(BASE_DIR, 'badminton_shot_classifier_normalized.h5')
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data_Normalized')

# --- STATIC CONFIGURATION ---
SEQUENCE_LENGTH = 40
CROP_CONFIG = { "top": 0.10, "bottom": 0.45, "left": 0.25, "right": 0.25 }

def get_label_map(data_dir):
    """Generates a map of class indices to shot names from the data directory."""
    if not os.path.isdir(data_dir):
        print(f"\nFATAL ERROR: Data directory '{data_dir}' not found.")
        print("Please ensure the 'Data_Normalized' directory is located alongside the 'Code' directory.")
        return None
    
    shot_types = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not shot_types:
        print(f"\nFATAL ERROR: No subdirectories found in '{data_dir}'. Cannot generate labels.")
        return None
        
    return {i: label for i, label in enumerate(shot_types)}

def run_realtime_inference(model, label_map, video_source):
    """Runs real-time inference using the specified video source."""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    is_webcam = (video_source == '0')
    cap = cv2.VideoCapture(int(video_source) if is_webcam else video_source)

    if not cap.isOpened():
        print(f"\nFATAL ERROR: Could not open video source: {video_source}")
        return

    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    prediction_buffer = deque(maxlen=15)
    display_prediction = "Waiting for data..."

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or camera feed.")
            break
        
        if is_webcam:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        start_row, end_row = int(h * CROP_CONFIG["top"]), h - int(h * CROP_CONFIG["bottom"])
        start_col, end_col = int(w * CROP_CONFIG["left"]), w - int(w * CROP_CONFIG["right"])
        frame_cropped = frame[start_row:end_row, start_col:end_col]

        if frame_cropped.size == 0:
            continue

        image_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame_cropped, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            frame_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            sequence_buffer.append(frame_landmarks)

            if len(sequence_buffer) == SEQUENCE_LENGTH:
                normalized_sequence = normalize_sequence(np.array(sequence_buffer))
                reshaped_sequence = normalized_sequence.reshape(1, SEQUENCE_LENGTH, -1)
                
                prediction = model.predict(reshaped_sequence, verbose=0)[0]
                prediction_buffer.append(np.argmax(prediction))

        if prediction_buffer:
            most_common_index, _ = Counter(prediction_buffer).most_common(1)[0]
            display_prediction = label_map.get(most_common_index, "Unknown")
        
        cv2.rectangle(frame, (start_col, start_row), (end_col, end_row), (0, 255, 0), 2)
        cv2.rectangle(frame, (0, 0), (450, 40), (245, 117, 16), -1)
        cv2.putText(frame, f'PREDICTION: {display_prediction}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Smashifix - Real-time Inference', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    print(display_prediction)
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run real-time badminton shot inference.")
    parser.add_argument("video", type=str, nargs='?', default='0', help="Path to the video file or '0' for webcam (default).")
    args = parser.parse_args()
    
    label_map = get_label_map(DATA_DIR)

    if not os.path.exists(MODEL_PATH):
        print(f"\nFATAL ERROR: Model file not found at '{MODEL_PATH}'.")
        sys.exit()

    if label_map:
        print(f"Loading model: {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        print(f"Model loaded. Starting inference on source: {args.video}...")
        run_realtime_inference(model, label_map, args.video)
    else:
        # get_label_map will have already printed the specific error
        print("Could not start inference.")