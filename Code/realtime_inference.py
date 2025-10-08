import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import os

# --- Directly import from your utility file ---
# This ensures the normalization logic is identical to your training pipeline.
try:
    from badminton_utils import normalize_landmarks
except ImportError:
    print("FATAL ERROR: badminton_utils.py not found.")
    print("Please ensure this script is in the same directory as your utility file.")
    exit()

def get_project_config():
    """
    Reads all necessary configuration from your project files and folder structure.
    """
    # --- Configuration from: smayankulkarni/.../Code/train.py ---
    DATA_PATH = "/home/smayan/Desktop/IPD/Data"
    SEQUENCE_LENGTH = 500
    MODEL_NAME = "badminton_shot_classifier_v3_sliced.h5"

    # --- CROP CONFIGURATION ---
    # This should match the configuration used for preprocessing your training data.
    # It defines the percentage of the frame to crop from each side.
    CROP_CONFIG = {
    "top": 0.10, "bottom": 0.45, "left": 0.25, "right": 0.25
}

    # --- Automatically generate LABEL_MAP from data folder ---
    try:
        shot_types = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
        if not shot_types:
            raise FileNotFoundError(f"No subdirectories found in {DATA_PATH}.")
        label_map = {num: label for num, label in enumerate(shot_types)}
        print(f"Successfully generated Label Map: {label_map}")
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not find data path '{DATA_PATH}'. {e}")
        exit()
        
    return MODEL_NAME, SEQUENCE_LENGTH, label_map, CROP_CONFIG


def run_realtime_inference(model, label_map, sequence_length, crop_config):
    """
    Runs a real-time inference loop with cropping and functions from badminton_utils.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.2, min_tracking_confidence=0.2, model_complexity=1)
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture("/home/smayan/Desktop/IPD/Data/backhand_net_shot/017.mp4")

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    sequence_buffer = deque(maxlen=sequence_length)
    current_prediction = "Waiting..."

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: continue

        frame = cv2.flip(frame, 1)

        # --- APPLY CROPPING LOGIC from badminton_utils.py ---
        h, w, _ = frame.shape
        start_row = int(h * crop_config["top"])
        end_row = h - int(h * crop_config["bottom"])
        start_col = int(w * crop_config["left"])
        end_col = w - int(w * crop_config["right"])
        frame_cropped = frame[start_row:end_row, start_col:end_col]
        # --- END CROPPING ---

        if frame_cropped.size == 0: continue

        image_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_world_landmarks:
            mp_drawing.draw_landmarks(frame_cropped, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            sequence_buffer.append(results.pose_world_landmarks.landmark)

            if len(sequence_buffer) == sequence_length:
                landmark_array = np.array([[[lm.x, lm.y, lm.z] for lm in frame_lms] for frame_lms in sequence_buffer])
                
                # USE YOUR FUNCTION for normalization
                normalized_landmarks = normalize_landmarks(landmark_array)
                
                if normalized_landmarks is not None:
                    frame_features = normalized_landmarks.reshape(sequence_length, -1)
                    input_data = np.expand_dims(frame_features, axis=0)
                    prediction = model.predict(input_data, verbose=0)
                    current_prediction = label_map.get(np.argmax(prediction), "Unknown")
        
        # Draw visualization on the original frame
        cv2.rectangle(frame, (start_col, start_row), (end_col, end_row), (0, 255, 0), 2)
        cv2.rectangle(frame, (0, 0), (320, 40), (245, 117, 16), -1)
        cv2.putText(frame, f'PREDICTION: {current_prediction}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Real-time Badminton Prediction', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == '__main__':
    MODEL_PATH, SEQUENCE_LENGTH, LABEL_MAP, CROP_CONFIG = get_project_config()
    
    print(f"\nLoading model: {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        run_realtime_inference(model, LABEL_MAP, SEQUENCE_LENGTH, CROP_CONFIG)
    except (FileNotFoundError, IOError):
        print(f"FATAL ERROR: Model file not found at '{MODEL_PATH}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")