import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import os

# --- Import from your utility file ---
try:
    from badminton_utils2 import normalize_landmarks
except ImportError:
    print("FATAL ERROR: badminton_utils2.py not found.")
    exit()

def get_project_config():
    """
    Reads all necessary configuration from your project files and folder structure.
    """
    DATA_PATH = "/home/smayan/Desktop/IPD/Data"
    # !! This value MUST match the sequence length used for training your model !!
    SEQUENCE_LENGTH = 40 
    MODEL_NAME = "2_bigger_window_reduced_data_badminton_shot_classifier_v5.h5"
    CROP_CONFIG = { "top": 0.10, "bottom": 0.45, "left": 0.25, "right": 0.25 }

    try:
        shot_types = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
        label_map = {num: label for num, label in enumerate(shot_types)}
        print(f"Successfully generated Label Map: {label_map}")
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not find data path '{DATA_PATH}'. {e}")
        exit()
        
    return MODEL_NAME, SEQUENCE_LENGTH, label_map, CROP_CONFIG


def run_realtime_inference(model, label_map, sequence_length, crop_config):
    """
    Runs real-time inference with settings matched to the batch script for accuracy.
    """
    PREDICTION_INTERVAL = 10 
    
    mp_pose = mp.solutions.pose
    # --- CHANGED: Matched model_complexity to the batch script (2) ---
    pose = mp_pose.Pose(
        static_image_mode=False, 
        model_complexity=2, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture("/home/smayan/Desktop/IPD/Data/forehand_net_shot/052.mp4")

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    sequence_buffer = deque(maxlen=sequence_length)
    predictions_buffer = deque(maxlen=15) 
    display_prediction = "Waiting for sequence..."
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            print("End of video.")
            break

        # --- REMOVED: Do not flip the frame, it mismatches the training data ---
        # frame = cv2.flip(frame, 1) 

        h, w, _ = frame.shape
        start_row, end_row = int(h * crop_config["top"]), h - int(h * crop_config["bottom"])
        start_col, end_col = int(w * crop_config["left"]), w - int(w * crop_config["right"])
        frame_cropped = frame[start_row:end_row, start_col:end_col]
        if frame_cropped.size == 0: continue

        image_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_world_landmarks:
            mp_drawing.draw_landmarks(frame_cropped, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            sequence_buffer.append(results.pose_world_landmarks.landmark)
            
            print(f"Buffer size: {len(sequence_buffer)}/{sequence_length}", end='\r')

            if len(sequence_buffer) == sequence_length and frame_counter % PREDICTION_INTERVAL == 0:
                landmark_array = np.array([[[lm.x, lm.y, lm.z] for lm in frame_lms] for frame_lms in sequence_buffer])
                normalized_landmarks = normalize_landmarks(landmark_array)
                
                if normalized_landmarks is not None:
                    frame_features = normalized_landmarks.reshape(sequence_length, -1)
                    input_data = np.expand_dims(frame_features, axis=0)
                    probabilities = model.predict(input_data, verbose=0)[0]
                    predictions_buffer.append(probabilities)
        
        if predictions_buffer:
            avg_probabilities = np.mean(np.array(predictions_buffer), axis=0)
            final_prediction_index = np.argmax(avg_probabilities)
            display_prediction = label_map.get(final_prediction_index, "Unknown")
        
        frame_counter += 1
        
        cv2.rectangle(frame, (start_col, start_row), (end_col, end_row), (0, 255, 0), 2)
        cv2.rectangle(frame, (0, 0), (450, 40), (245, 117, 16), -1)
        cv2.putText(frame, f'PREDICTION: {display_prediction}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Real-time Badminton Prediction', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'): break

    print("\nDone.")
    print(display_prediction)
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == '__main__':
    MODEL_PATH, SEQUENCE_LENGTH, LABEL_MAP, CROP_CONFIG = get_project_config()
    
    print(f"\nUsing SEQUENCE_LENGTH: {SEQUENCE_LENGTH}")
    print(f"Loading model: {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        run_realtime_inference(model, LABEL_MAP, SEQUENCE_LENGTH, CROP_CONFIG)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")