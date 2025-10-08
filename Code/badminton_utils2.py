# badminton_utils.py
import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical

def extract_3d_landmarks_from_video(video_path, crop_config=None, target_width=720, min_detection_confidence=0.2, min_tracking_confidence=0.2):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False, model_complexity=2,
        min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence
    )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 

    all_landmarks = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if crop_config:
            h, w, _ = frame.shape
            start_row, end_row = int(h * crop_config.get("top", 0.0)), h - int(h * crop_config.get("bottom", 0.0))
            start_col, end_col = int(w * crop_config.get("left", 0.0)), w - int(w * crop_config.get("right", 0.0))
            frame_cropped = frame[start_row:end_row, start_col:end_col]
        else:
            frame_cropped = frame
            
        h, w, _ = frame_cropped.shape
        if h == 0 or w == 0: continue
        
        frame_resized = cv2.resize(frame_cropped, (int(w * target_width / w), int(h * target_width / w)))
        image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            frame_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            all_landmarks.append(frame_landmarks)
            
    cap.release()
    pose.close()
    
    return np.array(all_landmarks) if all_landmarks else None, fps

def normalize_landmarks(landmarks):
    if landmarks is None or len(landmarks) == 0: return None
    hip_center = (landmarks[:, 23] + landmarks[:, 24]) / 2
    return landmarks - hip_center[:, np.newaxis, :]

def load_and_preprocess_data_for_training(data_path, sequence_length=50, stride=10):
    """
    MODIFIED: Implements a sliding window for feature extraction.
    """
    labels, sequences = [], []
    shot_types = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    label_map = {label: num for num, label in enumerate(shot_types)}

    for shot_type, shot_type_index in label_map.items():
        shot_path = os.path.join(data_path, shot_type)
        for file_name in os.listdir(shot_path):
            if not file_name.endswith(".npz"): continue

            data = np.load(os.path.join(shot_path, file_name))
            raw_landmarks = data['landmarks']

            normalized_landmarks = normalize_landmarks(raw_landmarks)
            if normalized_landmarks is None: continue
            
            frame_features = normalized_landmarks.reshape(normalized_landmarks.shape[0], -1)

            if len(frame_features) >= sequence_length:
                for i in range(0, len(frame_features) - sequence_length + 1, stride):
                    window = frame_features[i: i + sequence_length]
                    sequences.append(window)
                    labels.append(shot_type_index)

    X = np.array(sequences)
    y = to_categorical(np.array(labels)).astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42), label_map

def build_lstm_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        LSTM(64, return_sequences=True, activation='relu'),
        Dropout(0.2),
        LSTM(128, return_sequences=False, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model