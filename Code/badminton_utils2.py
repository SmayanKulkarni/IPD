import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical

def normalize_landmarks(landmarks):
    """
    Normalizes landmarks by centering them around the hip center.
    """
    if landmarks is None or len(landmarks) == 0:
        return None
    # Calculate hip center using landmarks at indices 23 (left) and 24 (right)
    hip_center = (landmarks[:, 23] + landmarks[:, 24]) / 2
    # Broadcast subtraction to normalize all landmarks
    return landmarks - hip_center[:, np.newaxis, :]

def load_and_preprocess_data_for_training(data_path):
    """
    Loads pre-windowed .npz files directly by checking for '_window_' in the filename.
    """
    labels, sequences = [], []
    shot_types = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    label_map = {label: num for num, label in enumerate(shot_types)}

    for shot_type, shot_type_index in label_map.items():
        shot_path = os.path.join(data_path, shot_type)
        for file_name in os.listdir(shot_path):
            # --- THIS IS THE MODIFIED LINE ---
            # It now checks for the "_window_" naming convention
            if not file_name.endswith(".npz") or "_window_" not in file_name:
                continue
            # --- END OF CHANGE ---

            data = np.load(os.path.join(shot_path, file_name))
            raw_landmarks = data['landmarks']

            normalized_landmarks = normalize_landmarks(raw_landmarks)
            if normalized_landmarks is None:
                continue

            # Reshape from (frames, landmarks, coords) -> (frames, features)
            frame_features = normalized_landmarks.reshape(normalized_landmarks.shape[0], -1)

            sequences.append(frame_features)
            labels.append(shot_type_index)

    if not sequences:
        print("Warning: No sequences were loaded. Check if your .npz files are named correctly (e.g., 'video_name_window_0.npz').")
        return (np.array([]), np.array([]), np.array([]), np.array([])), {}


    X = np.array(sequences)
    y = to_categorical(np.array(labels)).astype(int)

    return train_test_split(X, y, test_size=0.2, random_state=42), label_map

def build_lstm_model(input_shape, num_classes):
    """
    Builds and compiles the Conv1D + LSTM model.
    """
    model = Sequential([
        Conv1D(filters=128, kernel_size=4, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=3),
        Dropout(0.2),
        LSTM(128, return_sequences=True, activation='relu'),
        Dropout(0.5),
        LSTM(64, return_sequences=False, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model    