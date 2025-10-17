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

def normalize_pose(keypoints_3d):
    """
    Normalizes a single 3D pose to a standard, person-centric coordinate system.
    The output pose will be centered, upright, and scaled.
    
    :param keypoints_3d: A numpy array of shape (num_keypoints, 3) for a single frame.
    :return: Normalized keypoints as a numpy array of the same shape.
    """
    # Keypoint indices from MediaPipe
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    # --- 1. Centering ---
    # Calculate the hip center and move it to the origin (0,0,0).
    hip_center = (keypoints_3d[LEFT_HIP] + keypoints_3d[RIGHT_HIP]) / 2.0
    centered_keypoints = keypoints_3d - hip_center

    # --- 2. Alignment ---
    # Create a new coordinate system based on the body's orientation.
    shoulder_center = (centered_keypoints[LEFT_SHOULDER] + centered_keypoints[RIGHT_SHOULDER]) / 2.0
    
    # Avoid division-by-zero if shoulders are too close
    if np.linalg.norm(shoulder_center) < 1e-6:
        return centered_keypoints # Return centered keypoints if spine length is negligible

    new_y = shoulder_center / np.linalg.norm(shoulder_center)

    right_shoulder_vec = centered_keypoints[RIGHT_SHOULDER] - centered_keypoints[LEFT_SHOULDER]
    proj_on_y = np.dot(right_shoulder_vec, new_y) * new_y
    new_x = right_shoulder_vec - proj_on_y

    # Avoid division-by-zero if shoulders are perfectly aligned with the spine
    if np.linalg.norm(new_x) < 1e-6:
         # Create an arbitrary orthogonal vector if the shoulder vector is not usable
        if abs(new_y[0]) > 0.5:
             new_x = np.cross(new_y, [0, 1, 0])
        else:
            new_x = np.cross(new_y, [1, 0, 0])
    
    new_x = new_x / np.linalg.norm(new_x)
    new_z = np.cross(new_x, new_y)

    rotation_matrix = np.array([new_x, new_y, new_z])
    aligned_keypoints = np.dot(centered_keypoints, rotation_matrix.T)

    # --- 3. Scaling ---
    spine_length = np.linalg.norm(shoulder_center)
    if spine_length > 1e-6:
        normalized_keypoints = aligned_keypoints / spine_length
    else:
        normalized_keypoints = aligned_keypoints
        
    return normalized_keypoints

def normalize_sequence(keypoints_sequence):
    """
    Applies pose normalization to an entire sequence of frames.
    
    :param keypoints_sequence: A numpy array of shape (num_frames, num_keypoints, 3).
    :return: Normalized sequence of keypoints of the same shape.
    """
    return np.array([normalize_pose(frame) for frame in keypoints_sequence])