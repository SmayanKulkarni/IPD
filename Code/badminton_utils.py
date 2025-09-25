import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ==============================================================================
# PART 1: DATA EXTRACTION AND PREPARATION
# ==============================================================================

def extract_3d_landmarks_from_video(video_path, crop_config=None, target_width=720):
    """
    Crops a video frame based on a config dict and extracts 3D pose landmarks.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
        
    all_landmarks = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- 4-SIDED CROP LOGIC ---
        if crop_config:
            h, w, _ = frame.shape
            start_row = int(h * crop_config.get("top", 0.0))
            end_row = h - int(h * crop_config.get("bottom", 0.0))
            start_col = int(w * crop_config.get("left", 0.0))
            end_col = w - int(w * crop_config.get("right", 0.0))
            frame_cropped = frame[start_row:end_row, start_col:end_col]
        else:
            frame_cropped = frame
        # --- END CROP ---

        h, w, _ = frame_cropped.shape
        if h == 0 or w == 0: continue
        
        scale = target_width / w
        new_h, new_w = int(h * scale), int(w * scale)
        frame_resized = cv2.resize(frame_cropped, (new_w, new_h))
            
        image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks and results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            frame_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            all_landmarks.append(frame_landmarks)
            
    cap.release()
    pose.close()
    
    return np.array(all_landmarks) if all_landmarks else None

def normalize_landmarks(landmarks):
    if landmarks is None or len(landmarks) == 0: return None
    LEFT_HIP, RIGHT_HIP = 23, 24
    hip_center = (landmarks[:, LEFT_HIP] + landmarks[:, RIGHT_HIP]) / 2
    return landmarks - hip_center[:, np.newaxis, :]

# ==============================================================================
# PART 2: KSI & DTW
# ==============================================================================
def dynamic_time_warping(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            last_min = min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
            dtw_matrix[i, j] = cost + last_min
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        i, j = min((i-1, j), (i, j-1), (i-1, j-1), key=lambda x: dtw_matrix[x[0], x[1]])
    path.reverse()
    seq1_aligned = np.array([seq1[i] for i, j in path])
    seq2_aligned = np.array([seq2[j] for i, j in path])
    return seq1_aligned, seq2_aligned

def calculate_ksi(expert_seq, user_seq, weights={'pose': 0.4, 'velocity': 0.4, 'acceleration': 0.2}, alpha=0.1, beta=0.1):
    expert_aligned, user_aligned = dynamic_time_warping(expert_seq, user_seq)
    def cosine_similarity(v1, v2):
        epsilon = 1e-8
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + epsilon)
    v_upper_arm_exp = expert_aligned[:, 1] - expert_aligned[:, 0]
    v_forearm_exp = expert_aligned[:, 2] - expert_aligned[:, 1]
    v_upper_arm_user = user_aligned[:, 1] - user_aligned[:, 0]
    v_forearm_user = user_aligned[:, 2] - user_aligned[:, 1]
    pose_sims = [(cosine_similarity(v_upper_arm_exp[i], v_upper_arm_user[i]) + cosine_similarity(v_forearm_exp[i], v_forearm_user[i])) / 2 for i in range(len(v_upper_arm_exp))]
    s_pose = np.mean(pose_sims)
    wrist_exp, wrist_user = expert_aligned[:, 2], user_aligned[:, 2]
    vel_exp = np.diff(wrist_exp, axis=0, prepend=wrist_exp[0:1])
    vel_user = np.diff(wrist_user, axis=0, prepend=wrist_user[0:1])
    vel_sims = [cosine_similarity(vel_exp[i], vel_user[i]) * np.exp(-alpha * (np.linalg.norm(vel_exp[i]) - np.linalg.norm(vel_user[i]))**2) for i in range(len(vel_exp))]
    s_velocity = np.mean(vel_sims)
    accel_exp = np.diff(vel_exp, axis=0, prepend=vel_exp[0:1])
    accel_user = np.diff(vel_user, axis=0, prepend=vel_user[0:1])
    accel_sims = [np.exp(-beta * (np.linalg.norm(accel_exp[i]) - np.linalg.norm(accel_user[i]))**2) for i in range(len(accel_exp))]
    s_acceleration = np.mean(accel_sims)
    ksi_score = (weights['pose'] * s_pose + weights['velocity'] * s_velocity + weights['acceleration'] * s_acceleration)
    return {'ksi_total': ksi_score, 'pose_similarity': s_pose, 'velocity_coherence': s_velocity, 'acceleration_profile': s_acceleration}

# ==============================================================================
# PART 3: DEEP LEARNING MODEL
# ==============================================================================

def load_and_preprocess_data_for_training(data_path, sequence_length=50):
    labels, sequences = [], []
    shot_types = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    label_map = {label: num for num, label in enumerate(shot_types)}
    for shot_type, shot_type_index in label_map.items():
        shot_path = os.path.join(data_path, shot_type)
        for file_name in os.listdir(shot_path):
            if not file_name.endswith(".npy"): continue
            raw_landmarks = np.load(os.path.join(shot_path, file_name))
            normalized_landmarks = normalize_landmarks(raw_landmarks)
            if normalized_landmarks is None: continue
            if len(normalized_landmarks) > sequence_length:
                normalized_landmarks = normalized_landmarks[:sequence_length]
            else:
                padding = np.zeros((sequence_length - len(normalized_landmarks), 33, 3))
                normalized_landmarks = np.concatenate([normalized_landmarks, padding])
            sequences.append(normalized_landmarks)
            labels.append(shot_type_index)
    X = np.array(sequences)
    y = to_categorical(np.array(labels)).astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42), label_map

def build_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        LSTM(128, return_sequences=False, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
