import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Input
from collections import deque, Counter
import os
import argparse
import sys

# --- Import normalization function ---
try:
    from badminton_utils2 import normalize_sequence
except ImportError:
    print("FATAL ERROR: badminton_utils2.py not found.")
    print("Please ensure this file is in the same directory.")
    sys.exit()

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(BASE_DIR, 'badminton_shot_classifier_hybrid.h5')
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data_Normalized_Hybrid')

# --- STATIC CONFIGS ---
SEQUENCE_LENGTH = 40
CROP_CONFIG = {"top": 0.10, "bottom": 0.45, "left": 0.25, "right": 0.25}


def get_label_map(data_dir):
    """Generates label map based on folder structure."""
    if not os.path.isdir(data_dir):
        print(f"[FATAL] Data directory not found: {data_dir}")
        return None

    shot_types = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not shot_types:
        print(f"[FATAL] No subdirectories found in {data_dir}")
        return None

    return {i: label for i, label in enumerate(shot_types)}


def run_realtime_inference(model, label_map, video_source):
    """Runs hybrid CNN + Pose real-time inference."""

    # --- MediaPipe Pose ---
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # --- CNN Feature Extractor + 128D Projection ---
    base_cnn = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    cnn_input = Input(shape=(base_cnn.output_shape[-1],))
    cnn_proj = Dense(128, activation='relu', name='cnn_projection')(cnn_input)
    cnn_projector = Model(inputs=cnn_input, outputs=cnn_proj)
    print(f"[INFO] CNN extractor ready (1280→128 projection)")

    # --- Initialize video ---
    is_webcam = (video_source == '0')
    cap = cv2.VideoCapture(int(video_source) if is_webcam else video_source)
    if not cap.isOpened():
        print(f"[FATAL] Could not open video source: {video_source}")
        return

    # --- Buffers for sequence and prediction smoothing ---
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    prediction_buffer = deque(maxlen=15)
    display_prediction = "Waiting..."

    print("[INFO] Starting real-time hybrid inference... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video or stream.")
            break

        if is_webcam:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        # --- Adaptive crop (skip if vertical video) ---
        if h > w:
            frame_cropped = frame.copy()
            crop_applied = False
        else:
            start_row = int(h * CROP_CONFIG["top"])
            end_row = h - int(h * CROP_CONFIG["bottom"])
            start_col = int(w * CROP_CONFIG["left"])
            end_col = w - int(w * CROP_CONFIG["right"])
            frame_cropped = frame[start_row:end_row, start_col:end_col]
            crop_applied = True

        if frame_cropped.size == 0:
            continue

        # --- Pose Detection ---
        image_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # --- Draw Landmarks on correct frame ---
        if results.pose_landmarks:
            if crop_applied:
                mp_drawing.draw_landmarks(frame_cropped, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            else:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # --- Extract hybrid features ---
        if results.pose_world_landmarks:
            # CNN features
            cnn_in = cv2.resize(frame_cropped, (224, 224))
            cnn_in = preprocess_input(np.expand_dims(cnn_in[..., ::-1], axis=0))
            cnn_raw = base_cnn.predict(cnn_in, verbose=0)[0]
            cnn_feat = cnn_projector.predict(cnn_raw[np.newaxis, :], verbose=0)[0]
            cnn_feat = cnn_feat / (np.linalg.norm(cnn_feat) + 1e-6)

            # Pose features
            pose_arr = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark])
            pose_flat = pose_arr.flatten()
            pose_norm = (pose_flat - np.mean(pose_flat)) / (np.std(pose_flat) + 1e-6)

            fused_features = np.concatenate([pose_norm, cnn_feat])
            sequence_buffer.append(fused_features)

            # --- Predict once buffer full ---
            if len(sequence_buffer) == SEQUENCE_LENGTH:
                seq = np.array(sequence_buffer)
                prediction = model.predict(seq[np.newaxis, ...], verbose=0)[0]
                prediction_buffer.append(np.argmax(prediction))

        # --- Stability smoothing ---
        if prediction_buffer:
            most_common_idx, _ = Counter(prediction_buffer).most_common(1)[0]
            display_prediction = label_map.get(most_common_idx, "Unknown")

        # --- Draw UI ---
        cv2.rectangle(frame, (0, 0), (500, 45), (245, 117, 16), -1)
        cv2.putText(frame, f'PREDICTION: {display_prediction}', (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        print(f"[DEBUG] Current Prediction: {display_prediction}")

        # Draw crop box only for landscape
        if crop_applied:
            cv2.rectangle(frame, (start_col, start_row), (end_col, end_row), (0, 255, 0), 2)

        cv2.imshow('Badminton Shot Classifier (Hybrid CNN+Pose)', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    print("[INFO] Inference finished.")
    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run real-time badminton shot inference.")
    parser.add_argument("video", type=str, nargs='?', default='0', help="Path to video file or '0' for webcam.")
    args = parser.parse_args()

    label_map = get_label_map(DATA_DIR)
    if not label_map:
        sys.exit()

    if not os.path.exists(MODEL_PATH):
        print(f"[FATAL] Model file not found at '{MODEL_PATH}'.")
        sys.exit()

    print(f"[INFO] Loading model from '{MODEL_PATH}'...")
    model = load_model(MODEL_PATH)
    print(f"[INFO] Model loaded successfully.")
    run_realtime_inference(model, label_map, args.video)
