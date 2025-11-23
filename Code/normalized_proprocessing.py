import os
import numpy as np
import cv2
import mediapipe as mp
from badminton_utils2 import normalize_sequence
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

def extract_pose_and_cnn_features(video_path, crop_config=None,
                                  min_detection_confidence=0.2, min_tracking_confidence=0.2):
    """
    Extract 3D pose landmarks (world coordinates) + projected CNN features (128D) from a video.
    Returns (fused_features, fps)
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence)

    # --- CNN feature extractor ---
    base_cnn = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    cnn_input = Input(shape=(base_cnn.output_shape[-1],))
    cnn_proj = Dense(128, activation='relu', name='cnn_projection')(cnn_input)
    cnn_projector = Model(inputs=cnn_input, outputs=cnn_proj)
    print(f"[INFO] CNN feature extractor loaded (1280→128 projection)")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fused_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- Crop frame as per config ---
        if crop_config:
            h, w, _ = frame.shape
            start_row = int(h * crop_config.get("top", 0.0))
            end_row = h - int(h * crop_config.get("bottom", 0.0))
            start_col = int(w * crop_config.get("left", 0.0))
            end_col = w - int(w * crop_config.get("right", 0.0))
            frame = frame[start_row:end_row, start_col:end_col]

        if frame.size == 0:
            continue

        # --- CNN visual features ---
        img_resized = cv2.resize(frame, (224, 224))
        cnn_in = preprocess_input(np.expand_dims(img_resized[..., ::-1], axis=0))
        cnn_raw = base_cnn.predict(cnn_in, verbose=0)[0]
        cnn_features = cnn_projector.predict(cnn_raw[np.newaxis, :], verbose=0)[0]

        # --- Pose features (3D world landmarks) ---
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        pose_features = np.zeros(33 * 3)

        if results.pose_world_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark])
            pose_features = landmarks.flatten()

        # --- Normalize pose & scale CNN ---
        pose_norm = (pose_features - np.mean(pose_features)) / (np.std(pose_features) + 1e-6)
        cnn_scaled = cnn_features / (np.linalg.norm(cnn_features) + 1e-6)

        fused_features = np.concatenate([pose_norm, cnn_scaled])
        fused_frames.append(fused_features)

    cap.release()
    pose.close()
    return np.array(fused_frames) if fused_frames else None, fps


def preprocess_and_normalize_videos(data_path, normalized_data_path,
                                    sequence_length, stride, crop_config):
    print(f"Scanning for videos in: {data_path}")
    os.makedirs(normalized_data_path, exist_ok=True)

    for shot_type in os.listdir(data_path):
        shot_dir = os.path.join(data_path, shot_type)
        if not os.path.isdir(shot_dir):
            continue
        out_dir = os.path.join(normalized_data_path, shot_type)
        os.makedirs(out_dir, exist_ok=True)

        for fname in os.listdir(shot_dir):
            if not fname.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
                continue
            path = os.path.join(shot_dir, fname)
            print(f"\nProcessing: {path}")

            fused, fps = extract_pose_and_cnn_features(path, crop_config=crop_config)
            if fused is None or len(fused) < sequence_length:
                print(f"  -> Skipped (frames: {0 if fused is None else len(fused)})")
                continue

            count = 0
            for i in range(0, len(fused) - sequence_length + 1, stride):
                window = fused[i:i + sequence_length]
                out_name = os.path.splitext(fname)[0] + f"_win_{count}_hybrid.npz"
                np.savez(os.path.join(out_dir, out_name), features=window, fps=fps)
                count += 1
            print(f"  -> Saved {count} windows to '{out_dir}'")

if __name__ == "__main__":
    DATA_PATH = "/home/smayan/Desktop/IPD/Data"
    NORMALIZED_DATA_PATH = "/home/smayan/Desktop/IPD/Data_Normalized_Hybrid"
    SEQ_LEN, STRIDE = 40, 5
    CROP_CONFIG = {"top": 0.10, "bottom": 0.45, "left": 0.25, "right": 0.25}

    print("--- Starting hybrid normalization & preprocessing ---")
    preprocess_and_normalize_videos(DATA_PATH, NORMALIZED_DATA_PATH, SEQ_LEN, STRIDE, CROP_CONFIG)
    print("\n--- Completed! ---")
