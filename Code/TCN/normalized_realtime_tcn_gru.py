import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Input
from collections import deque, Counter
import os, sys

# --- Import from upper dir ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from badminton_utils2 import normalize_sequence

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = "/home/smayan/Desktop/IPD"
MODEL_PATH = os.path.join(BASE_DIR, "badminton_shot_classifier_tcn_gru_dual_temporal_pose.h5")
DATA_DIR = os.path.join(PROJECT_ROOT, "Data_Normalized_Hybrid")
VIDEO_PATH = "/home/smayan/Desktop/IPD/Test/lift2 (online-video-cutter.com).mp4"

SEQ_LENGTH = 40
CNN_FEATURE_DIM = 128
CROP_CONFIG = {"top": 0.10, "bottom": 0.45, "left": 0.25, "right": 0.25}

def get_label_map(data_dir):
    if not os.path.isdir(data_dir): return None
    shot_types = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    return {i: label for i, label in enumerate(shot_types)} if shot_types else None

def run_realtime_inference(model, label_map):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.3, min_tracking_confidence=0.3
    )
    mp_drawing = mp.solutions.drawing_utils

    base_cnn = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
    cnn_input = Input(shape=(base_cnn.output_shape[-1],))
    cnn_proj = Dense(CNN_FEATURE_DIM, activation="relu", name="cnn_projection")(cnn_input)
    cnn_projector = Model(inputs=cnn_input, outputs=cnn_proj)
    print("[INFO] CNN extractor ready (1280→128 projection)")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[FATAL] Could not open video source: {VIDEO_PATH}")
        return

    cnn_seq_buffer = deque(maxlen=SEQ_LENGTH)
    pose_seq_buffer = deque(maxlen=SEQ_LENGTH)
    prediction_buffer = deque(maxlen=15)
    display_prediction = "Waiting..."

    print("[INFO] Starting dual-temporal inference (CNN TCN + Pose GRU)... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        h, w, _ = frame.shape
        if h > w:
            frame_cropped, crop_applied = frame.copy(), False
        else:
            start_row = int(h*CROP_CONFIG["top"]); end_row = h - int(h*CROP_CONFIG["bottom"])
            start_col = int(w*CROP_CONFIG["left"]); end_col = w - int(w*CROP_CONFIG["right"])
            frame_cropped = frame[start_row:end_row, start_col:end_col]; crop_applied = True
        if frame_cropped.size == 0: continue

        image_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            target = frame_cropped if crop_applied else frame
            mp_drawing.draw_landmarks(target, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if results.pose_world_landmarks:
            # CNN features
            cnn_in = cv2.resize(frame_cropped, (224, 224))
            cnn_in = preprocess_input(np.expand_dims(cnn_in[..., ::-1], axis=0))
            cnn_raw = base_cnn.predict(cnn_in, verbose=0)[0]
            cnn_feat = cnn_projector.predict(cnn_raw[np.newaxis, :], verbose=0)[0]
            cnn_feat = cnn_feat / (np.linalg.norm(cnn_feat) + 1e-6)
            cnn_seq_buffer.append(cnn_feat)

            # Pose features
            pose_arr = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark])
            pose_flat = pose_arr.flatten()
            pose_norm = (pose_flat - np.mean(pose_flat)) / (np.std(pose_flat) + 1e-6)
            pose_seq_buffer.append(pose_norm)

            # Predict when full sequence available
            if len(cnn_seq_buffer) == SEQ_LENGTH:
                cnn_seq = np.array(cnn_seq_buffer).reshape(1, SEQ_LENGTH, CNN_FEATURE_DIM)
                pose_seq = np.array(pose_seq_buffer).reshape(1, SEQ_LENGTH, -1)
                pred = model.predict([cnn_seq, pose_seq], verbose=0)[0]
                prediction_buffer.append(np.argmax(pred))

        if prediction_buffer:
            idx, _ = Counter(prediction_buffer).most_common(1)[0]
            display_prediction = label_map.get(idx, "Unknown")

        cv2.rectangle(frame, (0, 0), (500, 45), (245, 117, 16), -1)
        cv2.putText(frame, f'PREDICTION: {display_prediction}', (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if crop_applied:
            cv2.rectangle(frame, (start_col, start_row), (end_col, end_row), (0, 255, 0), 2)
        print(display_prediction)
        cv2.imshow("Badminton Shot Classifier (Dual-Temporal)", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    print("[INFO] Inference finished.")
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    label_map = get_label_map(DATA_DIR)
    if not label_map: sys.exit("[FATAL] Could not build label map.")
    if not os.path.exists(MODEL_PATH): sys.exit(f"[FATAL] Model not found: {MODEL_PATH}")
    print(f"[INFO] Loading model from '{MODEL_PATH}'...")
    model = load_model(MODEL_PATH)
    print("[INFO] Model loaded successfully.")
    run_realtime_inference(model, label_map)
