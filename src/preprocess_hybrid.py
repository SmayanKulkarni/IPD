"""
Hybrid Feature Preprocessing Pipeline
======================================

Streaming video processor for extracting fused pose+CNN features from raw
badminton footage. Combines MediaPipe pose landmarks with MobileNetV2 visual
embeddings for hybrid classification.

Key Features:
    - Dual-feature extraction: 3D pose (99D) + CNN visual (128D)
    - Pose-guided ROI cropping for CNN input
    - Raw landmark preservation for KSI evaluation
    - Memory-efficient streaming processing
    - Sliding window segmentation with stride
    - Temporal smoothing via bounding box tracking

Processing Pipeline:
    1. Load video and determine segment bounds
    2. For each frame in segment:
       a. Apply crop configuration
       b. Extract 3D pose via MediaPipe
       c. Compute pose-guided ROI bounding box
       d. Extract CNN features via MobileNetV2
       e. Fuse pose + CNN features
       f. Store raw landmarks for KSI
    3. Save windows with features and landmarks
    4. Cleanup resources

Output Format:
    .npz files with:
    - 'features': (T, 99+CNN_DIM) fused pose+CNN features
    - 'raw_landmarks': (T, 33, 3) normalized pose for KSI
    - 'fps': Original video frame rate

Dependencies:
    External: cv2, numpy, tensorflow, yaml, tqdm
    Internal: features.HybridFeatureExtractor, utils.normalize_pose

Configuration (params.yaml):
    hybrid_pipeline:
        data_path: Output directory for processed features
        cnn_feature_dim: CNN embedding dimension (default: 128)
        cnn_input_size: CNN input resolution (default: 224)
        sequence_length: Frames per window
        stride: Sliding window step size
        crop_config: Frame cropping parameters
        cnn_roi: Pose-guided ROI configuration
    mediapipe: MediaPipe Pose configuration

Usage:
    python preprocess_hybrid.py

Author: IPD Research Team
Version: 1.0.0
"""

import os
import yaml
import cv2
import numpy as np
import gc
from tqdm import tqdm
from collections import deque
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from features import HybridFeatureExtractor
from utils import normalize_pose, should_skip_crop, get_segment_bounds, resolve_crop_config_for_video


def process_video_streaming(
    video_path,
    output_dir,
    extractor,
    seq_len,
    stride,
    crop_config,
    segment_rules=None,
    roi_cfg=None,
):
    """Stream video -> fused features -> sliding windows saved to disk.
    
    Saves both:
    - 'features': engineered features (pose + CNN) for model training
    - 'raw_landmarks': raw (T, 33, 3) pose landmarks for KSI evaluation
    """
    filename = os.path.basename(video_path)
    file_id = os.path.splitext(filename)[0]

    if os.path.exists(os.path.join(output_dir, f"{file_id}_win_0.npz")):
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    start_frame, segment_frames = get_segment_bounds(
        video_path,
        fps,
        total_frames,
        default_seconds=1.75,
        segment_cfg=segment_rules,
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    skip_crop = should_skip_crop(filename)
    zeros_pose = np.zeros(99, dtype=np.float32)
    zeros_landmarks = np.zeros((33, 3), dtype=np.float32)
    last_pose = None
    last_landmarks = None
    last_box = None

    window_buffer = deque(maxlen=seq_len)
    landmarks_buffer = deque(maxlen=seq_len)
    saved_count = 0
    frame_idx = 0

    try:
        while frame_idx < int(segment_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if skip_crop:
                frame_cropped = frame
            else:
                h, w = frame.shape[:2]
                start_row = int(h * crop_config['top'])
                end_row = h - int(h * crop_config['bottom'])
                start_col = int(w * crop_config['left'])
                end_col = w - int(w * crop_config['right'])
                frame_cropped = frame[start_row:end_row, start_col:end_col]
                if frame_cropped.size == 0:
                    continue

            res = extractor.pose.process(cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lm = np.array([[l.x, l.y, l.z] for l in res.pose_landmarks.landmark], dtype=np.float32)
                pose_flat = normalize_pose(lm).astype(np.float32).flatten()
                last_pose = pose_flat
                last_landmarks = lm.copy()
            else:
                pose_flat = last_pose if last_pose is not None else zeros_pose
                lm = last_landmarks if last_landmarks is not None else zeros_landmarks

            h2, w2 = frame_cropped.shape[:2]
            box = extractor._compute_pose_roi_box(
                res.pose_landmarks if hasattr(res, 'pose_landmarks') else None,
                w2,
                h2,
                roi_cfg,
                last_box=last_box,
            )
            last_box = box if box is not None else last_box
            roi_frame = extractor._crop_with_box(frame_cropped, box)

            img_size = getattr(extractor, 'cnn_input_size', 224)
            img = cv2.resize(roi_frame, (img_size, img_size))
            img = preprocess_input(np.expand_dims(img[..., ::-1], axis=0))
            cnn_feat = extractor.rgb_model.predict(img, verbose=0)[0].astype(np.float32)

            fused = np.concatenate([pose_flat, cnn_feat], axis=0)
            window_buffer.append(fused)
            landmarks_buffer.append(lm)  # NEW: append raw landmarks

            # Save windows on fixed stride relative to the segment start
            if len(window_buffer) == seq_len and ((frame_idx - seq_len) % stride == 0):
                save_path = os.path.join(output_dir, f"{file_id}_win_{saved_count}.npz")
                # NEW: save both features and raw_landmarks
                np.savez(
                    save_path, 
                    features=np.asarray(window_buffer, dtype=np.float32),
                    raw_landmarks=np.asarray(landmarks_buffer, dtype=np.float32),
                    fps=float(fps)
                )
                saved_count += 1

            del frame
            del frame_cropped
            del img
            del fused

    finally:
        cap.release()
        del window_buffer
        del landmarks_buffer  # NEW: cleanup
        gc.collect()


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    raw_dir = params['base']['raw_data_path']
    cfg = params['hybrid_pipeline']
    mp_cfg = params['mediapipe']
    segment_rules = params.get('segment_rules', {})
    crop_overrides = params.get('crop_overrides', {})
    out_dir = cfg['data_path']

    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(raw_dir):
        return

    extractor = HybridFeatureExtractor(
        mp_cfg,
        cnn_dim=cfg['cnn_feature_dim'],
        cnn_input_size=cfg.get('cnn_input_size', 224),
    )
    try:
        for cls in os.listdir(raw_dir):
            cls_in = os.path.join(raw_dir, cls)
            if not os.path.isdir(cls_in):
                continue

            cls_out = os.path.join(out_dir, cls)
            os.makedirs(cls_out, exist_ok=True)

            videos = [v for v in os.listdir(cls_in) if v.lower().endswith(('.mp4', '.avi', '.mov', '.webm'))]
            for vid in tqdm(videos, desc=f"Hybrid Prep {cls}"):
                video_path = os.path.join(cls_in, vid)
                crop_cfg = resolve_crop_config_for_video(video_path, cfg['crop_config'], crop_overrides)
                process_video_streaming(
                    video_path,
                    cls_out,
                    extractor,
                    cfg['sequence_length'],
                    cfg['stride'],
                    crop_cfg,
                    segment_rules,
                    roi_cfg=cfg.get('cnn_roi'),
                )
    finally:
        try:
            extractor.pose.close()
        except Exception:
            pass
        del extractor
        gc.collect()


if __name__ == "__main__":
    main()