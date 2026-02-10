"""
Feature Extraction for Pose and Hybrid Pipelines
=================================================

Provides feature extraction classes for converting raw video frames into
model-ready representations. Supports both pure pose and hybrid (pose+CNN)
feature extraction strategies.

Classes:
    1. HybridFeatureExtractor
       - Dual-stream feature extraction: 3D pose + CNN visual
       - MobileNetV2 backbone with L2-normalized embeddings
       - Pose-guided ROI cropping for focused visual features
       - Temporal smoothing of bounding box for stability
       
    2. PoseFeatureExtractor
       - Pure MediaPipe pose landmark extraction
       - Geometric normalization to person-centric coordinates
       - Suitable for pose-only classification pipeline

Pose-Guided ROI Algorithm:
    1. Detect full-body joints (shoulders → ankles)
    2. Filter by visibility threshold (default: 0.5)
    3. Compute bounding box with margin expansion
    4. Apply temporal smoothing with previous frame's box
    5. Fallback to full frame if insufficient joints detected

CNN Architecture:
    - Base: MobileNetV2 (ImageNet pretrained)
    - Pooling: Global average
    - Projection: Dense(cnn_dim) + ReLU
    - Normalization: L2 on output embedding

Dependencies:
    External: cv2, numpy, mediapipe, tensorflow
    Internal: utils.normalize_pose

Configuration:
    mp_config: MediaPipe Pose configuration
    cnn_dim: Output dimension for CNN features (default: 128)
    cnn_input_size: CNN input resolution (default: 224)
    roi_cfg: Pose-guided ROI configuration

Author: IPD Research Team
Version: 1.0.0
"""

import cv2
import os
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from utils import normalize_pose, should_skip_crop
from rsn import build_rsn_feature_extractor

class HybridFeatureExtractor:
    def __init__(self, mp_config, cnn_dim=128, cnn_input_size=224, rsn_weights_path=None):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,  # Use static mode for deterministic results
            model_complexity=mp_config['model_complexity'],
            min_detection_confidence=mp_config['min_detection_confidence'],
            min_tracking_confidence=mp_config['min_tracking_confidence']
        )
        
        # CNN Setup
        self.cnn_input_size = int(cnn_input_size)
        self.cnn_dim = int(cnn_dim)
        
        # Build CNN Backbone (Residual-Shuffle Network)
        print("🔧 Initializing Residual-Shuffle Network (RSN) backbone...")
        self.rgb_model = build_rsn_feature_extractor(
            input_shape=(self.cnn_input_size, self.cnn_input_size, 3),
            feature_dim=self.cnn_dim,
            weights_path=rsn_weights_path
        )
        print(f"✅ RSN ready: output dim = {self.cnn_dim}")

        # Backwards-compatible attributes
        self.base_cnn = self.rgb_model
        self.cnn_model = self.rgb_model

    @staticmethod
    def _clamp_int(v, lo, hi):
        return int(max(lo, min(hi, v)))

    def _compute_pose_roi_box(self, pose_landmarks, width, height, roi_cfg, last_box=None):
        """Compute an ROI box (x1,y1,x2,y2) on the current (already-cropped) frame.

        Forgiving ROI logic:
          - Uses full-body joints by default (shoulders → ankles).
          - Filters by visibility threshold.
          - Requires a minimum number of visible joints; otherwise falls back.
          - Applies temporal smoothing with previous box.
          - Falls back to full frame if detection is too uncertain.
        """
        if not roi_cfg or not roi_cfg.get('enabled', False):
            return None

        # ─────────────── Config knobs ───────────────
        # Full-body default: shoulders (11,12), elbows (13,14), wrists (15,16),
        # hips (23,24), knees (25,26), ankles (27,28)
        joint_ids = roi_cfg.get('joint_ids')
        if not joint_ids:
            joint_ids = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

        visibility_thresh = float(roi_cfg.get('visibility_thresh', 0.3))
        min_joints = int(roi_cfg.get('min_joints', 4))
        margin = float(roi_cfg.get('margin', 0.30))  # larger default margin
        min_size_frac = float(roi_cfg.get('min_size_frac', 0.50))  # larger min box
        smoothing = float(roi_cfg.get('smoothing', 0.3))  # EMA towards last_box
        fallback_full_frame = roi_cfg.get('fallback_full_frame', True)
        use_last_on_missing = roi_cfg.get('use_last_box_on_missing_pose', True)

        # ─────────────── Gather valid landmarks ───────────────
        # ─────────────── TORSO-ANCHORED LOGIC ───────────────
        # Identify torso joints: Shoulders(11,12) and Hips(23,24) defined in MediaPipe
        # We use these to stabilize the center.
        torso_ids = [11, 12, 23, 24]
        
        txs, tys = [], []
        all_xs, all_ys = [], []

        for j in joint_ids:
            try:
                lm = pose_landmarks.landmark[int(j)]
            except Exception:
                continue
            if getattr(lm, 'visibility', 1.0) < visibility_thresh:
                continue
                
            px, py = lm.x * width, lm.y * height
            all_xs.append(px)
            all_ys.append(py)
            
            if j in torso_ids:
                txs.append(px)
                tys.append(py)

        # Fallback if insufficient joints
        if len(all_xs) < min_joints:
             if use_last_on_missing and last_box is not None: return last_box
             if fallback_full_frame: return (0, 0, int(width), int(height))
             return None

        # 1. Determine Box Center (Stable)
        # If we have torso joints, use them for the center. If not, use all joints.
        if len(txs) >= 2:
            cx = np.mean(txs)
            cy = np.mean(tys)
        else:
            cx = np.mean(all_xs)
            cy = np.mean(all_ys)

        # 2. Determine Box Scale (Dynamic to include all limbs)
        # Find max extent from center to any visible joint
        # This ensures we encompass the racket arm / feet even if they stretch far
        max_dx = max([abs(x - cx) for x in all_xs])
        max_dy = max([abs(y - cy) for y in all_ys])
        
        # Base dimensions (2 * extent)
        w_box = 2 * max_dx
        h_box = 2 * max_dy
        
        # 3. Apply Margin & Minimum Size
        # Minimum size based on frame fraction
        min_w = float(width) * min_size_frac
        min_h = float(height) * min_size_frac
        
        w_box = max(w_box * (1 + margin), min_w)
        h_box = max(h_box * (1 + margin), min_h)

        # 4. Convert to corners
        x1 = cx - w_box / 2
        x2 = cx + w_box / 2
        y1 = cy - h_box / 2
        y2 = cy + h_box / 2

        # 5. Temporal Smoothing (EMA)
        # CRITICAL: Even with smoothing > 0, we can be deterministic if the sequence is deterministic.
        # This significantly reduces jitter.
        if last_box is not None and smoothing > 0:
            lx1, ly1, lx2, ly2 = last_box
            x1 = lx1 * smoothing + x1 * (1 - smoothing)
            y1 = ly1 * smoothing + y1 * (1 - smoothing)
            x2 = lx2 * smoothing + x2 * (1 - smoothing)
            y2 = ly2 * smoothing + y2 * (1 - smoothing)

        # 6. Clamp to image bounds
        x1i = self._clamp_int(x1, 0, width - 1)
        y1i = self._clamp_int(y1, 0, height - 1)
        x2i = self._clamp_int(x2, x1i + 1, width)
        y2i = self._clamp_int(y2, y1i + 1, height)
        
        return (x1i, y1i, x2i, y2i)

    @staticmethod
    def _crop_with_box(frame, box):
        if box is None:
            return frame
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]
        return roi if roi.size else frame

    def extract(self, video_path, crop_config, roi_cfg=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None, 30
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []

        # Decide whether to skip cropping for files like 'name (1).mp4'
        filename = os.path.basename(video_path)
        skip_crop = should_skip_crop(filename)

        last_box = None

        while True:
            ret, frame = cap.read()
            if not ret: break

            if not skip_crop:
                h, w = frame.shape[:2]
                frame = frame[
                    int(h*crop_config['top']):h-int(h*crop_config['bottom']),
                    int(w*crop_config['left']):w-int(w*crop_config['right'])
                ]
                if frame.size == 0: continue

            # Pose (get image-space landmarks for ROI + normalized pose)
            res = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lm = np.array([[l.x, l.y, l.z] for l in res.pose_landmarks.landmark])
                # Use Geometric Normalization (Compatible with KSI)
                pose_norm = normalize_pose(lm).flatten()
            else:
                pose_norm = np.zeros(99)

            # CNN Features (optionally pose-guided ROI crop)
            h2, w2 = frame.shape[:2]
            box = self._compute_pose_roi_box(
                res.pose_landmarks if hasattr(res, 'pose_landmarks') else None,
                w2,
                h2,
                roi_cfg,
                last_box=last_box,
            )
            last_box = box if box is not None else last_box
            roi_frame = self._crop_with_box(frame, box)

            img = cv2.resize(roi_frame, (self.cnn_input_size, self.cnn_input_size))
            img = preprocess_input(np.expand_dims(img[..., ::-1], axis=0))
            cnn_norm = self.rgb_model.predict(img, verbose=0)[0]

            frames.append(np.concatenate([pose_norm, cnn_norm]))
            
        cap.release()
        return np.array(frames), fps

class PoseFeatureExtractor:
    def __init__(self, mp_config):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,  # Use static mode for reliability/determinism
            model_complexity=mp_config['model_complexity'],
            min_detection_confidence=mp_config['min_detection_confidence'],
            min_tracking_confidence=mp_config['min_tracking_confidence']
        )

    def extract_full_sequence(self, video_path, crop_config=None):
        """Extract pose sequence with optional cropping."""
        if crop_config is None:
            crop_config = {'top': 0.0, 'bottom': 0.0, 'left': 0.0, 'right': 0.0}
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None
        frames = []
        # Decide whether to skip cropping for files like 'name (1).mp4'
        filename = os.path.basename(video_path)
        skip_crop = should_skip_crop(filename)

        while True:
            ret, frame = cap.read()
            if not ret: break

            # Apply crop (unless skipping)
            if skip_crop:
                frame_cropped = frame
            else:
                h, w = frame.shape[:2]
                frame_cropped = frame[
                    int(h*crop_config['top']):h-int(h*crop_config['bottom']),
                    int(w*crop_config['left']):w-int(w*crop_config['right'])
                ]
                if frame_cropped.size == 0: continue
            
            res = self.pose.process(cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lm = np.array([[l.x, l.y, l.z] for l in res.pose_landmarks.landmark])
                frames.append(normalize_pose(lm))
        cap.release()
        return np.array(frames) if frames else None