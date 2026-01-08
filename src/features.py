import cv2
import os
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
# Local import
from utils import normalize_pose, should_skip_crop

class HybridFeatureExtractor:
    def __init__(self, mp_config, cnn_dim=128, cnn_input_size=224):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, 
            model_complexity=mp_config['model_complexity'],
            min_detection_confidence=mp_config['min_detection_confidence'],
            min_tracking_confidence=mp_config['min_tracking_confidence']
        )
        
        # CNN Setup
        self.cnn_input_size = int(cnn_input_size)
        base_cnn = MobileNetV2(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(self.cnn_input_size, self.cnn_input_size, 3),
        )

        img_in = Input(shape=(self.cnn_input_size, self.cnn_input_size, 3))
        x = base_cnn(img_in, training=False)
        x = Dense(cnn_dim, activation='relu')(x)
        x = Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))(x)
        self.rgb_model = Model(inputs=img_in, outputs=x)

        # Backwards-compatible attributes (older code used base_cnn + cnn_model separately)
        self.base_cnn = base_cnn
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
        if pose_landmarks is None:
            if use_last_on_missing and last_box is not None:
                return last_box
            if fallback_full_frame:
                return (0, 0, int(width), int(height))
            return None

        xs = []
        ys = []
        for j in joint_ids:
            try:
                lm = pose_landmarks.landmark[int(j)]
            except Exception:
                continue
            if lm is None:
                continue
            vis = getattr(lm, 'visibility', 1.0)
            if vis < visibility_thresh:
                continue
            if not np.isfinite(lm.x) or not np.isfinite(lm.y):
                continue
            xs.append(lm.x * float(width))
            ys.append(lm.y * float(height))

        # Not enough visible joints → fallback
        if len(xs) < min_joints:
            if use_last_on_missing and last_box is not None:
                return last_box
            if fallback_full_frame:
                return (0, 0, int(width), int(height))
            return None

        x1 = float(np.min(xs))
        x2 = float(np.max(xs))
        y1 = float(np.min(ys))
        y2 = float(np.max(ys))

        # ─────────────── Expand by margin ───────────────
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        x1 -= bw * margin
        x2 += bw * margin
        y1 -= bh * margin
        y2 += bh * margin

        # ─────────────── Enforce minimum size ───────────────
        min_size = min(float(width), float(height)) * min_size_frac
        if (x2 - x1) < min_size:
            cx = 0.5 * (x1 + x2)
            x1 = cx - 0.5 * min_size
            x2 = cx + 0.5 * min_size
        if (y2 - y1) < min_size:
            cy = 0.5 * (y1 + y2)
            y1 = cy - 0.5 * min_size
            y2 = cy + 0.5 * min_size

        # ─────────────── Temporal smoothing ───────────────
        if last_box is not None and smoothing > 0:
            lx1, ly1, lx2, ly2 = last_box
            alpha = 1.0 - smoothing
            x1 = alpha * x1 + smoothing * lx1
            y1 = alpha * y1 + smoothing * ly1
            x2 = alpha * x2 + smoothing * lx2
            y2 = alpha * y2 + smoothing * ly2

        # ─────────────── Clamp to frame ───────────────
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
            static_image_mode=False, 
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