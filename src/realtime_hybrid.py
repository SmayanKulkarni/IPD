import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

import tensorflow as tf

from features import HybridFeatureExtractor
from utils import should_skip_crop, resolve_crop_config_for_video


def _load_params(params_path: str) -> dict:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is not installed. Install it with `pip install pyyaml` (or `conda install pyyaml`)"
        )
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def _resolve_classes(params: dict) -> list[str]:
    cfg = params.get("hybrid_pipeline", {})
    candidates = [
        cfg.get("data_path"),
        params.get("base", {}).get("raw_data_path"),
    ]

    for root in candidates:
        if not root or not os.path.isdir(root):
            continue
        classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        classes = sorted(classes)
        if classes:
            return classes

    return []


def _open_capture(source: str):
    # If it's an int like "0", use webcam
    if source.isdigit() and len(source) <= 2:
        return cv2.VideoCapture(int(source)), True
    return cv2.VideoCapture(source), False


def _apply_crop(frame: np.ndarray, crop_cfg: dict) -> np.ndarray:
    if crop_cfg is None:
        return frame

    h, w = frame.shape[:2]
    start_row = int(h * float(crop_cfg.get("top", 0.0)))
    end_row = h - int(h * float(crop_cfg.get("bottom", 0.0)))
    start_col = int(w * float(crop_cfg.get("left", 0.0)))
    end_col = w - int(w * float(crop_cfg.get("right", 0.0)))

    cropped = frame[start_row:end_row, start_col:end_col]
    return cropped if cropped.size else frame


def _smooth_signal(signal, window_size=3):
    """Exponential moving average for real-time signal smoothing."""
    if len(signal) == 0:
        return signal
    alpha = 2.0 / (window_size + 1)
    smoothed = [signal[0]]
    for val in signal[1:]:
        smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
    return np.array(smoothed)


def _detect_contact_realtime(landmarks_buffer):
    """
    Real-time contact detection based on multi-joint acceleration.
    
    Detects the frame within the buffer with peak arm acceleration
    (wrist + elbow + shoulder composite motion).
    
    Args:
        landmarks_buffer: deque of (33, 3) landmark arrays
    
    Returns:
        contact_frame_idx: Index within buffer (0 to len-1) with highest acceleration
    """
    if len(landmarks_buffer) < 3:  # Need at least 3 frames for meaningful acceleration
        return 0
    
    lm_array = np.array(landmarks_buffer)  # (T, 33, 3)
    
    # Get right arm joints: shoulder (12), elbow (14), wrist (16)
    shoulder_pos = lm_array[:, 12, :2]  # (T, 2)
    elbow_pos = lm_array[:, 14, :2]
    wrist_pos = lm_array[:, 16, :2]
    
    # Calculate velocities
    shoulder_vel = np.linalg.norm(np.diff(shoulder_pos, axis=0), axis=1)
    elbow_vel = np.linalg.norm(np.diff(elbow_pos, axis=0), axis=1)
    wrist_vel = np.linalg.norm(np.diff(wrist_pos, axis=0), axis=1)
    
    # Composite velocity (weighted: wrist primary, elbow secondary)
    composite_vel = 0.5 * wrist_vel + 0.3 * elbow_vel + 0.2 * shoulder_vel
    
    # Smooth for real-time robustness
    composite_vel_smooth = _smooth_signal(composite_vel, window_size=3)
    
    # Find frame with peak acceleration
    if len(composite_vel_smooth) > 1:
        acceleration = np.diff(composite_vel_smooth)
        contact_idx = np.argmax(np.abs(acceleration)) + 1  # +1 because diff reduces length
        return min(contact_idx, len(landmarks_buffer) - 1)
    
    return np.argmax(composite_vel)


def _prepare_model_inputs(model: tf.keras.Model, x_fused: np.ndarray, cnn_dim: int):
    """Prepare inputs for different historical model signatures.

    Supported cases:
      - 1 input: (None, T, fused_dim)
      - 2 inputs: any pairing of (cnn_dim, pose_dim, fused_dim)
    """
    if x_fused.ndim != 3:
        raise ValueError(f"Expected x_fused shape (1,T,D), got {x_fused.shape}")

    fused_dim = int(x_fused.shape[-1])
    x_cnn = x_fused[..., -cnn_dim:] if cnn_dim > 0 else x_fused[..., :0]
    x_pose = x_fused[..., :-cnn_dim] if cnn_dim > 0 else x_fused

    candidates = {
        int(x_cnn.shape[-1]): x_cnn,
        int(x_pose.shape[-1]): x_pose,
        int(x_fused.shape[-1]): x_fused,
    }

    if len(model.inputs) == 1:
        expected = int(model.inputs[0].shape[-1])
        if expected in candidates:
            return [candidates[expected]]
        return [x_fused]

    expected_dims = []
    for inp in model.inputs:
        try:
            expected_dims.append(int(inp.shape[-1]))
        except Exception:
            expected_dims.append(None)

    prepared = []
    for d in expected_dims:
        if d is None:
            prepared.append(x_fused)
            continue
        if d not in candidates:
            raise ValueError(
                f"Model expects input dim {d}, but available are {sorted(candidates.keys())}. "
                f"(fused_dim={fused_dim}, cnn_dim={cnn_dim})"
            )
        prepared.append(candidates[d])

    return prepared


def main():
    parser = argparse.ArgumentParser(description="Realtime hybrid inference (webcam or video)")
    parser.add_argument(
        "--source",
        default="0",
        help='Webcam index (e.g. "0") or video path (e.g. data/raw/forehand_clear/002.mp4)',
    )
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--model", default=None, help="Override hybrid model path")
    parser.add_argument("--no-roi", action="store_true", help="Disable pose-guided ROI cropping for CNN")
    parser.add_argument("--topk", type=int, default=3, help="Show top-K classes")
    parser.add_argument("--max-fps", type=float, default=0.0, help="Cap processing FPS (0 = no cap)")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening a window (prints a few predictions and exits)",
    )
    parser.add_argument(
        "--headless-frames",
        type=int,
        default=200,
        help="Frames to process in --headless mode",
    )
    args = parser.parse_args()

    params = _load_params(args.params)
    cfg = params["hybrid_pipeline"]
    mp_cfg = params["mediapipe"]

    model_path = args.model or cfg["model_path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Hybrid model not found: {model_path}")

    classes = _resolve_classes(params)

    seq_len = int(cfg["sequence_length"])
    cnn_dim = int(cfg["cnn_feature_dim"])
    crop_cfg_default = cfg.get("crop_config")
    crop_overrides = params.get("crop_overrides", {})

    roi_cfg = cfg.get("cnn_roi") or {}
    if args.no_roi:
        roi_cfg = {"enabled": False}

    extractor = HybridFeatureExtractor(
        mp_cfg,
        cnn_dim=cnn_dim,
        cnn_input_size=int(cfg.get("cnn_input_size", 224)),
    )

    try:
        model = tf.keras.models.load_model(model_path)

        cap, is_webcam = _open_capture(args.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {args.source}")

        # Decide crop config for input video (webcam uses default)
        if is_webcam:
            crop_cfg = crop_cfg_default
            skip_crop = False
        else:
            crop_cfg = resolve_crop_config_for_video(args.source, crop_cfg_default, crop_overrides)
            skip_crop = should_skip_crop(os.path.basename(args.source))

        window = deque(maxlen=seq_len)
        landmark_buffer = deque(maxlen=seq_len)  # Track landmarks for contact detection
        last_pose = None
        zeros_pose = np.zeros(99, dtype=np.float32)
        last_box = None
        
        contact_frame_idx = -1  # Index in window where contact occurs

        prev_tick = time.time()
        shown_fps = 0.0

        processed = 0
        printed = 0

        while True:
            loop_start = time.time()

            ok, frame = cap.read()
            if not ok:
                break

            frame_in = frame

            if skip_crop:
                frame_cropped = frame_in
            else:
                frame_cropped = _apply_crop(frame_in, crop_cfg)

            res = extractor.pose.process(cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB))

            # Use image-space landmarks (pose_landmarks) for consistency with templates
            if res.pose_landmarks:
                lm = np.array(
                    [[l.x, l.y, l.z] for l in res.pose_landmarks.landmark],
                    dtype=np.float32,
                )
                from utils import normalize_pose

                pose_flat = normalize_pose(lm).astype(np.float32).flatten()
                last_pose = pose_flat
                landmark_buffer.append(lm)
            else:
                pose_flat = last_pose if last_pose is not None else zeros_pose
                landmark_buffer.append(np.zeros((33, 3), dtype=np.float32))

            h2, w2 = frame_cropped.shape[:2]
            box = extractor._compute_pose_roi_box(
                getattr(res, "pose_landmarks", None),
                w2,
                h2,
                roi_cfg,
                last_box=last_box,
            )
            last_box = box if box is not None else last_box
            roi_frame = extractor._crop_with_box(frame_cropped, box)

            img_size = getattr(extractor, "cnn_input_size", 224)
            img = cv2.resize(roi_frame, (img_size, img_size))
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

            img = preprocess_input(np.expand_dims(img[..., ::-1], axis=0))
            cnn_feat = extractor.rgb_model.predict(img, verbose=0)[0].astype(np.float32)

            fused = np.concatenate([pose_flat, cnn_feat], axis=0)
            window.append(fused)

            pred_text_lines = ["warming up..."]
            contact_indicator = ""

            if len(window) == seq_len:
                # Detect contact moment for real-time insight
                contact_frame_idx = _detect_contact_realtime(landmark_buffer)
                
                x = np.asarray(window, dtype=np.float32)[None, ...]

                model_inputs = _prepare_model_inputs(model, x_fused=x, cnn_dim=cnn_dim)
                probs = model.predict(model_inputs, verbose=0)[0]
                topk = max(1, int(args.topk))
                top_idx = np.argsort(probs)[::-1][:topk]

                pred_text_lines = []
                for j, idx in enumerate(top_idx):
                    name = classes[idx] if classes and idx < len(classes) else f"class_{int(idx)}"
                    pred_text_lines.append(f"{j+1}. {name}: {float(probs[idx]):.3f}")
                
                # Mark contact frame
                contact_indicator = f" [Contact: frame {contact_frame_idx+1}/{seq_len}]"

                if args.headless and printed < 5:
                    print(" | ".join(pred_text_lines) + contact_indicator)
                    printed += 1

            # Update FPS display
            now = time.time()
            dt = now - prev_tick
            if dt > 0:
                shown_fps = 0.9 * shown_fps + 0.1 * (1.0 / dt)
            prev_tick = now

            # Overlay
            overlay = frame_in.copy()
            y = 30
            cv2.putText(
                overlay,
                f"FPS: {shown_fps:.1f}" + contact_indicator,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y += 28
            for line in pred_text_lines:
                cv2.putText(
                    overlay,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y += 28

            # Optionally visualize ROI box on the cropped coordinate system by drawing on overlay
            # (we keep it minimal; box is mainly for debugging locality)
            h, w = frame_in.shape[:2]
            top_off = int(h * float(crop_cfg.get("top", 0.0))) if not skip_crop else 0
            left_off = int(w * float(crop_cfg.get("left", 0.0))) if not skip_crop else 0

            if box is not None and not skip_crop:
                # Map box from cropped coords to original frame coords
                x1, y1, x2, y2 = box
                cv2.rectangle(
                    overlay,
                    (left_off + x1, top_off + y1),
                    (left_off + x2, top_off + y2),
                    (0, 255, 255),
                    2,
                )

            # Draw MediaPipe skeleton on the overlay (mapped to original frame coords)
            if res.pose_landmarks:
                mp_drawing = mp.solutions.drawing_utils
                mp_pose = mp.solutions.pose
                # We draw on a region of the overlay corresponding to the cropped area
                # Create a sub-image view, draw skeleton, then it reflects on overlay
                h_crop, w_crop = frame_cropped.shape[:2]
                # Draw directly onto overlay in the cropped region
                for conn in mp_pose.POSE_CONNECTIONS:
                    idx0, idx1 = conn
                    lm0 = res.pose_landmarks.landmark[idx0]
                    lm1 = res.pose_landmarks.landmark[idx1]
                    # Convert normalized coords to pixel coords in cropped frame, then offset
                    pt0 = (int(lm0.x * w_crop) + left_off, int(lm0.y * h_crop) + top_off)
                    pt1 = (int(lm1.x * w_crop) + left_off, int(lm1.y * h_crop) + top_off)
                    cv2.line(overlay, pt0, pt1, (0, 255, 0), 2, cv2.LINE_AA)
                # Draw landmarks as circles
                for lm in res.pose_landmarks.landmark:
                    cx = int(lm.x * w_crop) + left_off
                    cy = int(lm.y * h_crop) + top_off
                    cv2.circle(overlay, (cx, cy), 4, (255, 0, 255), -1, cv2.LINE_AA)

            if not args.headless:
                cv2.imshow("Hybrid Realtime", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            processed += 1
            if args.headless and processed >= int(args.headless_frames):
                break

            # FPS cap
            if args.max_fps and args.max_fps > 0:
                target_dt = 1.0 / float(args.max_fps)
                elapsed = time.time() - loop_start
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)

        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

    finally:
        try:
            extractor.pose.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
