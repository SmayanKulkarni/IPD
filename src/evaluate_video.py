#!/usr/bin/env python3
"""
Evaluate a single video file with KSI metrics and natural language coaching.
Supports both file paths and real-time webcam input.
"""

import argparse
import os
import yaml
import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import mediapipe as mp

from ksi_v2 import EnhancedKSI, ShotPhase
from features import HybridFeatureExtractor
from utils import normalize_pose, resolve_crop_config_for_video, should_skip_crop

try:
    from natural_language_coach import generate_coaching_report
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False


def load_params():
    """Load configuration from params.yaml"""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def load_expert_templates(params):
    """Load expert reference templates"""
    template_path = params['expert_pipeline']['output_path']
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Templates not found at {template_path}")
    return np.load(template_path, allow_pickle=True)


def _smooth_signal(signal, window_size=5):
    """Apply exponential moving average for smoothing noisy signals."""
    if len(signal) == 0:
        return signal
    alpha = 2.0 / (window_size + 1)
    smoothed = [signal[0]]
    for val in signal[1:]:
        smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
    return np.array(smoothed)


def find_contact_moment(all_landmarks, seq_len):
    """
    Identify the contact moment using multi-joint acceleration (real-time optimized).
    
    Contact detection:
    - Combines wrist (16), elbow (14), shoulder (12) joints
    - Calculates composite "arm acceleration" (rate of velocity change)
    - Finds peak acceleration with temporal smoothing for robustness
    - Returns the window with highest acceleration (predictive of contact)
    
    Returns:
        contact_window_idx: Index of window containing contact
        contact_frame_in_window: Frame within that window where contact occurs (0-seq_len)
    """
    max_acceleration = 0
    contact_window = 0
    contact_frame = 0
    
    for win_idx, landmarks_window in enumerate(all_landmarks):
        # landmarks_window: (T, 33, 3)
        
        # Get right arm joints: shoulder (12), elbow (14), wrist (16)
        shoulder_pos = landmarks_window[:, 12, :2]  # (T, 2)
        elbow_pos = landmarks_window[:, 14, :2]
        wrist_pos = landmarks_window[:, 16, :2]
        
        # Calculate velocities for each segment
        shoulder_vel = np.linalg.norm(np.diff(shoulder_pos, axis=0), axis=1)  # (T-1,)
        elbow_vel = np.linalg.norm(np.diff(elbow_pos, axis=0), axis=1)
        wrist_vel = np.linalg.norm(np.diff(wrist_pos, axis=0), axis=1)
        
        # Composite arm velocity (wrist is primary, elbow secondary, shoulder stabilizer)
        # Weighted: more weight on distal joints (wrist is fastest)
        composite_vel = 0.5 * wrist_vel + 0.3 * elbow_vel + 0.2 * shoulder_vel
        
        # Smooth velocity for real-time robustness
        composite_vel_smooth = _smooth_signal(composite_vel, window_size=3)
        
        # Calculate acceleration (change in composite velocity)
        if len(composite_vel_smooth) > 1:
            acceleration = np.linalg.norm(np.diff(composite_vel_smooth))
            
            # Check if this window has maximum acceleration
            if acceleration > max_acceleration:
                max_acceleration = acceleration
                contact_window = win_idx
                # Find frame in this window with max composite velocity
                contact_frame = np.argmax(composite_vel_smooth)
    
    return contact_window, contact_frame


def predict_shot_type_at_contact(all_windows, all_landmarks, model, classes, cnn_dim, pipeline_type='hybrid'):
    """
    Instead of consensus across all windows, predict based on the window
    containing the contact moment (highest acceleration).
    
    Returns:
        predictions: All predictions for reference
        best_prediction: The prediction at contact moment
        best_class: Shot class at contact
        contact_info: Dict with contact window and frame info
    """
    # Find contact moment
    seq_len = all_windows[0].shape[0] if all_windows else 40
    contact_window_idx, contact_frame = find_contact_moment(all_landmarks, seq_len)
    
    # Get all predictions first
    features = np.array(all_windows)  # (N, T, D)
    model_inputs = _prepare_model_inputs(model, x_fused=features, cnn_dim=cnn_dim)
    all_probs = model.predict(model_inputs, verbose=0)
    
    all_predictions = []
    for i, probs in enumerate(all_probs):
        pred_idx = np.argmax(probs)
        predicted_class = classes[pred_idx]
        confidence = float(probs[pred_idx])
        all_predictions.append({
            'window': i,
            'class': predicted_class,
            'confidence': confidence,
            'all_scores': {classes[j]: float(probs[j]) for j in range(len(classes))}
        })
    
    # Get prediction at contact
    best_prediction = all_predictions[contact_window_idx]
    best_class = best_prediction['class']
    
    contact_info = {
        'contact_window': contact_window_idx,
        'contact_frame': contact_frame,
        'total_windows': len(all_windows),
        'seq_len': seq_len
    }
    
    return all_predictions, best_prediction, best_class, contact_info


def extract_features_from_video(video_source, extractor, params, pipeline_type='hybrid'):
    """
    Extract features and landmarks from video using sliding window (like realtime_hybrid).
    Uses image-space landmarks (pose_landmarks) so units match expert templates.
    Skips low-quality windows to avoid zeroed KSI.
    
    Args:
        video_source: File path or webcam index (0, 1, etc)
        extractor: Feature extractor (HybridFeatureExtractor)
        params: Configuration dict
        pipeline_type: 'hybrid' or 'pose'
    
    Returns:
        all_windows: List of fused feature windows
        all_landmarks: List of (T, 33, 3) landmarks per window
        frame_count: Total frames processed
    """
    # Open video or webcam
    if isinstance(video_source, str) and video_source.isdigit():
        cap = cv2.VideoCapture(int(video_source))
        is_webcam = True
    else:
        cap = cv2.VideoCapture(video_source)
        is_webcam = False
    
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {video_source}")
    
    cfg = params['hybrid_pipeline']
    seq_len = cfg['sequence_length']
    cnn_dim = cfg['cnn_feature_dim']
    crop_cfg = resolve_crop_config_for_video(video_source, params)
    roi_cfg = cfg.get("cnn_roi") or {}
    
    # Use extractor's pose model
    window = deque(maxlen=seq_len)
    landmark_window = deque(maxlen=seq_len)
    valid_mask_window = deque(maxlen=seq_len)
    
    all_windows = []
    all_landmarks = []
    all_valid_ratios = []
    last_pose = None
    last_box = None
    frame_count = 0
    
    print(f"📹 Processing video from: {video_source}")
    print(f"   Sequence length: {seq_len} | CNN features: {cnn_dim}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop if needed
        if crop_cfg:
            frame = _apply_crop(frame, crop_cfg)
        
        # Pose detection using extractor's MediaPipe
        res = extractor.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_count += 1
        
        # Extract pose landmarks (image-space) so scale matches templates
        if res.pose_landmarks:
            lm = np.array(
                [[l.x, l.y, l.z] for l in res.pose_landmarks.landmark],
                dtype=np.float32,
            )
            pose_flat = normalize_pose(lm).astype(np.float32).flatten()
            last_pose = pose_flat
            landmark_window.append(lm)
            valid_mask_window.append(1)
        else:
            # Reuse last good pose; if none, mark invalid
            zeros_pose = np.zeros(99, dtype=np.float32)
            pose_flat = last_pose if last_pose is not None else zeros_pose
            landmark_window.append(np.zeros((33, 3), dtype=np.float32))
            valid_mask_window.append(0)
        
        # Extract CNN features from frame
        h, w = frame.shape[:2]
        box = extractor._compute_pose_roi_box(
            getattr(res, "pose_landmarks", None),
            w,
            h,
            roi_cfg,
            last_box=last_box,
        )
        last_box = box if box is not None else last_box
        roi_frame = extractor._crop_with_box(frame, box)
        
        img_size = cfg.get("cnn_input_size", 224)
        img = cv2.resize(roi_frame, (img_size, img_size))
        img = preprocess_input(np.expand_dims(img[..., ::-1], axis=0))
        cnn_feat = extractor.rgb_model.predict(img, verbose=0)[0].astype(np.float32)
        
        # Fuse pose and CNN features
        fused = np.concatenate([pose_flat, cnn_feat], axis=0)
        window.append(fused)
        
        # When window is full, save it
        if len(window) == seq_len:
            valid_ratio = sum(valid_mask_window) / float(seq_len)
            all_windows.append(np.array(list(window)))
            all_landmarks.append(np.array(list(landmark_window)))
            all_valid_ratios.append(valid_ratio)
    
    cap.release()
    extractor.pose.close()
    
    if not all_windows:
        raise RuntimeError("No valid windows extracted from video")
    
    # Filter out low-quality windows (too many missing poses or NaNs)
    filtered_windows = []
    filtered_landmarks = []
    for win, lm, ratio in zip(all_windows, all_landmarks, all_valid_ratios):
        if ratio < 0.7:  # require at least 70% frames with pose
            continue
        if not np.isfinite(win).all() or not np.isfinite(lm).all():
            continue
        if np.allclose(lm, 0):  # avoid all-zero landmark windows
            continue
        filtered_windows.append(win)
        filtered_landmarks.append(lm)
    
    if not filtered_windows:
        raise RuntimeError("All windows were filtered out due to low pose quality; try a clearer video")
    
    print(f"✅ Extracted {frame_count} frames into {len(filtered_windows)} valid windows (from {len(all_windows)} total)")
    print(f"   Window shape: {filtered_windows[0].shape}")
    print(f"   Landmarks shape: {filtered_landmarks[0].shape}")
    
    return filtered_windows, filtered_landmarks, frame_count
    print(f"   Features shape: {features.shape}")
    print(f"   Landmarks shape: {raw_landmarks.shape}")
    
    return features, raw_landmarks, frame_count


def _apply_crop(frame, crop_cfg):
    """Apply crop to frame"""
    if crop_cfg is None:
        return frame
    
    h, w = frame.shape[:2]
    start_row = int(h * float(crop_cfg.get("top", 0.0)))
    end_row = h - int(h * float(crop_cfg.get("bottom", 0.0)))
    start_col = int(w * float(crop_cfg.get("left", 0.0)))
    end_col = w - int(w * float(crop_cfg.get("right", 0.0)))
    
    cropped = frame[start_row:end_row, start_col:end_col]
    return cropped if cropped.size else frame


def _prepare_model_inputs(model, x_fused, cnn_dim):
    """Prepare inputs for model (handles different input signatures like realtime_hybrid)."""
    if x_fused.ndim != 3:
        raise ValueError(f"Expected x_fused shape (N, T, D), got {x_fused.shape}")
    
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


def predict_shot_type(all_windows, model, classes, cnn_dim, pipeline_type='hybrid'):
    """
    Predict shot type for all windows using proper model input preparation.
    
    Returns:
        predictions: List of dicts with 'class', 'confidence', and 'all_scores'
        best_class: Most common predicted class (consensus)
    """
    predictions = []
    pred_indices = []
    
    # Convert list of windows to array
    features = np.array(all_windows)  # (N, T, D)
    
    # Prepare and predict
    model_inputs = _prepare_model_inputs(model, x_fused=features, cnn_dim=cnn_dim)
    all_probs = model.predict(model_inputs, verbose=0)
    
    for probs in all_probs:
        pred_idx = np.argmax(probs)
        pred_indices.append(pred_idx)
        predicted_class = classes[pred_idx]
        confidence = float(probs[pred_idx])
        predictions.append({
            'class': predicted_class,
            'confidence': confidence,
            'all_scores': {classes[i]: float(probs[i]) for i in range(len(classes))}
        })
    
    # Find most common prediction (consensus)
    from collections import Counter
    pred_counts = Counter(pred_indices)
    best_idx = pred_counts.most_common(1)[0][0]
    best_class = classes[best_idx]
    
    return predictions, best_class


def evaluate_video(
    video_source,
    model_path,
    pipeline_type='hybrid',
    nlp_skill_level='intermediate',
    generate_report=True
):
    """
    Main evaluation function for single video.
    
    Args:
        video_source: File path or webcam index (0, 1, etc)
        model_path: Path to trained model
        pipeline_type: 'hybrid' or 'pose'
        nlp_skill_level: Skill level for coaching ('beginner', 'intermediate', 'advanced', 'expert')
        generate_report: Whether to generate coaching report
    """
    params = load_params()
    cfg = params[f'{pipeline_type}_pipeline']
    ksi_cfg = params.get('ksi', {'weights': {'pose': 0.5, 'velocity': 0.3, 'acceleration': 0.2}})
    
    # Load model
    print(f"\n🔄 Loading model: {model_path}")
    model = load_model(model_path)
    
    # Load classes
    data_path = cfg['data_path']
    classes = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    print(f"📋 Classes: {classes}")
    
    # Feature extractor
    mp_config = params['mediapipe']
    extractor = HybridFeatureExtractor(
        mp_config=mp_config,
        cnn_dim=cfg['cnn_feature_dim'],
        cnn_input_size=cfg['cnn_input_size']
    )
    
    # Get sequence parameters
    seq_len = cfg['sequence_length']
    stride = cfg['stride']
    
    # Extract features from video
    print(f"\n{'='*70}")
    print(f"EXTRACTING FEATURES FROM VIDEO")
    print(f"{'='*70}")
    all_windows, all_landmarks, frame_count = extract_features_from_video(
        video_source, extractor, params, pipeline_type
    )
    
    # Predict shot type
    print(f"\n{'='*70}")
    print(f"PREDICTING SHOT TYPE (ALL WINDOWS)")
    print(f"{'='*70}")
    all_predictions, best_prediction, best_class, contact_info = predict_shot_type_at_contact(
        all_windows, all_landmarks, model, classes, cfg['cnn_feature_dim'], pipeline_type
    )
    
    print(f"\n📊 Total predictions: {len(all_predictions)}")
    print(f"{'─'*70}")
    
    # Group by class and show statistics
    from collections import Counter
    pred_classes = [p['class'] for p in all_predictions]
    class_counts = Counter(pred_classes)
    
    print(f"\n🎯 PREDICTION SUMMARY (all windows):")
    for shot_class in sorted(class_counts.keys()):
        count = class_counts[shot_class]
        percentage = 100 * count / len(all_predictions)
        confs = [p['confidence'] for p in all_predictions if p['class'] == shot_class]
        avg_conf = np.mean(confs)
        print(f"   {shot_class:20s}: {count:3d} predictions ({percentage:5.1f}%) | Avg confidence: {avg_conf:.2%}")
    
    # Show contact-based prediction
    print(f"\n{'─'*70}")
    print(f"⚡ CONTACT-BASED PREDICTION (MOST RELIABLE):")
    print(f"{'─'*70}")
    print(f"   Contact occurs at: Window {contact_info['contact_window']} (frame {contact_info['contact_frame']}/{contact_info['seq_len']})")
    print(f"\n   🎯 Predicted at contact: {best_prediction['class']}")
    print(f"   Confidence: {best_prediction['confidence']:.2%}")
    print(f"   All scores at contact:")
    for cls, score in sorted(best_prediction['all_scores'].items(), key=lambda x: x[1], reverse=True):
        print(f"      {cls:20s}: {score:.2%}")
    
    # Show first 10 detailed predictions
    print(f"\n{'─'*70}")
    print(f"📋 DETAILED PREDICTIONS (first 10 windows):")
    print(f"{'─'*70}")
    for i, pred in enumerate(all_predictions[:10]):
        marker = " ⚡ CONTACT" if i == contact_info['contact_window'] else ""
        print(f"\n   Window {i+1:2d}: {pred['class']:20s} ({pred['confidence']:.2%}){marker}")
        sorted_scores = sorted(pred['all_scores'].items(), key=lambda x: x[1], reverse=True)
        for cls, score in sorted_scores[:3]:
            print(f"              {cls:20s}: {score:.2%}")
    
    if len(all_predictions) > 10:
        print(f"\n   ... and {len(all_predictions) - 10} more predictions")
    
    # Calculate KSI
    print(f"\n{'='*70}")
    print(f"CALCULATING KSI METRICS")
    print(f"{'='*70}")
    
    templates = load_expert_templates(params)
    ksi_calc = EnhancedKSI()
    
    # Get expert template for consensus class
    template_key = best_class
    if template_key not in templates.files:
        template_key = f'{best_class}_variant1'
    
    if template_key not in templates.files:
        print(f"⚠️  Template not found for {best_class}")
        return
    
    expert_template = templates[template_key]
    if expert_template.ndim == 2 and expert_template.shape[1] == 99:
        expert_lm = expert_template.reshape(-1, 33, 3)
    else:
        expert_lm = expert_template
    
    # Calculate KSI for each window and average
    ksi_scores = []
    for i, user_lm in enumerate(all_landmarks):
        result = ksi_calc.calculate(
            expert_landmarks=expert_lm,
            user_landmarks=user_lm,
            weights=ksi_cfg['weights'],
        )
        ksi_scores.append(result)
    
    # Use average KSI result
    avg_ksi_total = np.mean([r.ksi_total for r in ksi_scores])
    avg_ksi_weighted = np.mean([r.ksi_weighted for r in ksi_scores])
    
    # Prefer contact window if valid, otherwise highest KSI
    contact_idx = contact_info['contact_window'] if ksi_scores else 0
    if ksi_scores and np.isfinite(ksi_scores[contact_idx].ksi_total) and ksi_scores[contact_idx].ksi_total > 0:
        result = ksi_scores[contact_idx]
        chosen_idx = contact_idx
        chosen_reason = "contact window"
    else:
        best_idx = int(np.argmax([r.ksi_total for r in ksi_scores])) if ksi_scores else 0
        result = ksi_scores[best_idx]
        chosen_idx = best_idx
        chosen_reason = "highest KSI"
    
    print(f"📊 KSI Analysis ({len(ksi_scores)} windows):")
    print(f"   Average KSI Total: {avg_ksi_total:.3f}")
    print(f"   Average KSI Weighted: {avg_ksi_weighted:.3f}")
    print(f"   Using window #{chosen_idx + 1} ({chosen_reason}) for report")
    print(f"\n   Selected KSI Score: {result.ksi_total:.3f}")
    print(f"   KSI Weighted: {result.ksi_weighted:.3f}")
    print(f"   Phase scores: {result.phase_scores}")
    print(f"   Component scores: {result.components}")
    
    # Generate coaching report
    if generate_report and NLP_AVAILABLE:
        print(f"\n{'='*70}")
        print(f"GENERATING COACHING REPORT")
        print(f"{'='*70}")
        
        os.makedirs("coaching_reports", exist_ok=True)
        
        report = generate_coaching_report(
            ksi_result=result,
            shot_type_str=best_class,
            skill_level_str=nlp_skill_level,
            output_format='text',
            simplified=True
        )
        
        report_filename = f"coaching_reports/{best_class}_video_ksi{result.ksi_total:.3f}_report.txt"
        with open(report_filename, 'w') as f:
            f.write(report)
        
        print(f"✅ Report saved: {report_filename}")
        print(f"\n{'='*70}")
        print("📄 COACHING REPORT PREVIEW")
        print(f"{'='*70}")
        print(report)
    
    print(f"\n{'='*70}")
    print("✨ EVALUATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a single video with KSI metrics and coaching")
    parser.add_argument("video", type=str, help="Video file path or webcam index (0, 1, etc)")
    parser.add_argument("--type", choices=['pose', 'hybrid'], default='hybrid', help="Pipeline type (default: hybrid)")
    parser.add_argument("--model", type=str, default="models/tcn_hybrid_tuned.h5", help="Model path")
    parser.add_argument("--skill", type=str, default='intermediate',
                        choices=['beginner', 'intermediate', 'advanced', 'expert'],
                        help="Skill level for coaching (default: intermediate)")
    parser.add_argument("--no-report", action='store_true', help="Skip coaching report generation")
    
    args = parser.parse_args()
    
    evaluate_video(
        video_source=args.video,
        model_path=args.model,
        pipeline_type=args.type,
        nlp_skill_level=args.skill,
        generate_report=not args.no_report
    )
