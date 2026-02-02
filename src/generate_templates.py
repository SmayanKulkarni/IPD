"""
Expert Template Generation for KSI Evaluation
==============================================

Generates reference templates from expert player videos for biomechanical
comparison in the KSI evaluation pipeline. Templates represent the "gold
standard" motion patterns for each shot type.

Key Features:
    - Multi-video template averaging with quality weighting
    - Dynamic Time Warping (DTW) for temporal alignment
    - Quality-based video filtering (motion smoothness, stability)
    - Reference video selection (minimum total DTW distance)
    - Phase-specific template extraction (contact phase)
    - Temporal resampling for FPS normalization

Template Generation Pipeline:
    1. Load expert videos with same preprocessing as hybrid pipeline
    2. Extract pose landmarks and enhanced features
    3. Quality filter: remove low-quality recordings
    4. Find best reference video (centroid selection)
    5. DTW-align all videos to reference
    6. Compute quality-weighted average template
    7. Extract phase-specific templates (optional)
    8. Save as compressed .npz with metadata

Output Format:
    .npz file with:
    - '<class_name>': (T, 33, 3) raw landmark template
    - '<class_name>_variant1/2/3': Top-3 variant templates
    - '<class_name>_contact': Contact phase template
    - '_metadata_json': Generation metadata

Quality Metrics:
    - Motion smoothness (low velocity variance)
    - Feature stability (low feature variance)
    - Minimum motion threshold (detects static frames)

Dependencies:
    External: cv2, numpy, scipy, yaml
    Internal: features.PoseFeatureExtractor, ksi_v2, utils

Configuration (params.yaml):
    expert_pipeline:
        raw_path: Path to expert video directory
        output_path: Output path for template file

Usage:
    python generate_templates.py

Author: IPD Research Team
Version: 1.0.0
"""

import os
import yaml
import cv2
import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime
from features import PoseFeatureExtractor
from ksi_v2 import extract_enhanced_features, EnhancedKSI, ShotPhaseSegmenter, ShotPhase, dynamic_time_warping_optimized
from utils import normalize_pose, should_skip_crop, get_segment_bounds, resolve_crop_config_for_video


def temporal_resample(sequence, original_fps, target_fps):
    """
    Resample sequence to target FPS to normalize temporal scale.
    This ensures all videos are compared at the same temporal resolution.
    """
    if abs(original_fps - target_fps) < 0.1:
        return sequence
    
    n_frames_original = len(sequence)
    duration = n_frames_original / original_fps
    n_frames_target = int(duration * target_fps)
    
    if n_frames_target < 2:
        return sequence
    
    # Temporal interpolation
    old_times = np.linspace(0, 1, n_frames_original)
    new_times = np.linspace(0, 1, n_frames_target)
    
    # Handle both 2D (T, F) and 3D (T, J, 3) arrays
    if sequence.ndim == 2:
        resampled = np.array([
            interp1d(old_times, sequence[:, i], kind='linear', fill_value='extrapolate')(new_times)
            for i in range(sequence.shape[1])
        ]).T
    elif sequence.ndim == 3:
        # For landmarks (T, 33, 3) - interpolate each joint coordinate
        n_joints, n_coords = sequence.shape[1], sequence.shape[2]
        resampled = np.zeros((n_frames_target, n_joints, n_coords))
        for j in range(n_joints):
            for c in range(n_coords):
                resampled[:, j, c] = interp1d(
                    old_times, sequence[:, j, c], 
                    kind='linear', fill_value='extrapolate'
                )(new_times)
    else:
        return sequence
    
    return resampled


def filter_quality(video_data_list):
    """
    Filter out low-quality expert videos based on multiple criteria.
    Returns filtered list with quality scores.
    """
    if not video_data_list:
        return []
    
    filtered = []
    
    for vid_data in video_data_list:
        seq = vid_data['features']
        lms = vid_data['landmarks']
        
        # Criterion 1: Minimum length (at least 10 frames)
        if len(seq) < 10:
            continue
        
        # Criterion 2: Pose confidence - SKIP for normalized landmarks
        # (Normalized pose landmarks don't preserve visibility scores)
        mean_confidence = 1.0
        
        # Criterion 3: Motion smoothness (penalize jitter)
        velocity = np.linalg.norm(np.diff(seq, axis=0), axis=1)
        if len(velocity) > 0:
            smoothness = 1.0 / (1.0 + np.std(velocity))
        else:
            smoothness = 0.5
        
        # Criterion 4: Feature stability (no extreme outliers)
        feature_std = np.std(seq, axis=0)
        stability = 1.0 / (1.0 + np.mean(feature_std))
        
        # Criterion 5: Check for zero/invalid frames
        has_motion = np.any(velocity > 0.01) if len(velocity) > 0 else True
        if not has_motion:
            continue
        
        # Combined quality score
        quality_score = smoothness * 0.6 + stability * 0.4
        
        vid_data['quality_score'] = quality_score
        filtered.append(vid_data)
    
    # Sort by quality (best first)
    filtered.sort(key=lambda x: x['quality_score'], reverse=True)
    
    return filtered


def find_best_reference(sequences, ksi_calc):
    """
    Find the most representative video as reference for alignment.
    Uses minimum total DTW distance to all other videos.
    """
    if len(sequences) == 1:
        return 0
    
    n = len(sequences)
    pairwise_distances = np.zeros((n, n))
    
    print(f"    Computing pairwise DTW distances...")
    for i in range(n):
        for j in range(i + 1, n):
            try:
                _, _, dist = dynamic_time_warping_optimized(sequences[i], sequences[j])
                pairwise_distances[i, j] = dist
                pairwise_distances[j, i] = dist
            except Exception as e:
                print(f"    ⚠ DTW error between video {i} and {j}: {e}")
                pairwise_distances[i, j] = 1e6
                pairwise_distances[j, i] = 1e6
    
    # Find video with minimum total distance
    total_distances = np.sum(pairwise_distances, axis=1)
    best_idx = np.argmin(total_distances)
    
    return best_idx


def weighted_average_templates(aligned_sequences, quality_scores):
    """
    Compute weighted average of aligned sequences based on quality scores.
    """
    if not aligned_sequences:
        return None
    
    weights = np.array(quality_scores)
    weights = weights / np.sum(weights)  # Normalize
    
    # Weighted average
    weighted_avg = np.average(aligned_sequences, axis=0, weights=weights)
    
    return weighted_avg


def extract_phase_specific_template(landmarks_list, features_list, target_fps):
    """
    Extract contact-phase specific template for enhanced evaluation.
    """
    if not landmarks_list or not features_list:
        return None
    
    segmenter = ShotPhaseSegmenter(fps=target_fps)
    contact_features = []
    
    for lms, feats in zip(landmarks_list, features_list):
        if lms is None or len(lms) < 10:
            continue
        
        try:
            phases = segmenter.segment(lms)
            if ShotPhase.CONTACT.value in phases:
                contact_start, contact_end = phases[ShotPhase.CONTACT.value]
                if contact_start < contact_end and contact_end <= len(feats):
                    contact_features.append(feats[contact_start:contact_end])
        except Exception as e:
            print(f"    ⚠ Phase segmentation failed: {e}")
            continue
    
    if not contact_features:
        return None
    
    # Align and average contact phases
    if len(contact_features) == 1:
        return contact_features[0]
    
    # Use first as reference, align others
    ref = contact_features[0]
    
    aligned = [ref]
    for cf in contact_features[1:]:
        try:
            _, aligned_cf, _ = dynamic_time_warping_optimized(ref, cf)
            if len(aligned_cf) != len(ref):
                # Resample to match reference
                old_indices = np.linspace(0, 1, len(aligned_cf))
                new_indices = np.linspace(0, 1, len(ref))
                resampled = np.array([
                    interp1d(old_indices, aligned_cf[:, i], kind='linear')(new_indices)
                    for i in range(aligned_cf.shape[1])
                ]).T
                aligned.append(resampled)
            else:
                aligned.append(aligned_cf)
        except Exception:
            continue
    
    if aligned:
        return np.mean(aligned, axis=0)
    return None


def dtw_align_landmarks(ref_landmarks, query_landmarks, ref_features, query_features):
    """
    Align query landmarks to reference using DTW path computed from features.
    This ensures landmarks stay aligned with features.
    """
    n, m = len(ref_features), len(query_features)
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.linalg.norm(ref_features[i-1] - query_features[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    # Backtrack to get alignment path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        steps = [dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1]]
        best = np.argmin(steps)
        if best == 0:
            i -= 1
        elif best == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    
    path.reverse()
    
    if not path:
        return query_landmarks
    
    _, idx_query = zip(*path)
    return query_landmarks[list(idx_query)]


def temporal_resample_landmarks(landmarks, target_length):
    """
    Resample landmarks (T, 33, 3) to target length using interpolation.
    """
    if len(landmarks) == target_length:
        return landmarks
    
    n_frames = len(landmarks)
    old_times = np.linspace(0, 1, n_frames)
    new_times = np.linspace(0, 1, target_length)
    
    n_joints, n_coords = landmarks.shape[1], landmarks.shape[2]
    resampled = np.zeros((target_length, n_joints, n_coords))
    
    for j in range(n_joints):
        for c in range(n_coords):
            resampled[:, j, c] = interp1d(
                old_times, landmarks[:, j, c],
                kind='linear', fill_value='extrapolate'
            )(new_times)
    
    return resampled


def weighted_average_landmarks(aligned_landmarks, quality_scores):
    """
    Compute weighted average of aligned landmark sequences based on quality scores.
    """
    if not aligned_landmarks:
        return None
    
    weights = np.array(quality_scores)
    weights = weights / np.sum(weights)  # Normalize
    
    # Weighted average across all landmarks
    weighted_avg = np.average(aligned_landmarks, axis=0, weights=weights)
    
    return weighted_avg


def extract_phase_specific_landmarks(landmarks_list, target_fps):
    """
    Extract contact-phase specific RAW LANDMARKS for template.
    """
    if not landmarks_list:
        return None
    
    segmenter = ShotPhaseSegmenter(fps=target_fps)
    contact_landmarks = []
    
    for lms in landmarks_list:
        if lms is None or len(lms) < 10:
            continue
        
        try:
            phases = segmenter.segment(lms)
            if ShotPhase.CONTACT.value in phases:
                contact_start, contact_end = phases[ShotPhase.CONTACT.value]
                if contact_start < contact_end and contact_end <= len(lms):
                    contact_landmarks.append(lms[contact_start:contact_end])
        except Exception as e:
            print(f"    ⚠ Phase segmentation failed: {e}")
            continue
    
    if not contact_landmarks:
        return None
    
    # Align and average contact phases
    if len(contact_landmarks) == 1:
        return contact_landmarks[0]
    
    # Use first as reference
    ref = contact_landmarks[0]
    aligned = [ref]
    
    for cl in contact_landmarks[1:]:
        try:
            # Extract features for DTW alignment
            ref_feats = np.array([extract_enhanced_features(f) for f in ref])
            cl_feats = np.array([extract_enhanced_features(f) for f in cl])
            
            aligned_lm = dtw_align_landmarks(ref, cl, ref_feats, cl_feats)
            
            if len(aligned_lm) != len(ref):
                aligned_lm = temporal_resample_landmarks(aligned_lm, len(ref))
            
            aligned.append(aligned_lm)
        except Exception:
            continue
    
    if aligned:
        return np.mean(aligned, axis=0)
    return None


def extract_landmarks_with_preprocessing(
    video_path: str,
    extractor: PoseFeatureExtractor,
    base_crop_cfg: dict,
    segment_rules: dict = None,
    crop_overrides: dict = None,
):
    """
    Extract landmarks from a video using the same preprocessing logic as preprocess_hybrid.py.
    
    Handles:
    - Segment bounds (tail/middle extraction based on shot type)
    - Skip-crop detection for files like "name (N).mp4"
    - Per-video crop overrides based on video number
    - FPS extraction
    
    Returns:
        Tuple of (landmarks, fps) where landmarks is (T, 33, 3) or None
    """
    filename = os.path.basename(video_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 30.0
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    # Get segment bounds (same logic as preprocess_hybrid)
    start_frame, segment_frames = get_segment_bounds(
        video_path,
        fps,
        total_frames,
        default_seconds=1.75,
        segment_cfg=segment_rules,
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    
    # Check if we should skip cropping
    skip_crop = should_skip_crop(filename)
    
    # Get effective crop config with per-video overrides
    effective_crop = resolve_crop_config_for_video(video_path, base_crop_cfg, crop_overrides)
    
    frames = []
    frame_idx = 0
    
    while frame_idx < int(segment_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        # Apply crop (unless skipping)
        if skip_crop:
            frame_cropped = frame
        else:
            h, w = frame.shape[:2]
            start_row = int(h * effective_crop.get('top', 0.0))
            end_row = h - int(h * effective_crop.get('bottom', 0.0))
            start_col = int(w * effective_crop.get('left', 0.0))
            end_col = w - int(w * effective_crop.get('right', 0.0))
            frame_cropped = frame[start_row:end_row, start_col:end_col]
            if frame_cropped.size == 0:
                continue
        
        # Extract pose
        res = extractor.pose.process(cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            lm = np.array([[l.x, l.y, l.z] for l in res.pose_landmarks.landmark])
            frames.append(normalize_pose(lm))
    
    cap.release()
    return np.array(frames) if frames else None, fps


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    
    cfg, mp_cfg = params['expert_pipeline'], params['mediapipe']
    
    # Use the same crop config as hybrid pipeline
    base_crop_cfg = params.get('hybrid_pipeline', {}).get('crop_config', {
        'top': 0.13, 'bottom': 0.35, 'left': 0.25, 'right': 0.25
    })
    segment_rules = params.get('segment_rules', None)
    crop_overrides = params.get('crop_overrides', None)
    
    # Target FPS for normalization (use median of all videos or default 30)
    target_fps = 30.0
    
    extractor = PoseFeatureExtractor(mp_cfg)
    ksi_calc = EnhancedKSI(fps=target_fps)
    
    templates = {}
    metadata = {
        'feature_version': 'raw_landmarks',  # KSI calculator extracts features internally
        'feature_shape': '(T, 33, 3)',
        'target_fps': target_fps,
        'generation_date': datetime.now().isoformat(),
        'num_videos_per_class': {},
        'quality_metrics': {}
    }
    
    if not os.path.exists(cfg['raw_path']):
        print("No expert data found.")
        return
    
    print(f"\n{'='*70}")
    print(f"EXPERT TEMPLATE GENERATION (KSI v2.0 Enhanced)")
    print(f"{'='*70}")
    print(f"Using same preprocessing as hybrid pipeline:")
    print(f"  • Segment rules: {segment_rules is not None}")
    print(f"  • Crop overrides: {crop_overrides is not None}")
    print(f"  • Base crop: top={base_crop_cfg.get('top', 0)}, bottom={base_crop_cfg.get('bottom', 0)}")
    print(f"{'='*70}\n")
    
    for cls in sorted(os.listdir(cfg['raw_path'])):
        cls_path = os.path.join(cfg['raw_path'], cls)
        if not os.path.isdir(cls_path):
            continue
        
        print(f"📊 Processing class: {cls}")
        
        # Step 1: Load all videos with same preprocessing as hybrid pipeline
        video_data = []
        video_files = sorted([v for v in os.listdir(cls_path) if v.endswith(('.mp4', '.avi', '.mov'))])
        
        for vid in video_files:
            vid_path = os.path.join(cls_path, vid)
            
            # Extract landmarks with full preprocessing logic
            lms, fps = extract_landmarks_with_preprocessing(
                vid_path,
                extractor,
                base_crop_cfg,
                segment_rules,
                crop_overrides
            )
            
            if lms is not None and len(lms) > 0:
                # Extract 32 enhanced features (for quality filtering and alignment)
                features = np.array([extract_enhanced_features(f) for f in lms])
                
                # Temporal resampling to target FPS
                features_resampled = temporal_resample(features, fps, target_fps)
                lms_resampled = temporal_resample(lms, fps, target_fps)
                
                video_data.append({
                    'filename': vid,
                    'original_fps': fps,
                    'features': features_resampled,  # For alignment
                    'landmarks': lms_resampled  # Raw landmarks (T, 33, 3) - stored in template
                })
        
        if not video_data:
            print(f"  ⚠ No valid videos found for class '{cls}'\n")
            continue
        
        print(f"  Loaded {len(video_data)} videos (FPS range: {min(v['original_fps'] for v in video_data):.1f}-{max(v['original_fps'] for v in video_data):.1f})")
        
        # Step 2: Quality filtering
        filtered_data = filter_quality(video_data)
        
        if not filtered_data:
            print(f"  ⚠ All videos filtered out due to low quality\n")
            continue
        
        print(f"  Quality filtered: {len(filtered_data)}/{len(video_data)} videos retained")
        quality_scores_str = [f"{v['quality_score']:.3f}" for v in filtered_data[:5]]
        print(f"  Quality scores: {quality_scores_str}")
        
        # Step 3: Find best reference video
        sequences = [v['features'] for v in filtered_data]  # Use features for alignment
        landmarks_list = [v['landmarks'] for v in filtered_data]  # Raw landmarks to store
        ref_idx = find_best_reference(sequences, ksi_calc)
        ref_features = sequences[ref_idx]
        ref_landmarks = landmarks_list[ref_idx]
        
        print(f"  Reference: video {ref_idx+1}/{len(sequences)} ('{filtered_data[ref_idx]['filename']}')") 
        
        # Step 4: DTW alignment (align both features and landmarks)
        print(f"  Aligning {len(sequences)} sequences...")
        aligned_features = [ref_features]
        aligned_landmarks = [ref_landmarks]
        quality_scores = [filtered_data[ref_idx]['quality_score']]
        
        for i, (seq, lms, vid_data) in enumerate(zip(sequences, landmarks_list, filtered_data)):
            if i == ref_idx:
                continue
            
            try:
                # Align features using optimized DTW
                _, aligned_feat, _ = dynamic_time_warping_optimized(ref_features, seq)
                
                # Align landmarks using same DTW path
                aligned_lm = dtw_align_landmarks(ref_landmarks, lms, ref_features, seq)
                
                # Ensure same length as reference
                if len(aligned_feat) != len(ref_features):
                    # Resample features
                    old_indices = np.linspace(0, 1, len(aligned_feat))
                    new_indices = np.linspace(0, 1, len(ref_features))
                    resampled_feat = np.array([
                        interp1d(old_indices, aligned_feat[:, j], kind='linear')(new_indices)
                        for j in range(aligned_feat.shape[1])
                    ]).T
                    aligned_features.append(resampled_feat)
                    
                    # Resample landmarks
                    resampled_lm = temporal_resample_landmarks(aligned_lm, len(ref_landmarks))
                    aligned_landmarks.append(resampled_lm)
                else:
                    aligned_features.append(aligned_feat)
                    aligned_landmarks.append(aligned_lm)
                quality_scores.append(vid_data['quality_score'])
            except Exception as e:
                print(f"    ⚠ Alignment failed for video {i+1}: {e}")
                continue
        
        # Step 5: Weighted averaging of RAW LANDMARKS (not features)
        template_avg = weighted_average_landmarks(aligned_landmarks, quality_scores)
        templates[cls] = template_avg
        
        print(f"  ✓ Main template shape: {template_avg.shape} (raw landmarks)")
        
        # Step 6: Store top-3 variants (if enough videos)
        if len(aligned_landmarks) >= 3:
            templates[f'{cls}_variant1'] = aligned_landmarks[0]  # Reference (best)
            templates[f'{cls}_variant2'] = aligned_landmarks[len(aligned_landmarks) // 2]  # Middle quality
            templates[f'{cls}_variant3'] = aligned_landmarks[-1]  # Different style
            print(f"  ✓ Stored 3 template variants")
        
        # Step 7: Phase-specific template (contact) - use raw landmarks
        contact_template = extract_phase_specific_landmarks(
            aligned_landmarks,
            target_fps
        )
        
        if contact_template is not None:
            templates[f'{cls}_contact'] = contact_template
            print(f"  ✓ Contact-phase template: {contact_template.shape}")
        
        # Store metadata
        metadata['num_videos_per_class'][cls] = len(filtered_data)
        metadata['quality_metrics'][cls] = {
            'mean_quality': float(np.mean(quality_scores)),
            'best_quality': float(np.max(quality_scores)),
            'fps_range': [float(min(v['original_fps'] for v in video_data)), 
                         float(max(v['original_fps'] for v in video_data))]
        }
        
        print(f"  ✅ Completed '{cls}'\n")
    
    # Save templates with metadata
    if templates:
        # Store metadata as arrays (npz limitation)
        templates['_metadata_json'] = np.array([str(metadata)])
        
        np.savez_compressed(cfg['output_path'], **templates)
        
        print(f"\n{'='*70}")
        print(f"✅ SAVED {len([k for k in templates.keys() if not k.startswith('_')])} TEMPLATES")
        print(f"{'='*70}")
        print(f"Output: {cfg['output_path']}")
        print(f"Format: Raw landmarks (T, 33, 3) - KSI calculator extracts features internally")
        print(f"Normalized FPS: {target_fps}")
        print(f"\nTemplate summary:")
        for cls in sorted(set(k.split('_')[0] for k in templates.keys() if not k.startswith('_'))):
            variants = [k for k in templates.keys() if k.startswith(cls)]
            print(f"  • {cls}: {len(variants)} templates ({', '.join([k.replace(cls+'_', '') if '_' in k else 'main' for k in sorted(variants)])})")
        print(f"{'='*70}\n")
    else:
        print("\n❌ No templates generated - no expert videos found!")


if __name__ == "__main__":
    main()