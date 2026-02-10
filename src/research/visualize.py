# Updated visualize.py
# --- DETERMINISM FIXES (MUST BE BEFORE IMPORTS) ---
import os
import sys

# Check for GPU flag early
_use_gpu = '--gpu' in sys.argv

if not _use_gpu:
    # Force CPU mode for deterministic predictions
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
    print("🔒 Running in CPU mode for deterministic predictions (use --gpu to enable GPU)")

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import cv2
import yaml
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Dict
import os
# Local imports
from utils import normalize_sequence, should_skip_crop, get_segment_bounds, resolve_crop_config_for_video
from features import PoseFeatureExtractor
from ksi_v2 import (
    EnhancedKSI, 
    ShotPhaseSegmenter, 
    extract_sequence_features,
    FEATURE_NAMES,
    ShotPhase
)


def load_config():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def visualize_2d(video_path: str, crop_config: Dict, speed: float = 1.0, segment_rules: Optional[Dict] = None):
    """2D skeleton overlay visualization."""
    print(f"\n--- 2D Visualization: {video_path} ---")
    print("Press 'q' to quit.")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame, tail_frames = get_segment_bounds(video_path, fps, total_frames, default_seconds=1.75, segment_cfg=segment_rules)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    base_delay_ms = max(1, int(1000.0 / fps))

    # Decide whether to skip cropping for files like 'name (1).mp4'
    filename = os.path.basename(video_path)
    skip_crop = should_skip_crop(filename)

    processed = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed += 1
        # Apply Crop (skip if filename matches the '(N)' pattern)
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

        # Detect
        image_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        # Draw
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame_cropped, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        cv2.imshow('Badminton 2D View (Cropped)', frame_cropped)
        delay = max(1, int(base_delay_ms / max(speed, 1e-3)))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

        if processed >= int(tail_frames):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


def visualize_3d(video_path: str, crop_config: Dict, mp_config: Dict, speed: float = 1.0, segment_rules: Optional[Dict] = None):
    """3D skeleton animation with phase annotations."""
    print(f"\n--- 3D Visualization: {video_path} ---")
    print("Extracting landmarks...")
    
    extractor = PoseFeatureExtractor(mp_config)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame, tail_frames = get_segment_bounds(video_path, fps, total_frames, default_seconds=1.75, segment_cfg=segment_rules)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    # Decide whether to skip cropping for files like 'name (1).mp4'
    filename = os.path.basename(video_path)
    skip_crop = should_skip_crop(filename)

    raw_landmarks = []

    processed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed += 1
        
        if not skip_crop:
            h, w = frame.shape[:2]
            frame = frame[
                int(h * crop_config['top']):h - int(h * crop_config['bottom']),
                int(w * crop_config['left']):w - int(w * crop_config['right'])
            ]
            if frame.size == 0:
                continue

        res = extractor.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            lm = np.array([[l.x, l.y, l.z] for l in res.pose_landmarks.landmark])
            raw_landmarks.append(lm)

        if processed >= int(tail_frames):
            break
            
    cap.release()
    
    if not raw_landmarks:
        print("No landmarks found.")
        return

    print(f"Normalizing {len(raw_landmarks)} frames...")
    normalized_seq = normalize_sequence(raw_landmarks)  # [T, 33, 3]
    
    # Segment phases using ksi_v2
    segmenter = ShotPhaseSegmenter(fps=fps)
    phases = segmenter.segment(normalized_seq)
    
    # Phase colors for visualization
    phase_colors = {
        ShotPhase.PREPARATION.value: 'gray',
        ShotPhase.LOADING.value: 'blue',
        ShotPhase.ACCELERATION.value: 'orange',
        ShotPhase.CONTACT.value: 'red',
        ShotPhase.FOLLOW_THROUGH.value: 'green'
    }
    
    def get_phase_for_frame(frame_idx: int) -> str:
        for phase_name, (start, end) in phases.items():
            if start <= frame_idx < end:
                return phase_name
        return ShotPhase.FOLLOW_THROUGH.value

    # Animation Setup
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    pose_connections = mp.solutions.pose.POSE_CONNECTIONS

    def update(frame_idx):
        ax.clear()
        keypoints = normalized_seq[frame_idx]
        current_phase = get_phase_for_frame(frame_idx)
        phase_color = phase_colors.get(current_phase, 'gray')
        
        # Plot Joints
        ax.scatter(keypoints[:, 0], keypoints[:, 2], keypoints[:, 1], 
                   c=phase_color, marker='o', s=30)

        # Plot Bones
        for conn in pose_connections:
            start, end = conn
            ax.plot([keypoints[start, 0], keypoints[end, 0]],
                    [keypoints[start, 2], keypoints[end, 2]],
                    [keypoints[start, 1], keypoints[end, 1]], 
                    color=phase_color, linewidth=2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_title(f"Frame {frame_idx}/{len(normalized_seq)} | Phase: {current_phase.upper()}")
        ax.view_init(elev=20, azim=-60)

    print("Starting Animation Window...")
    print(f"Detected phases: {phases}")
    interval_ms = max(1, int((1000.0 / fps) / max(speed, 1e-3)))
    ani = FuncAnimation(fig, update, frames=len(normalized_seq), interval=interval_ms)
    plt.show()


def visualize_ksi_analysis(user_landmarks: np.ndarray, 
                           expert_landmarks: np.ndarray,
                           fps: float = 30.0,
                           weights: Optional[Dict[str, float]] = None):
    """
    Visualize KSI analysis with phase scores, velocity profiles, and confidence.
    Uses the new ksi_v2 EnhancedKSI with contact-centered windowing.
    """
    if weights is None:
        weights = {'pose': 0.5, 'velocity': 0.3, 'acceleration': 0.2}
    
    # Initialize enhanced KSI calculator with contact-centered windowing
    ksi_calc = EnhancedKSI(
        fps=fps,
        contact_window_pre_frames=18,  # ±18 frames around contact
        contact_window_post_frames=18,
        bootstrap_min=50,
        bootstrap_max=200,
        ranking_margin=0.05
    )
    
    # Calculate KSI
    result = ksi_calc.calculate(expert_landmarks, user_landmarks, weights)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"KSI Analysis | Score: {result.ksi_total:.3f} (Weighted: {result.ksi_weighted:.3f})", 
                 fontsize=14, fontweight='bold')
    
    # 1. Component scores bar chart
    ax1 = axes[0, 0]
    components = ['pose', 'velocity', 'acceleration', 'jerk']
    scores = [result.components.get(c, 0) for c in components]
    colors = ['steelblue', 'coral', 'seagreen', 'orchid']
    ax1.bar(components, scores, color=colors)
    ax1.set_ylabel('Score')
    ax1.set_title('Component Scores')
    ax1.set_ylim([0, 1])
    for i, v in enumerate(scores):
        ax1.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
    
    # 2. Phase scores
    ax2 = axes[0, 1]
    if result.phase_scores:
        phase_names = list(result.phase_scores.keys())
        phase_vals = list(result.phase_scores.values())
        phase_colors = ['gray', 'blue', 'orange', 'red', 'green']
        ax2.barh(phase_names, phase_vals, color=phase_colors[:len(phase_names)])
        ax2.set_xlim([0, 1])
        ax2.set_xlabel('Score')
        ax2.set_title('Phase Scores')
    else:
        ax2.text(0.5, 0.5, 'No phase data', ha='center', va='center')
        ax2.set_title('Phase Scores')
    
    # 3. Velocity profile comparison
    ax3 = axes[0, 2]
    if result.velocity_analysis:
        vel_expert = result.velocity_analysis.get('velocity_profile_expert', [])
        vel_user = result.velocity_analysis.get('velocity_profile_user', [])
        if vel_expert and vel_user:
            ax3.plot(vel_expert, label='Expert', color='green', linewidth=2)
            ax3.plot(vel_user, label='User', color='blue', linewidth=2, alpha=0.7)
            # Mark peaks
            expert_peak = result.velocity_analysis.get('expert_peak_frame', 0)
            user_peak = result.velocity_analysis.get('user_peak_frame', 0)
            if expert_peak < len(vel_expert):
                ax3.axvline(expert_peak, color='green', linestyle='--', alpha=0.5, label='Expert peak')
            if user_peak < len(vel_user):
                ax3.axvline(user_peak, color='blue', linestyle='--', alpha=0.5, label='User peak')
            ax3.legend(fontsize=8)
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Velocity')
        ax3.set_title(f"Velocity Profile (Timing offset: {result.velocity_analysis.get('timing_offset_ms', 0):.0f}ms)")
    else:
        ax3.text(0.5, 0.5, 'No velocity data', ha='center', va='center')
        ax3.set_title('Velocity Profile')
    
    # 4. Top joint errors
    ax4 = axes[1, 0]
    if result.per_joint_errors:
        # Sort by mean error, take top 8
        sorted_errors = sorted(
            result.per_joint_errors.items(),
            key=lambda x: x[1].mean_error,
            reverse=True
        )[:8]
        joint_names = [e[0].replace('_', '\n') for e in sorted_errors]
        mean_errors = [e[1].mean_error for e in sorted_errors]
        ci_lower = [e[1].confidence_interval[0] for e in sorted_errors]
        ci_upper = [e[1].confidence_interval[1] for e in sorted_errors]
        yerr = [[m - l for m, l in zip(mean_errors, ci_lower)],
                [u - m for m, u in zip(mean_errors, ci_upper)]]
        
        ax4.barh(joint_names, mean_errors, xerr=yerr, color='salmon', capsize=3)
        ax4.set_xlabel('Mean Error (degrees/ratio)')
        ax4.set_title('Top Joint Errors (with 95% CI)')
    else:
        ax4.text(0.5, 0.5, 'No error data', ha='center', va='center')
        ax4.set_title('Joint Errors')
    
    # 5. Confidence summary
    ax5 = axes[1, 1]
    if result.confidence:
        conf_items = [
            ('KSI Mean', result.confidence.get('mean', 0)),
            ('KSI Std', result.confidence.get('std', 0)),
            ('CI Lower', result.confidence.get('ci_95_lower', 0)),
            ('CI Upper', result.confidence.get('ci_95_upper', 0)),
            ('Uncertainty', result.confidence.get('uncertainty_scalar', 0)),
            ('Valid Ratio', result.confidence.get('valid_frame_ratio', 0)),
        ]
        labels = [c[0] for c in conf_items]
        values = [c[1] for c in conf_items]
        ax5.barh(labels, values, color='teal')
        ax5.set_title(f"Confidence (Reliable: {result.confidence.get('reliable', False)}, "
                      f"n_bootstrap: {result.confidence.get('n_bootstrap', 'N/A')})")
    else:
        ax5.text(0.5, 0.5, 'No confidence data', ha='center', va='center')
        ax5.set_title('Confidence')
    
    # 6. Recommendations
    ax6 = axes[1, 2]
    ax6.axis('off')
    rec_text = "RECOMMENDATIONS:\n\n"
    for i, rec in enumerate(result.recommendations[:5], 1):
        rec_text += f"{i}. {rec}\n\n"
    ax6.text(0.05, 0.95, rec_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', wrap=True,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax6.set_title('Top Recommendations')
    
    plt.tight_layout()
    plt.savefig('dvclive/ksi_analysis.png', dpi=150)
    print("Saved KSI analysis to dvclive/ksi_analysis.png")
    plt.show()
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Path to video file")
    parser.add_argument("--mode", choices=['2d', '3d', 'both', 'ksi'], default='both')
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (e.g., 0.5 for half speed)")
    parser.add_argument("--user_npz", help="User poses .npz for KSI analysis")
    parser.add_argument("--expert_npz", help="Expert poses .npz for KSI analysis")
    parser.add_argument("--gpu", action='store_true', help="Enable GPU acceleration (less deterministic)")
    args = parser.parse_args()

    params = load_config()
    crop_cfg = params['pose_pipeline']['crop_config']
    crop_overrides = params.get('crop_overrides', {})
    mp_cfg = params['mediapipe']
    segment_rules = params.get('segment_rules', {})

    if args.mode == 'ksi':
        if not args.user_npz or not args.expert_npz:
            print("KSI mode requires --user_npz and --expert_npz")
        else:
            user_data = np.load(args.user_npz)
            expert_data = np.load(args.expert_npz)
            user_lm = user_data['features'] if 'features' in user_data else user_data['arr_0']
            expert_lm = expert_data['features'] if 'features' in expert_data else expert_data['arr_0']
            # Reshape to [T, 33, 3] if needed
            if user_lm.ndim == 2 and user_lm.shape[1] == 99:
                user_lm = user_lm.reshape(-1, 33, 3)
            if expert_lm.ndim == 2 and expert_lm.shape[1] == 99:
                expert_lm = expert_lm.reshape(-1, 33, 3)
            visualize_ksi_analysis(user_lm, expert_lm, fps=params.get('fps', 30.0),
                                   weights=params.get('ksi', {}).get('weights'))
    else:
        if not args.video:
            print("Video mode requires --video")
        else:
            if args.mode in ['2d', 'both']:
                eff_crop_cfg = resolve_crop_config_for_video(args.video, crop_cfg, crop_overrides)
                visualize_2d(args.video, eff_crop_cfg, speed=args.speed, segment_rules=segment_rules)
                if args.mode in ['3d', 'both']:
                    visualize_3d(args.video, eff_crop_cfg, mp_cfg, speed=args.speed, segment_rules=segment_rules)