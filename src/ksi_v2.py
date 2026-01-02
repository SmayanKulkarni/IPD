"""
KSI v2.0 - Enhanced Kinematic Similarity Index
==============================================
Research-grade biomechanical analysis with:
- 32 hierarchical features (vs original 12)
- Temporal attention weighting
- Per-joint confidence intervals
- Phase-aware comparison
- Derivative analysis (velocity, acceleration, jerk)

Author: IPD Research Team
Version: 2.0.0
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ShotPhase(Enum):
    PREPARATION = "preparation"
    LOADING = "loading"
    ACCELERATION = "acceleration"
    CONTACT = "contact"
    FOLLOW_THROUGH = "follow_through"


@dataclass
class JointError:
    """Structured representation of per-joint error analysis."""
    joint_name: str
    mean_error: float
    max_error: float
    std_error: float
    critical_frame: int
    critical_phase: ShotPhase
    error_trajectory: np.ndarray
    confidence_interval: Tuple[float, float]


@dataclass
class KSIResult:
    """Complete KSI analysis result."""
    ksi_total: float
    ksi_weighted: float  # Attention-weighted version
    components: Dict[str, float]
    per_joint_errors: Dict[str, JointError]
    phase_scores: Dict[str, float]
    velocity_analysis: Dict
    temporal_analysis: Dict
    confidence: Dict
    recommendations: List[str]


# =============================================================================
# FEATURE EXTRACTION - 32 Hierarchical Biomechanical Features
# =============================================================================

MEDIAPIPE_LANDMARKS = {
    'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
    'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10,
    'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16, 'left_pinky': 17, 'right_pinky': 18,
    'left_index': 19, 'right_index': 20, 'left_thumb': 21, 'right_thumb': 22,
    'left_hip': 23, 'right_hip': 24, 'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28, 'left_heel': 29, 'right_heel': 30,
    'left_foot_index': 31, 'right_foot_index': 32
}

# Indices for quick access
L_S, R_S = 11, 12  # Shoulders
L_E, R_E = 13, 14  # Elbows
L_W, R_W = 15, 16  # Wrists
L_H, R_H = 23, 24  # Hips
L_K, R_K = 25, 26  # Knees
L_A, R_A = 27, 28  # Ankles

# Feature names for interpretability
FEATURE_NAMES = [
    # Joint Angles (12)
    'left_elbow_angle', 'right_elbow_angle',
    'left_shoulder_elevation', 'right_shoulder_elevation',
    'left_shoulder_abduction', 'right_shoulder_abduction',
    'left_knee_angle', 'right_knee_angle',
    'left_hip_flexion', 'right_hip_flexion',
    'left_ankle_angle', 'right_ankle_angle',
    
    # Limb Ratios (6)
    'left_arm_extension', 'right_arm_extension',
    'left_leg_extension', 'right_leg_extension',
    'left_forearm_ratio', 'right_forearm_ratio',
    
    # Spinal Alignment (4)
    'spine_forward_lean', 'spine_lateral_lean',
    'spine_twist_angle', 'spine_length_normalized',
    
    # Rotational Dynamics (4)
    'hip_shoulder_separation', 'hip_rotation_angle',
    'shoulder_rotation_angle', 'trunk_rotation_velocity',
    
    # Wrist Dynamics (3)
    'wrist_height_normalized', 'wrist_horizontal_reach',
    'wrist_radial_deviation',
    
    # Center of Mass (3)
    'com_x_normalized', 'com_y_normalized', 'com_z_normalized',
]


def calculate_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculate angle at point B formed by points A-B-C in 3D space.
    Returns angle in degrees.
    """
    v1 = a - b
    v2 = c - b
    
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product < 1e-8:
        return 0.0
    
    cosine_angle = np.dot(v1, v2) / norm_product
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def calculate_dihedral_angle(p1: np.ndarray, p2: np.ndarray, 
                             p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Calculate dihedral (torsion) angle between planes P1-P2-P3 and P2-P3-P4.
    Used for measuring rotational alignment (e.g., hip-shoulder separation).
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    
    if n1_norm < 1e-8 or n2_norm < 1e-8:
        return 0.0
    
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    
    return np.degrees(np.arctan2(y, x))


def extract_enhanced_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Extract 32 hierarchical biomechanical features from a single frame.
    
    Args:
        landmarks: (33, 3) array of MediaPipe pose landmarks
    
    Returns:
        (32,) array of normalized features
    """
    p = lambda i: landmarks[i]
    
    # Reference points
    mid_shoulder = (p(L_S) + p(R_S)) / 2
    mid_hip = (p(L_H) + p(R_H)) / 2
    torso_len = np.linalg.norm(mid_shoulder - mid_hip) + 1e-6
    
    features = []
    
    # === JOINT ANGLES (12) ===
    # Elbow angles
    features.append(calculate_angle_3d(p(L_S), p(L_E), p(L_W)))
    features.append(calculate_angle_3d(p(R_S), p(R_E), p(R_W)))
    
    # Shoulder elevation (angle from vertical)
    features.append(calculate_angle_3d(mid_hip, p(L_S), p(L_E)))
    features.append(calculate_angle_3d(mid_hip, p(R_S), p(R_E)))
    
    # Shoulder abduction (angle from body midline)
    features.append(calculate_angle_3d(p(R_S), p(L_S), p(L_E)))
    features.append(calculate_angle_3d(p(L_S), p(R_S), p(R_E)))
    
    # Knee angles
    features.append(calculate_angle_3d(p(L_H), p(L_K), p(L_A)))
    features.append(calculate_angle_3d(p(R_H), p(R_K), p(R_A)))
    
    # Hip flexion
    features.append(calculate_angle_3d(p(L_K), p(L_H), mid_shoulder))
    features.append(calculate_angle_3d(p(R_K), p(R_H), mid_shoulder))
    
    # Ankle angles
    features.append(calculate_angle_3d(p(L_K), p(L_A), p(31)))  # left_foot_index
    features.append(calculate_angle_3d(p(R_K), p(R_A), p(32)))  # right_foot_index
    
    # === LIMB RATIOS (6) ===
    # Full arm extension (shoulder to wrist)
    features.append(np.linalg.norm(p(L_S) - p(L_W)) / torso_len)
    features.append(np.linalg.norm(p(R_S) - p(R_W)) / torso_len)
    
    # Full leg extension (hip to ankle)
    features.append(np.linalg.norm(p(L_H) - p(L_A)) / torso_len)
    features.append(np.linalg.norm(p(R_H) - p(R_A)) / torso_len)
    
    # Forearm to upper arm ratio
    left_upper = np.linalg.norm(p(L_S) - p(L_E)) + 1e-6
    left_fore = np.linalg.norm(p(L_E) - p(L_W)) + 1e-6
    features.append(left_fore / left_upper)
    
    right_upper = np.linalg.norm(p(R_S) - p(R_E)) + 1e-6
    right_fore = np.linalg.norm(p(R_E) - p(R_W)) + 1e-6
    features.append(right_fore / right_upper)
    
    # === SPINAL ALIGNMENT (4) ===
    spine_vector = mid_shoulder - mid_hip
    spine_len = np.linalg.norm(spine_vector) + 1e-6
    
    # Forward lean (sagittal plane)
    features.append(np.degrees(np.arctan2(spine_vector[2], spine_vector[1])))
    
    # Lateral lean (frontal plane)
    features.append(np.degrees(np.arctan2(spine_vector[0], spine_vector[1])))
    
    # Spine twist (transverse plane)
    features.append(calculate_angle_3d(p(L_S), mid_shoulder, p(R_S)))
    
    # Normalized spine length
    features.append(spine_len)
    
    # === ROTATIONAL DYNAMICS (4) ===
    # Hip-shoulder separation (key for power generation)
    hip_vector = p(R_H) - p(L_H)
    shoulder_vector = p(R_S) - p(L_S)
    
    hip_angle = np.degrees(np.arctan2(hip_vector[2], hip_vector[0]))
    shoulder_angle = np.degrees(np.arctan2(shoulder_vector[2], shoulder_vector[0]))
    features.append(shoulder_angle - hip_angle)  # Separation angle
    
    features.append(hip_angle)
    features.append(shoulder_angle)
    
    # Trunk rotation velocity proxy (will be computed across frames)
    features.append(0.0)  # Placeholder - computed in sequence analysis
    
    # === WRIST DYNAMICS (3) ===
    # Wrist height relative to body
    features.append((p(R_W)[1] - mid_hip[1]) / torso_len)
    
    # Horizontal reach
    features.append(np.linalg.norm(p(R_W)[:2] - mid_shoulder[:2]) / torso_len)
    
    # Radial deviation (wrist angle relative to forearm)
    forearm_vec = p(R_W) - p(R_E)
    hand_vec = p(20) - p(R_W)  # right_index - right_wrist
    features.append(calculate_angle_3d(p(R_E), p(R_W), p(20)) - 180)
    
    # === CENTER OF MASS (3) ===
    # Approximate CoM as weighted average of major body segments
    com = (mid_hip * 0.5 + mid_shoulder * 0.3 + 
           (p(L_K) + p(R_K)) / 2 * 0.2)
    
    features.append((com[0] - mid_hip[0]) / torso_len)
    features.append((com[1] - mid_hip[1]) / torso_len)
    features.append((com[2] - mid_hip[2]) / torso_len)
    
    return np.array(features)


def extract_sequence_features(landmarks_sequence: np.ndarray) -> np.ndarray:
    """
    Extract features from entire sequence, including temporal features.
    
    Args:
        landmarks_sequence: (T, 33, 3) array
    
    Returns:
        (T, 32) array of features
    """
    features = np.array([extract_enhanced_features(frame) for frame in landmarks_sequence])
    
    # Compute trunk rotation velocity (feature index 25)
    hip_shoulder_sep = features[:, 22]  # hip_shoulder_separation
    rotation_velocity = np.gradient(hip_shoulder_sep)
    rotation_velocity_smooth = gaussian_filter1d(rotation_velocity, sigma=2)
    features[:, 25] = rotation_velocity_smooth
    
    return features


# =============================================================================
# DYNAMIC TIME WARPING - Enhanced with Sakoe-Chiba Band
# =============================================================================

def dynamic_time_warping_optimized(seq1: np.ndarray, seq2: np.ndarray, 
                                   band_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Optimized DTW with Sakoe-Chiba band constraint.
    
    Args:
        seq1: Reference sequence (expert)
        seq2: Query sequence (user)
        band_ratio: Constraint band as ratio of sequence length
    
    Returns:
        aligned_seq1, aligned_seq2, dtw_distance
    """
    n, m = len(seq1), len(seq2)
    band = max(int(band_ratio * max(n, m)), abs(n - m))
    
    # Initialize cost matrix with infinity
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    
    # Fill cost matrix with band constraint
    for i in range(1, n + 1):
        j_start = max(1, i - band)
        j_end = min(m + 1, i + band + 1)
        
        for j in range(j_start, j_end):
            cost = np.linalg.norm(seq1[i-1] - seq2[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    # Backtrack to find optimal path
    path = []
    i, j = n, m
    
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        
        candidates = [
            (dtw[i-1, j], i-1, j),
            (dtw[i, j-1], i, j-1),
            (dtw[i-1, j-1], i-1, j-1)
        ]
        
        best = min(candidates, key=lambda x: x[0])
        i, j = best[1], best[2]
    
    path.reverse()
    
    if not path:
        return seq1, seq2, float('inf')
    
    idx1, idx2 = zip(*path)
    aligned_seq1 = seq1[list(idx1)]
    aligned_seq2 = seq2[list(idx2)]
    
    dtw_distance = dtw[n, m]
    
    return aligned_seq1, aligned_seq2, dtw_distance


# =============================================================================
# SHOT PHASE SEGMENTATION - Automatic Detection
# =============================================================================

class ShotPhaseSegmenter:
    """
    Automatically segments shot into biomechanically meaningful phases.
    Uses velocity profiles and key pose markers.
    """
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps
    
    def segment(self, landmarks_sequence: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """
        Detect shot phases from landmarks sequence.
        
        Returns:
            Dictionary mapping phase names to (start_frame, end_frame) tuples
        """
        # Extract wrist trajectory (racket arm)
        wrist_positions = landmarks_sequence[:, R_W, :]
        
        # Compute velocity magnitude
        velocity = np.linalg.norm(np.diff(wrist_positions, axis=0), axis=1)
        velocity = np.concatenate([[0], velocity])  # Pad to match length
        velocity_smooth = gaussian_filter1d(velocity, sigma=2)
        
        n_frames = len(landmarks_sequence)
        
        # Find peak velocity (impact moment)
        peaks, properties = find_peaks(velocity_smooth, height=np.max(velocity_smooth) * 0.5)
        peak_velocity_frame = peaks[np.argmax(properties['peak_heights'])] if len(peaks) > 0 else n_frames // 2
        
        # Find backswing endpoint (minimum forward position before peak)
        wrist_z = wrist_positions[:peak_velocity_frame, 2]  # Forward/backward
        backswing_end = np.argmin(wrist_z) if len(wrist_z) > 0 else peak_velocity_frame // 2
        
        # Find hip rotation onset
        hip_rotation_start = self._detect_hip_rotation_onset(
            landmarks_sequence[:peak_velocity_frame]
        )
        
        # Define contact window (peak ± uncertainty)
        contact_window = max(2, int(0.05 * self.fps))  # ~50ms window
        contact_start = max(0, peak_velocity_frame - contact_window)
        contact_end = min(n_frames, peak_velocity_frame + contact_window)
        
        # Find motion end (velocity drops to 10% of peak)
        motion_end = n_frames - 1
        threshold = 0.1 * np.max(velocity_smooth)
        for i in range(peak_velocity_frame, n_frames):
            if velocity_smooth[i] < threshold:
                motion_end = i
                break
        
        phases = {
            ShotPhase.PREPARATION.value: (0, backswing_end),
            ShotPhase.LOADING.value: (backswing_end, hip_rotation_start),
            ShotPhase.ACCELERATION.value: (hip_rotation_start, contact_start),
            ShotPhase.CONTACT.value: (contact_start, contact_end),
            ShotPhase.FOLLOW_THROUGH.value: (contact_end, motion_end)
        }
        
        return phases
    
    def _detect_hip_rotation_onset(self, landmarks_sequence: np.ndarray) -> int:
        """
        Detect when hip rotation begins (initiation of power phase).
        """
        if len(landmarks_sequence) < 3:
            return len(landmarks_sequence) // 2
        
        # Calculate hip rotation angle across frames
        hip_angles = []
        for frame in landmarks_sequence:
            hip_vector = frame[R_H] - frame[L_H]
            hip_angle = np.arctan2(hip_vector[2], hip_vector[0])
            hip_angles.append(hip_angle)
        
        # Find significant angular velocity change
        angular_velocity = np.abs(np.gradient(hip_angles))
        angular_velocity_smooth = gaussian_filter1d(angular_velocity, sigma=2)
        
        threshold = np.mean(angular_velocity_smooth) + 1.5 * np.std(angular_velocity_smooth)
        
        rotation_frames = np.where(angular_velocity_smooth > threshold)[0]
        
        return rotation_frames[0] if len(rotation_frames) > 0 else len(landmarks_sequence) // 2


# =============================================================================
# TEMPORAL ATTENTION MECHANISM
# =============================================================================

class TemporalAttention:
    """
    Attention-based weighting for critical shot phases.
    Contact and acceleration phases receive higher weights.
    """
    
    def __init__(self):
        # Phase importance weights (learned from expert analysis)
        self.phase_weights = {
            ShotPhase.PREPARATION.value: 0.8,
            ShotPhase.LOADING.value: 1.2,
            ShotPhase.ACCELERATION.value: 2.0,
            ShotPhase.CONTACT.value: 3.0,  # Most critical
            ShotPhase.FOLLOW_THROUGH.value: 1.0
        }
    
    def compute_attention_weights(self, sequence_length: int, 
                                  phases: Dict[str, Tuple[int, int]]) -> np.ndarray:
        """
        Compute per-frame attention weights based on shot phases.
        """
        weights = np.ones(sequence_length)
        
        for phase_name, (start, end) in phases.items():
            if start < end and end <= sequence_length:
                phase_weight = self.phase_weights.get(phase_name, 1.0)
                
                # Apply Gaussian-shaped weight within phase
                phase_center = (start + end) / 2
                phase_width = (end - start) / 4 + 1
                
                for i in range(start, min(end, sequence_length)):
                    gaussian_weight = np.exp(-0.5 * ((i - phase_center) / phase_width) ** 2)
                    weights[i] = phase_weight * (0.5 + 0.5 * gaussian_weight)
        
        # Normalize to maintain scale
        weights = weights * sequence_length / np.sum(weights)
        
        return weights


# =============================================================================
# CONFIDENCE FILTERING
# =============================================================================

class LandmarkConfidenceFilter:
    """
    Filter unreliable frames based on detection confidence and biomechanical plausibility.
    """
    
    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        
        # Expected limb length ratios (from anthropometric data)
        self.expected_ratios = {
            'upper_arm': (0.15, 0.25),  # As fraction of height
            'forearm': (0.13, 0.22),
            'thigh': (0.20, 0.30),
            'shin': (0.18, 0.28)
        }
    
    def filter_sequence(self, landmarks_sequence: np.ndarray,
                       visibility_scores: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[int]]:
        """
        Filter frames based on confidence and plausibility.
        
        Returns:
            filtered_sequence, valid_indices
        """
        valid_frames = []
        valid_indices = []
        
        prev_frame = None
        
        for i, frame in enumerate(landmarks_sequence):
            # Check visibility if available
            if visibility_scores is not None:
                if np.mean(visibility_scores[i]) < self.min_confidence:
                    continue
            
            # Check biomechanical plausibility
            if not self._is_plausible(frame):
                continue
            
            # Check temporal consistency (no teleportation)
            if prev_frame is not None:
                max_displacement = np.max(np.linalg.norm(frame - prev_frame, axis=1))
                if max_displacement > 0.3:  # 30cm sudden jump = artifact
                    continue
            
            valid_frames.append(frame)
            valid_indices.append(i)
            prev_frame = frame
        
        return np.array(valid_frames) if valid_frames else landmarks_sequence, valid_indices
    
    def _is_plausible(self, landmarks: np.ndarray) -> bool:
        """
        Check if pose is anatomically plausible.
        """
        # Check limb length consistency
        left_upper = np.linalg.norm(landmarks[L_S] - landmarks[L_E])
        left_fore = np.linalg.norm(landmarks[L_E] - landmarks[L_W])
        
        if left_upper > 0 and left_fore > 0:
            ratio = left_fore / left_upper
            if ratio < 0.5 or ratio > 2.0:
                return False
        
        # Check for self-intersection (left hand shouldn't be far right of right shoulder, etc.)
        if landmarks[L_W][0] > landmarks[R_S][0] + 0.3:
            return False
        
        return True


# =============================================================================
# MAIN KSI CALCULATION ENGINE
# =============================================================================

class EnhancedKSI:
    """
    Enhanced Kinematic Similarity Index Calculator.
    Combines all advanced components for research-grade analysis.
    """
    
    def __init__(self, fps: float = 30.0, contact_window_pre_frames: int = 18,
                 contact_window_post_frames: int = 18, bootstrap_min: int = 50,
                 bootstrap_max: int = 200, ranking_margin: float = 0.05):
        self.fps = fps
        self.segmenter = ShotPhaseSegmenter(fps)
        self.attention = TemporalAttention()
        self.confidence_filter = LandmarkConfidenceFilter()
        self.contact_window_pre = contact_window_pre_frames
        self.contact_window_post = contact_window_post_frames
        self.bootstrap_min = bootstrap_min
        self.bootstrap_max = bootstrap_max
        self.ranking_margin = ranking_margin
    
    def calculate(self, expert_landmarks: np.ndarray, 
                 user_landmarks: np.ndarray,
                 weights: Dict[str, float],
                 expert_visibility: Optional[np.ndarray] = None,
                 user_visibility: Optional[np.ndarray] = None,
                 baseline_ksi: Optional[float] = None) -> KSIResult:
        """
        Calculate comprehensive KSI analysis.
        
        Args:
            expert_landmarks: (T1, 33, 3) expert pose sequence
            user_landmarks: (T2, 33, 3) user pose sequence
            weights: Component weights {'pose': w1, 'velocity': w2, 'acceleration': w3}
            expert_visibility: Optional visibility scores
            user_visibility: Optional visibility scores
        
        Returns:
            KSIResult with complete analysis
        """
        # Step 1: Filter unreliable frames
        expert_filtered, expert_valid_idx = self.confidence_filter.filter_sequence(
            expert_landmarks, expert_visibility
        )
        user_filtered, user_valid_idx = self.confidence_filter.filter_sequence(
            user_landmarks, user_visibility
        )
        
        if len(expert_filtered) < 10 or len(user_filtered) < 10:
            return self._empty_result("Insufficient valid frames for analysis")
        
        # Step 2: Segment shot phases (pre-window) to locate contact
        expert_phases = self.segmenter.segment(expert_filtered)
        user_phases = self.segmenter.segment(user_filtered)
        
        # Step 3: Apply contact-centered windowing
        expert_windowed, expert_phases = self._apply_contact_window(expert_filtered, expert_phases)
        user_windowed, user_phases = self._apply_contact_window(user_filtered, user_phases)
        
        if len(expert_windowed) < 5 or len(user_windowed) < 5:
            return self._empty_result("Insufficient frames after contact windowing")
        
        # Step 4: Extract features on windowed sequences
        expert_features = extract_sequence_features(expert_windowed)
        user_features = extract_sequence_features(user_windowed)
        
        # Step 5: DTW alignment
        expert_aligned, user_aligned, dtw_distance = dynamic_time_warping_optimized(
            expert_features, user_features
        )
        
        # Step 6: Compute attention weights
        attention_weights = self.attention.compute_attention_weights(
            len(expert_aligned), expert_phases
        )
        
        # Step 7: Calculate component scores
        s_pose = self._calculate_pose_similarity(expert_aligned, user_aligned, attention_weights)
        s_velocity = self._calculate_velocity_similarity(expert_aligned, user_aligned, attention_weights)
        s_acceleration = self._calculate_acceleration_similarity(expert_aligned, user_aligned, attention_weights)
        s_jerk = self._calculate_jerk_similarity(expert_aligned, user_aligned, attention_weights)
        
        # Step 8: Per-joint error analysis
        per_joint_errors = self._analyze_per_joint_errors(
            expert_aligned, user_aligned, attention_weights, expert_phases
        )
        
        # Step 9: Phase-specific scores
        phase_scores = self._calculate_phase_scores(
            expert_aligned, user_aligned, expert_phases
        )
        
        # Step 10: Velocity analysis
        velocity_analysis = self._analyze_velocity_profile(
            expert_aligned, user_aligned, expert_phases
        )
        
        # Step 11: Confidence estimation
        confidence = self._estimate_confidence(
            expert_aligned, user_aligned, len(user_valid_idx) / len(user_landmarks)
        )
        
        # Step 12: Weighted KSI total
        ksi_total = (
            weights['pose'] * s_pose +
            weights['velocity'] * s_velocity +
            weights['acceleration'] * s_acceleration
        )
        
        # Attention-weighted version (for research comparison)
        ksi_weighted = self._calculate_weighted_ksi(
            expert_aligned, user_aligned, weights, attention_weights
        )

        ranking_hinge = 0.0
        if baseline_ksi is not None:
            ranking_hinge = self._ranking_hinge_loss(ksi_total, baseline_ksi)
        
        # Generate initial recommendations
        recommendations = self._generate_recommendations(per_joint_errors, phase_scores, velocity_analysis)
        
        return KSIResult(
            ksi_total=float(ksi_total),
            ksi_weighted=float(ksi_weighted),
            components={
                'pose': float(s_pose),
                'velocity': float(s_velocity),
                'acceleration': float(s_acceleration),
                'jerk': float(s_jerk),
                'dtw_distance': float(dtw_distance),
                'ranking_hinge': float(ranking_hinge)
            },
            per_joint_errors=per_joint_errors,
            phase_scores=phase_scores,
            velocity_analysis=velocity_analysis,
            temporal_analysis={
                'attention_weights': attention_weights.tolist(),
                'expert_phases': {k: v for k, v in expert_phases.items()},
                'user_phases': {k: v for k, v in user_phases.items()},
                'alignment_ratio': len(expert_aligned) / len(expert_features)
            },
            confidence=confidence,
            recommendations=recommendations
        )

    def _apply_contact_window(self, landmarks_sequence: np.ndarray,
                              phases: Dict[str, Tuple[int, int]]) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
        """
        Focus analysis on a tight window around contact to reduce noise and compute load.
        """
        n_frames = len(landmarks_sequence)
        if n_frames == 0:
            return landmarks_sequence, phases

        contact_phase = phases.get(ShotPhase.CONTACT.value)
        if contact_phase and contact_phase[0] < contact_phase[1]:
            contact_center = (contact_phase[0] + contact_phase[1]) // 2
        else:
            contact_center = n_frames // 2
        
        start = max(0, contact_center - self.contact_window_pre)
        end = min(n_frames, contact_center + self.contact_window_post)
        if end - start < 5:  # Fallback to avoid too-short slices
            return landmarks_sequence, phases
        
        windowed = landmarks_sequence[start:end]
        adjusted_phases = {}
        for phase_name, (p_start, p_end) in phases.items():
            adj_start = max(0, p_start - start)
            adj_end = max(0, p_end - start)
            adj_start = min(len(windowed), adj_start)
            adj_end = min(len(windowed), adj_end)
            if adj_start < adj_end:
                adjusted_phases[phase_name] = (adj_start, adj_end)
        
        if ShotPhase.CONTACT.value not in adjusted_phases:
            center = len(windowed) // 2
            adjusted_phases[ShotPhase.CONTACT.value] = (max(0, center - 1), min(len(windowed), center + 1))
        
        return windowed, adjusted_phases
    
    def _calculate_pose_similarity(self, expert: np.ndarray, user: np.ndarray,
                                   weights: np.ndarray) -> float:
        """
        Cosine similarity of pose features with attention weighting.
        """
        similarities = []
        
        for i in range(len(expert)):
            e_norm = np.linalg.norm(expert[i])
            u_norm = np.linalg.norm(user[i])
            
            if e_norm > 1e-8 and u_norm > 1e-8:
                sim = np.dot(expert[i], user[i]) / (e_norm * u_norm)
                similarities.append(sim * weights[i])
            else:
                similarities.append(0.0)
        
        return np.sum(similarities) / np.sum(weights)
    
    def _calculate_velocity_similarity(self, expert: np.ndarray, user: np.ndarray,
                                       weights: np.ndarray) -> float:
        """
        Velocity similarity considering both direction and magnitude.
        """
        vel_expert = np.diff(expert, axis=0, prepend=expert[0:1])
        vel_user = np.diff(user, axis=0, prepend=user[0:1])
        
        similarities = []
        
        for i in range(len(vel_expert)):
            e_norm = np.linalg.norm(vel_expert[i])
            u_norm = np.linalg.norm(vel_user[i])
            
            # Direction similarity
            if e_norm > 1e-8 and u_norm > 1e-8:
                dir_sim = np.dot(vel_expert[i], vel_user[i]) / (e_norm * u_norm)
            else:
                dir_sim = 1.0
            
            # Magnitude similarity (Gaussian penalty)
            mag_sim = np.exp(-0.1 * (e_norm - u_norm) ** 2)
            
            similarities.append(dir_sim * mag_sim * weights[i])
        
        return np.sum(similarities) / np.sum(weights)
    
    def _calculate_acceleration_similarity(self, expert: np.ndarray, user: np.ndarray,
                                           weights: np.ndarray) -> float:
        """
        Acceleration similarity for power/explosiveness analysis.
        """
        vel_expert = np.diff(expert, axis=0, prepend=expert[0:1])
        vel_user = np.diff(user, axis=0, prepend=user[0:1])
        
        acc_expert = np.diff(vel_expert, axis=0, prepend=vel_expert[0:1])
        acc_user = np.diff(vel_user, axis=0, prepend=vel_user[0:1])
        
        similarities = []
        
        for i in range(len(acc_expert)):
            e_norm = np.linalg.norm(acc_expert[i])
            u_norm = np.linalg.norm(acc_user[i])
            
            # Magnitude similarity
            sim = np.exp(-0.1 * (e_norm - u_norm) ** 2)
            similarities.append(sim * weights[i])
        
        return np.sum(similarities) / np.sum(weights)
    
    def _calculate_jerk_similarity(self, expert: np.ndarray, user: np.ndarray,
                                   weights: np.ndarray) -> float:
        """
        Jerk (rate of acceleration change) for smoothness analysis.
        Novel metric for motion quality.
        """
        vel_expert = np.diff(expert, axis=0, prepend=expert[0:1])
        vel_user = np.diff(user, axis=0, prepend=user[0:1])
        
        acc_expert = np.diff(vel_expert, axis=0, prepend=vel_expert[0:1])
        acc_user = np.diff(vel_user, axis=0, prepend=vel_user[0:1])
        
        jerk_expert = np.diff(acc_expert, axis=0, prepend=acc_expert[0:1])
        jerk_user = np.diff(acc_user, axis=0, prepend=acc_user[0:1])
        
        # Lower jerk = smoother motion
        expert_smoothness = np.mean([np.linalg.norm(j) for j in jerk_expert])
        user_smoothness = np.mean([np.linalg.norm(j) for j in jerk_user])
        
        # Penalize user if significantly less smooth
        smoothness_ratio = min(expert_smoothness, user_smoothness) / (max(expert_smoothness, user_smoothness) + 1e-8)
        
        return smoothness_ratio
    
    def _analyze_per_joint_errors(self, expert: np.ndarray, user: np.ndarray,
                                  weights: np.ndarray, phases: Dict) -> Dict[str, JointError]:
        """
        Detailed per-joint error analysis.
        """
        errors = {}
        n_frames = len(expert)
        
        for i, feature_name in enumerate(FEATURE_NAMES):
            if i >= expert.shape[1]:
                break
            
            expert_feat = expert[:, i]
            user_feat = user[:, i]
            
            # Error trajectory
            error_trajectory = np.abs(expert_feat - user_feat)
            weighted_error = error_trajectory * weights
            
            # Statistics
            mean_error = np.mean(error_trajectory)
            max_error = np.max(error_trajectory)
            std_error = np.std(error_trajectory)
            
            # Critical frame
            critical_frame = int(np.argmax(weighted_error))
            
            # Identify phase of critical frame
            critical_phase = ShotPhase.CONTACT
            for phase_name, (start, end) in phases.items():
                if start <= critical_frame < end:
                    critical_phase = ShotPhase(phase_name)
                    break
            
            # Adaptive bootstrap confidence interval
            n_bootstrap = min(self.bootstrap_max, max(self.bootstrap_min, len(error_trajectory) * 3))
            bootstrap_errors = []
            for _ in range(n_bootstrap):
                idx = np.random.choice(len(error_trajectory), size=len(error_trajectory), replace=True)
                bootstrap_errors.append(np.mean(error_trajectory[idx]))
            ci_lower = float(np.percentile(bootstrap_errors, 2.5))
            ci_upper = float(np.percentile(bootstrap_errors, 97.5))
            
            errors[feature_name] = JointError(
                joint_name=feature_name,
                mean_error=float(mean_error),
                max_error=float(max_error),
                std_error=float(std_error),
                critical_frame=critical_frame,
                critical_phase=critical_phase,
                error_trajectory=error_trajectory,
                confidence_interval=(ci_lower, ci_upper)
            )
        
        return errors
    
    def _calculate_phase_scores(self, expert: np.ndarray, user: np.ndarray,
                               phases: Dict) -> Dict[str, float]:
        """
        Calculate KSI score for each shot phase.
        """
        phase_scores = {}
        
        for phase_name, (start, end) in phases.items():
            if start >= end or end > len(expert):
                phase_scores[phase_name] = 0.0
                continue
            
            expert_phase = expert[start:end]
            user_phase = user[start:end]
            
            if len(expert_phase) == 0:
                phase_scores[phase_name] = 0.0
                continue
            
            # Simple cosine similarity for phase
            similarities = []
            for i in range(len(expert_phase)):
                e_norm = np.linalg.norm(expert_phase[i])
                u_norm = np.linalg.norm(user_phase[i])
                
                if e_norm > 1e-8 and u_norm > 1e-8:
                    sim = np.dot(expert_phase[i], user_phase[i]) / (e_norm * u_norm)
                    similarities.append(sim)
            
            phase_scores[phase_name] = float(np.mean(similarities)) if similarities else 0.0
        
        return phase_scores
    
    def _analyze_velocity_profile(self, expert: np.ndarray, user: np.ndarray,
                                  phases: Dict) -> Dict:
        """
        Detailed velocity profile analysis.
        """
        vel_expert = np.linalg.norm(np.diff(expert, axis=0), axis=1)
        vel_user = np.linalg.norm(np.diff(user, axis=0), axis=1)
        
        # Smooth
        vel_expert_smooth = gaussian_filter1d(vel_expert, sigma=2)
        vel_user_smooth = gaussian_filter1d(vel_user, sigma=2)
        
        # Find peaks
        expert_peak_frame = int(np.argmax(vel_expert_smooth))
        user_peak_frame = int(np.argmax(vel_user_smooth))
        
        expert_peak_velocity = float(vel_expert_smooth[expert_peak_frame])
        user_peak_velocity = float(vel_user_smooth[user_peak_frame])
        
        return {
            'expert_peak_frame': expert_peak_frame,
            'user_peak_frame': user_peak_frame,
            'timing_offset_frames': user_peak_frame - expert_peak_frame,
            'timing_offset_ms': (user_peak_frame - expert_peak_frame) * 1000 / self.fps,
            'expert_peak_velocity': expert_peak_velocity,
            'user_peak_velocity': user_peak_velocity,
            'peak_velocity_ratio': user_peak_velocity / (expert_peak_velocity + 1e-8),
            'velocity_profile_expert': vel_expert_smooth.tolist(),
            'velocity_profile_user': vel_user_smooth.tolist()
        }
    
    def _estimate_confidence(self, expert: np.ndarray, user: np.ndarray,
                            valid_frame_ratio: float) -> Dict:
        """
        Estimate confidence of KSI calculation.
        """
        # Adaptive bootstrap variance
        n_bootstrap = min(self.bootstrap_max, max(self.bootstrap_min, len(expert) * 2))
        bootstrap_ksi = []
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(expert), size=len(expert), replace=True)
            exp_boot = expert[idx]
            usr_boot = user[idx]
            
            # Quick KSI calculation
            sims = []
            for i in range(len(exp_boot)):
                e_norm = np.linalg.norm(exp_boot[i])
                u_norm = np.linalg.norm(usr_boot[i])
                if e_norm > 1e-8 and u_norm > 1e-8:
                    sims.append(np.dot(exp_boot[i], usr_boot[i]) / (e_norm * u_norm))
            
            if sims:
                bootstrap_ksi.append(np.mean(sims))
        
        mean_ksi = float(np.mean(bootstrap_ksi)) if bootstrap_ksi else 0.0
        std_ksi = float(np.std(bootstrap_ksi)) if bootstrap_ksi else 1.0
        ci_lower = float(np.percentile(bootstrap_ksi, 2.5)) if bootstrap_ksi else 0.0
        ci_upper = float(np.percentile(bootstrap_ksi, 97.5)) if bootstrap_ksi else 1.0
        uncertainty = (ci_upper - ci_lower) / (abs(mean_ksi) + 1e-6)
        
        return {
            'mean': mean_ksi,
            'std': std_ksi,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'uncertainty_scalar': float(uncertainty),
            'valid_frame_ratio': float(valid_frame_ratio),
            'reliable': (std_ksi < 0.1 and uncertainty < 1.0) if bootstrap_ksi else False,
            'n_bootstrap': int(n_bootstrap)
        }
    
    def _calculate_weighted_ksi(self, expert: np.ndarray, user: np.ndarray,
                               weights: Dict, attention: np.ndarray) -> float:
        """
        Calculate attention-weighted KSI.
        """
        frame_ksi = []
        
        for i in range(len(expert)):
            e_norm = np.linalg.norm(expert[i])
            u_norm = np.linalg.norm(user[i])
            
            if e_norm > 1e-8 and u_norm > 1e-8:
                sim = np.dot(expert[i], user[i]) / (e_norm * u_norm)
                frame_ksi.append(sim * attention[i])
            else:
                frame_ksi.append(0.0)
        
        return np.sum(frame_ksi) / np.sum(attention)

    def _ranking_hinge_loss(self, user_score: float, baseline_score: float) -> float:
        """
        Margin-based ranking hinge to compare user vs. baseline/reference.
        Positive when user underperforms the baseline by more than margin.
        """
        gap = user_score - baseline_score
        return float(max(0.0, self.ranking_margin - gap))
    
    def _generate_recommendations(self, errors: Dict[str, JointError],
                                  phase_scores: Dict, velocity: Dict) -> List[str]:
        """
        Generate initial recommendations based on analysis.
        """
        recommendations = []
        
        # Sort errors by magnitude
        sorted_errors = sorted(errors.items(), key=lambda x: x[1].mean_error, reverse=True)
        
        # Top 3 errors
        for name, error in sorted_errors[:3]:
            if error.mean_error > 10:
                recommendations.append(f"CRITICAL: {name} deviation of {error.mean_error:.1f}° during {error.critical_phase.value}")
        
        # Timing issues
        if abs(velocity['timing_offset_frames']) > 3:
            if velocity['timing_offset_frames'] < 0:
                recommendations.append("TIMING: Peak velocity occurs too early - delay forward swing")
            else:
                recommendations.append("TIMING: Peak velocity occurs too late - speed up preparation")
        
        # Power issues
        if velocity['peak_velocity_ratio'] < 0.7:
            recommendations.append(f"POWER: Only achieving {velocity['peak_velocity_ratio']:.0%} of expert speed")
        
        # Phase-specific
        worst_phase = min(phase_scores.items(), key=lambda x: x[1])
        if worst_phase[1] < 0.7:
            recommendations.append(f"PHASE: {worst_phase[0]} phase needs most work (score: {worst_phase[1]:.2f})")
        
        return recommendations
    
    def _empty_result(self, message: str) -> KSIResult:
        """Return empty result with error message."""
        return KSIResult(
            ksi_total=0.0,
            ksi_weighted=0.0,
            components={'pose': 0.0, 'velocity': 0.0, 'acceleration': 0.0, 'jerk': 0.0, 'dtw_distance': float('inf')},
            per_joint_errors={},
            phase_scores={},
            velocity_analysis={},
            temporal_analysis={},
            confidence={'reliable': False, 'error': message},
            recommendations=[message]
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ksi_to_dict(result: KSIResult) -> Dict:
    """
    Convert KSIResult to JSON-serializable dictionary.
    """
    return {
        'ksi_total': result.ksi_total,
        'ksi_weighted': result.ksi_weighted,
        'components': result.components,
        'per_joint_errors': {
            k: {
                'joint_name': v.joint_name,
                'mean_error': v.mean_error,
                'max_error': v.max_error,
                'std_error': v.std_error,
                'critical_frame': v.critical_frame,
                'critical_phase': v.critical_phase.value,
                'confidence_interval': v.confidence_interval
            }
            for k, v in result.per_joint_errors.items()
        },
        'phase_scores': result.phase_scores,
        'velocity_analysis': result.velocity_analysis,
        'temporal_analysis': {
            k: v for k, v in result.temporal_analysis.items()
            if k != 'attention_weights'  # Exclude large arrays
        },
        'confidence': result.confidence,
        'recommendations': result.recommendations
    }
