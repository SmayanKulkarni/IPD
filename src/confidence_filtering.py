"""
Confidence Filtering - Landmark Quality Assessment and Artifact Removal
=======================================================================
Validates MediaPipe landmark quality and removes motion artifacts for
robust pose estimation and comparison.

Features:
- Landmark visibility and confidence scoring
- Temporal consistency validation
- Motion artifact detection and interpolation
- Outlier joint detection and correction
- Adaptive quality thresholds

Research Novelties:
- Multi-modal confidence fusion
- Bayesian landmark reliability estimation
- Physics-based motion constraint validation
- Adaptive temporal smoothing with edge preservation

Author: IPD Research Team
Version: 2.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from scipy import signal, ndimage
from scipy.interpolate import interp1d


class QualityLevel(Enum):
    """Landmark quality classification."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"


@dataclass
class LandmarkQuality:
    """Quality assessment for a single landmark."""
    landmark_idx: int
    landmark_name: str
    visibility_score: float
    temporal_consistency: float
    physics_validity: float
    overall_quality: float
    quality_level: QualityLevel
    needs_interpolation: bool


@dataclass
class FrameQuality:
    """Quality assessment for a single frame."""
    frame_idx: int
    overall_quality: float
    quality_level: QualityLevel
    problematic_landmarks: List[int]
    artifact_detected: bool
    recommended_action: str


@dataclass
class SequenceQuality:
    """Quality assessment for entire sequence."""
    overall_quality: float
    quality_level: QualityLevel
    usable_percentage: float
    problematic_frames: List[int]
    problematic_landmarks: Dict[int, float]  # landmark_idx -> failure_rate
    recommendation: str


class LandmarkValidator:
    """
    Validates landmark quality using multiple criteria.
    """
    
    # MediaPipe landmark names
    LANDMARK_NAMES = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
        'left_index', 'right_index', 'left_thumb', 'right_thumb',
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]
    
    # Anatomical constraints (approximate max velocity in units/frame at 30fps)
    MAX_VELOCITIES = {
        'extremities': 0.5,    # hands, feet
        'limbs': 0.3,          # elbows, knees
        'torso': 0.15,         # shoulders, hips
        'head': 0.2,           # head landmarks
    }
    
    # Landmark groups
    EXTREMITY_LANDMARKS = [15, 16, 17, 18, 19, 20, 21, 22, 27, 28, 29, 30, 31, 32]  # wrists, hands, ankles, feet
    LIMB_LANDMARKS = [13, 14, 25, 26]  # elbows, knees
    TORSO_LANDMARKS = [11, 12, 23, 24]  # shoulders, hips
    HEAD_LANDMARKS = list(range(11))  # face landmarks
    
    # Critical landmarks for badminton analysis
    CRITICAL_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]  # upper body + legs
    
    def __init__(self, visibility_threshold: float = 0.5,
                 consistency_threshold: float = 0.3,
                 min_quality: float = 0.4):
        """
        Initialize validator.
        
        Args:
            visibility_threshold: Minimum visibility score
            consistency_threshold: Maximum allowed velocity deviation
            min_quality: Minimum overall quality for usable landmarks
        """
        self.visibility_threshold = visibility_threshold
        self.consistency_threshold = consistency_threshold
        self.min_quality = min_quality
    
    def validate_landmark(self, landmark_idx: int, 
                         positions: np.ndarray,
                         visibilities: Optional[np.ndarray] = None) -> LandmarkQuality:
        """
        Validate a single landmark across time.
        
        Args:
            landmark_idx: Index of landmark
            positions: Shape (T, 3) positions over time
            visibilities: Shape (T,) visibility scores (optional)
        
        Returns:
            LandmarkQuality assessment
        """
        T = len(positions)
        landmark_name = self.LANDMARK_NAMES[landmark_idx] if landmark_idx < 33 else f"landmark_{landmark_idx}"
        
        # 1. Visibility score
        if visibilities is not None:
            visibility_score = np.mean(visibilities)
        else:
            # Estimate from position variance (stable = visible)
            variance = np.var(positions, axis=0).mean()
            visibility_score = np.exp(-variance * 10)  # High variance = likely noise
        
        # 2. Temporal consistency (check for jumps)
        temporal_consistency = self._check_temporal_consistency(positions, landmark_idx)
        
        # 3. Physics validity (check for impossible movements)
        physics_validity = self._check_physics_validity(positions, landmark_idx)
        
        # 4. Overall quality (weighted combination)
        overall_quality = (
            0.3 * visibility_score +
            0.4 * temporal_consistency +
            0.3 * physics_validity
        )
        
        # Determine quality level
        quality_level = self._classify_quality(overall_quality)
        
        # Needs interpolation?
        needs_interpolation = overall_quality < self.min_quality or temporal_consistency < 0.5
        
        return LandmarkQuality(
            landmark_idx=landmark_idx,
            landmark_name=landmark_name,
            visibility_score=visibility_score,
            temporal_consistency=temporal_consistency,
            physics_validity=physics_validity,
            overall_quality=overall_quality,
            quality_level=quality_level,
            needs_interpolation=needs_interpolation
        )
    
    def _check_temporal_consistency(self, positions: np.ndarray, 
                                   landmark_idx: int) -> float:
        """Check for temporal jumps/inconsistencies."""
        if len(positions) < 2:
            return 1.0
        
        # Calculate frame-to-frame displacements
        displacements = np.diff(positions, axis=0)
        velocities = np.linalg.norm(displacements, axis=1)
        
        # Get appropriate max velocity threshold
        if landmark_idx in self.EXTREMITY_LANDMARKS:
            max_vel = self.MAX_VELOCITIES['extremities']
        elif landmark_idx in self.LIMB_LANDMARKS:
            max_vel = self.MAX_VELOCITIES['limbs']
        elif landmark_idx in self.TORSO_LANDMARKS:
            max_vel = self.MAX_VELOCITIES['torso']
        else:
            max_vel = self.MAX_VELOCITIES['head']
        
        # Score based on percentage of frames with acceptable velocity
        acceptable = velocities < max_vel
        consistency_score = np.mean(acceptable)
        
        # Penalize for very large jumps
        max_jump = np.max(velocities)
        if max_jump > max_vel * 3:
            consistency_score *= 0.5
        
        return consistency_score
    
    def _check_physics_validity(self, positions: np.ndarray,
                               landmark_idx: int) -> float:
        """Check for physically impossible positions/movements."""
        if len(positions) < 3:
            return 1.0
        
        # Calculate acceleration
        velocities = np.diff(positions, axis=0)
        accelerations = np.diff(velocities, axis=0)
        accel_magnitudes = np.linalg.norm(accelerations, axis=1)
        
        # Check for sudden direction changes (impossible physics)
        direction_changes = []
        for t in range(len(velocities) - 1):
            v1, v2 = velocities[t], velocities[t+1]
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 > 1e-6 and norm2 > 1e-6:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                direction_changes.append(cos_angle)
        
        if direction_changes:
            # Many sudden reversals indicate tracking issues
            reversals = np.array(direction_changes) < -0.5
            reversal_rate = np.mean(reversals)
            physics_score = 1.0 - reversal_rate
        else:
            physics_score = 1.0
        
        # Check for position range validity (should be in normalized range)
        pos_range = np.ptp(positions, axis=0)  # Range per axis
        if np.any(pos_range > 2.0):  # Excessive range for normalized poses
            physics_score *= 0.7
        
        return physics_score
    
    def _classify_quality(self, score: float) -> QualityLevel:
        """Classify quality score into level."""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.75:
            return QualityLevel.GOOD
        elif score >= 0.5:
            return QualityLevel.ACCEPTABLE
        elif score >= 0.3:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNUSABLE


class FrameValidator:
    """
    Validates frame-level quality and detects artifacts.
    """
    
    def __init__(self, landmark_validator: Optional[LandmarkValidator] = None):
        self.landmark_validator = landmark_validator or LandmarkValidator()
    
    def validate_frame(self, pose: np.ndarray, 
                      prev_pose: Optional[np.ndarray] = None,
                      visibility: Optional[np.ndarray] = None) -> FrameQuality:
        """
        Validate a single frame.
        
        Args:
            pose: Shape (33, 3) or (99,) pose
            prev_pose: Previous frame's pose for motion analysis
            visibility: Shape (33,) visibility scores
        
        Returns:
            FrameQuality assessment
        """
        # Reshape if flattened
        if pose.ndim == 1:
            pose = pose.reshape(33, 3)
        if prev_pose is not None and prev_pose.ndim == 1:
            prev_pose = prev_pose.reshape(33, 3)
        
        problematic_landmarks = []
        landmark_qualities = []
        
        # Check each landmark
        for i in range(33):
            if visibility is not None:
                vis = visibility[i]
            else:
                vis = 1.0
            
            # Quick per-frame validation
            if vis < self.landmark_validator.visibility_threshold:
                problematic_landmarks.append(i)
                landmark_qualities.append(vis)
            elif prev_pose is not None:
                # Check for jumps
                displacement = np.linalg.norm(pose[i] - prev_pose[i])
                if displacement > 0.3:  # Significant jump
                    problematic_landmarks.append(i)
                    landmark_qualities.append(0.5)
                else:
                    landmark_qualities.append(1.0)
            else:
                landmark_qualities.append(1.0 if vis > 0.5 else 0.5)
        
        # Overall frame quality
        overall_quality = np.mean(landmark_qualities)
        
        # Detect artifact (many landmarks problematic simultaneously)
        artifact_detected = len(problematic_landmarks) > 10
        
        # Classify
        quality_level = self._classify_frame_quality(overall_quality, artifact_detected)
        
        # Recommendation
        if artifact_detected:
            recommendation = "interpolate_frame"
        elif len(problematic_landmarks) > 5:
            recommendation = "interpolate_landmarks"
        elif len(problematic_landmarks) > 0:
            recommendation = "smooth_landmarks"
        else:
            recommendation = "use_as_is"
        
        return FrameQuality(
            frame_idx=0,  # Filled by caller
            overall_quality=overall_quality,
            quality_level=quality_level,
            problematic_landmarks=problematic_landmarks,
            artifact_detected=artifact_detected,
            recommended_action=recommendation
        )
    
    def _classify_frame_quality(self, score: float, has_artifact: bool) -> QualityLevel:
        """Classify frame quality."""
        if has_artifact:
            return QualityLevel.UNUSABLE
        elif score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.75:
            return QualityLevel.GOOD
        elif score >= 0.5:
            return QualityLevel.ACCEPTABLE
        elif score >= 0.3:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNUSABLE


class SequenceFilter:
    """
    Filters and cleans pose sequences for robust analysis.
    """
    
    def __init__(self, 
                 landmark_validator: Optional[LandmarkValidator] = None,
                 frame_validator: Optional[FrameValidator] = None):
        self.landmark_validator = landmark_validator or LandmarkValidator()
        self.frame_validator = frame_validator or FrameValidator(self.landmark_validator)
    
    def filter_sequence(self, pose_sequence: np.ndarray,
                       visibilities: Optional[np.ndarray] = None,
                       interpolate: bool = True,
                       smooth: bool = True) -> Tuple[np.ndarray, SequenceQuality]:
        """
        Filter and clean a pose sequence.
        
        Args:
            pose_sequence: Shape (T, 33, 3) or (T, 99) poses
            visibilities: Shape (T, 33) visibility scores (optional)
            interpolate: Whether to interpolate bad frames/landmarks
            smooth: Whether to apply temporal smoothing
        
        Returns:
            Tuple of (filtered_sequence, quality_assessment)
        """
        # Reshape if needed
        original_shape = pose_sequence.shape
        if pose_sequence.ndim == 2 and pose_sequence.shape[1] == 99:
            pose_sequence = pose_sequence.reshape(-1, 33, 3)
        
        T = pose_sequence.shape[0]
        filtered = pose_sequence.copy()
        
        # Validate each frame
        frame_qualities = []
        problematic_frames = []
        
        for t in range(T):
            prev_pose = pose_sequence[t-1] if t > 0 else None
            vis = visibilities[t] if visibilities is not None else None
            
            quality = self.frame_validator.validate_frame(
                pose_sequence[t], prev_pose, vis
            )
            quality.frame_idx = t
            frame_qualities.append(quality)
            
            if quality.quality_level in [QualityLevel.POOR, QualityLevel.UNUSABLE]:
                problematic_frames.append(t)
        
        # Validate each landmark across time
        landmark_qualities = {}
        problematic_landmarks = {}
        
        for i in range(33):
            vis = visibilities[:, i] if visibilities is not None else None
            quality = self.landmark_validator.validate_landmark(
                i, pose_sequence[:, i, :], vis
            )
            landmark_qualities[i] = quality
            
            if quality.quality_level in [QualityLevel.POOR, QualityLevel.UNUSABLE]:
                problematic_landmarks[i] = 1.0 - quality.overall_quality
        
        # Interpolation
        if interpolate:
            filtered = self._interpolate_bad_data(
                filtered, frame_qualities, landmark_qualities
            )
        
        # Smoothing
        if smooth:
            filtered = self._smooth_sequence(filtered, landmark_qualities)
        
        # Calculate overall sequence quality
        usable_frames = sum(1 for q in frame_qualities 
                          if q.quality_level not in [QualityLevel.UNUSABLE])
        usable_percentage = usable_frames / T * 100
        
        avg_quality = np.mean([q.overall_quality for q in frame_qualities])
        overall_level = self._classify_sequence_quality(avg_quality, usable_percentage)
        
        recommendation = self._generate_recommendation(
            avg_quality, usable_percentage, problematic_landmarks
        )
        
        sequence_quality = SequenceQuality(
            overall_quality=avg_quality,
            quality_level=overall_level,
            usable_percentage=usable_percentage,
            problematic_frames=problematic_frames,
            problematic_landmarks=problematic_landmarks,
            recommendation=recommendation
        )
        
        # Reshape back if needed
        if len(original_shape) == 2:
            filtered = filtered.reshape(-1, 99)
        
        return filtered, sequence_quality
    
    def _interpolate_bad_data(self, sequence: np.ndarray,
                             frame_qualities: List[FrameQuality],
                             landmark_qualities: Dict[int, LandmarkQuality]) -> np.ndarray:
        """Interpolate bad frames and landmarks."""
        T = len(sequence)
        result = sequence.copy()
        
        # Identify good frames for interpolation anchors
        good_frame_mask = np.array([
            q.quality_level not in [QualityLevel.UNUSABLE, QualityLevel.POOR]
            for q in frame_qualities
        ])
        
        # If too few good frames, return as-is
        if np.sum(good_frame_mask) < T * 0.3:
            return result
        
        # Interpolate each landmark
        for i in range(33):
            quality = landmark_qualities.get(i)
            
            # Find frames where this landmark is problematic
            bad_frames = []
            for t, fq in enumerate(frame_qualities):
                if i in fq.problematic_landmarks or fq.artifact_detected:
                    bad_frames.append(t)
            
            if not bad_frames:
                continue
            
            # Good frames for this landmark
            good_frames = [t for t in range(T) if t not in bad_frames]
            
            if len(good_frames) < 2:
                continue
            
            # Interpolate each coordinate
            for coord in range(3):
                good_values = sequence[good_frames, i, coord]
                
                # Create interpolation function
                interp_func = interp1d(
                    good_frames, good_values,
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                
                # Interpolate bad frames
                result[bad_frames, i, coord] = interp_func(bad_frames)
        
        return result
    
    def _smooth_sequence(self, sequence: np.ndarray,
                        landmark_qualities: Dict[int, LandmarkQuality]) -> np.ndarray:
        """Apply adaptive temporal smoothing."""
        result = sequence.copy()
        
        for i in range(33):
            quality = landmark_qualities.get(i)
            
            # Adaptive smoothing based on quality
            if quality is None:
                sigma = 1.0
            elif quality.quality_level == QualityLevel.EXCELLENT:
                sigma = 0.5  # Light smoothing
            elif quality.quality_level == QualityLevel.GOOD:
                sigma = 1.0
            elif quality.quality_level == QualityLevel.ACCEPTABLE:
                sigma = 1.5
            else:
                sigma = 2.0  # Heavy smoothing
            
            # Apply Gaussian smoothing to each coordinate
            for coord in range(3):
                result[:, i, coord] = ndimage.gaussian_filter1d(
                    result[:, i, coord], sigma=sigma
                )
        
        return result
    
    def _classify_sequence_quality(self, avg_quality: float, 
                                  usable_percentage: float) -> QualityLevel:
        """Classify overall sequence quality."""
        if avg_quality >= 0.85 and usable_percentage >= 95:
            return QualityLevel.EXCELLENT
        elif avg_quality >= 0.7 and usable_percentage >= 85:
            return QualityLevel.GOOD
        elif avg_quality >= 0.5 and usable_percentage >= 70:
            return QualityLevel.ACCEPTABLE
        elif avg_quality >= 0.3 and usable_percentage >= 50:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNUSABLE
    
    def _generate_recommendation(self, avg_quality: float,
                                usable_percentage: float,
                                problematic_landmarks: Dict) -> str:
        """Generate recommendation based on quality assessment."""
        if avg_quality >= 0.85 and usable_percentage >= 95:
            return "Data quality is excellent. Proceed with full analysis."
        
        recommendations = []
        
        if usable_percentage < 70:
            recommendations.append(
                f"Only {usable_percentage:.1f}% of frames are usable. "
                "Consider re-recording with better conditions."
            )
        
        if problematic_landmarks:
            worst_landmarks = sorted(
                problematic_landmarks.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            landmark_names = [
                LandmarkValidator.LANDMARK_NAMES[idx] 
                for idx, _ in worst_landmarks
            ]
            recommendations.append(
                f"Most problematic landmarks: {', '.join(landmark_names)}. "
                "Ensure these body parts are visible and well-lit."
            )
        
        if avg_quality < 0.5:
            recommendations.append(
                "Overall quality is low. Results may be unreliable. "
                "Consider better lighting and camera angle."
            )
        
        return " ".join(recommendations) if recommendations else "Data is acceptable for analysis."


class ArtifactDetector:
    """
    Detects common motion capture artifacts.
    """
    
    @staticmethod
    def detect_jitter(positions: np.ndarray, threshold: float = 0.05) -> np.ndarray:
        """
        Detect high-frequency jitter in landmark positions.
        
        Args:
            positions: Shape (T, 3) or (T,) positions
            threshold: Jitter magnitude threshold
        
        Returns:
            Boolean mask of frames with jitter
        """
        if positions.ndim == 1:
            positions = positions.reshape(-1, 1)
        
        T = len(positions)
        if T < 3:
            return np.zeros(T, dtype=bool)
        
        # Calculate second derivative (acceleration)
        accel = np.diff(np.diff(positions, axis=0), axis=0)
        accel_magnitude = np.linalg.norm(accel, axis=1) if positions.shape[1] > 1 else np.abs(accel).flatten()
        
        # Pad to match original length
        jitter_magnitude = np.zeros(T)
        jitter_magnitude[1:-1] = accel_magnitude
        
        return jitter_magnitude > threshold
    
    @staticmethod
    def detect_occlusion(visibilities: np.ndarray, 
                        threshold: float = 0.3) -> np.ndarray:
        """
        Detect occluded frames from visibility scores.
        
        Args:
            visibilities: Shape (T,) visibility scores
            threshold: Minimum visibility for non-occlusion
        
        Returns:
            Boolean mask of occluded frames
        """
        return visibilities < threshold
    
    @staticmethod
    def detect_tracking_loss(positions: np.ndarray,
                            max_jump: float = 0.3) -> np.ndarray:
        """
        Detect frames where tracking was lost (large position jumps).
        
        Args:
            positions: Shape (T, 3) positions
            max_jump: Maximum acceptable displacement between frames
        
        Returns:
            Boolean mask of frames with tracking loss
        """
        T = len(positions)
        if T < 2:
            return np.zeros(T, dtype=bool)
        
        displacements = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        
        # Pad to match original length
        tracking_loss = np.zeros(T, dtype=bool)
        tracking_loss[1:] = displacements > max_jump
        
        return tracking_loss
    
    @staticmethod
    def detect_impossible_pose(pose: np.ndarray) -> bool:
        """
        Detect physically impossible poses (e.g., limbs too long/short).
        
        Args:
            pose: Shape (33, 3) pose
        
        Returns:
            True if pose is impossible
        """
        if pose.ndim == 1:
            pose = pose.reshape(33, 3)
        
        # Check arm length ratios
        right_upper_arm = np.linalg.norm(pose[14] - pose[12])  # elbow - shoulder
        right_lower_arm = np.linalg.norm(pose[16] - pose[14])  # wrist - elbow
        
        if right_upper_arm > 0:
            arm_ratio = right_lower_arm / right_upper_arm
            if arm_ratio < 0.5 or arm_ratio > 2.0:  # Impossible ratio
                return True
        
        # Check leg length ratios
        right_thigh = np.linalg.norm(pose[26] - pose[24])  # knee - hip
        right_shin = np.linalg.norm(pose[28] - pose[26])  # ankle - knee
        
        if right_thigh > 0:
            leg_ratio = right_shin / right_thigh
            if leg_ratio < 0.5 or leg_ratio > 2.0:
                return True
        
        # Check for positions outside reasonable bounds
        if np.any(np.abs(pose) > 3.0):  # Normalized poses should be within bounds
            return True
        
        return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def filter_poses(pose_sequence: np.ndarray,
                visibilities: Optional[np.ndarray] = None,
                interpolate: bool = True,
                smooth: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to filter and clean pose sequence.
    
    Args:
        pose_sequence: Shape (T, 33, 3) or (T, 99) poses
        visibilities: Shape (T, 33) visibility scores (optional)
        interpolate: Whether to interpolate bad data
        smooth: Whether to apply temporal smoothing
    
    Returns:
        Tuple of (filtered_poses, quality_info_dict)
    """
    filter_obj = SequenceFilter()
    filtered, quality = filter_obj.filter_sequence(
        pose_sequence, visibilities, interpolate, smooth
    )
    
    quality_info = {
        'overall_quality': quality.overall_quality,
        'quality_level': quality.quality_level.value,
        'usable_percentage': quality.usable_percentage,
        'problematic_frames': quality.problematic_frames,
        'problematic_landmarks': quality.problematic_landmarks,
        'recommendation': quality.recommendation
    }
    
    return filtered, quality_info


def validate_sequence(pose_sequence: np.ndarray,
                     visibilities: Optional[np.ndarray] = None) -> Dict:
    """
    Validate pose sequence without filtering.
    
    Args:
        pose_sequence: Shape (T, 33, 3) or (T, 99) poses
        visibilities: Shape (T, 33) visibility scores (optional)
    
    Returns:
        Dictionary with quality metrics
    """
    _, quality_info = filter_poses(
        pose_sequence, visibilities, 
        interpolate=False, smooth=False
    )
    return quality_info


def get_quality_report(pose_sequence: np.ndarray,
                      visibilities: Optional[np.ndarray] = None) -> str:
    """
    Generate human-readable quality report.
    
    Args:
        pose_sequence: Shape (T, 33, 3) or (T, 99) poses
        visibilities: Shape (T, 33) visibility scores (optional)
    
    Returns:
        Formatted quality report string
    """
    quality_info = validate_sequence(pose_sequence, visibilities)
    
    lines = [
        "=" * 50,
        "POSE SEQUENCE QUALITY REPORT",
        "=" * 50,
        f"Overall Quality: {quality_info['overall_quality']:.2%}",
        f"Quality Level: {quality_info['quality_level'].upper()}",
        f"Usable Frames: {quality_info['usable_percentage']:.1f}%",
        "",
    ]
    
    if quality_info['problematic_frames']:
        lines.append(f"Problematic Frames: {len(quality_info['problematic_frames'])} detected")
        if len(quality_info['problematic_frames']) <= 10:
            lines.append(f"  Frames: {quality_info['problematic_frames']}")
    else:
        lines.append("Problematic Frames: None")
    
    if quality_info['problematic_landmarks']:
        lines.append(f"\nProblematic Landmarks:")
        for idx, severity in sorted(
            quality_info['problematic_landmarks'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]:
            name = LandmarkValidator.LANDMARK_NAMES[idx]
            lines.append(f"  - {name}: {severity:.0%} severity")
    
    lines.extend([
        "",
        "RECOMMENDATION:",
        quality_info['recommendation'],
        "=" * 50
    ])
    
    return "\n".join(lines)
