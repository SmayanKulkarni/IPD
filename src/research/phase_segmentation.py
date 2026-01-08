"""
Phase Segmentation - Automatic Badminton Shot Phase Detection
=============================================================
Automatically segments badminton shots into distinct phases using
velocity profiles, pose configuration, and biomechanical markers.

Phases Detected:
1. Preparation - Ready position, shuttle tracking
2. Loading/Backswing - Weight transfer, racket back
3. Acceleration - Forward swing initiation
4. Contact - Impact moment with shuttle
5. Follow-through - Deceleration and recovery

Research Novelties:
- Velocity-based phase boundary detection
- Multi-joint velocity synchronization analysis
- Pose configuration clustering
- Adaptive thresholds based on shot type
- Phase duration analysis for timing feedback

Author: IPD Research Team
Version: 2.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy import signal
from scipy.ndimage import gaussian_filter1d


class ShotPhase(Enum):
    """Badminton shot phases."""
    PREPARATION = "preparation"
    LOADING = "loading"  
    ACCELERATION = "acceleration"
    CONTACT = "contact"
    FOLLOW_THROUGH = "follow_through"


@dataclass
class PhaseSegment:
    """Represents a detected phase segment."""
    phase: ShotPhase
    start_frame: int
    end_frame: int
    duration_frames: int
    duration_ms: float  # Assuming 30 fps by default
    confidence: float
    key_features: Dict[str, float]


@dataclass
class PhaseBoundary:
    """Represents a boundary between phases."""
    frame: int
    from_phase: ShotPhase
    to_phase: ShotPhase
    confidence: float
    marker_type: str  # What marker triggered this boundary


@dataclass
class PhaseAnalysisResult:
    """Complete phase analysis result."""
    phases: List[PhaseSegment]
    boundaries: List[PhaseBoundary]
    contact_frame: int
    total_duration_ms: float
    phase_timing: Dict[str, float]  # Normalized timing per phase
    velocity_profile: np.ndarray
    quality_scores: Dict[str, float]  # Per-phase quality


class PhaseSegmenter:
    """
    Automatic phase segmentation for badminton shots.
    Uses velocity profiles, pose configuration, and biomechanical markers.
    """
    
    # MediaPipe landmark indices
    LANDMARKS = {
        'right_wrist': 16,
        'right_elbow': 14,
        'right_shoulder': 12,
        'right_hip': 24,
        'right_knee': 26,
        'right_ankle': 28,
        'left_wrist': 15,
        'left_elbow': 13,
        'left_shoulder': 11,
        'left_hip': 23,
        'left_knee': 25,
        'left_ankle': 27,
    }
    
    # Typical phase durations (ms) for different shot types
    EXPECTED_DURATIONS = {
        'forehand_clear': {
            ShotPhase.PREPARATION: (100, 300),
            ShotPhase.LOADING: (150, 400),
            ShotPhase.ACCELERATION: (80, 200),
            ShotPhase.CONTACT: (20, 50),
            ShotPhase.FOLLOW_THROUGH: (100, 300)
        },
        'forehand_drive': {
            ShotPhase.PREPARATION: (50, 200),
            ShotPhase.LOADING: (80, 250),
            ShotPhase.ACCELERATION: (50, 150),
            ShotPhase.CONTACT: (15, 40),
            ShotPhase.FOLLOW_THROUGH: (80, 200)
        },
        'forehand_net_shot': {
            ShotPhase.PREPARATION: (80, 200),
            ShotPhase.LOADING: (50, 150),
            ShotPhase.ACCELERATION: (30, 100),
            ShotPhase.CONTACT: (10, 30),
            ShotPhase.FOLLOW_THROUGH: (50, 150)
        },
        'backhand_drive': {
            ShotPhase.PREPARATION: (80, 250),
            ShotPhase.LOADING: (100, 300),
            ShotPhase.ACCELERATION: (60, 180),
            ShotPhase.CONTACT: (15, 40),
            ShotPhase.FOLLOW_THROUGH: (80, 200)
        }
    }
    
    def __init__(self, fps: float = 30.0, shot_type: str = 'forehand_clear'):
        """
        Initialize phase segmenter.
        
        Args:
            fps: Video frame rate
            shot_type: Type of shot for adapted thresholds
        """
        self.fps = fps
        self.ms_per_frame = 1000.0 / fps
        self.shot_type = shot_type
        self.expected_durations = self.EXPECTED_DURATIONS.get(
            shot_type, self.EXPECTED_DURATIONS['forehand_clear']
        )
    
    def segment(self, pose_sequence: np.ndarray) -> PhaseAnalysisResult:
        """
        Segment pose sequence into phases.
        
        Args:
            pose_sequence: Shape (T, 33, 3) or (T, 99) normalized poses
        
        Returns:
            PhaseAnalysisResult with detected phases
        """
        # Reshape if flattened
        if pose_sequence.ndim == 2 and pose_sequence.shape[1] == 99:
            pose_sequence = pose_sequence.reshape(-1, 33, 3)
        
        T = pose_sequence.shape[0]
        
        # Calculate velocity profiles
        velocities = self._calculate_velocities(pose_sequence)
        
        # Get key joint velocities
        wrist_velocity = velocities['right_wrist']
        elbow_velocity = velocities['right_elbow']
        shoulder_velocity = velocities['right_shoulder']
        hip_velocity = velocities['right_hip']
        
        # Combined racket arm velocity (weighted)
        arm_velocity = (0.5 * wrist_velocity + 0.3 * elbow_velocity + 0.2 * shoulder_velocity)
        
        # Smooth velocity for robust peak detection
        arm_velocity_smooth = gaussian_filter1d(arm_velocity, sigma=2)
        
        # Detect contact frame (peak velocity)
        contact_frame = self._detect_contact(arm_velocity_smooth)
        
        # Detect phase boundaries
        boundaries = self._detect_boundaries(
            pose_sequence, velocities, arm_velocity_smooth, contact_frame
        )
        
        # Create phase segments
        phases = self._create_segments(boundaries, T, contact_frame)
        
        # Calculate phase timing
        phase_timing = self._calculate_phase_timing(phases)
        
        # Calculate quality scores per phase
        quality_scores = self._calculate_phase_quality(pose_sequence, phases, velocities)
        
        return PhaseAnalysisResult(
            phases=phases,
            boundaries=boundaries,
            contact_frame=contact_frame,
            total_duration_ms=T * self.ms_per_frame,
            phase_timing=phase_timing,
            velocity_profile=arm_velocity_smooth,
            quality_scores=quality_scores
        )
    
    def _calculate_velocities(self, pose_sequence: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate velocity for each tracked joint."""
        velocities = {}
        
        for joint_name, idx in self.LANDMARKS.items():
            # Extract joint positions
            positions = pose_sequence[:, idx, :]  # (T, 3)
            
            # Calculate velocity (frame-to-frame displacement)
            velocity = np.zeros(len(positions))
            for t in range(1, len(positions)):
                displacement = positions[t] - positions[t-1]
                velocity[t] = np.linalg.norm(displacement) * self.fps  # units/second
            
            # First frame gets second frame's velocity
            velocity[0] = velocity[1] if len(velocity) > 1 else 0
            
            velocities[joint_name] = velocity
        
        return velocities
    
    def _detect_contact(self, velocity: np.ndarray) -> int:
        """Detect contact frame as peak velocity moment."""
        # Find global maximum
        contact_frame = np.argmax(velocity)
        
        # Validate: contact should be in middle-late portion of sequence
        T = len(velocity)
        if contact_frame < T * 0.2:
            # Too early, likely noise - find peak in later portion
            search_start = int(T * 0.3)
            contact_frame = search_start + np.argmax(velocity[search_start:])
        
        return contact_frame
    
    def _detect_boundaries(self, pose_sequence: np.ndarray, 
                          velocities: Dict[str, np.ndarray],
                          arm_velocity: np.ndarray, 
                          contact_frame: int) -> List[PhaseBoundary]:
        """Detect phase boundaries using multiple markers."""
        boundaries = []
        T = len(arm_velocity)
        
        # 1. PREPARATION → LOADING boundary
        # Detected by: initial upward movement of racket arm, weight shift
        loading_start = self._detect_loading_start(
            pose_sequence, velocities, arm_velocity, contact_frame
        )
        boundaries.append(PhaseBoundary(
            frame=loading_start,
            from_phase=ShotPhase.PREPARATION,
            to_phase=ShotPhase.LOADING,
            confidence=0.8,
            marker_type="velocity_threshold"
        ))
        
        # 2. LOADING → ACCELERATION boundary  
        # Detected by: velocity inflection point (minimum before peak)
        accel_start = self._detect_acceleration_start(arm_velocity, contact_frame, loading_start)
        boundaries.append(PhaseBoundary(
            frame=accel_start,
            from_phase=ShotPhase.LOADING,
            to_phase=ShotPhase.ACCELERATION,
            confidence=0.85,
            marker_type="velocity_minimum"
        ))
        
        # 3. ACCELERATION → CONTACT boundary
        # Detected by: 90% of peak velocity (approach to contact)
        contact_start = self._detect_contact_start(arm_velocity, contact_frame)
        boundaries.append(PhaseBoundary(
            frame=contact_start,
            from_phase=ShotPhase.ACCELERATION,
            to_phase=ShotPhase.CONTACT,
            confidence=0.9,
            marker_type="velocity_threshold_90"
        ))
        
        # 4. CONTACT → FOLLOW_THROUGH boundary
        # Detected by: velocity drops below 90% of peak
        followthrough_start = self._detect_followthrough_start(arm_velocity, contact_frame)
        boundaries.append(PhaseBoundary(
            frame=followthrough_start,
            from_phase=ShotPhase.CONTACT,
            to_phase=ShotPhase.FOLLOW_THROUGH,
            confidence=0.9,
            marker_type="velocity_threshold_90"
        ))
        
        return boundaries
    
    def _detect_loading_start(self, pose_sequence: np.ndarray,
                             velocities: Dict[str, np.ndarray],
                             arm_velocity: np.ndarray,
                             contact_frame: int) -> int:
        """Detect start of loading/backswing phase."""
        # Look for initial significant movement
        threshold = np.percentile(arm_velocity[:contact_frame], 25)
        
        for t in range(min(contact_frame, len(arm_velocity) - 1)):
            if arm_velocity[t] > threshold:
                return max(0, t - 2)  # Small buffer
        
        return 0
    
    def _detect_acceleration_start(self, arm_velocity: np.ndarray,
                                   contact_frame: int, 
                                   loading_start: int) -> int:
        """Detect start of acceleration phase (velocity minimum before peak)."""
        # Search between loading start and contact
        search_region = arm_velocity[loading_start:contact_frame]
        
        if len(search_region) < 3:
            return loading_start + 1
        
        # Find local minima
        minima_idx = signal.argrelmin(search_region, order=3)[0]
        
        if len(minima_idx) > 0:
            # Take last minimum before contact (end of backswing)
            local_idx = minima_idx[-1]
            return loading_start + local_idx
        
        # Fallback: find point where velocity starts consistently increasing
        for t in range(len(search_region) - 3):
            if all(search_region[t+i] < search_region[t+i+1] for i in range(3)):
                return loading_start + t
        
        # Last resort: middle point
        return loading_start + len(search_region) // 2
    
    def _detect_contact_start(self, arm_velocity: np.ndarray, 
                             contact_frame: int) -> int:
        """Detect when contact phase begins (90% peak velocity approaching)."""
        peak_velocity = arm_velocity[contact_frame]
        threshold = 0.9 * peak_velocity
        
        for t in range(contact_frame - 1, -1, -1):
            if arm_velocity[t] < threshold:
                return t + 1
        
        return max(0, contact_frame - 2)
    
    def _detect_followthrough_start(self, arm_velocity: np.ndarray,
                                   contact_frame: int) -> int:
        """Detect when follow-through begins (velocity drops below 90% peak)."""
        peak_velocity = arm_velocity[contact_frame]
        threshold = 0.9 * peak_velocity
        T = len(arm_velocity)
        
        for t in range(contact_frame + 1, T):
            if arm_velocity[t] < threshold:
                return t
        
        return min(T - 1, contact_frame + 2)
    
    def _create_segments(self, boundaries: List[PhaseBoundary], 
                        total_frames: int,
                        contact_frame: int) -> List[PhaseSegment]:
        """Create phase segments from boundaries."""
        segments = []
        
        # Sort boundaries by frame
        boundaries = sorted(boundaries, key=lambda b: b.frame)
        
        # Create segments
        phase_order = [ShotPhase.PREPARATION, ShotPhase.LOADING, 
                      ShotPhase.ACCELERATION, ShotPhase.CONTACT, 
                      ShotPhase.FOLLOW_THROUGH]
        
        start_frame = 0
        for i, phase in enumerate(phase_order):
            if i < len(boundaries):
                end_frame = boundaries[i].frame
            else:
                end_frame = total_frames
            
            # For FOLLOW_THROUGH, extend to end
            if phase == ShotPhase.FOLLOW_THROUGH:
                end_frame = total_frames
            
            duration_frames = end_frame - start_frame
            duration_ms = duration_frames * self.ms_per_frame
            
            # Calculate confidence based on expected durations
            expected = self.expected_durations.get(phase, (50, 200))
            if expected[0] <= duration_ms <= expected[1]:
                confidence = 0.95
            elif duration_ms < expected[0]:
                confidence = max(0.5, 1.0 - (expected[0] - duration_ms) / expected[0])
            else:
                confidence = max(0.5, 1.0 - (duration_ms - expected[1]) / expected[1])
            
            segments.append(PhaseSegment(
                phase=phase,
                start_frame=start_frame,
                end_frame=end_frame,
                duration_frames=duration_frames,
                duration_ms=duration_ms,
                confidence=confidence,
                key_features={'contact_relative': contact_frame - start_frame}
            ))
            
            start_frame = end_frame
        
        return segments
    
    def _calculate_phase_timing(self, phases: List[PhaseSegment]) -> Dict[str, float]:
        """Calculate normalized timing for each phase."""
        total_duration = sum(p.duration_ms for p in phases)
        
        if total_duration == 0:
            return {}
        
        return {
            p.phase.value: p.duration_ms / total_duration
            for p in phases
        }
    
    def _calculate_phase_quality(self, pose_sequence: np.ndarray,
                                phases: List[PhaseSegment],
                                velocities: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate quality score for each phase."""
        quality = {}
        
        for segment in phases:
            phase_poses = pose_sequence[segment.start_frame:segment.end_frame]
            
            if len(phase_poses) == 0:
                quality[segment.phase.value] = 0.0
                continue
            
            # Quality metrics per phase
            if segment.phase == ShotPhase.PREPARATION:
                # Check for stable ready position
                score = self._assess_preparation(phase_poses)
            elif segment.phase == ShotPhase.LOADING:
                # Check for proper backswing
                score = self._assess_loading(phase_poses, velocities, segment)
            elif segment.phase == ShotPhase.ACCELERATION:
                # Check for smooth acceleration
                score = self._assess_acceleration(velocities, segment)
            elif segment.phase == ShotPhase.CONTACT:
                # Check for optimal contact position
                score = self._assess_contact(phase_poses)
            else:  # FOLLOW_THROUGH
                # Check for complete follow-through
                score = self._assess_followthrough(phase_poses, velocities, segment)
            
            quality[segment.phase.value] = score
        
        return quality
    
    def _assess_preparation(self, poses: np.ndarray) -> float:
        """Assess quality of preparation phase."""
        if len(poses) == 0:
            return 0.0
        
        # Check for stability (low variance)
        variance = np.mean(np.std(poses, axis=0))
        stability_score = np.exp(-variance * 10)  # Low variance = high score
        
        # Check for balanced stance
        avg_pose = np.mean(poses, axis=0)
        left_hip = avg_pose[self.LANDMARKS['left_hip']]
        right_hip = avg_pose[self.LANDMARKS['right_hip']]
        hip_level = 1.0 - abs(left_hip[1] - right_hip[1])  # Should be level
        
        return (stability_score + hip_level) / 2
    
    def _assess_loading(self, poses: np.ndarray, 
                       velocities: Dict[str, np.ndarray],
                       segment: PhaseSegment) -> float:
        """Assess quality of loading/backswing phase."""
        if len(poses) == 0:
            return 0.0
        
        # Check for upward arm movement
        start_wrist = poses[0, self.LANDMARKS['right_wrist'], 1]  # Y coordinate
        end_wrist = poses[-1, self.LANDMARKS['right_wrist'], 1]
        arm_raised = min(1.0, max(0.0, (start_wrist - end_wrist) * 10))  # Negative Y is up
        
        # Check for hip-shoulder separation
        shoulder_vel = velocities['right_shoulder'][segment.start_frame:segment.end_frame]
        hip_vel = velocities['right_hip'][segment.start_frame:segment.end_frame]
        if len(shoulder_vel) > 0 and len(hip_vel) > 0:
            separation = np.mean(np.abs(shoulder_vel - hip_vel))
            separation_score = min(1.0, separation * 5)
        else:
            separation_score = 0.5
        
        return (arm_raised + separation_score) / 2
    
    def _assess_acceleration(self, velocities: Dict[str, np.ndarray],
                            segment: PhaseSegment) -> float:
        """Assess quality of acceleration phase."""
        wrist_vel = velocities['right_wrist'][segment.start_frame:segment.end_frame]
        
        if len(wrist_vel) < 2:
            return 0.0
        
        # Check for consistent acceleration (increasing velocity)
        velocity_increases = np.diff(wrist_vel) > 0
        consistency = np.mean(velocity_increases)
        
        # Check for smooth acceleration (low jerk)
        acceleration = np.diff(wrist_vel)
        if len(acceleration) > 1:
            jerk = np.diff(acceleration)
            smoothness = np.exp(-np.std(jerk) * 10)
        else:
            smoothness = 0.5
        
        return (consistency * 0.6 + smoothness * 0.4)
    
    def _assess_contact(self, poses: np.ndarray) -> float:
        """Assess quality of contact phase."""
        if len(poses) == 0:
            return 0.0
        
        # Use middle pose as contact moment
        contact_pose = poses[len(poses) // 2]
        
        # Check wrist height (should be high)
        wrist_y = contact_pose[self.LANDMARKS['right_wrist'], 1]
        shoulder_y = contact_pose[self.LANDMARKS['right_shoulder'], 1]
        arm_extension = min(1.0, max(0.0, (shoulder_y - wrist_y) * 5 + 0.5))
        
        # Check elbow angle (should be extending)
        elbow = contact_pose[self.LANDMARKS['right_elbow']]
        shoulder = contact_pose[self.LANDMARKS['right_shoulder']]
        wrist = contact_pose[self.LANDMARKS['right_wrist']]
        
        v1 = shoulder - elbow
        v2 = wrist - elbow
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        elbow_angle = np.arccos(np.clip(cos_angle, -1, 1))
        elbow_score = min(1.0, elbow_angle / np.pi)  # More extension = higher score
        
        return (arm_extension * 0.5 + elbow_score * 0.5)
    
    def _assess_followthrough(self, poses: np.ndarray,
                             velocities: Dict[str, np.ndarray],
                             segment: PhaseSegment) -> float:
        """Assess quality of follow-through phase."""
        if len(poses) == 0:
            return 0.0
        
        # Check for smooth deceleration
        wrist_vel = velocities['right_wrist'][segment.start_frame:segment.end_frame]
        
        if len(wrist_vel) < 2:
            return 0.0
        
        # Velocity should decrease
        velocity_decreases = np.diff(wrist_vel) < 0
        decel_consistency = np.mean(velocity_decreases)
        
        # Arm should cross body
        start_wrist_x = poses[0, self.LANDMARKS['right_wrist'], 0]
        end_wrist_x = poses[-1, self.LANDMARKS['right_wrist'], 0]
        cross_body = min(1.0, abs(start_wrist_x - end_wrist_x) * 5)
        
        return (decel_consistency * 0.5 + cross_body * 0.5)


class MultiJointSynchronization:
    """
    Analyzes synchronization of multiple joints during shot phases.
    Key for kinetic chain analysis.
    """
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps
    
    def analyze_kinetic_chain(self, pose_sequence: np.ndarray,
                             phase_result: PhaseAnalysisResult) -> Dict:
        """
        Analyze kinetic chain timing during acceleration phase.
        
        Returns timing delays between joints (proximal-to-distal sequencing).
        """
        # Get acceleration phase
        accel_phase = None
        for phase in phase_result.phases:
            if phase.phase == ShotPhase.ACCELERATION:
                accel_phase = phase
                break
        
        if accel_phase is None:
            return {'error': 'No acceleration phase found'}
        
        # Extract phase data
        start = accel_phase.start_frame
        end = accel_phase.end_frame
        
        # Calculate velocity for kinetic chain joints
        chain_joints = ['right_hip', 'right_shoulder', 'right_elbow', 'right_wrist']
        velocities = {}
        peak_frames = {}
        
        for joint in chain_joints:
            idx = PhaseSegmenter.LANDMARKS[joint]
            positions = pose_sequence[start:end, idx, :]
            
            # Calculate velocity
            velocity = np.zeros(len(positions))
            for t in range(1, len(positions)):
                velocity[t] = np.linalg.norm(positions[t] - positions[t-1]) * self.fps
            
            velocities[joint] = velocity
            peak_frames[joint] = np.argmax(velocity) + start
        
        # Calculate delays (relative to hip)
        hip_peak = peak_frames['right_hip']
        delays = {
            joint: (peak_frames[joint] - hip_peak) * (1000.0 / self.fps)
            for joint in chain_joints
        }
        
        # Assess sequencing quality
        # Ideal: hip → shoulder → elbow → wrist (positive delays increasing)
        expected_order = ['right_hip', 'right_shoulder', 'right_elbow', 'right_wrist']
        is_sequential = all(
            delays[expected_order[i]] <= delays[expected_order[i+1]]
            for i in range(len(expected_order) - 1)
        )
        
        # Calculate total delay (should be 80-150ms from hip to wrist)
        total_delay = delays['right_wrist'] - delays['right_hip']
        
        if 80 <= total_delay <= 150:
            timing_quality = 'optimal'
        elif 50 <= total_delay <= 200:
            timing_quality = 'acceptable'
        else:
            timing_quality = 'needs_improvement'
        
        return {
            'peak_frames': peak_frames,
            'delays_ms': delays,
            'is_sequential': is_sequential,
            'total_delay_ms': total_delay,
            'timing_quality': timing_quality,
            'feedback': self._generate_chain_feedback(delays, is_sequential, total_delay)
        }
    
    def _generate_chain_feedback(self, delays: Dict, is_sequential: bool, 
                                total_delay: float) -> str:
        """Generate natural language feedback for kinetic chain."""
        if is_sequential and 80 <= total_delay <= 150:
            return ("Excellent kinetic chain sequencing! Your hip leads, followed by "
                   "shoulder, elbow, and wrist in proper order with optimal timing.")
        
        feedback_parts = []
        
        if not is_sequential:
            feedback_parts.append("Your kinetic chain is firing out of sequence. ")
            
            # Identify specific issues
            if delays['right_shoulder'] < delays['right_hip']:
                feedback_parts.append("Your shoulder is moving before your hip - start with hip rotation. ")
            if delays['right_wrist'] < delays['right_elbow']:
                feedback_parts.append("Your wrist is snapping before elbow extends - let elbow lead. ")
        
        if total_delay < 80:
            feedback_parts.append(f"Your chain is firing too fast ({total_delay:.0f}ms). "
                                "This 'all at once' pattern loses energy. Try to sequence it out over 100-150ms.")
        elif total_delay > 150:
            feedback_parts.append(f"Your chain timing is too slow ({total_delay:.0f}ms). "
                                "You're losing momentum between segments. Tighten the sequence to 100-150ms.")
        
        return "".join(feedback_parts) if feedback_parts else "Kinetic chain timing is acceptable."


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def segment_shot(pose_sequence: np.ndarray, 
                shot_type: str = 'forehand_clear',
                fps: float = 30.0) -> PhaseAnalysisResult:
    """
    Convenience function to segment a shot into phases.
    
    Args:
        pose_sequence: Shape (T, 33, 3) or (T, 99) normalized poses
        shot_type: Type of badminton shot
        fps: Video frame rate
    
    Returns:
        PhaseAnalysisResult with detected phases
    """
    segmenter = PhaseSegmenter(fps=fps, shot_type=shot_type)
    return segmenter.segment(pose_sequence)


def analyze_kinetic_chain(pose_sequence: np.ndarray,
                         phase_result: Optional[PhaseAnalysisResult] = None,
                         shot_type: str = 'forehand_clear',
                         fps: float = 30.0) -> Dict:
    """
    Convenience function to analyze kinetic chain timing.
    
    Args:
        pose_sequence: Shape (T, 33, 3) or (T, 99) normalized poses
        phase_result: Optional pre-computed phase segmentation
        shot_type: Type of badminton shot
        fps: Video frame rate
    
    Returns:
        Dictionary with kinetic chain analysis
    """
    if phase_result is None:
        phase_result = segment_shot(pose_sequence, shot_type, fps)
    
    analyzer = MultiJointSynchronization(fps=fps)
    return analyzer.analyze_kinetic_chain(pose_sequence, phase_result)


def get_phase_feedback(phase_result: PhaseAnalysisResult) -> Dict[str, str]:
    """
    Generate natural language feedback for each phase.
    
    Args:
        phase_result: PhaseAnalysisResult from segmentation
    
    Returns:
        Dictionary mapping phase names to feedback strings
    """
    feedback = {}
    
    for phase in phase_result.phases:
        quality = phase_result.quality_scores.get(phase.phase.value, 0.5)
        duration_ms = phase.duration_ms
        
        if quality >= 0.8:
            quality_text = "excellent"
        elif quality >= 0.6:
            quality_text = "good"
        elif quality >= 0.4:
            quality_text = "needs work"
        else:
            quality_text = "needs significant improvement"
        
        phase_name = phase.phase.value.replace('_', ' ').title()
        feedback[phase.phase.value] = (
            f"{phase_name}: Quality is {quality_text} (score: {quality:.2f}). "
            f"Duration: {duration_ms:.0f}ms."
        )
    
    return feedback
