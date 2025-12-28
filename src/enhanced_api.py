"""
Enhanced Pose Analysis API - Integrated Interface
=================================================
Clean API for accessing all enhanced pose analysis features.
Provides unified interface for KSI calculation, coaching feedback,
phase segmentation, and quality filtering.

Usage:
    from enhanced_api import analyze_shot, compare_to_expert
    
    result = analyze_shot(user_poses, 'forehand_clear', skill_level='intermediate')
    print(result.coaching_report)

Author: IPD Research Team
Version: 2.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import json

# Import enhanced modules
from ksi_v2 import (
    EnhancedKSICalculator,
    BiomechanicalFeatureExtractor,
    calculate_ksi_with_confidence
)
from natural_language_coach import (
    NaturalLanguageCoach,
    generate_coaching_report,
    SkillLevel,
    ShotType
)
from phase_segmentation import (
    PhaseSegmenter,
    segment_shot,
    analyze_kinetic_chain,
    get_phase_feedback
)
from confidence_filtering import (
    SequenceFilter,
    filter_poses,
    validate_sequence,
    get_quality_report
)


@dataclass
class AnalysisResult:
    """Complete analysis result."""
    # Scores
    ksi_score: float
    ksi_confidence: Dict[str, float]
    phase_scores: Dict[str, float]
    quality_metrics: Dict[str, float]
    
    # Natural Language Output
    coaching_report: str
    coaching_json: Dict
    
    # Detailed Breakdowns
    per_joint_errors: Dict[str, Dict]
    phase_analysis: Dict[str, str]
    kinetic_chain_analysis: Dict
    
    # Recommendations
    priority_fixes: List[Dict]
    weekly_plan: Dict[str, List[str]]
    
    # Technical Data
    velocity_profile: np.ndarray
    contact_frame: int
    filtered_poses: np.ndarray
    
    # Metadata
    shot_type: str
    skill_level: str
    quality_passed: bool


class EnhancedPoseAnalyzer:
    """
    Main class for enhanced pose analysis.
    Integrates all v2.0 improvements.
    """
    
    def __init__(self, 
                 skill_level: str = 'intermediate',
                 fps: float = 30.0,
                 filter_poses: bool = True,
                 min_quality: float = 0.5):
        """
        Initialize analyzer.
        
        Args:
            skill_level: User skill level ('beginner', 'intermediate', 'advanced', 'expert')
            fps: Video frame rate
            filter_poses: Whether to filter/clean poses before analysis
            min_quality: Minimum pose quality for analysis
        """
        self.skill_level = skill_level
        self.fps = fps
        self.do_filter = filter_poses
        self.min_quality = min_quality
        
        # Initialize components
        self.ksi_calculator = EnhancedKSICalculator()
        self.feature_extractor = BiomechanicalFeatureExtractor()
        self.coach = NaturalLanguageCoach(skill_level=SkillLevel(skill_level))
        self.sequence_filter = SequenceFilter()
    
    def analyze(self, user_poses: np.ndarray,
               expert_poses: np.ndarray,
               shot_type: str = 'forehand_clear',
               user_name: Optional[str] = None,
               visibilities: Optional[np.ndarray] = None) -> AnalysisResult:
        """
        Perform complete analysis of user shot compared to expert.
        
        Args:
            user_poses: Shape (T, 33, 3) or (T, 99) user poses
            expert_poses: Shape (T, 33, 3) or (T, 99) expert template
            shot_type: Type of badminton shot
            user_name: Optional user name for personalization
            visibilities: Optional visibility scores for user poses
        
        Returns:
            AnalysisResult with complete analysis
        """
        # Reshape if needed
        if user_poses.ndim == 2 and user_poses.shape[1] == 99:
            user_poses = user_poses.reshape(-1, 33, 3)
        if expert_poses.ndim == 2 and expert_poses.shape[1] == 99:
            expert_poses = expert_poses.reshape(-1, 33, 3)
        
        # 1. Quality filtering
        if self.do_filter:
            filtered_user, quality_info = filter_poses(
                user_poses, visibilities, interpolate=True, smooth=True
            )
            quality_passed = quality_info['overall_quality'] >= self.min_quality
        else:
            filtered_user = user_poses
            quality_info = validate_sequence(user_poses, visibilities)
            quality_passed = True
        
        # 2. Phase segmentation
        phase_result = segment_shot(filtered_user, shot_type, self.fps)
        phase_scores = phase_result.quality_scores
        
        # 3. Kinetic chain analysis
        kinetic_chain = analyze_kinetic_chain(
            filtered_user, phase_result, shot_type, self.fps
        )
        
        # 4. Enhanced KSI calculation
        ksi_result = self.ksi_calculator.calculate_detailed_ksi(
            filtered_user, expert_poses, fps=self.fps
        )
        
        # 5. KSI with confidence intervals
        ksi_with_confidence = calculate_ksi_with_confidence(
            filtered_user, expert_poses
        )
        
        # 6. Generate coaching feedback
        feedback = self.coach.generate_feedback(
            ksi_result,
            ShotType(shot_type),
            user_name
        )
        
        # Format outputs
        coaching_report = self.coach.format_feedback_text(feedback)
        coaching_json = json.loads(self.coach.format_feedback_json(feedback))
        
        # Convert per-joint errors to serializable format
        per_joint_dict = {}
        for joint_name, error_data in ksi_result.per_joint_errors.items():
            if hasattr(error_data, 'mean_error'):
                per_joint_dict[joint_name] = {
                    'mean_error': float(error_data.mean_error),
                    'max_error': float(error_data.max_error),
                    'critical_frame': int(error_data.critical_frame)
                }
            else:
                per_joint_dict[joint_name] = dict(error_data)
        
        # Phase analysis text
        phase_analysis = get_phase_feedback(phase_result)
        
        # Priority fixes as dicts
        priority_fixes = [
            {
                'issue': f.issue,
                'severity': f.severity.value,
                'fix': f.fix,
                'drill': f.drill
            }
            for f in feedback.priority_fixes
        ]
        
        return AnalysisResult(
            ksi_score=ksi_result.ksi_total,
            ksi_confidence={
                'ci_lower': ksi_with_confidence['ci_95_lower'],
                'ci_upper': ksi_with_confidence['ci_95_upper'],
                'reliable': ksi_with_confidence['reliable']
            },
            phase_scores=phase_scores,
            quality_metrics={
                'overall_quality': quality_info['overall_quality'],
                'usable_percentage': quality_info['usable_percentage'],
                'quality_level': quality_info['quality_level']
            },
            coaching_report=coaching_report,
            coaching_json=coaching_json,
            per_joint_errors=per_joint_dict,
            phase_analysis=phase_analysis,
            kinetic_chain_analysis=kinetic_chain,
            priority_fixes=priority_fixes,
            weekly_plan=feedback.weekly_plan,
            velocity_profile=phase_result.velocity_profile,
            contact_frame=phase_result.contact_frame,
            filtered_poses=filtered_user,
            shot_type=shot_type,
            skill_level=self.skill_level,
            quality_passed=quality_passed
        )
    
    def quick_score(self, user_poses: np.ndarray,
                   expert_poses: np.ndarray) -> Tuple[float, str]:
        """
        Quick KSI score without full analysis.
        
        Returns:
            Tuple of (ksi_score, rating_text)
        """
        ksi_result = self.ksi_calculator.calculate_detailed_ksi(
            user_poses, expert_poses
        )
        
        score = ksi_result.ksi_total
        
        if score >= 0.90:
            rating = "Excellent"
        elif score >= 0.75:
            rating = "Good"
        elif score >= 0.60:
            rating = "Developing"
        else:
            rating = "Needs Work"
        
        return score, rating


def analyze_shot(user_poses: np.ndarray,
                expert_poses: np.ndarray,
                shot_type: str = 'forehand_clear',
                skill_level: str = 'intermediate',
                user_name: Optional[str] = None,
                fps: float = 30.0) -> AnalysisResult:
    """
    Convenience function for complete shot analysis.
    
    Args:
        user_poses: User pose sequence
        expert_poses: Expert template poses
        shot_type: Type of shot
        skill_level: User skill level
        user_name: Optional user name
        fps: Video frame rate
    
    Returns:
        Complete AnalysisResult
    """
    analyzer = EnhancedPoseAnalyzer(skill_level=skill_level, fps=fps)
    return analyzer.analyze(user_poses, expert_poses, shot_type, user_name)


def get_quick_feedback(user_poses: np.ndarray,
                      expert_poses: np.ndarray,
                      shot_type: str = 'forehand_clear',
                      skill_level: str = 'intermediate') -> str:
    """
    Get quick natural language feedback without full analysis.
    
    Returns:
        Concise feedback string
    """
    # Quick KSI
    ksi_calc = EnhancedKSICalculator()
    ksi_result = ksi_calc.calculate_detailed_ksi(user_poses, expert_poses)
    score = ksi_result.ksi_total
    
    # Phase segmentation for timing
    phase_result = segment_shot(user_poses, shot_type)
    
    # Build feedback
    if score >= 0.90:
        intro = "Excellent technique! "
    elif score >= 0.75:
        intro = "Good form overall. "
    elif score >= 0.60:
        intro = "Decent attempt. "
    else:
        intro = "Let's work on the fundamentals. "
    
    # Top issue
    top_issues = sorted(
        ksi_result.per_joint_errors.items(),
        key=lambda x: x[1].mean_error if hasattr(x[1], 'mean_error') else x[1].get('mean_error', 0),
        reverse=True
    )[:1]
    
    if top_issues:
        joint_name = top_issues[0][0].replace('_', ' ')
        issue_text = f"Focus on your {joint_name} positioning. "
    else:
        issue_text = ""
    
    # Timing
    contact_frame = phase_result.contact_frame
    velocity = phase_result.velocity_profile
    if len(velocity) > 0:
        peak_frame = np.argmax(velocity)
        if abs(peak_frame - contact_frame) < 3:
            timing_text = "Your timing is good!"
        elif peak_frame < contact_frame:
            timing_text = "Try to delay your peak speed slightly."
        else:
            timing_text = "Speed up your swing initiation."
    else:
        timing_text = ""
    
    return f"{intro}{issue_text}{timing_text} Score: {score:.2f}/1.00"


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Pose Analysis')
    parser.add_argument('--user', type=str, required=True, help='Path to user poses (.npz)')
    parser.add_argument('--expert', type=str, required=True, help='Path to expert poses (.npz)')
    parser.add_argument('--shot', type=str, default='forehand_clear', help='Shot type')
    parser.add_argument('--level', type=str, default='intermediate', help='Skill level')
    parser.add_argument('--name', type=str, default=None, help='User name')
    parser.add_argument('--output', type=str, choices=['text', 'json'], default='text')
    
    args = parser.parse_args()
    
    # Load poses
    user_data = np.load(args.user)
    expert_data = np.load(args.expert)
    
    user_poses = user_data['poses'] if 'poses' in user_data else user_data['arr_0']
    expert_poses = expert_data['poses'] if 'poses' in expert_data else expert_data['arr_0']
    
    # Analyze
    result = analyze_shot(
        user_poses, expert_poses,
        shot_type=args.shot,
        skill_level=args.level,
        user_name=args.name
    )
    
    # Output
    if args.output == 'json':
        print(json.dumps(result.coaching_json, indent=2))
    else:
        print(result.coaching_report)


if __name__ == '__main__':
    main()
