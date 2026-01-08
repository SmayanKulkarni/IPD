"""
Natural Language Coach - AI-Powered Badminton Coaching Assistant
================================================================
Generates human-like, personalized coaching feedback from KSI analysis.

Features:
- Multi-level skill adaptation (beginner → expert)
- Causal reasoning for errors
- Progressive correction strategies
- Motivational psychology integration
- Context-aware explanations
- Multi-language support foundation

Research Novelties:
- Biomechanical causality chains
- Temporal error attribution
- Personalized learning path generation
- Explainable AI for sports coaching

Author: IPD Research Team
Version: 2.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import re


class SkillLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ErrorSeverity(Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    NEGLIGIBLE = "negligible"


class ShotType(Enum):
    FOREHAND_CLEAR = "forehand_clear"
    FOREHAND_DRIVE = "forehand_drive"
    FOREHAND_LIFT = "forehand_lift"
    FOREHAND_NET_SHOT = "forehand_net_shot"
    BACKHAND_DRIVE = "backhand_drive"
    BACKHAND_NET_SHOT = "backhand_net_shot"


@dataclass
class CorrectionItem:
    """Structured correction with explanation and drill."""
    issue: str
    severity: ErrorSeverity
    cause: str
    explanation: str
    fix: str
    drill: str
    visual_cue: str
    common_mistake: str
    success_indicator: str


@dataclass
class CoachingFeedback:
    """Complete coaching feedback structure."""
    overall_score: float
    rating: str
    summary: str
    priority_fixes: List[CorrectionItem]
    phase_analysis: Dict[str, str]
    timing_feedback: str
    power_feedback: str
    strengths: List[str]
    improvement_areas: List[str]
    weekly_plan: Dict[str, List[str]]
    motivational_message: str
    technical_notes: str  # For advanced users
    video_timestamps: List[Dict]  # Key moments to review


class BiomechanicalKnowledgeBase:
    """
    Expert knowledge base for biomechanical error analysis.
    Contains causality chains, fixes, and drills for each error type.
    """
    
    def __init__(self):
        self._build_knowledge_base()
    
    def _build_knowledge_base(self):
        """Build comprehensive error-cause-fix mappings."""
        
        # =====================================================================
        # CAUSAL CHAINS: Error → Possible Causes → Underlying Issues
        # =====================================================================
        self.causal_chains = {
            # ELBOW ERRORS
            'right_elbow_angle': {
                'error_name': 'Racket Arm Elbow',
                'symptoms': ['elbow too straight', 'elbow too bent', 'elbow dropping'],
                'causes': {
                    'too_straight': [
                        ('tension', 'Arm muscles too tense, preventing natural bend'),
                        ('rushing', 'Starting forward swing before completing backswing'),
                        ('weak_grip', 'Gripping racket too tightly, restricting movement')
                    ],
                    'too_bent': [
                        ('late_preparation', 'Not getting racket up in time'),
                        ('wrong_grip', 'Incorrect grip limiting wrist/elbow freedom'),
                        ('footwork', 'Poor positioning forcing awkward arm angle')
                    ],
                    'dropping': [
                        ('fatigue', 'Shoulder muscles tiring mid-rally'),
                        ('habit', 'Developed bad habit from incorrect practice'),
                        ('weak_rotator', 'Rotator cuff not strong enough to maintain position')
                    ]
                },
                'biomechanical_principle': 'The elbow acts as a velocity multiplier in the kinetic chain. Optimal angle (90-120°) allows maximum angular velocity transfer from shoulder to wrist.'
            },
            
            'left_elbow_angle': {
                'error_name': 'Non-Racket Arm Elbow',
                'symptoms': ['arm not used for balance', 'arm too low', 'arm crossing body'],
                'causes': {
                    'not_engaged': [
                        ('focus', 'Over-focusing on hitting arm, neglecting balance arm'),
                        ('habit', 'Never learned to use non-racket arm'),
                        ('timing', 'Arm drops before contact')
                    ]
                },
                'biomechanical_principle': 'The non-racket arm provides counterbalance and helps generate rotational momentum. When extended up and back during preparation, it creates a longer lever for rotation.'
            },
            
            # SHOULDER ERRORS
            'right_shoulder_elevation': {
                'error_name': 'Shoulder Height',
                'symptoms': ['shoulder too low', 'shoulder shrugging', 'uneven shoulders'],
                'causes': {
                    'too_low': [
                        ('late_arrival', 'Not reaching shuttle at optimal height'),
                        ('weak_deltoid', 'Shoulder muscles cant maintain elevation'),
                        ('footwork', 'Poor positioning preventing full extension')
                    ],
                    'shrugging': [
                        ('tension', 'Stress/anxiety causing muscle tension'),
                        ('overcompensation', 'Trying too hard to generate power')
                    ]
                },
                'biomechanical_principle': 'Shoulder elevation determines the angle of attack on the shuttle. Higher contact = more downward trajectory options.'
            },
            
            # HIP-SHOULDER SEPARATION
            'hip_shoulder_separation': {
                'error_name': 'Hip-Shoulder Separation',
                'symptoms': ['no rotation', 'hips and shoulders move together', 'arm-only swing'],
                'causes': {
                    'insufficient': [
                        ('core_weakness', 'Core muscles cant create separation'),
                        ('timing', 'Hips and shoulders fire simultaneously'),
                        ('habit', 'Never learned rotational mechanics'),
                        ('stiff_trunk', 'Limited thoracic spine mobility')
                    ]
                },
                'biomechanical_principle': 'Hip-shoulder separation creates elastic potential energy in the torso. The "X-factor" (15-25° separation) generates 30-40% of racket head speed through the stretch-shortening cycle.'
            },
            
            # WRIST DYNAMICS
            'wrist_height_normalized': {
                'error_name': 'Contact Point Height',
                'symptoms': ['hitting too low', 'reaching down', 'contact behind body'],
                'causes': {
                    'too_low': [
                        ('late_timing', 'Not moving to shuttle quickly enough'),
                        ('poor_judgment', 'Misjudging shuttle trajectory'),
                        ('footwork', 'Feet not in position to hit high'),
                        ('preparation', 'Racket not up during ready position')
                    ]
                },
                'biomechanical_principle': 'Contact height determines shot angle and power. Every 10cm lower contact reduces power by ~15% and limits shot options.'
            },
            
            # KINETIC CHAIN
            'chain_straightness': {
                'error_name': 'Kinetic Chain Alignment',
                'symptoms': ['broken chain', 'power leakage', 'inefficient transfer'],
                'causes': {
                    'broken': [
                        ('timing', 'Segments not firing in sequence'),
                        ('muscle_imbalance', 'Some links stronger than others'),
                        ('coordination', 'Neural coordination not developed')
                    ]
                },
                'biomechanical_principle': 'Power flows from ground through legs → hips → trunk → shoulder → elbow → wrist. Each segment accelerates the next. Breaking the chain anywhere loses all downstream power.'
            },
            
            # SPINE ALIGNMENT
            'spine_forward_lean': {
                'error_name': 'Trunk Posture',
                'symptoms': ['leaning too far forward', 'leaning back', 'unstable core'],
                'causes': {
                    'excessive_forward': [
                        ('reaching', 'Reaching for shuttle instead of moving feet'),
                        ('balance', 'Weight too far forward'),
                        ('habit', 'Developed compensation for poor footwork')
                    ],
                    'excessive_back': [
                        ('late', 'Shuttle behind optimal contact point'),
                        ('defensive', 'Being pushed back by opponent')
                    ]
                },
                'biomechanical_principle': 'Trunk angle affects force vector direction. Forward lean >15° shifts power forward; backward lean reduces control. Optimal is 5-10° forward at contact.'
            },
            
            # KNEE ANGLES
            'left_knee_angle': {
                'error_name': 'Front Leg (Left Knee)',
                'symptoms': ['leg too straight', 'knee collapsed', 'unstable base'],
                'causes': {
                    'straight': [
                        ('landing', 'Landing with straight leg absorbs no force'),
                        ('habit', 'Not bending into shots')
                    ],
                    'collapsed': [
                        ('weak_quad', 'Quadriceps cant support load'),
                        ('fatigue', 'Legs tired late in match')
                    ]
                },
                'biomechanical_principle': 'Front knee acts as a brake and pivot. Slight bend (150-170°) allows energy transfer from ground. Straight leg = energy leak; collapsed = unstable platform.'
            },
            
            'right_knee_angle': {
                'error_name': 'Back Leg (Right Knee)', 
                'symptoms': ['no push-off', 'flat-footed', 'passive leg'],
                'causes': {
                    'passive': [
                        ('footwork', 'Not using leg drive'),
                        ('habit', 'Arm-dominant technique'),
                        ('fatigue', 'Legs not conditioned')
                    ]
                },
                'biomechanical_principle': 'Back leg initiates ground reaction force. Push-off generates 20-25% of shot power. Passive back leg = arm-only shot.'
            }
        }
        
        # =====================================================================
        # FIXES: Skill-level appropriate corrections
        # =====================================================================
        self.fixes = {
            'right_elbow_angle': {
                SkillLevel.BEGINNER: "Keep your elbow at about 90 degrees - like you're carrying a tray of drinks at shoulder height. Don't straighten until you swing forward.",
                SkillLevel.INTERMEDIATE: "Maintain 90-120° elbow angle through the backswing. Think 'throw' not 'push' - let the elbow lead, then snap straight at contact.",
                SkillLevel.ADVANCED: "Focus on the pronation-supination timing. Elbow extends just before contact (30-50ms), with forearm pronation adding final racket head acceleration.",
                SkillLevel.EXPERT: "Optimize your elbow angular velocity peak. Target 1200-1500°/s at extension. Late extension loses energy to deceleration before contact."
            },
            
            'hip_shoulder_separation': {
                SkillLevel.BEGINNER: "Turn your body like you're throwing a ball sideways. Hips face forward first, then shoulders catch up. Feel the twist in your stomach.",
                SkillLevel.INTERMEDIATE: "Create 'X-factor' stretch: rotate hips 30° toward target while shoulders stay back, then unwind explosively. Think of winding up a spring.",
                SkillLevel.ADVANCED: "Target 15-25° separation at peak. Initiate with hip rotation from ground force, let shoulders lag 40-60ms. Feel stretch in obliques.",
                SkillLevel.EXPERT: "Optimize stretch-shortening cycle timing. Peak separation should occur 80-100ms before contact. Earlier = power leaks; later = incomplete transfer."
            },
            
            'wrist_height_normalized': {
                SkillLevel.BEGINNER: "Hit the shuttle as high as you can reach - pretend you're picking an apple from a high branch. Move your feet to get under it.",
                SkillLevel.INTERMEDIATE: "Contact above forehead height. If you're reaching down, you've arrived too late. Better footwork = higher contact.",
                SkillLevel.ADVANCED: "Optimal contact is at 95-100% of maximum reach. This gives best angle while maintaining control. Adjust with split-step timing.",
                SkillLevel.EXPERT: "Maximize contact height variance based on shot selection. Clears at 100% reach, drives at 85-90%, drops with slight forward lean for deception."
            },
            
            'chain_straightness': {
                SkillLevel.BEGINNER: "Move your body in order: push with legs → turn hips → turn shoulders → swing arm → flick wrist. Like cracking a whip from handle to tip.",
                SkillLevel.INTERMEDIATE: "Sequential acceleration: each body part speeds up the next. Ground → legs → hips → torso → shoulder → elbow → wrist. Pause 50ms between each.",
                SkillLevel.ADVANCED: "Focus on 'proximal-to-distal' sequencing. Monitor for simultaneous activation (power leak). Each segment should peak velocity just as next begins.",
                SkillLevel.EXPERT: "Optimize inter-segmental timing: leg drive 150ms before contact, hip rotation 100ms, shoulder 70ms, elbow 35ms, wrist 15ms. ±10ms variance acceptable."
            },
            
            'spine_forward_lean': {
                SkillLevel.BEGINNER: "Keep your chest up and back relatively straight. Don't bend at the waist to reach - move your feet instead.",
                SkillLevel.INTERMEDIATE: "Maintain 5-10° forward lean at contact. Core engaged (like bracing for a punch). Excessive lean = weak shots.",
                SkillLevel.ADVANCED: "Trunk angle affects force vector. Use slight forward lean for power, neutral for control. Never exceed 15° forward or you lose posterior chain.",
                SkillLevel.EXPERT: "Dynamically adjust trunk angle: 0-5° for deceptive shots, 10-15° for maximum power. Integrate with hip-shoulder separation timing."
            }
        }
        
        # =====================================================================
        # DRILLS: Progressive exercises for each error
        # =====================================================================
        self.drills = {
            'right_elbow_angle': {
                'name': "Shadow Elbow Drill",
                'beginner': "Stand in front of mirror. Practice raising racket to 'L' position (90° elbow) 20 times. Hold each position for 2 seconds. Feel the angle.",
                'intermediate': "Shadow swings with freeze: swing halfway, freeze with bent elbow, check angle, continue to contact. 3 sets of 15.",
                'advanced': "Resistance band elbow extensions: anchor band, practice explosive elbow snap with resistance. 3 sets of 20. Focus on speed.",
                'equipment': "Mirror, resistance band (intermediate+)",
                'duration': "10 minutes daily for 2 weeks"
            },
            
            'hip_shoulder_separation': {
                'name': "Rotation Separation Drill",
                'beginner': "Medicine ball throws: stand sideways to wall, rotate hips first then throw. Feel the hip-shoulder stretch. 3 sets of 10.",
                'intermediate': "Split-stance rotations: hold racket across shoulders, rotate hips 45° while keeping shoulders still, then release. 3 sets of 15.",
                'advanced': "Cable wood chops: low-to-high rotation with cable machine, emphasizing hip lead. 3 sets of 12 each side.",
                'equipment': "Medicine ball, cable machine (advanced)",
                'duration': "15 minutes, 3x per week"
            },
            
            'wrist_height_normalized': {
                'name': "High Contact Point Drill",
                'beginner': "Balloon tap drill: keep balloon in air using only overhead hits. Forces you to move feet and hit high. 5 minute rallies.",
                'intermediate': "Target practice: hang target (shuttle tube) at maximum reach height. Practice hitting it consistently. 50 hits per session.",
                'advanced': "Multi-shuttle feed: coach feeds shuttles at varying heights. Move and hit every one at highest possible point. 3 sets of 20.",
                'equipment': "Balloon, hanging target, multi-shuttle setup",
                'duration': "15 minutes daily"
            },
            
            'chain_straightness': {
                'name': "Sequential Activation Drill", 
                'beginner': "Towel whip: snap towel to make cracking sound. Requires sequential activation. Practice until consistent crack. 50 snaps.",
                'intermediate': "Throw and hit: throw shuttle high with non-racket hand, hit at peak. Forces full kinetic chain. 3 sets of 20.",
                'advanced': "Slow motion breakdown: perform shot at 25% speed, counting 1-2-3-4-5 for legs-hips-trunk-arm-wrist. 3 sets of 10.",
                'equipment': "Towel, shuttles",
                'duration': "10 minutes daily"
            },
            
            'spine_forward_lean': {
                'name': "Posture Control Drill",
                'beginner': "Wall alignment: stand with back to wall, maintain contact through swing motion. Builds awareness. 3 sets of 10.",
                'intermediate': "Broomstick overhead: hold stick overhead during shadow footwork. Must maintain upright posture. 5 minutes continuous.",
                'advanced': "Video analysis: record yourself, draw vertical line through hip, check deviation at contact. Target <15°.",
                'equipment': "Wall, broomstick, phone for video",
                'duration': "5 minutes before each session"
            }
        }
        
        # =====================================================================
        # VISUAL CUES: Mental images for correct technique
        # =====================================================================
        self.visual_cues = {
            'right_elbow_angle': "Imagine you're a waiter holding a tray at shoulder height - your elbow naturally bends to ~90°",
            'hip_shoulder_separation': "Picture wringing out a wet towel - hips twist one way, shoulders the other, then release",
            'wrist_height_normalized': "Reach up to change a light bulb on the ceiling - that's your contact height",
            'chain_straightness': "Crack a bullwhip - the handle moves first, energy flows to the tip where it snaps",
            'spine_forward_lean': "Imagine a string attached to the top of your head, pulling you tall and slightly forward",
            'left_elbow_angle': "Your non-racket arm is a tightrope walker's balance pole - keep it wide and active"
        }
        
        # =====================================================================
        # COMMON MISTAKES: What NOT to do
        # =====================================================================
        self.common_mistakes = {
            'right_elbow_angle': "Don't 'arm' the shot with a straight elbow - this is all arm, no power. Also don't lock your elbow at the top of backswing.",
            'hip_shoulder_separation': "Don't rotate everything together like a robot. Don't start rotation from shoulders - always hips first.",
            'wrist_height_normalized': "Don't reach down for low shuttles - MOVE YOUR FEET. Don't let shuttle drop below shoulder height if you can help it.",
            'chain_straightness': "Don't fire everything at once - that's a push, not a whip. Don't skip leg drive - it's 25% of your power.",
            'spine_forward_lean': "Don't bend at the waist to reach. Don't lean back when hitting - you lose all power."
        }
        
        # =====================================================================
        # SUCCESS INDICATORS: How to know you're doing it right
        # =====================================================================
        self.success_indicators = {
            'right_elbow_angle': "You'll hear a cleaner 'crack' sound at contact. The shuttle will feel lighter to hit. Your arm won't tire as quickly.",
            'hip_shoulder_separation': "You'll feel a stretch in your obliques/side muscles during backswing. Power will feel 'effortless' - less arm work, more rotation.",
            'wrist_height_normalized': "Opponent will have less time to react. Your clears will travel further with less effort. Steeper smash angles.",
            'chain_straightness': "Effortless power - like the racket is doing the work. No arm soreness after long sessions. Consistent shot depth.",
            'spine_forward_lean': "Better balance on recovery. More consistent shot direction. Less lower back fatigue."
        }
        
        # =====================================================================
        # SHOT-SPECIFIC PRIORITIES: What matters most for each shot
        # =====================================================================
        self.shot_priorities = {
            ShotType.FOREHAND_CLEAR: ['wrist_height_normalized', 'hip_shoulder_separation', 'chain_straightness', 'right_elbow_angle'],
            ShotType.FOREHAND_DRIVE: ['wrist_height_normalized', 'spine_forward_lean', 'right_elbow_angle', 'timing'],
            ShotType.FOREHAND_LIFT: ['left_knee_angle', 'wrist_height_normalized', 'spine_forward_lean'],
            ShotType.FOREHAND_NET_SHOT: ['wrist_height_normalized', 'spine_forward_lean', 'right_elbow_angle'],
            ShotType.BACKHAND_DRIVE: ['right_shoulder_elevation', 'right_elbow_angle', 'spine_twist'],
            ShotType.BACKHAND_NET_SHOT: ['spine_forward_lean', 'right_elbow_angle', 'footwork']
        }


class NaturalLanguageCoach:
    """
    Advanced Natural Language Generation for Badminton Coaching.
    Produces human-like, personalized feedback from KSI analysis.
    """
    
    def __init__(self, skill_level: SkillLevel = SkillLevel.INTERMEDIATE):
        self.skill_level = skill_level
        self.knowledge_base = BiomechanicalKnowledgeBase()
        
        # Error severity thresholds (in degrees or normalized units)
        self.thresholds = {
            ErrorSeverity.CRITICAL: 25.0,
            ErrorSeverity.MAJOR: 15.0,
            ErrorSeverity.MINOR: 8.0,
            ErrorSeverity.NEGLIGIBLE: 8.0  # Below this
        }
        
        # Rating scale
        self.rating_scale = [
            (0.95, "⭐⭐⭐⭐⭐ World Class", "elite"),
            (0.90, "⭐⭐⭐⭐⭐ Excellent", "excellent"),
            (0.80, "⭐⭐⭐⭐ Very Good", "very_good"),
            (0.70, "⭐⭐⭐⭐ Good", "good"),
            (0.60, "⭐⭐⭐ Developing", "developing"),
            (0.50, "⭐⭐ Needs Work", "needs_work"),
            (0.0, "⭐ Foundation Building", "foundation")
        ]
    
    def generate_feedback(self, ksi_result, shot_type: ShotType,
                         user_name: Optional[str] = None) -> CoachingFeedback:
        """
        Generate comprehensive natural language coaching feedback.
        
        Args:
            ksi_result: KSIResult from enhanced KSI calculation
            shot_type: Type of badminton shot
            user_name: Optional user name for personalization
        
        Returns:
            CoachingFeedback with complete analysis
        """
        # Extract key data
        ksi_total = ksi_result.ksi_total
        per_joint_errors = ksi_result.per_joint_errors
        phase_scores = ksi_result.phase_scores
        velocity_analysis = ksi_result.velocity_analysis
        
        # Determine rating
        rating, rating_text, rating_key = self._get_rating(ksi_total)
        
        # Classify errors
        classified_errors = self._classify_errors(per_joint_errors, shot_type)
        
        # Generate priority fixes (top 3 most important)
        priority_fixes = self._generate_priority_fixes(classified_errors, shot_type)
        
        # Analyze phases
        phase_analysis = self._analyze_phases(phase_scores)
        
        # Timing and power feedback
        timing_feedback = self._generate_timing_feedback(velocity_analysis)
        power_feedback = self._generate_power_feedback(velocity_analysis, ksi_result.components)
        
        # Identify strengths
        strengths = self._identify_strengths(per_joint_errors, phase_scores)
        
        # Improvement areas
        improvement_areas = self._identify_improvement_areas(classified_errors)
        
        # Generate weekly plan
        weekly_plan = self._generate_weekly_plan(classified_errors, shot_type)
        
        # Summary
        summary = self._generate_summary(
            ksi_total, rating_text, classified_errors, shot_type, user_name
        )
        
        # Motivation
        motivational_message = self._generate_motivation(
            ksi_total, strengths, classified_errors, rating_key
        )
        
        # Technical notes (for advanced users)
        technical_notes = self._generate_technical_notes(ksi_result) if self.skill_level in [SkillLevel.ADVANCED, SkillLevel.EXPERT] else ""
        
        # Video timestamps
        video_timestamps = self._identify_key_moments(ksi_result)
        
        return CoachingFeedback(
            overall_score=ksi_total,
            rating=rating_text,
            summary=summary,
            priority_fixes=priority_fixes,
            phase_analysis=phase_analysis,
            timing_feedback=timing_feedback,
            power_feedback=power_feedback,
            strengths=strengths,
            improvement_areas=improvement_areas,
            weekly_plan=weekly_plan,
            motivational_message=motivational_message,
            technical_notes=technical_notes,
            video_timestamps=video_timestamps
        )
    
    def _get_rating(self, ksi_total: float) -> Tuple[str, str, str]:
        """Get rating based on KSI score."""
        for threshold, text, key in self.rating_scale:
            if ksi_total >= threshold:
                return text, text, key
        return self.rating_scale[-1]
    
    def _classify_errors(self, per_joint_errors: Dict, shot_type: ShotType) -> Dict[ErrorSeverity, List]:
        """
        Classify errors by severity with shot-type weighting.
        """
        classified = {s: [] for s in ErrorSeverity}
        
        # Get shot-specific priorities
        priorities = self.knowledge_base.shot_priorities.get(shot_type, [])
        
        for joint_name, error_data in per_joint_errors.items():
            # Get error magnitude
            if hasattr(error_data, 'mean_error'):
                magnitude = error_data.mean_error
            elif isinstance(error_data, dict):
                magnitude = error_data.get('mean_error', 0)
            else:
                continue
            
            # Boost priority joints
            effective_magnitude = magnitude
            if joint_name in priorities[:2]:  # Top 2 priorities
                effective_magnitude *= 1.5
            elif joint_name in priorities[2:4]:  # Secondary priorities
                effective_magnitude *= 1.2
            
            # Classify
            if effective_magnitude >= self.thresholds[ErrorSeverity.CRITICAL]:
                severity = ErrorSeverity.CRITICAL
            elif effective_magnitude >= self.thresholds[ErrorSeverity.MAJOR]:
                severity = ErrorSeverity.MAJOR
            elif effective_magnitude >= self.thresholds[ErrorSeverity.MINOR]:
                severity = ErrorSeverity.MINOR
            else:
                severity = ErrorSeverity.NEGLIGIBLE
            
            classified[severity].append({
                'joint_name': joint_name,
                'magnitude': magnitude,
                'effective_magnitude': effective_magnitude,
                'error_data': error_data,
                'is_priority': joint_name in priorities[:4]
            })
        
        # Sort each category by effective magnitude
        for severity in classified:
            classified[severity].sort(key=lambda x: x['effective_magnitude'], reverse=True)
        
        return classified
    
    def _generate_priority_fixes(self, classified_errors: Dict, 
                                 shot_type: ShotType) -> List[CorrectionItem]:
        """
        Generate detailed corrections for top priority errors.
        """
        priority_fixes = []
        
        # Take top errors: 2 critical (if any), 1 major
        errors_to_fix = (
            classified_errors[ErrorSeverity.CRITICAL][:2] +
            classified_errors[ErrorSeverity.MAJOR][:1]
        )[:3]
        
        for error in errors_to_fix:
            joint_name = error['joint_name']
            magnitude = error['magnitude']
            
            correction = self._create_correction_item(joint_name, magnitude, shot_type)
            if correction:
                priority_fixes.append(correction)
        
        return priority_fixes
    
    def _create_correction_item(self, joint_name: str, magnitude: float,
                               shot_type: ShotType) -> Optional[CorrectionItem]:
        """
        Create a detailed correction item for a specific error.
        """
        kb = self.knowledge_base
        
        # Get knowledge base entry
        if joint_name not in kb.causal_chains:
            return self._create_generic_correction(joint_name, magnitude)
        
        chain = kb.causal_chains[joint_name]
        
        # Determine error type based on joint and magnitude
        error_type = self._determine_error_type(joint_name, magnitude)
        
        # Get causes
        causes = chain['causes'].get(error_type, chain['causes'].get(list(chain['causes'].keys())[0], []))
        primary_cause = causes[0] if causes else ('unknown', 'Cause unclear')
        
        # Build issue description
        issue = self._build_issue_description(chain['error_name'], magnitude, error_type)
        
        # Get appropriate fix for skill level
        fix = kb.fixes.get(joint_name, {}).get(self.skill_level, "Focus on correcting this through deliberate practice.")
        
        # Get drill
        drill_info = kb.drills.get(joint_name, {})
        drill_key = self.skill_level.value if self.skill_level != SkillLevel.EXPERT else 'advanced'
        drill = drill_info.get(drill_key, drill_info.get('beginner', "Practice slowly in front of mirror."))
        
        return CorrectionItem(
            issue=issue,
            severity=ErrorSeverity.CRITICAL if magnitude > self.thresholds[ErrorSeverity.CRITICAL] else ErrorSeverity.MAJOR,
            cause=primary_cause[0],
            explanation=f"**Why this happens:** {primary_cause[1]}\n\n**Biomechanics:** {chain['biomechanical_principle']}",
            fix=fix,
            drill=f"**{drill_info.get('name', 'Practice Drill')}:** {drill}\n\n*Duration: {drill_info.get('duration', '10 minutes daily')}*",
            visual_cue=kb.visual_cues.get(joint_name, "Focus on the correct form."),
            common_mistake=kb.common_mistakes.get(joint_name, "Avoid rushing the movement."),
            success_indicator=kb.success_indicators.get(joint_name, "Movement feels smoother and more powerful.")
        )
    
    def _create_generic_correction(self, joint_name: str, magnitude: float) -> CorrectionItem:
        """Create generic correction for joints not in knowledge base."""
        readable_name = joint_name.replace('_', ' ').title()
        
        return CorrectionItem(
            issue=f"Your {readable_name} shows a {magnitude:.1f}° deviation from optimal technique.",
            severity=ErrorSeverity.MAJOR,
            cause="technical",
            explanation=f"Your {readable_name} positioning needs adjustment for better performance.",
            fix=f"Focus on your {readable_name} during practice. Use video analysis to compare with expert footage.",
            drill=f"Practice the movement slowly, focusing on {readable_name} position. Use a mirror or video for feedback.",
            visual_cue=f"Imagine the correct {readable_name} position and try to replicate it.",
            common_mistake=f"Don't ignore {readable_name} positioning - it affects overall technique.",
            success_indicator="You'll feel more comfortable and powerful when the position is correct."
        )
    
    def _determine_error_type(self, joint_name: str, magnitude: float) -> str:
        """Determine specific error type based on joint and magnitude."""
        # This would ideally use the actual angle values, not just magnitude
        # Simplified logic for now
        if 'elbow' in joint_name:
            return 'too_bent' if magnitude > 20 else 'dropping'
        elif 'hip_shoulder' in joint_name:
            return 'insufficient'
        elif 'wrist' in joint_name:
            return 'too_low'
        elif 'spine' in joint_name:
            return 'excessive_forward' if magnitude > 0 else 'excessive_back'
        elif 'knee' in joint_name:
            return 'straight' if magnitude > 15 else 'collapsed'
        return list(self.knowledge_base.causal_chains.get(joint_name, {}).get('causes', {'default': []}).keys())[0]
    
    def _build_issue_description(self, error_name: str, magnitude: float, error_type: str) -> str:
        """Build human-readable issue description."""
        severity_word = "significantly" if magnitude > 20 else "noticeably" if magnitude > 12 else "slightly"
        
        type_descriptions = {
            'too_bent': 'is bending too much',
            'too_straight': 'is too straight/locked',
            'dropping': 'is dropping too low',
            'insufficient': 'shows insufficient rotation',
            'too_low': 'is too low at contact',
            'excessive_forward': 'shows too much forward lean',
            'excessive_back': 'shows backward lean',
            'straight': 'is too straight (locked)',
            'collapsed': 'is collapsing inward',
            'passive': 'is not actively engaged',
            'not_engaged': 'is not being used effectively'
        }
        
        type_desc = type_descriptions.get(error_type, 'is not optimal')
        
        return f"🔴 **{error_name}** {severity_word} {type_desc} ({magnitude:.1f}° deviation)"
    
    def _analyze_phases(self, phase_scores: Dict) -> Dict[str, str]:
        """Generate natural language analysis for each shot phase."""
        analysis = {}
        
        phase_descriptions = {
            'preparation': {
                'good': "Your ready position and initial setup look solid. Good foundation.",
                'bad': "Your preparation phase needs work. You're not setting up properly before the swing. This affects everything that follows."
            },
            'loading': {
                'good': "Nice backswing and loading phase. You're storing energy well.",
                'bad': "Your loading/backswing phase is incomplete. You're not generating enough potential energy for a powerful shot."
            },
            'acceleration': {
                'good': "Excellent acceleration through the ball! This is where champions are made.",
                'bad': "Your acceleration phase shows issues. You're not building speed effectively toward contact."
            },
            'contact': {
                'good': "Clean contact! Your technique at impact is working well.",
                'bad': "Contact point issues detected. This is critical - you need to make cleaner contact."
            },
            'follow_through': {
                'good': "Good follow-through. You're completing your shots properly.",
                'bad': "Incomplete follow-through. You're cutting your swing short, which reduces power and increases injury risk."
            }
        }
        
        for phase, score in phase_scores.items():
            if phase in phase_descriptions:
                quality = 'good' if score >= 0.75 else 'bad'
                base_text = phase_descriptions[phase][quality]
                analysis[phase] = f"{base_text} (Phase score: {score:.2f})"
        
        return analysis
    
    def _generate_timing_feedback(self, velocity_analysis: Dict) -> str:
        """Generate natural language timing feedback."""
        timing_offset = velocity_analysis.get('timing_offset_frames', 0)
        timing_ms = velocity_analysis.get('timing_offset_ms', 0)
        
        feedback_parts = []
        
        if abs(timing_offset) <= 2:
            feedback_parts.append("✅ **Excellent timing!** Your peak acceleration aligns well with the expert template.")
        elif timing_offset < -2:
            feedback_parts.append(f"⏰ **Early peak:** You're reaching maximum speed {abs(timing_ms):.0f}ms too early.")
            feedback_parts.append("This means you're decelerating by the time you hit the shuttle.")
            
            if self.skill_level == SkillLevel.BEGINNER:
                feedback_parts.append("**Fix:** Count '1-2-3' during your swing. The '3' should be at contact, not before.")
            else:
                feedback_parts.append("**Fix:** Delay your forward swing initiation. Hold the loaded position 50-100ms longer.")
        else:
            feedback_parts.append(f"⏰ **Late peak:** You're reaching maximum speed {timing_ms:.0f}ms after optimal.")
            feedback_parts.append("You're still accelerating after contact, wasting energy.")
            
            if self.skill_level == SkillLevel.BEGINNER:
                feedback_parts.append("**Fix:** Start your forward swing earlier. React faster to the shuttle.")
            else:
                feedback_parts.append("**Fix:** Speed up your preparation phase. Better anticipation = earlier swing initiation.")
        
        return "\n".join(feedback_parts)
    
    def _generate_power_feedback(self, velocity_analysis: Dict, components: Dict) -> str:
        """Generate natural language power/velocity feedback."""
        peak_ratio = velocity_analysis.get('peak_velocity_ratio', 1.0)
        
        feedback_parts = []
        
        if peak_ratio >= 0.95:
            feedback_parts.append("💪 **Excellent power generation!** You're matching expert racket head speed.")
        elif peak_ratio >= 0.80:
            feedback_parts.append(f"💪 **Good power:** You're generating {peak_ratio:.0%} of expert speed.")
            feedback_parts.append("Room for improvement, but solid foundation.")
        elif peak_ratio >= 0.60:
            feedback_parts.append(f"⚡ **Power deficit:** Only achieving {peak_ratio:.0%} of potential speed.")
            feedback_parts.append(f"You're leaving {(1-peak_ratio)*100:.0f}% of power on the table!")
            
            # Diagnose cause
            if components.get('acceleration', 1.0) < 0.6:
                feedback_parts.append("**Cause:** Insufficient acceleration - you're not exploding through contact.")
            else:
                feedback_parts.append("**Cause:** Likely kinetic chain issues - power not transferring efficiently.")
        else:
            feedback_parts.append(f"🔴 **Significant power gap:** Only {peak_ratio:.0%} of expert speed.")
            feedback_parts.append("Focus on fundamentals before power - technique first, then speed.")
        
        return "\n".join(feedback_parts)
    
    def _identify_strengths(self, per_joint_errors: Dict, phase_scores: Dict) -> List[str]:
        """Identify what the user is doing well."""
        strengths = []
        
        # Check per-joint errors for low errors
        for joint_name, error_data in per_joint_errors.items():
            magnitude = error_data.mean_error if hasattr(error_data, 'mean_error') else error_data.get('mean_error', 999)
            
            if magnitude < 5:
                readable_name = joint_name.replace('_', ' ').replace('right ', '').replace('left ', '').title()
                strengths.append(f"✅ Excellent {readable_name} positioning")
        
        # Check phase scores
        for phase, score in phase_scores.items():
            if score >= 0.85:
                strengths.append(f"✅ Strong {phase.replace('_', ' ')} phase")
        
        return strengths[:5]  # Top 5 strengths
    
    def _identify_improvement_areas(self, classified_errors: Dict) -> List[str]:
        """List areas needing improvement."""
        areas = []
        
        for severity in [ErrorSeverity.CRITICAL, ErrorSeverity.MAJOR]:
            for error in classified_errors[severity][:3]:
                readable_name = error['joint_name'].replace('_', ' ').title()
                areas.append(f"• {readable_name} ({error['magnitude']:.1f}° off)")
        
        return areas
    
    def _generate_weekly_plan(self, classified_errors: Dict, shot_type: ShotType) -> Dict[str, List[str]]:
        """Generate a progressive weekly training plan."""
        
        # Get top errors to address
        top_errors = (
            classified_errors[ErrorSeverity.CRITICAL][:2] +
            classified_errors[ErrorSeverity.MAJOR][:2]
        )[:3]
        
        plan = {
            'week_1': [
                "🎯 **Focus: Foundation & Awareness**",
                "• Review video analysis, identify your key errors",
                "• 15 min daily: Slow-motion shadow swings focusing on correct positions"
            ],
            'week_2': [
                "🎯 **Focus: Isolated Corrections**",
                "• Target one error at a time (don't try to fix everything)",
            ],
            'week_3': [
                "🎯 **Focus: Integration**",
                "• Combine corrections into full shot",
                "• Controlled multi-shuttle practice (50% speed)",
                "• Video yourself and compare to week 1"
            ],
            'week_4': [
                "🎯 **Focus: Game Speed**",
                "• Match-pace drills with focus on maintaining corrections",
                "• Point play with technical focus",
                "• Re-test: Record another video for KSI comparison"
            ]
        }
        
        # Add specific drills for top errors
        for i, error in enumerate(top_errors):
            joint_name = error['joint_name']
            drill_info = self.knowledge_base.drills.get(joint_name, {})
            drill_name = drill_info.get('name', f"{joint_name} drill")
            
            if i < 2:  # First 2 errors in week 1-2
                plan['week_1'].append(f"• {drill_name} (from knowledge base)")
                plan['week_2'].append(f"• Continue {drill_name} with progression")
            else:  # 3rd error in week 2-3
                plan['week_2'].append(f"• Introduce {drill_name}")
                plan['week_3'].append(f"• {drill_name} at game speed")
        
        return plan
    
    def _generate_summary(self, ksi_total: float, rating: str, 
                         classified_errors: Dict, shot_type: ShotType,
                         user_name: Optional[str]) -> str:
        """Generate opening summary paragraph."""
        
        name_prefix = f"Hi {user_name}! " if user_name else ""
        shot_name = shot_type.value.replace('_', ' ').title()
        
        n_critical = len(classified_errors[ErrorSeverity.CRITICAL])
        n_major = len(classified_errors[ErrorSeverity.MAJOR])
        
        if ksi_total >= 0.90:
            return f"""{name_prefix}Excellent work on this **{shot_name}**! 🏆

Your technique scores **{ksi_total:.2f}** ({rating}), which means you're executing at near-expert level. You have the fundamentals mastered - now it's about fine-tuning the details for that extra 5-10% performance gain."""
        
        elif ksi_total >= 0.75:
            return f"""{name_prefix}Solid **{shot_name}** technique! 👍

Your score of **{ksi_total:.2f}** ({rating}) shows you understand the mechanics well. There {'is 1 major area' if n_major == 1 else f'are {n_major} areas'} to refine, which will unlock the next level of your game."""
        
        elif ksi_total >= 0.60:
            return f"""{name_prefix}Your **{shot_name}** shows potential! 📈

Scoring **{ksi_total:.2f}** ({rating}) indicates room for significant improvement. I've identified {n_critical + n_major} key technical issues. Don't worry - these are very fixable with focused practice."""
        
        else:
            return f"""{name_prefix}Let's rebuild your **{shot_name}** from the ground up. 💪

Your score of **{ksi_total:.2f}** ({rating}) indicates fundamental technique issues, but that's actually good news - it means small changes will create big improvements! Focus on the basics below, and you'll see rapid progress."""
    
    def _generate_motivation(self, ksi_total: float, strengths: List[str],
                            classified_errors: Dict, rating_key: str) -> str:
        """Generate closing motivational message."""
        
        n_strengths = len(strengths)
        
        messages = {
            'elite': "You're performing at an elite level. Stay sharp, keep analyzing, and maintain your edge! 🏆",
            'excellent': f"With {n_strengths} strong technical elements, you're on the path to excellence. Keep pushing! 🌟",
            'very_good': "You're doing many things right! Focus on the priority fixes above and watch your game transform. 💪",
            'good': "Solid foundation in place. The corrections above will accelerate your improvement dramatically. 📈",
            'developing': "You're learning and improving - that's what matters! Take it one fix at a time. Consistency beats intensity. 🎯",
            'needs_work': "Every expert was once a beginner. The fact that you're analyzing your technique shows you're serious. Trust the process! 🚀",
            'foundation': "The best players rebuild their technique multiple times. This is YOUR foundation moment. Small steps, big results! 💯"
        }
        
        return messages.get(rating_key, "Keep practicing, keep analyzing, keep improving! 💪")
    
    def _generate_technical_notes(self, ksi_result) -> str:
        """Generate technical notes for advanced users."""
        
        components = ksi_result.components
        confidence = ksi_result.confidence
        
        notes = [
            "### Technical Analysis (Advanced)",
            f"- **Pose Similarity:** {components.get('pose', 0):.3f}",
            f"- **Velocity Coherence:** {components.get('velocity', 0):.3f}",
            f"- **Acceleration Profile:** {components.get('acceleration', 0):.3f}",
            f"- **Jerk (Smoothness):** {components.get('jerk', 0):.3f}",
            f"- **DTW Distance:** {components.get('dtw_distance', 0):.2f}",
            "",
            f"- **Confidence Interval (95%):** [{confidence.get('ci_95_lower', 0):.3f}, {confidence.get('ci_95_upper', 0):.3f}]",
            f"- **Analysis Reliability:** {'High' if confidence.get('reliable', False) else 'Moderate'}"
        ]
        
        return "\n".join(notes)
    
    def _identify_key_moments(self, ksi_result) -> List[Dict]:
        """Identify key frames/moments for video review."""
        
        timestamps = []
        velocity = ksi_result.velocity_analysis
        
        # Peak velocity moment
        if 'user_peak_frame' in velocity:
            timestamps.append({
                'frame': velocity['user_peak_frame'],
                'label': 'Peak Velocity (Contact)',
                'importance': 'critical',
                'note': 'This is your impact moment - check body position here'
            })
        
        # Expert peak for comparison
        if 'expert_peak_frame' in velocity:
            timestamps.append({
                'frame': velocity['expert_peak_frame'],
                'label': 'Expert Peak (Reference)',
                'importance': 'reference',
                'note': 'Expert reaches peak velocity here'
            })
        
        # Worst frame for top error
        for joint_name, error_data in list(ksi_result.per_joint_errors.items())[:3]:
            critical_frame = error_data.critical_frame if hasattr(error_data, 'critical_frame') else error_data.get('critical_frame', 0)
            timestamps.append({
                'frame': critical_frame,
                'label': f'Error Peak: {joint_name}',
                'importance': 'error',
                'note': f'Largest deviation in {joint_name}'
            })
        
        return timestamps
    
    def format_feedback_text(self, feedback: CoachingFeedback, simplified: bool = False) -> str:
        """
        Format CoachingFeedback into readable text output.
        
        Args:
            feedback: CoachingFeedback object
            simplified: If True, removes weekly plan and shortens output
        """
        sections = []
        
        # Header
        sections.append("=" * 70)
        sections.append("🏸 BADMINTON COACHING ANALYSIS REPORT")
        sections.append("=" * 70)
        sections.append("")
        
        # Summary
        sections.append(feedback.summary)
        sections.append("")
        
        # Overall Score
        sections.append(f"**Overall Score: {feedback.overall_score:.2f}** | {feedback.rating}")
        sections.append("-" * 70)
        sections.append("")
        
        # Priority Fixes (limit to 2 in simplified mode)
        max_fixes = 2 if simplified else 3
        sections.append("## 🎯 PRIORITY FIXES")
        sections.append("")
        
        for i, fix in enumerate(feedback.priority_fixes[:max_fixes], 1):
            sections.append(f"### {i}. {fix.issue}")
            sections.append("")
            if not simplified:  # Full explanation only in non-simplified mode
                sections.append(fix.explanation)
                sections.append("")
            sections.append(f"**How to fix:** {fix.fix}")
            sections.append("")
            sections.append(fix.drill)
            sections.append("")
            if not simplified:
                sections.append(f"💡 **Visual cue:** {fix.visual_cue}")
                sections.append(f"⚠️ **Avoid:** {fix.common_mistake}")
            sections.append(f"✅ **Success looks like:** {fix.success_indicator}")
            sections.append("")
            sections.append("-" * 40)
            sections.append("")
        
        # Timing & Power
        sections.append("## ⏱️ TIMING ANALYSIS")
        sections.append(feedback.timing_feedback)
        sections.append("")
        sections.append("## 💪 POWER ANALYSIS")
        sections.append(feedback.power_feedback)
        sections.append("")
        
        # Phase Analysis
        sections.append("## 📊 PHASE BREAKDOWN")
        for phase, analysis in feedback.phase_analysis.items():
            sections.append(f"**{phase.replace('_', ' ').title()}:** {analysis}")
        sections.append("")
        
        # Strengths (limit to 3 in simplified mode)
        if feedback.strengths:
            max_strengths = 3 if simplified else 5
            sections.append("## 🌟 YOUR STRENGTHS")
            for strength in feedback.strengths[:max_strengths]:
                sections.append(strength)
            sections.append("")
        
        # Weekly Plan (skip in simplified mode)
        if not simplified:
            sections.append("## 📅 4-WEEK IMPROVEMENT PLAN")
            for week, activities in feedback.weekly_plan.items():
                sections.append(f"\n**{week.replace('_', ' ').title()}**")
                for activity in activities:
                    sections.append(activity)
            sections.append("")
        
        # Technical Notes (if present)
        if feedback.technical_notes and not simplified:
            sections.append(feedback.technical_notes)
            sections.append("")
        
        # Motivation
        sections.append("-" * 70)
        sections.append(f"## 💡 {feedback.motivational_message}")
        sections.append("=" * 70)
        
        return "\n".join(sections)
    
    def format_feedback_json(self, feedback: CoachingFeedback, simplified: bool = False) -> str:
        """Format feedback as JSON for API responses.
        
        Args:
            feedback: CoachingFeedback object
            simplified: If True, excludes weekly plan
        """
        data = {
            'overall_score': feedback.overall_score,
            'rating': feedback.rating,
            'summary': feedback.summary,
            'priority_fixes': [
                {
                    'issue': f.issue,
                    'severity': f.severity.value,
                    'cause': f.cause,
                    'explanation': f.explanation if not simplified else None,
                    'fix': f.fix,
                    'drill': f.drill,
                    'visual_cue': f.visual_cue if not simplified else None,
                    'common_mistake': f.common_mistake if not simplified else None,
                    'success_indicator': f.success_indicator
                }
                for f in feedback.priority_fixes[:(2 if simplified else 3)]
            ],
            'phase_analysis': feedback.phase_analysis,
            'timing_feedback': feedback.timing_feedback,
            'power_feedback': feedback.power_feedback,
            'strengths': feedback.strengths[:(3 if simplified else 5)],
            'improvement_areas': feedback.improvement_areas,
            'motivational_message': feedback.motivational_message,
            'technical_notes': feedback.technical_notes if not simplified else None
        }
        
        # Include weekly plan only if not simplified
        if not simplified:
            data['weekly_plan'] = feedback.weekly_plan
            data['video_timestamps'] = feedback.video_timestamps
        
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        
        return json.dumps(data, indent=2)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_coaching_report(ksi_result, shot_type_str: str, 
                            skill_level_str: str = 'intermediate',
                            user_name: Optional[str] = None,
                            output_format: str = 'text',
                            simplified: bool = False) -> str:
    """
    Convenience function to generate complete coaching report.
    
    Args:
        ksi_result: KSIResult from ksi_v2.py
        shot_type_str: Shot type as string (e.g., 'forehand_clear')
        skill_level_str: User skill level ('beginner', 'intermediate', 'advanced', 'expert')
        user_name: Optional user name for personalization
        output_format: 'text' or 'json'
        simplified: If True, removes weekly plan and shortens output
    
    Returns:
        Formatted coaching report as string
    """
    # Parse enums
    try:
        shot_type = ShotType(shot_type_str)
    except ValueError:
        shot_type = ShotType.FOREHAND_CLEAR
    
    try:
        skill_level = SkillLevel(skill_level_str)
    except ValueError:
        skill_level = SkillLevel.INTERMEDIATE
    
    # Create coach and generate feedback
    coach = NaturalLanguageCoach(skill_level=skill_level)
    feedback = coach.generate_feedback(ksi_result, shot_type, user_name)
    
    # Format output
    if output_format == 'json':
        return coach.format_feedback_json(feedback, simplified=simplified)
    else:
        return coach.format_feedback_text(feedback, simplified=simplified)
