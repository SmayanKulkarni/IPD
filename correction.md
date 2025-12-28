# Technical Specification: Enhanced 3D Badminton Posture Analysis & Correction System

## Version 2.0 - Research Edition

This document provides comprehensive technical documentation of the enhanced system architecture, novel research contributions, and the advanced **Kinetic Similarity Index (KSI) v2.0** with natural language coaching output.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Research Novelties](#2-research-novelties)
3. [Mathematical Foundation](#3-mathematical-foundation)
4. [Enhanced KSI v2.0](#4-enhanced-ksi-v20)
5. [Natural Language Coaching System](#5-natural-language-coaching-system)
6. [Phase Segmentation](#6-phase-segmentation)
7. [Confidence Filtering](#7-confidence-filtering)
8. [API Reference](#8-api-reference)

---

## 1. System Architecture Overview

### 1.1 Enhanced Pipeline Flow

The system is designed as a multi-stage pipeline that converts raw video into actionable biomechanical feedback with natural language coaching output.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT: Video Stream                              │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    POSE EXTRACTION (MediaPipe)                           │
│  • model_complexity=2 for accuracy                                       │
│  • 33 landmarks × 3 coordinates = 99 features per frame                  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    QUALITY FILTERING [NEW v2.0]                          │
│  • Landmark visibility validation                                        │
│  • Temporal consistency checking                                         │
│  • Physics-based motion constraint validation                            │
│  • Adaptive interpolation and smoothing                                  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE SEGMENTATION [NEW v2.0]                         │
│  • Automatic shot phase detection                                        │
│  • Preparation → Loading → Acceleration → Contact → Follow-through       │
│  • Velocity-based boundary detection                                     │
│  • Per-phase quality scoring                                             │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ENHANCED KSI CALCULATION [NEW v2.0]                   │
│  • 36-feature biomechanical extraction (vs. 12 original)                 │
│  • Per-joint error breakdown with critical frame identification          │
│  • Temporal attention weighting                                          │
│  • Bootstrap confidence intervals                                        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    NATURAL LANGUAGE COACH [NEW v2.0]                     │
│  • Skill-level adaptive feedback                                         │
│  • Causal reasoning for errors                                           │
│  • Biomechanical knowledge base                                          │
│  • Personalized drill recommendations                                    │
│  • Weekly training plan generation                                       │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT: Coaching Report                          │
│  • Human-readable feedback                                               │
│  • Priority fixes with explanations                                      │
│  • Video timestamps for review                                           │
│  • Progress tracking metrics                                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Original Pipeline (Reference)

1. **Input:** Dual-player broadcast or practice footage.
2. **ROI Filtering:** Segmenting the "Player Behind" via Y-coordinate centroids.
3. **Skeletal Extraction:** Generating a time-series of 3D coordinates.
4. **Action Classification:** Identifying the shot type (e.g., Smash vs. Clear) using an LSTM network.
5. **Dynamic Alignment:** Syncing user timing with expert timing.
6. **Biomechanical Scoring:** Applying the KSI formula to calculate deviations.
7. **Correction Logic:** Mapping score ranges to natural language feedback.

---

## 2. Mathematical Foundation

### 2.1 Spatial Normalization (Pose Invariance)

To compare a user to an expert, we must eliminate variables like camera distance and height. We transform raw coordinates  into a relative space centered at the Mid-Hip .

Where  is the distance between the left and right shoulders, acting as a scale constant.

### 2.2 Dynamic Time Warping (DTW)

Since players move at different speeds, we find a warping path  that minimizes the distance between the expert sequence  and user sequence .

The cost function for the alignment is:



where  is the Euclidean distance between joints in 3D space.

---

## 3. The Kinetic Similarity Index (KSI)

The KSI is the core novelty of this system. It evaluates motion across three physical dimensions: **Shape**, **Flow**, and **Force**.

### 3.1 Postural Component ()

Uses **Cosine Similarity** to measure angular accuracy. For a joint like the elbow, we define two vectors:  (Shoulder to Elbow) and  (Elbow to Wrist).

* **Logic:** If the user’s elbow is tucked or flared incorrectly, the vector direction changes, lowering the score regardless of speed.

### 3.2 Velocity Coherence ()

Analyzes the **direction and speed** of movement using finite differences.


The similarity is calculated as:


* **Logic:** The Gaussian part  ensures that if the user moves in the right direction but is too slow, the score drops gracefully.

### 3.3 Acceleration Profile ()

Captures the **explosiveness** or "whip" of the shot.


* **Logic:** In a smash, peak acceleration happens in a split second. This formula detects if the user's "power spike" matches the expert's timing and intensity.

---

## 4. Deep Learning Classification

To ensure we compare a "Smash" to a "Smash," an LSTM (Long Short-Term Memory) network is used. LSTMs are preferred because they maintain a "memory" of previous frames, which is essential for movement.

### Model Architecture:

* **Input Layer:**  — (33 landmarks  3 coordinates).
* **LSTM Layer 1:** 64 units (extracts basic temporal patterns).
* **LSTM Layer 2:** 128 units (captures complex shot dynamics).
* **Dropout Layer:** 0.2 (prevents overfitting to specific players).
* **Dense Output:** Softmax activation for shot probability.

---

## 5. Enhanced KSI v2.0 [NEW]

### 5.1 Extended Feature Set (36 Features)

The enhanced system extracts 36 biomechanical features:

**Joint Angles (12 features)**
- `right_elbow`, `left_elbow`
- `right_shoulder`, `left_shoulder`
- `right_hip`, `left_hip`
- `right_knee`, `left_knee`
- `right_ankle`, `left_ankle`
- `spine_mid`, `neck`

**Advanced Metrics (12 features)**
- `hip_shoulder_separation` - Rotational stretch (X-factor)
- `trunk_rotation` - Torso twist angle
- `spine_forward_lean` - Sagittal plane posture
- `spine_lateral_lean` - Frontal plane posture
- `shoulder_plane_angle` - Shoulder alignment
- `hip_plane_angle` - Hip alignment
- `chain_straightness` - Kinetic chain alignment
- `center_of_mass_x/y/z` - Balance metrics (3)
- `base_of_support` - Stance width
- `vertical_oscillation` - Up-down movement

**Limb Ratios (8 features)**
- `right_arm_ratio`, `left_arm_ratio` - Upper/lower arm
- `right_leg_ratio`, `left_leg_ratio` - Thigh/shin
- `arm_leg_ratio_right/left` - Cross-limb ratios
- `wingspan_height_ratio` - Body proportions
- `shoulder_hip_ratio`

**Contact Point Metrics (4 features)**
- `wrist_height_normalized` - Contact height
- `wrist_forward_reach` - Extension
- `racket_arm_extension` - Full extension %
- `contact_body_distance` - Clearance from body

### 5.2 Temporal Attention Mechanism

Phase-based weights prioritize critical moments:

| Phase | Weight | Rationale |
|-------|--------|-----------|
| Preparation | 0.10 | Setup foundation |
| Loading | 0.15 | Energy storage |
| Acceleration | 0.25 | Power generation |
| Contact | 0.30 | Impact quality |
| Follow-through | 0.20 | Completion/safety |

---

## 6. Natural Language Coaching System [NEW]

### 6.1 Skill-Level Adaptation

| Level | Vocabulary | Detail Level | Focus |
|-------|------------|--------------|-------|
| Beginner | Simple, analogies | Low | Basic positions |
| Intermediate | Technical terms introduced | Medium | Timing, sequencing |
| Advanced | Full technical | High | Optimization |
| Expert | Biomechanical | Maximum | Millisecond timing |

### 6.2 Causal Reasoning Knowledge Base

The system includes a comprehensive knowledge base mapping errors to causes and fixes:
- Error → Possible Causes → Root Issues
- Skill-appropriate fixes and drills
- Visual cues and success indicators
- Common mistakes to avoid

---

## 7. Phase Segmentation [NEW]

### 7.1 Automatic Phase Detection

| Phase | Duration (typical) | Key Markers |
|-------|-------------------|-------------|
| Preparation | 100-300ms | Ready position, shuttle tracking |
| Loading | 150-400ms | Weight transfer, racket back |
| Acceleration | 80-200ms | Forward swing initiation |
| Contact | 20-50ms | Impact moment |
| Follow-through | 100-300ms | Deceleration, recovery |

### 7.2 Kinetic Chain Analysis

Proximal-to-distal sequencing validation:
- Optimal: Hip → Shoulder → Elbow → Wrist
- Delays: 0ms → 50ms → 85ms → 120ms

---

## 8. Correction Dataset Logic

The system maps the mathematical output of the KSI components to a **Correction Dataset**. This allows the model to produce "Human-in-the-loop" style feedback.

### Feedback Matrix:

| Component | Threshold | Error Detection | Correction Statement |
| --- | --- | --- | --- |
|  |  | Geometric mismatch | "Elbow movement is inaccurate: check your arm angle." |
|  |  | Wrong swing path | "Swing trajectory is off: follow through across your body." |
|  |  | Lack of power/snap | "Inadequate wrist snap: accelerate faster at contact." |

---

## 9. ROI Filtering Strategy

Since the dataset contains two players, the system uses a **Spatial Centroid Filter**.
For every frame , we calculate the average  position:


If  (normalized coordinate), the detection is discarded as the "Player in Front." Only  is passed to the analysis engine.

---

## 10. API Reference [NEW]

### Quick Start

```python
from enhanced_api import analyze_shot, get_quick_feedback

# Full analysis
result = analyze_shot(
    user_poses,           # Shape (T, 33, 3)
    expert_poses,         # Shape (T, 33, 3)
    shot_type='forehand_clear',
    skill_level='intermediate',
    user_name='Player'
)

print(result.ksi_score)           # 0.78
print(result.coaching_report)     # Full text report
print(result.priority_fixes)      # List of top issues
```

### Module Overview

| Module | File | Purpose |
|--------|------|---------|
| Enhanced KSI | `src/ksi_v2.py` | 36-feature biomechanical analysis |
| NL Coach | `src/natural_language_coach.py` | Natural language feedback generation |
| Phase Segmentation | `src/phase_segmentation.py` | Automatic shot phase detection |
| Confidence Filtering | `src/confidence_filtering.py` | Pose quality validation |
| Unified API | `src/enhanced_api.py` | Clean interface for all features |

---

## 11. Research Novelties Summary

1. **Extended Biomechanical Feature Set** - 36 vs 12 original features
2. **Per-Joint Error Attribution** - Fine-grained error localization
3. **Temporal Attention Mechanism** - Phase-based importance weighting
4. **Biomechanical Causal Reasoning** - Knowledge base for error explanations
5. **Adaptive Natural Language Generation** - Four skill levels
6. **Confidence-Aware Analysis** - Bootstrap confidence intervals

---

## 12. Future Expansion: Differential Similarity

The next phase of this project will replace the Gaussian magnitude similarity with a **Differential Similarity Metric**, treating the joints as a system of Ordinary Differential Equations (ODEs):



By comparing the damping coefficient () and stiffness () of a user's limb to an expert's, we can provide feedback on muscle tension and fluidity.

---

*Document Version: 2.0.0*
*Last Updated: 2024*
*Authors: IPD Research Team*