# Graph Report - /home/smayan/Desktop/IPD  (2026-04-19)

## Corpus Check
- 28 files · ~9,039,126 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 433 nodes · 767 edges · 18 communities detected
- Extraction: 75% EXTRACTED · 25% INFERRED · 0% AMBIGUOUS · INFERRED: 194 edges (avg confidence: 0.64)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]

## God Nodes (most connected - your core abstractions)
1. `EnhancedKSI` - 50 edges
2. `HybridFeatureExtractor` - 29 edges
3. `ShotPhase` - 27 edges
4. `NaturalLanguageCoach` - 24 edges
5. `ShotPhaseSegmenter` - 23 edges
6. `MLflowRunManager` - 22 edges
7. `PhaseSegmenter` - 20 edges
8. `PoseFeatureExtractor` - 19 edges
9. `main()` - 15 edges
10. `main()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `PredictionSmoother` --uses--> `HybridFeatureExtractor`  [INFERRED]
  /home/smayan/Desktop/IPD/app.py → /home/smayan/Desktop/IPD/src/features.py
- `PredictionSmoother` --uses--> `EnhancedKSI`  [INFERRED]
  /home/smayan/Desktop/IPD/app.py → /home/smayan/Desktop/IPD/src/ksi_v2.py
- `ServiceState` --uses--> `HybridFeatureExtractor`  [INFERRED]
  /home/smayan/Desktop/IPD/app.py → /home/smayan/Desktop/IPD/src/features.py
- `ServiceState` --uses--> `EnhancedKSI`  [INFERRED]
  /home/smayan/Desktop/IPD/app.py → /home/smayan/Desktop/IPD/src/ksi_v2.py
- `_startup()` --calls--> `HybridFeatureExtractor`  [INFERRED]
  /home/smayan/Desktop/IPD/app.py → /home/smayan/Desktop/IPD/src/features.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (36): Enum, BiomechanicalKnowledgeBase, CoachingFeedback, CorrectionItem, ErrorSeverity, generate_coaching_report(), NaturalLanguageCoach, Natural Language Coach - AI-Powered Badminton Coaching Assistant =============== (+28 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (48): _apply_crop(), evaluate_video(), extract_features_from_video(), find_contact_moment(), load_expert_templates(), load_params(), predict_shot_type(), predict_shot_type_at_contact() (+40 more)

### Community 2 - "Community 2"
Cohesion: 0.04
Nodes (37): analyze_kinetic_chain(), get_phase_feedback(), MultiJointSynchronization, PhaseAnalysisResult, PhaseBoundary, PhaseSegment, PhaseSegmenter, Phase Segmentation - Automatic Badminton Shot Phase Detection ================== (+29 more)

### Community 3 - "Community 3"
Cohesion: 0.08
Nodes (46): PoseFeatureExtractor, dtw_align_landmarks(), extract_landmarks_with_preprocessing(), extract_phase_specific_landmarks(), extract_phase_specific_template(), filter_quality(), find_best_reference(), main() (+38 more)

### Community 4 - "Community 4"
Cohesion: 0.06
Nodes (28): EnhancedKSI, JointError, KSIResult, LandmarkConfidenceFilter, Calculate attention-weighted KSI., Margin-based ranking hinge to compare user vs. baseline/reference.         Posit, Generate initial recommendations based on analysis., Return empty result with error message. (+20 more)

### Community 5 - "Community 5"
Cohesion: 0.07
Nodes (27): evaluate(), Model Evaluation Pipeline with KSI v2.0 Metrics ================================, Evaluate model with enhanced KSI v2 metrics.     Logs phase scores, confidence i, MLflowRunManager, MLflow utility module for enhanced experiment tracking and run management. Provi, Check if run name is unique within experiment., Log comprehensive run metadata as tags., Get current git branch. (+19 more)

### Community 6 - "Community 6"
Cohesion: 0.14
Nodes (14): _apply_crop(), _detect_contact_realtime(), _extract_features_from_video(), _infer_video_sync(), _ksi_to_dict(), _load_params(), PredictionSmoother, realtime_ws() (+6 more)

### Community 7 - "Community 7"
Cohesion: 0.11
Nodes (18): get_pose_model(), main(), process_video_streaming(), Pose Feature Preprocessing Pipeline ====================================  Stream, Helper to initialize MediaPipe Pose., Processes video frame-by-frame and saves windows immediately.     Uses O(1) memo, extract_video_number(), get_segment_bounds() (+10 more)

### Community 8 - "Community 8"
Cohesion: 0.15
Nodes (14): build_rsn(), build_rsn_feature_extractor(), channel_shuffle(), Residual-Shuffle Network (RSN) for Feature Extraction.  A lightweight, efficient, Build the Residual-Shuffle Network backbone.          Architecture:         - St, Shuffle channels across groups.          This operation rearranges channels so t, Build RSN with projection head for feature extraction.          This wraps the b, Shuffle Unit - the core building block of RSN.          For stride=1 (identity s (+6 more)

### Community 9 - "Community 9"
Cohesion: 0.31
Nodes (7): build_hybrid_model(), HybridObjective, load_hybrid_data(), main(), Hyperparameter Tuning for Hybrid TCN Model =====================================, Build Attention-TCN model with Optuna-suggested hyperparameters., retrain_and_register()

### Community 10 - "Community 10"
Cohesion: 0.24
Nodes (9): compare_experiments(), compare_specific_runs(), get_best_run(), main(), Utility script for comparing MLflow runs across experiments. Provides easy compa, Compare specific runs by their run IDs.          Args:         run_ids: List of, Compare all runs across experiments.          Args:         experiment_names: Li, Get the best run from an experiment.          Args:         experiment_name: Nam (+1 more)

### Community 11 - "Community 11"
Cohesion: 0.33
Nodes (8): delete_video(), find_videos(), main(), prompt_keep_or_delete(), Find all video files in the given folder., Run visualize.py on a video.     Returns True if visualization succeeded, False, Prompt user to keep, delete, or skip the video.     Returns: 'keep', 'delete', o, visualize_video()

### Community 12 - "Community 12"
Cohesion: 0.36
Nodes (6): build_pose_model(), load_pose_data(), main(), PoseObjective, Hyperparameter Tuning for Pose LSTM Model ======================================, retrain_and_register()

### Community 13 - "Community 13"
Cohesion: 0.67
Nodes (2): draw_neural_net(), Draw a neural network cartoon using matplotlib.          Args:         ax: matpl

### Community 14 - "Community 14"
Cohesion: 0.67
Nodes (1): Evaluate the trained Attention-TCN model using the EXACT same data loading and s

### Community 15 - "Community 15"
Cohesion: 1.0
Nodes (2): extract_frames(), load_params()

### Community 16 - "Community 16"
Cohesion: 1.0
Nodes (1): Quick debug: check model input spec and test a prediction.

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **141 isolated node(s):** `Draw a neural network cartoon using matplotlib.          Args:         ax: matpl`, `Evaluate the trained Attention-TCN model using the EXACT same data loading and s`, `Quick debug: check model input spec and test a prediction.`, `Find all video files in the given folder.`, `Run visualize.py on a video.     Returns True if visualization succeeded, False` (+136 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 16`** (2 nodes): `Quick debug: check model input spec and test a prediction.`, `debug_model.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `EnhancedKSI` connect `Community 4` to `Community 1`, `Community 3`, `Community 5`, `Community 6`?**
  _High betweenness centrality (0.210) - this node is a cross-community bridge._
- **Why does `generate_coaching_report()` connect `Community 0` to `Community 1`, `Community 5`, `Community 6`?**
  _High betweenness centrality (0.153) - this node is a cross-community bridge._
- **Why does `HybridFeatureExtractor` connect `Community 1` to `Community 8`, `Community 6`?**
  _High betweenness centrality (0.139) - this node is a cross-community bridge._
- **Are the 33 inferred relationships involving `EnhancedKSI` (e.g. with `PredictionSmoother` and `ServiceState`) actually correct?**
  _`EnhancedKSI` has 33 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `HybridFeatureExtractor` (e.g. with `PredictionSmoother` and `ServiceState`) actually correct?**
  _`HybridFeatureExtractor` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 24 inferred relationships involving `ShotPhase` (e.g. with `Model Evaluation Pipeline with KSI v2.0 Metrics ================================` and `Evaluate model with enhanced KSI v2 metrics.     Logs phase scores, confidence i`) actually correct?**
  _`ShotPhase` has 24 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `ShotPhaseSegmenter` (e.g. with `Expert Template Generation for KSI Evaluation ==================================` and `Resample sequence to target FPS to normalize temporal scale.     This ensures al`) actually correct?**
  _`ShotPhaseSegmenter` has 17 INFERRED edges - model-reasoned connections that need verification._