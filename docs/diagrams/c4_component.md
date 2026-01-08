# C4 Component Diagram (Library)

```mermaid
flowchart LR
  subgraph Enhanced Analysis
    API[enhanced_api.py]
    KSI[ksi_v2.py]
    PHASE[phase_segmentation.py]
    NLC[natural_language_coach.py]
    CONF[confidence_filtering.py]
  end

  subgraph Feature Extraction
    FEAT[features.py]
    UTILS[utils.py]
  end

  subgraph Training & Tuning
    TRAINH[train_hybrid.py]
    TRAINP[train_pose.py]
    TUNEH[tune_hybrid.py]
    TUNEP[tune_pose.py]
    MODELS[models.py]
    MLFU[mlflow_utils.py]
  end

  subgraph CLI
    EV[src/evaluate_video.py]
    RT[src/realtime_hybrid.py]
    DEMO[demo_nlp_coaching.py]
    COMP[src/compare_runs.py]
  end

  API --> KSI
  API --> PHASE
  API --> NLC
  API --> CONF
  API --> UTILS

  FEAT --> MediaPipe
  FEAT --> OpenCV
  EV --> FEAT
  EV --> KSI
  EV --> NLC
  EV --> UTILS

  RT --> FEAT
  RT --> MODELS
  TRAINH --> MODELS
  TRAINH --> MLFU
  TRAINP --> MODELS
  TRAINP --> MLFU
  TUNEH --> MODELS
  TUNEP --> MODELS
  COMP --> MLRUNS
```

**Method Highlights**
- `ksi_v2.py` (EnhancedKSI / EnhancedKSICalculator): `calculate()`, `calculate_detailed_ksi()`, `calculate_with_confidence()` (bootstrap CIs), feature extraction helpers, DTW with contact-centered windowing.
- `phase_segmentation.py` (PhaseSegmenter): `segment()` (phases + contact frame), `_detect_boundaries()` (velocity thresholds), `_calculate_phase_quality()`, `MultiJointSynchronization.analyze_kinetic_chain()`; convenience `segment_shot()`, `analyze_kinetic_chain()`, `get_phase_feedback()`.
- `natural_language_coach.py` (NaturalLanguageCoach): `generate_feedback()` (severity classification + priority fixes), `format_feedback_text/json()`, helper `_generate_timing_feedback()`, `_generate_power_feedback()`, `_generate_weekly_plan()`; knowledge base spans causal chains, drills, visual cues.
