# C4 Container Diagram

```mermaid
flowchart TB
  subgraph CLI Applications
    EV["Evaluate Video CLI\n(src/evaluate_video.py)"]
    RT["Realtime Hybrid CLI\n(src/realtime_hybrid.py)"]
    DEMO["Demo NL Coaching\n(demo_nlp_coaching.py)"]
    CR["MLflow Compare Runs\n(src/compare_runs.py)"]
  end

  subgraph Library (src)
    FEAT["features.py\nPose + CNN fusion"]
    KSI["ksi_v2.py\nEnhanced KSI engine"]
    PHASE["phase_segmentation.py\nShot phases & kinetic chain"]
    NLC["natural_language_coach.py\nNL feedback generation"]
    API["enhanced_api.py\nUnified Analyzer API"]
    PREPH["preprocess_hybrid.py"]
    PREPP["preprocess_pose.py"]
    TRAINH["train_hybrid.py"]
    TRAINP["train_pose.py"]
    TUNEH["tune_hybrid.py"]
    TUNEP["tune_pose.py"]
    MODELS["models.py\nBuilders: LSTM/TCN"]
    MLFU["mlflow_utils.py\nRun manager & logging"]
    UTILS["utils.py\nNormalization, crop, segment bounds"]
    VIS["visualize.py\n2D/3D & KSI viz"]
  end

  subgraph Storage
    RAW["data/raw/* (videos)"]
    HYB["data/Data_Normalized_Hybrid/*/*.npz\nfeatures + raw_landmarks + fps"]
    EXP["data/expert_templates.npz"]
    MODELS_STORE["models/*.h5"]
    DVCSTORE["dvclive/* (metrics, plots)"]
    MLRUNS["mlruns/* (experiments)"]
    REPORTS["coaching_reports/*.txt, *.json"]
  end

  EV --> FEAT
  EV --> KSI
  EV --> NLC
  EV --> UTILS
  EV --> EXP

  DEMO --> KSI
  DEMO --> NLC
  DEMO --> EXP

  RT --> FEAT
  RT --> UTILS
  RT --> MODELS

  PREPH --> FEAT
  PREPH --> UTILS
  PREPP --> UTILS

  TRAINH --> MODELS
  TRAINH --> MLFU
  TRAINP --> MODELS
  TRAINP --> MLFU
  TUNEH --> MODELS
  TUNEP --> MODELS

  FEAT --> MediaPipe
  FEAT --> OpenCV
  API --> KSI
  API --> PHASE
  API --> NLC
  API --> UTILS

  PREPH --> HYB
  PREPP --> data_pose["data/pose_normalized/*.npz"]
  TRAINH --> HYB
  TRAINP --> data_pose
  TUNEH --> HYB
  TUNEP --> data_pose
  EV --> EXP
  EV --> REPORTS
  DEMO --> REPORTS

  MLFU --> MLRUNS
  TRAINH --> DVCSTORE
  TRAINP --> DVCSTORE
```
