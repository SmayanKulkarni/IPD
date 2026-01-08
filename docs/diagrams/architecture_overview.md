# Unified Architecture Diagram

```mermaid
flowchart TB
  %% Actors & Entry Points
  U["User / Coach / Analyst"]
  CLI_EVAL["evaluate_video.py<br/>(single video eval)"]
  CLI_RT["realtime_hybrid.py<br/>(webcam/video realtime)"]
  CLI_DEMO["demo_nlp_coaching.py<br/>(coaching demo)"]
  CLI_ML["train_* / tune_*<br/>(training & tuning)"]
  CLI_CMP["compare_runs.py<br/>(MLflow compare)"]

  U --> CLI_EVAL
  U --> CLI_RT
  U --> CLI_DEMO
  U --> CLI_ML
  U --> CLI_CMP

  %% Preprocessing
  subgraph Preprocessing
    PREP_H["preprocess_hybrid.py<br/>features + raw_landmarks"]
    PREP_P["preprocess_pose.py<br/>pose features"]
  end

  RAW --> PREP_P
  # Unified Architecture Diagram

  PREP_H --> HYB["data/Data_Normalized_Hybrid/*/*.npz<br/>(features, raw_landmarks, fps)"]
  PREP_P --> POSE["data/pose_normalized/*/*.npz<br/>(features, fps)"]

  %% Training & Tuning
  subgraph Train & Tune
    TRAIN_H["train_hybrid.py<br/>TCN hybrid"]
    TRAIN_P["train_pose.py<br/>LSTM pose"]
    TUNE_H["tune_hybrid.py<br/>Optuna"]
    TUNE_P["tune_pose.py<br/>Optuna"]
    MODELS["models/*.h5<br/>(tcn_hybrid*, lstm_pose*)"]
    MLFU["mlflow_utils.py<br/>run manager"]
  end

  HYB --> TRAIN_H
  HYB --> TUNE_H
  POSE --> TRAIN_P
  POSE --> TUNE_P
  TRAIN_H --> MODELS
  TRAIN_P --> MODELS
  TUNE_H --> MODELS
  TUNE_P --> MODELS
  TRAIN_H --> DVCLIVE["dvclive/* metrics, plots"]
  TRAIN_P --> DVCLIVE
  TRAIN_H --> MLRUNS["mlruns/* (MLflow tracking)"]
  TRAIN_P --> MLRUNS
  TUNE_H --> MLRUNS
  TUNE_P --> MLRUNS
  MLFU --> MLRUNS

  %% Core Library
  subgraph Core Library (src)
    FEAT["features.py<br/>Pose+CNN fusion"]
    UTILS["utils.py<br/>normalize, crop, segment bounds"]
    KSI["ksi_v2.py<br/>EnhancedKSI"]
    PHASE["phase_segmentation.py<br/>phase detection & kinetic chain"]
    NLC["natural_language_coach.py<br/>NL coaching"]
    CONF["confidence_filtering.py<br/>quality gates"]
    API["enhanced_api.py<br/>unified analyzer"]
  end

  %% Templates
  TEMPL["data/expert_templates.npz<br/>(class & variant templates)"]

  %% Evaluation paths
  CLI_EVAL --> FEAT
  CLI_EVAL --> UTILS
  CLI_EVAL --> MODELS
  CLI_EVAL --> TEMPL
  FEAT --> KSI
  FEAT --> PHASE
  KSI --> NLC
  PHASE --> NLC
  CONF --> API
  API --> NLC

  CLI_RT --> FEAT
  CLI_RT --> MODELS
  CLI_DEMO --> KSI
  CLI_DEMO --> NLC
  CLI_DEMO --> TEMPL

  %% Outputs
  NLC --> REPORTS["coaching_reports/*.txt, *.json"]
  CLI_EVAL --> REPORTS
  CLI_DEMO --> REPORTS
  CLI_CMP --> MLRUNS

  %% Storage interactions
  FEAT -.-> MediaPipe["MediaPipe Pose"]
  FEAT -.-> OpenCV["OpenCV Video IO"]
```
