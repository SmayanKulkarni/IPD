# C4 Context Diagram

```mermaid
flowchart LR
  User["User / Coach / Analyst"] --> CLI["CLI Scripts\n- evaluate_video.py\n- demo_nlp_coaching.py\n- realtime_hybrid.py"]
  CLI --> Lib["Python Library (src/*)\nCore analysis components"]

  subgraph External Systems
    MediaPipe["MediaPipe Pose"]
    OpenCV["OpenCV (Video IO)"]
    MLflow["MLflow Tracking (mlruns)"]
    DVC["DVC + DVCLive (metrics/plots)"]
  end

  Lib --> MediaPipe
  Lib --> OpenCV
  Lib --> MLflow
  Lib --> DVC

  Lib --> Data["Data Store\n- data/raw (videos)\n- data/Data_Normalized_Hybrid (npz)\n- data/expert_templates.npz"]
  Lib --> Models["Model Store\n- models/*.h5"]
  Lib --> Reports["Coaching Reports\n- coaching_reports/*.txt, *.json"]

  note right of CLI: Runs preprocess/train/evaluate/coaching workflows
  note right of MLflow: Params, metrics, artifacts, model registry
```
