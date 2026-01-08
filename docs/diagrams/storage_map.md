# Storage Map

- **data/raw/**: Original videos organized by shot type.
- **data/Data_Normalized_Hybrid/**: Per-class folders with `.npz` files containing:
  - **features**: `(T, D)` fused pose+CNN features where `D = 99 + cnn_dim`.
  - **raw_landmarks**: `(T, 33, 3)` image-space landmarks for KSI.
  - **fps**: float frames-per-second used during preprocessing.
- **data/expert_templates.npz**: Numpy archive; keys per shot class and variants (e.g., `forehand_clear`, `forehand_clear_variant1`). Values are `(T, 33, 3)` landmarks (or 99D flattened, reshaped during use).
- **models/**: Trained Keras models:
  - `lstm_pose.h5`, `lstm_pose_tuned.h5` (pose-only).
  - `tcn_hybrid.h5`, `tcn_hybrid_tuned.h5` (pose+CNN).
- **dvclive/**: Metrics JSON and plots from training/evaluation:
  - `metrics.json`, `hybrid_metrics.json`, `pose_metrics.json`, confusion matrices, `plots/metrics/*`.
- **mlruns/**: MLflow tracking data; experiments, runs, params, metrics, artifacts; local file-based URI.
- **coaching_reports/**: Generated coaching outputs in `.txt` and `.json`.

**Schemas & Contracts**
- **Hybrid .npz contract**: Must include `features`, `raw_landmarks`, `fps`.
- **Pose .npz contract**: Must include `features`, `fps`.
- **Templates contract**: Per-class landmark arrays; consumer reshapes 99D to `(T,33,3)` when needed.
- **Evaluate pipeline**: Requires trained model `.h5`, class list from `data_path`, expert templates, and MediaPipe runtime.
