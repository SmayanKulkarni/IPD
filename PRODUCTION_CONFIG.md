# Production Configuration - IPD Badminton Shot Classifier

**Status:** Production Ready  
**Model:** TCN Hybrid (Tuned)  
**Last Updated:** January 3, 2026

---

## Production Model

**Model File:** `models/tcn_hybrid_tuned.h5`  
**Architecture:** Dual-stream TCN + GRU with MobileNetV2 CNN features + MediaPipe Pose  
**Training:** Optuna hyperparameter optimization (20 trials)  
**Expected Performance:** ~85% test accuracy, 200ms latency

### Model Inputs
- **Pose Stream:** (batch, 40, 163) - MediaPipe landmarks + geometric features
- **CNN Stream:** (batch, 40, 64) - MobileNetV2 bottleneck features from ROI crops

### Classes (6)
1. `backhand_drive`
2. `backhand_net_shot`
3. `forehand_clear`
4. `forehand_drive`
5. `forehand_lift`
6. `forehand_net_shot`

---

## Expert Templates

**File:** `data/expert_templates.npz`  
**Generated:** January 3, 2026  
**Total Videos:** 111 expert demonstrations

### Template Breakdown
- `forehand_clear`: 15 videos → (39, 12) averaged template
- `forehand_lift`: 24 videos → (44, 12) averaged template
- `backhand_net_shot`: 29 videos → (56, 12) averaged template
- `backhand_drive`: 10 videos → (63, 12) averaged template
- `forehand_drive`: 23 videos → (12, 12) averaged template
- `forehand_net_shot`: 10 videos → (91, 12) averaged template

---

## Pipeline Configuration

### Data Processing
```yaml
sequence_length: 40 frames
stride: 5 frames
cnn_feature_dim: 64
cnn_input_size: 224x224
```

### ROI Detection
```yaml
enabled: true
joint_ids: [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # Full body
visibility_thresh: 0.3
margin: 0.35
min_size_frac: 0.65
smoothing: 0.3
```

### Frame Cropping
```yaml
top: 0.13
bottom: 0.35
left: 0.25
right: 0.25
```

---

## KSI Configuration

**Weights:**
- Pose similarity: 0.4
- Velocity coherence: 0.4
- Acceleration profile: 0.2

**Features:**
- Phase-aware segmentation (preparation, loading, acceleration, contact, follow-through)
- Confidence intervals (bootstrapping)
- Per-joint error analysis
- Velocity/acceleration profiles

---

## Usage

### Evaluation
```bash
# Evaluate production model
dvc repro

# Or directly:
python src/evaluate.py --type hybrid --model models/tcn_hybrid_tuned.h5
```

### Real-time Inference
```bash
# Webcam
python src/realtime_hybrid.py --source 0

# Video file
python src/realtime_hybrid.py --source path/to/video.mp4 --model models/tcn_hybrid_tuned.h5
```

### API Usage
```python
from enhanced_api import analyze_shot
import numpy as np

# Load user video poses
user_poses = extract_poses_from_video(video_path)  # (T, 33, 3)

# Analyze
result = analyze_shot(
    user_poses, 
    shot_type='forehand_clear',
    skill_level='intermediate',
    expert_template_path='data/expert_templates.npz'
)

# Get coaching feedback
print(result.coaching_report)
print(f"KSI Score: {result.ksi_score:.3f}")
```

---

## Deployment Checklist

- [x] Expert templates generated (111 videos)
- [x] Production model tuned and saved (`tcn_hybrid_tuned.h5`)
- [x] `params.yaml` updated to use tuned model
- [x] `dvc.yaml` simplified to production pipeline
- [x] Evaluation script supports custom model path
- [ ] Deploy to Hugging Face Spaces (optional)
- [ ] Set up REST API endpoint (optional)
- [ ] Mobile TFLite conversion (if needed)

---

## Development vs Production

### Development Models (Archived)
- `models/lstm_pose.h5` - Pose-only baseline (~80% accuracy)
- `models/lstm_pose_tuned.h5` - Tuned pose model
- `models/tcn_hybrid.h5` - Untuned hybrid model

### Production Model (Active)
- `models/tcn_hybrid_tuned.h5` - **USE THIS**

All scripts now default to the tuned model via `params.yaml`.

---

## Model Performance (Expected)

**Test Set Metrics:**
- Accuracy: ~85%
- Precision: ~85%
- Recall: ~85%
- F1-Score: ~85%

**Per-Class Performance:**
- Forehand shots: 85-90% accuracy
- Backhand shots: 80-85% accuracy
- Net shots: 75-80% accuracy (harder due to subtle differences)

**Inference Speed:**
- CPU: ~200ms per sequence
- GPU: ~50ms per sequence

---

## Maintenance

### Retraining Triggers
- New labeled data added to `data/raw/`
- Expert templates need updates
- Accuracy drops below 80% on validation set

### Update Procedure
1. Add new data to `data/raw/<shot_class>/`
2. Update expert videos in `data/expert_data/`
3. Run full pipeline: `dvc repro`
4. Evaluate: Check `dvclive/production_metrics.json`
5. If improved, replace `tcn_hybrid_tuned.h5`
6. Update this document with new metrics

---

## Contact

**Project:** IPD - Intelligent Performance Diagnostics  
**Repository:** SmayanKulkarni/IPD  
**Branch:** DeepResearch  
**Model Version:** v2.0 (TCN Hybrid Tuned)
