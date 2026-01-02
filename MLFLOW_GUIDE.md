# MLflow Enhanced Tracking Guide

## Overview

The IPD project now includes enhanced MLflow tracking with interactive run naming, rich metadata logging, and comprehensive comparison tools.

## New Features

### 1. Interactive Run Naming
Every time you start a training or evaluation run, you'll be prompted to provide:
- **Unique run name**: A descriptive identifier (e.g., `baseline_v1`, `tcn_high_lr`)
- **Run description**: Optional detailed description of what you're testing

### 2. Rich Metadata Logging
Each run automatically logs:
- **Dataset info**: Sample counts, class distribution, input shapes
- **Model architecture**: Layer counts, parameter counts, model summary
- **Training curves**: Loss and accuracy plots saved as artifacts
- **Git information**: Current branch and commit hash
- **User attribution**: Who ran the experiment
- **Timestamps**: When the run started

### 3. Enhanced Metrics
Beyond basic accuracy/loss, the system now logs:
- Best epoch and corresponding metrics
- Early stopping status
- Per-class sample counts
- Model signatures for deployment

## Usage

### Training with Interactive Prompts

```bash
# Train Pose-LSTM model
python src/train_pose.py

# You'll see:
# ======================================================================
# 🚀 MLFLOW RUN CONFIGURATION
# ======================================================================
# 
# 📊 Recent runs in 'Pose_LSTM_Experiment':
#    1. baseline_v1 | Status: FINISHED | Val Acc: 0.8842
#    2. lstm_run_20260101_143022 | Status: FINISHED | Val Acc: 0.8721
#
# 📝 Enter a unique run name (or press Enter for auto-generated):
#    Examples: 'baseline_v1', 'tcn_lr_experiment', 'production_candidate'
#    > improved_preprocessing_v1
#
# 📋 Enter run description (optional, press Enter to skip):
#    Default: LSTM-Pose pipeline training with geometric normalization
#    > Testing new crop overrides for better player framing
```

### Training with Auto-Generated Names

If you want to skip the prompts, modify the code to use `auto_name=True`:

```python
# In train_pose.py or train_hybrid.py
with run_manager.start_interactive_run(
    default_description="Your description",
    auto_name=True  # <-- Add this
):
```

Or set in params.yaml:
```yaml
mlflow:
  auto_name_runs: true
```

### Comparing Runs

```bash
# Compare all runs across both experiments
python src/compare_runs.py

# Compare specific experiments
python src/compare_runs.py --experiments Pose_LSTM_Experiment

# Show only top 5 runs
python src/compare_runs.py --top-n 5

# Sort by different metric
python src/compare_runs.py --metric best_val_loss

# Show only the best run from each experiment
python src/compare_runs.py --best
```

### Output Example

```
================================================================================
📊 EXPERIMENT COMPARISON
================================================================================

Total runs: 8
Experiments: Pose_LSTM_Experiment, Hybrid_TCN_Experiment

Top runs by best_val_accuracy:
--------------------------------------------------------------------------------
Run Name                    Experiment              Best Val Acc  Epochs  Status
improved_preprocessing_v1   Pose_LSTM_Experiment    0.9142       52      FINISHED
hybrid_baseline_v2          Hybrid_TCN_Experiment   0.9308       187     FINISHED
baseline_v1                 Pose_LSTM_Experiment    0.8842       80      FINISHED
--------------------------------------------------------------------------------

📈 STATISTICS:
   Best accuracy: 0.9308
   Mean accuracy: 0.8964
   Std accuracy: 0.0234
   Completed runs: 7
   Failed runs: 1
```

## Viewing in MLflow UI

```bash
# Launch the MLflow UI
mlflow ui --port 5000

# Open browser to: http://localhost:5000
```

In the UI you can:
- Compare runs side-by-side
- View training curves
- See model architectures
- Download trained models
- Check git commits and branches

## Best Practices

### 1. Naming Conventions

Use descriptive, hierarchical names:
- `baseline_v1`, `baseline_v2` - Initial experiments
- `exp_lr_tuning_v1` - Specific experiments
- `prod_candidate_20260101` - Production candidates
- `ablation_no_cnn_v1` - Ablation studies

### 2. Descriptions

Include key details:
- What changed compared to baseline
- Hypothesis being tested
- Expected outcome
- Any data/config changes

Example:
```
Testing impact of reduced batch size (16→8) on hybrid model. 
Expecting slower training but better generalization due to 
increased gradient noise. Using new crop overrides for videos 42-58.
```

### 3. Git Workflow

Always commit code changes before running experiments:
```bash
git add .
git commit -m "Add new preprocessing step"
python src/train_pose.py  # Now git info is logged
```

## Programmatic Access

You can also use the utilities in your own scripts:

```python
from mlflow_utils import MLflowRunManager
from compare_runs import get_best_run, compare_experiments

# Get best run from an experiment
best = get_best_run("Pose_LSTM_Experiment", metric="best_val_accuracy")
print(f"Best model: {best['run_name']} with accuracy {best['metric_value']:.4f}")

# Compare programmatically
df = compare_experiments(
    experiment_names=["Pose_LSTM_Experiment", "Hybrid_TCN_Experiment"],
    top_n=10,
    metric="best_val_accuracy"
)

# Filter for high-performing runs
good_runs = df[df['Best Val Acc'] > 0.90]
print(good_runs)
```

## Troubleshooting

### Issue: "Run name already exists"
Solution: Choose a different name or use auto-generation (press Enter)

### Issue: Git information shows "unknown"
Solution: Make sure you're running from within the git repository

### Issue: No recent runs displayed
Solution: This is normal if it's your first run in the experiment

### Issue: Want to disable prompts for automation
Solution: Set `auto_name=True` in the training scripts or use params.yaml:
```yaml
mlflow:
  auto_name_runs: true
```

## Integration with DVC

MLflow tracking works alongside DVC:
- **DVC**: Tracks data versions and pipeline dependencies
- **MLflow**: Tracks experiment runs and models
- **DVCLive**: Logs metrics to both DVC and MLflow

All three work together seamlessly!

## Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/index.html)
