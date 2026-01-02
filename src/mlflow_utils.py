"""
MLflow utility module for enhanced experiment tracking and run management.
Provides interactive run naming, rich metadata logging, and comparison tools.
"""

import mlflow
import os
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
import json
import subprocess


class MLflowRunManager:
    """Enhanced MLflow run management with interactive naming and rich metadata."""
    
    def __init__(self, experiment_name: str, params_file: str = "params.yaml"):
        self.experiment_name = experiment_name
        self.params_file = params_file
        mlflow.set_experiment(experiment_name)
        
    def start_interactive_run(
        self, 
        default_description: str = "",
        auto_name: bool = False
    ) -> mlflow.ActiveRun:
        """
        Start MLflow run with interactive naming and rich metadata.
        
        Args:
            default_description: Default description to show user
            auto_name: If True, auto-generate name without prompting
        
        Returns:
            mlflow.ActiveRun: Active MLflow run context
        """
        if auto_name:
            run_name = self._generate_auto_name()
            description = default_description
        else:
            run_name = self._prompt_run_name()
            description = self._prompt_description(default_description)
        
        # Create run with enhanced tags
        run = mlflow.start_run(run_name=run_name)
        
        # Log comprehensive metadata
        self._log_run_metadata(description, run_name)
        
        return run
    
    def _prompt_run_name(self) -> str:
        """Prompt user for a unique, descriptive run name."""
        print("\n" + "="*70)
        print("🚀 MLFLOW RUN CONFIGURATION")
        print("="*70)
        
        # Show existing runs for reference
        self._display_recent_runs()
        
        while True:
            print("\n📝 Enter a unique run name (or press Enter for auto-generated):")
            print("   Examples: 'baseline_v1', 'tcn_lr_experiment', 'production_candidate'")
            run_name = input("   > ").strip()
            
            if not run_name:
                # Auto-generate
                run_name = self._generate_auto_name()
                print(f"   ✓ Auto-generated: {run_name}")
                break
            
            # Validate uniqueness
            if self._is_run_name_unique(run_name):
                print(f"   ✓ Run name accepted: {run_name}")
                break
            else:
                print(f"   ✗ Run name '{run_name}' already exists. Please choose another.")
        
        return run_name
    
    def _prompt_description(self, default: str) -> str:
        """Prompt for run description."""
        print("\n📋 Enter run description (optional, press Enter to skip):")
        if default:
            print(f"   Default: {default}")
        description = input("   > ").strip()
        return description if description else default
    
    def _generate_auto_name(self) -> str:
        """Generate automatic run name with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_short = self.experiment_name.split("_")[0].lower()
        return f"{exp_short}_run_{timestamp}"
    
    def _display_recent_runs(self, n: int = 5):
        """Display recent runs for reference."""
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(self.experiment_name)
            
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=n
                )
                
                if runs:
                    print(f"\n📊 Recent runs in '{self.experiment_name}':")
                    for i, run in enumerate(runs, 1):
                        name = run.data.tags.get('mlflow.runName', 'unnamed')
                        status = run.info.status
                        metrics = run.data.metrics
                        acc = metrics.get('best_val_accuracy', metrics.get('val_accuracy', 'N/A'))
                        if isinstance(acc, float):
                            acc = f"{acc:.4f}"
                        print(f"   {i}. {name} | Status: {status} | Val Acc: {acc}")
        except Exception as e:
            print(f"   (Could not fetch recent runs: {e})")
    
    def _is_run_name_unique(self, run_name: str) -> bool:
        """Check if run name is unique within experiment."""
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(self.experiment_name)
            
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=f"tags.mlflow.runName = '{run_name}'"
                )
                return len(runs) == 0
            return True
        except:
            return True  # If check fails, allow name
    
    def _log_run_metadata(self, description: str, run_name: str):
        """Log comprehensive run metadata as tags."""
        mlflow.set_tags({
            "run_description": description,
            "run_name": run_name,
            "timestamp": datetime.now().isoformat(),
            "user": os.environ.get("USER", "unknown"),
            "git_branch": self._get_git_branch(),
            "git_commit": self._get_git_commit(),
        })
    
    def _get_git_branch(self) -> str:
        """Get current git branch."""
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=os.path.dirname(os.path.abspath(__file__))
            ).decode().strip()
        except:
            return "unknown"
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=os.path.dirname(os.path.abspath(__file__))
            ).decode().strip()
        except:
            return "unknown"
    
    def log_dataset_info(self, X_train, X_test, y_train, y_test, classes):
        """Log comprehensive dataset information."""
        mlflow.log_params({
            "dataset.train_samples": len(X_train),
            "dataset.test_samples": len(X_test),
            "dataset.train_test_split": f"{len(X_train)}/{len(X_test)}",
            "dataset.num_classes": len(classes),
            "dataset.classes": ",".join(classes),
            "dataset.input_shape": str(X_train.shape[1:]),
        })
        
        # Log class distribution
        from collections import Counter
        train_dist = Counter(np.argmax(y_train, axis=1))
        test_dist = Counter(np.argmax(y_test, axis=1))
        
        for i, cls in enumerate(classes):
            mlflow.log_metrics({
                f"dataset.class_{cls}.train_count": train_dist.get(i, 0),
                f"dataset.class_{cls}.test_count": test_dist.get(i, 0),
            })
    
    def log_model_architecture(self, model):
        """Log detailed model architecture information."""
        import io
        import tensorflow as tf
        
        # Model summary as text artifact
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        mlflow.log_text(stream.getvalue(), "model_summary.txt")
        
        # Model parameters
        mlflow.log_params({
            "model.total_params": int(model.count_params()),
            "model.trainable_params": int(sum([tf.size(w).numpy() for w in model.trainable_weights])),
            "model.layers": len(model.layers),
        })
    
    def log_training_artifacts(self, history, save_plots: bool = True):
        """Log comprehensive training artifacts."""
        import matplotlib.pyplot as plt
        
        # Save history as JSON
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        mlflow.log_dict(history_dict, "training_history.json")
        
        if save_plots:
            # Training curves
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Loss plot
            axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
            axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)
            
            # Accuracy plot
            axes[1].plot(history.history['accuracy'], label='Train Acc', linewidth=2)
            axes[1].plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Accuracy', fontsize=12)
            axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            mlflow.log_figure(fig, "training_curves.png")
            plt.close()
        
        # Log best metrics
        best_epoch = int(np.argmax(history.history['val_accuracy']))
        mlflow.log_metrics({
            "best_epoch": best_epoch,
            "best_val_accuracy": float(history.history['val_accuracy'][best_epoch]),
            "best_val_loss": float(history.history['val_loss'][best_epoch]),
            "final_train_accuracy": float(history.history['accuracy'][-1]),
            "final_train_loss": float(history.history['loss'][-1]),
            "epochs_trained": len(history.history['loss']),
        })
        
        # Check if early stopped
        try:
            configured_epochs = int(mlflow.get_run(mlflow.active_run().info.run_id).data.params.get('epochs', 0))
            mlflow.log_metric("early_stopped", int(len(history.history['loss']) < configured_epochs))
        except:
            pass
