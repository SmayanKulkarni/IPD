"""
Pose-LSTM Training Pipeline
============================

Training script for the pose-based LSTM classifier in the badminton shot
analysis system. Trains a Conv1D + LSTM architecture on normalized 3D pose
sequences extracted from MediaPipe.

Key Features:
    - LSTM-based temporal modeling of pose sequences
    - Train/validation/test split (70/10/20) with stratification
    - MLflow experiment tracking with interactive run naming
    - DVC live callback for experiment versioning
    - Early stopping with best model checkpointing
    - Automatic model registration to MLflow Model Registry

Pipeline Position:
    preprocess_pose.py → [train_pose.py] → evaluate.py
    
    Consumes preprocessed .npz files containing normalized pose features
    (T, 99) where T=sequence_length and 99=33 joints × 3 coordinates.

Dependencies:
    External: tensorflow, mlflow, sklearn, numpy, yaml, dvclive
    Internal: models.build_lstm_pose, mlflow_utils.MLflowRunManager

Configuration (params.yaml):
    pose_pipeline:
        data_path: Path to preprocessed pose data
        model_path: Output path for trained model
        sequence_length: Number of frames per sample
        epochs: Maximum training epochs
        batch_size: Training batch size
    base:
        random_state: Seed for reproducibility

Usage:
    python train_pose.py

Author: IPD Research Team
Version: 1.0.0
"""

import os
import yaml
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from dvclive.keras import DVCLiveCallback
from models import build_lstm_pose
from mlflow_utils import MLflowRunManager


def main():
    with open("params.yaml") as f: params = yaml.safe_load(f)
    cfg = params['pose_pipeline']
    
    run_manager = MLflowRunManager("Pose_LSTM_Experiment")
    mlflow.enable_system_metrics_logging()
    mlflow.tensorflow.autolog(log_models=False)

    with run_manager.start_interactive_run(
        default_description="LSTM-Pose pipeline training with geometric normalization"
    ):
        mlflow.log_params(cfg)
        mlflow.log_params(params['mediapipe'])
        mlflow.log_params(params['segment_rules'])
        mlflow.log_param("base.random_state", params['base']['random_state'])
        
        X, y = [], []
        if not os.path.exists(cfg['data_path']): 
            print(f"Data path {cfg['data_path']} not found.")
            return

        classes = sorted(os.listdir(cfg['data_path']))
        for i, cls in enumerate(classes):
            path = os.path.join(cfg['data_path'], cls)
            for f in os.listdir(path):
                X.append(np.load(os.path.join(path, f))['features'])
                y.append(i)
        
        if not X: 
            print("❌ No data loaded.")
            return
        
        X = np.array(X)
        y = np.array(y)
        y_cat = to_categorical(y, len(classes))


        X_trainval, X_test, y_trainval, y_test, y_trainval_lbl, y_test_lbl = train_test_split(
            X,
            y_cat,
            y,
            test_size=0.2,
            stratify=y,
            random_state=params['base']['random_state']
        )

        X_train, X_val, y_train, y_val, _, _ = train_test_split(
            X_trainval,
            y_trainval,
            y_trainval_lbl,
            test_size=0.125,
            stratify=y_trainval_lbl,
            random_state=params['base']['random_state']
        )
        

        run_manager.log_dataset_info(X_train, X_val, X_test, y_train, y_val, y_test, classes)
        

        model = build_lstm_pose(X_train.shape[1:], len(classes))
        

        run_manager.log_model_architecture(model)
        

        callbacks = [
            EarlyStopping(
                patience=10, 
                restore_best_weights=True,
                monitor='val_accuracy',
                verbose=1
            ),
            ModelCheckpoint(
                cfg['model_path'], 
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            ),
            DVCLiveCallback(save_dvc_exp=True)
        ]

        print("\n🚀 Starting Pose-LSTM Training...")
        print(f"   Train samples: {len(X_train)}")
        print(f"   Val samples: {len(X_val)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Classes: {classes}\n")
        
        history = model.fit(
            X_train, y_train, 
            validation_data=(X_val, y_val),
            epochs=cfg['epochs'], 
            batch_size=cfg['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        

        run_manager.log_training_artifacts(history, save_plots=True)
        
        print(f"\n✅ Training finished!")
        print(f"   Best Val Acc: {max(history.history['val_accuracy']):.4f}")


        print("\n📦 Logging and Registering Best Model to MLflow...")
        best_model = tf.keras.models.load_model(cfg['model_path'])
        
        mlflow.keras.log_model(
            best_model, 
            artifact_path="model", 
            registered_model_name="Pose_LSTM",
            signature=mlflow.models.infer_signature(X_train, model.predict(X_train[:1]))
        )
        print("✅ Model registered as 'Pose_LSTM'.")

if __name__ == "__main__": main()