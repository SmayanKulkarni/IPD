"""
Hybrid TCN Training Pipeline
=============================

Training script for the hybrid pose+CNN temporal convolutional network (TCN)
classifier. Combines geometric pose features with MobileNetV2 visual embeddings
for robust badminton shot classification.

Key Features:
    - Dual-input architecture: TCN for CNN features + GRU for pose features
    - Late fusion via concatenation for complementary feature integration
    - Train/validation/test split (70/10/20) with stratification
    - MLflow experiment tracking and model registry integration
    - DVC live callback for experiment versioning

Architecture:
    CNN Branch:
        Conv1D (causal, dilated) → BatchNorm → ReLU → GRU → Dense
    Pose Branch:
        GRU → BatchNorm → Dense → Dropout
    Fusion:
        Concatenate → Dense (softmax)

Pipeline Position:
    preprocess_hybrid.py → [train_hybrid.py] → evaluate.py
    
    Consumes preprocessed .npz files containing fused features:
    (T, 99+CNN_DIM) where T=sequence_length, 99=pose, CNN_DIM=visual embedding

Dependencies:
    External: tensorflow, mlflow, sklearn, numpy, yaml, dvclive
    Internal: models.build_tcn_hybrid, mlflow_utils.MLflowRunManager

Configuration (params.yaml):
    hybrid_pipeline:
        data_path: Path to preprocessed hybrid data
        model_path: Output path for trained model
        cnn_feature_dim: Dimension of CNN visual features (default: 128)
        sequence_length: Number of frames per sample
        epochs: Maximum training epochs
        batch_size: Training batch size
    base:
        random_state: Seed for reproducibility

Usage:
    python train_hybrid.py

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
from models import build_tcn_hybrid
from mlflow_utils import MLflowRunManager


def main():
    with open("params.yaml") as f: params = yaml.safe_load(f)
    cfg = params['hybrid_pipeline']
    
    run_manager = MLflowRunManager("Hybrid_TCN_Experiment")
    mlflow.enable_system_metrics_logging()
    mlflow.tensorflow.autolog(log_models=False)

    with run_manager.start_interactive_run(
        default_description="TCN-Hybrid pipeline training with pose + CNN fusion"
    ):
        mlflow.log_params(cfg)
        mlflow.log_params(params['mediapipe'])
        mlflow.log_params(params['segment_rules'])
        mlflow.log_param("base.random_state", params['base']['random_state'])
        
        X, y = [], []
        if not os.path.exists(cfg['data_path']): 
            print(f"❌ Data path {cfg['data_path']} not found.")
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
        

        cnn_dim = cfg['cnn_feature_dim']
        X_pose = X[..., :-cnn_dim]
        X_cnn = X[..., -cnn_dim:]
        
        idx_trainval, idx_test, y_trainval_lbl, y_test_lbl = train_test_split(
            np.arange(len(X)),
            y,
            test_size=0.2,
            stratify=y,
            random_state=params['base']['random_state']
        )

        idx_train, idx_val, y_train_lbl, y_val_lbl = train_test_split(
            idx_trainval,
            y_trainval_lbl,
            test_size=0.125,
            stratify=y_trainval_lbl,
            random_state=params['base']['random_state']
        )
        

        run_manager.log_dataset_info(
            X_pose[idx_train], X_pose[idx_val], X_pose[idx_test],
            y_cat[idx_train], y_cat[idx_val], y_cat[idx_test],
            classes
        )
        

        model = build_tcn_hybrid(X_pose.shape[1:], X_cnn.shape[1:], len(classes))
        

        run_manager.log_model_architecture(model)
        

        callbacks = [
            EarlyStopping(
                patience=25, 
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

        print("\n🚀 Starting Hybrid-TCN Training...")
        print(f"   Train samples: {len(idx_train)}")
        print(f"   Val samples: {len(idx_val)}")
        print(f"   Test samples: {len(idx_test)}")
        print(f"   Classes: {classes}")
        print(f"   Pose features: {X_pose.shape[1:]}")
        print(f"   CNN features: {X_cnn.shape[1:]}\n")
        
        history = model.fit(
            [X_cnn[idx_train], X_pose[idx_train]], y_cat[idx_train],
            validation_data=([X_cnn[idx_val], X_pose[idx_val]], y_cat[idx_val]),
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
            registered_model_name="Hybrid_TCN"
        )
        print("✅ Model registered as 'Hybrid_TCN'.")

if __name__ == "__main__": main()
