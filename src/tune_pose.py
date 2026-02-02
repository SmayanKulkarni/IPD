"""
Hyperparameter Tuning for Pose LSTM Model
==========================================

Optuna-based hyperparameter optimization for the pose-only LSTM classifier.
Searches over architectural and training hyperparameters to maximize
validation accuracy.

Key Features:
    - Bayesian hyperparameter search via Optuna
    - Median pruner for early stopping of unpromising trials
    - MLflow integration for trial tracking and comparison
    - Automatic retraining with best hyperparameters
    - Model registration to MLflow Model Registry

Search Space:
    Architecture:
        - conv_filters: 64-192 (step 32)
        - kernel_size: 3-7 (step 2)
        - lstm_units: [64, 96, 128, 192, 256]
        - dense_units: [32, 64, 96, 128, 192]
    
    Regularization:
        - dropout_conv: 0.1-0.5
        - dropout_lstm: 0.1-0.6
        - dropout_dense: 0.0-0.5
    
    Training:
        - learning_rate: 1e-5 to 5e-3 (log scale)
        - batch_size: [8, 12, 16, 24, 32, 48, 64]

Pipeline Position:
    preprocess_pose.py → [tune_pose.py] → evaluate.py
    
    Alternative to train_pose.py for finding optimal hyperparameters
    before production training.

Dependencies:
    External: optuna, tensorflow, sklearn, mlflow
    Internal: None (model defined inline for flexibility)

Configuration (params.yaml):
    pose_pipeline:
        data_path: Path to preprocessed pose data
        epochs: Maximum epochs per trial (default: 80)
    mlflow:
        tracking_uri: MLflow tracking URI

Usage:
    python tune_pose.py --trials 50

Author: IPD Research Team
Version: 1.0.0
"""

import argparse
import os
import tempfile
import yaml
import numpy as np
import optuna
import mlflow
import tensorflow as tf
try:
    from optuna_integration.tfkeras import TFKerasPruningCallback
except ModuleNotFoundError:
    from optuna.integration import TFKerasPruningCallback
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def load_pose_data(data_path: str):
    X, y = [], []
    classes = sorted(os.listdir(data_path))
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(data_path, cls)
        for fname in os.listdir(cls_dir):
            arr = np.load(os.path.join(cls_dir, fname))["features"]
            X.append(arr)
            y.append(idx)
    return np.array(X), np.array(y), classes


def build_pose_model(trial, input_shape, num_classes):
    filters = trial.suggest_int("conv_filters", 64, 192, step=32)
    kernel_size = trial.suggest_int("kernel_size", 3, 7, step=2)
    lstm_units = trial.suggest_categorical("lstm_units", [64, 96, 128, 192, 256])
    dense_units = trial.suggest_categorical("dense_units", [32, 64, 96, 128, 192])
    dropout_conv = trial.suggest_float("dropout_conv", 0.1, 0.5)
    dropout_lstm = trial.suggest_float("dropout_lstm", 0.1, 0.6)
    dropout_dense = trial.suggest_float("dropout_dense", 0.0, 0.5)
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)

    model = Sequential(
        [
            Conv1D(filters=filters, kernel_size=kernel_size, activation="relu", input_shape=input_shape),
            MaxPooling1D(pool_size=3),
            Dropout(dropout_conv),
            LSTM(lstm_units, return_sequences=True, activation="relu"),
            Dropout(dropout_lstm),
            BatchNormalization(),
            LSTM(max(lstm_units // 2, 32), activation="relu"),
            Dropout(dropout_lstm),
            Dense(dense_units, activation="relu"),
            Dropout(dropout_dense),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer=Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


class PoseObjective:
    def __init__(self, X, y, classes, cfg, random_state, experiment):
        self.X = X
        self.y = y
        self.classes = classes
        self.cfg = cfg
        self.random_state = random_state
        self.experiment = experiment

    def __call__(self, trial: optuna.Trial):
        mlflow.set_experiment(self.experiment)
        with mlflow.start_run(run_name=f"trial_{trial.number}"):
            mlflow.log_param("input_shape", str(self.X.shape[1:]))
            mlflow.log_param("num_classes", len(self.classes))

            X_train, X_val, y_train, y_val = train_test_split(
                self.X,
                to_categorical(self.y, len(self.classes)),
                test_size=0.2,
                stratify=self.y,
                random_state=self.random_state,
            )

            model = build_pose_model(trial, X_train.shape[1:], len(self.classes))
            batch_size = trial.suggest_categorical("batch_size", [8, 12, 16, 24, 32, 48, 64])

            ckpt_dir = tempfile.mkdtemp()
            ckpt_path = os.path.join(ckpt_dir, "pose_best.h5")
            callbacks = [
                EarlyStopping(monitor="val_accuracy", mode="max", patience=10, restore_best_weights=True, verbose=0),
                ModelCheckpoint(ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=0),
                TFKerasPruningCallback(trial, "val_accuracy"),
            ]

            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=self.cfg.get("epochs", 80),
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0,
                shuffle=True,
            )

            best_val_acc = float(np.max(history.history["val_accuracy"]))
            mlflow.log_params(
                {
                    "batch_size": batch_size,
                    "epochs": self.cfg.get("epochs", 80),
                }
            )
            for k, v in trial.params.items():
                mlflow.log_param(k, v)
            mlflow.log_metric("best_val_accuracy", best_val_acc)
            mlflow.log_metric("best_val_loss", float(np.min(history.history["val_loss"])))
            return best_val_acc


def retrain_and_register(X, y, classes, cfg, random_state, experiment, best_params):
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="best_params_retrain"):
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            to_categorical(y, len(classes)),
            test_size=0.2,
            stratify=y,
            random_state=random_state,
        )

        trial_stub = optuna.trial.FixedTrial(best_params)
        model = build_pose_model(trial_stub, X_train.shape[1:], len(classes))
        batch_size = int(best_params.get("batch_size", cfg.get("batch_size", 16)))

        ckpt_dir = tempfile.mkdtemp()
        ckpt_path = os.path.join(ckpt_dir, "pose_best.h5")
        callbacks = [
            EarlyStopping(monitor="val_accuracy", mode="max", patience=12, restore_best_weights=True, verbose=0),
            ModelCheckpoint(ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=0),
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=cfg.get("epochs", 80),
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
            shuffle=True,
        )

        best_model = tf.keras.models.load_model(ckpt_path)
        os.makedirs("models", exist_ok=True)
        final_path = os.path.join("models", "lstm_pose_tuned.h5")
        best_model.save(final_path)

        best_val_acc = float(np.max(history.history["val_accuracy"]))
        mlflow.log_params(best_params)
        mlflow.log_param("input_shape", str(X_train.shape[1:]))
        mlflow.log_param("num_classes", len(classes))
        mlflow.log_metric("best_val_accuracy", best_val_acc)
        mlflow.log_metric("best_val_loss", float(np.min(history.history["val_loss"])))
        mlflow.log_artifact(final_path)
        mlflow.keras.log_model(best_model, artifact_path="model", registered_model_name="Pose_LSTM_Tuned")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    args = parser.parse_args()

    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    cfg = params["pose_pipeline"]
    ml_cfg = params.get("mlflow", {})
    mlflow.set_tracking_uri(ml_cfg.get("tracking_uri", "file:./mlruns"))
    if ml_cfg.get("enable_system_metrics", False):
        mlflow.enable_system_metrics_logging()

    random_state = params["base"]["random_state"]
    tf.keras.utils.set_random_seed(random_state)

    data_path = cfg["data_path"]
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Pose data path not found: {data_path}")

    X, y, classes = load_pose_data(data_path)
    if len(X) == 0:
        raise RuntimeError("No pose samples were loaded; check data_path.")

    objective = PoseObjective(X, y, classes, cfg, random_state, "Pose_LSTM_Tuning")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=args.trials)

    retrain_and_register(X, y, classes, cfg, random_state, "Pose_LSTM_Tuning", study.best_trial.params)


if __name__ == "__main__":
    main()
