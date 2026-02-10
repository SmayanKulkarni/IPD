"""
Hyperparameter Tuning for Hybrid TCN Model
===========================================

Optuna-based hyperparameter optimization for the hybrid pose+CNN TCN classifier.
Searches over architectural and training hyperparameters for the dual-input
multi-stream architecture.

Key Features:
    - Bayesian hyperparameter search via Optuna
    - Median pruner for early stopping of unpromising trials
    - MLflow integration for trial tracking and comparison
    - Automatic retraining with best hyperparameters
    - Model registration to MLflow Model Registry

Search Space:
    Architecture:
        - conv_filters: 48-160 (step 16)
        - kernel_size: 3-7 (step 2)
        - gru_units: [48, 64, 96, 128, 160]
    
    Regularization:
        - dropout: 0.1-0.6
        - l2_weight: 1e-6 to 1e-3 (log scale)
    
    Training:
        - learning_rate: 1e-5 to 5e-3 (log scale)
        - batch_size: [4, 8, 12, 16, 24, 32]

Pipeline Position:
    preprocess_hybrid.py → [tune_hybrid.py] → evaluate.py
    
    Alternative to train_hybrid.py for finding optimal hyperparameters
    before production training.

Dependencies:
    External: optuna, tensorflow, sklearn, mlflow
    Internal: None (model defined inline for flexibility)

Configuration (params.yaml):
    hybrid_pipeline:
        data_path: Path to preprocessed hybrid data
        cnn_feature_dim: CNN embedding dimension
        epochs: Maximum epochs per trial (default: 250)
    mlflow:
        tracking_uri: MLflow tracking URI

Usage:
    python tune_hybrid.py --trials 50

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
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, SpatialDropout1D, GRU, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def load_hybrid_data(data_path: str):
    X, y = [], []
    classes = sorted(os.listdir(data_path))
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(data_path, cls)
        for fname in os.listdir(cls_dir):
            arr = np.load(os.path.join(cls_dir, fname))["features"]
            X.append(arr)
            y.append(idx)
    return np.array(X), np.array(y), classes


def build_hybrid_model(trial, pose_shape, cnn_shape, num_classes):
    """Build Attention-TCN model with Optuna-suggested hyperparameters."""
    from tensorflow.keras.layers import Add, MultiHeadAttention, LayerNormalization

    # --- Search Space ---
    tcn_filters = trial.suggest_int("tcn_filters", 32, 96, step=16)
    gru_units = trial.suggest_int("gru_units", 32, 96, step=16)
    num_heads = trial.suggest_categorical("attn_heads", [2, 4, 8])
    attn_key_dim = trial.suggest_categorical("attn_key_dim", [8, 16, 32])
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    attn_dropout = trial.suggest_float("attn_dropout", 0.1, 0.3)
    l2_weight = trial.suggest_float("l2", 1e-5, 5e-3, log=True)
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
    fusion_units = trial.suggest_int("fusion_units", 32, 96, step=16)
    n_dilations = trial.suggest_int("n_dilations", 2, 4)

    reg = l2(l2_weight)

    # --- CNN/Visual Branch: Residual Dilated TCN + Attention ---
    cnn_in = Input(shape=cnn_shape, name="cnn_input")
    x = Conv1D(tcn_filters, 1, kernel_regularizer=reg)(cnn_in)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    dilations = [2**i for i in range(n_dilations)]
    for d in dilations:
        residual = x
        x = Conv1D(tcn_filters, 3, padding="causal", dilation_rate=d, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = SpatialDropout1D(dropout)(x)
        from tensorflow.keras.layers import Add as AddLayer
        x = AddLayer()([x, residual])

    # Temporal Self-Attention
    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=attn_key_dim, dropout=attn_dropout)(x, x)
    x = Add()([x, attn_out])
    x = LayerNormalization()(x)

    x = GRU(gru_units, dropout=dropout)(x)
    x = Dense(max(gru_units // 2, 32), activation="relu", kernel_regularizer=reg)(x)
    x = Dropout(dropout)(x)

    # --- Pose Branch: Conv1D + Attention + GRU ---
    pose_in = Input(shape=pose_shape, name="pose_input")
    y = Conv1D(tcn_filters, 3, padding="causal", activation="relu", kernel_regularizer=reg)(pose_in)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(dropout)(y)

    pose_attn = MultiHeadAttention(num_heads=num_heads, key_dim=attn_key_dim, dropout=attn_dropout)(y, y)
    y = Add()([y, pose_attn])
    y = LayerNormalization()(y)

    y = GRU(gru_units, dropout=dropout)(y)
    y = BatchNormalization()(y)
    y = Dense(max(gru_units // 2, 32), activation="relu", kernel_regularizer=reg)(y)
    y = Dropout(dropout)(y)

    # --- Fusion ---
    fused = Concatenate()([x, y])
    fused = Dense(fusion_units, activation="relu", kernel_regularizer=reg)(fused)
    fused = Dropout(min(dropout + 0.1, 0.6))(fused)
    fused = Dense(fusion_units // 2, activation="relu", kernel_regularizer=reg)(fused)
    fused = Dropout(dropout)(fused)
    out = Dense(num_classes, activation="softmax")(fused)

    model = Model([cnn_in, pose_in], out)
    model.compile(
        optimizer=Adam(lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )
    return model


class HybridObjective:
    def __init__(self, X_cnn, X_pose, y, classes, cfg, random_state, experiment):
        self.X_cnn = X_cnn
        self.X_pose = X_pose
        self.y = y
        self.classes = classes
        self.cfg = cfg
        self.random_state = random_state
        self.experiment = experiment

    def __call__(self, trial: optuna.Trial):
        mlflow.set_experiment(self.experiment)
        with mlflow.start_run(run_name=f"trial_{trial.number}"):
            mlflow.log_param("pose_input_shape", str(self.X_pose.shape[1:]))
            mlflow.log_param("cnn_input_shape", str(self.X_cnn.shape[1:]))
            mlflow.log_param("num_classes", len(self.classes))

            idx_train, idx_val = train_test_split(
                np.arange(len(self.y)),
                test_size=0.2,
                stratify=self.y,
                random_state=self.random_state,
            )
            y_cat = to_categorical(self.y, len(self.classes))
            model = build_hybrid_model(trial, self.X_pose.shape[1:], self.X_cnn.shape[1:], len(self.classes))
            batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])

            # Compute class weights
            cw = compute_class_weight('balanced', classes=np.unique(self.y[idx_train]), y=self.y[idx_train])
            class_weight_dict = dict(enumerate(cw))

            ckpt_dir = tempfile.mkdtemp()
            ckpt_path = os.path.join(ckpt_dir, "hybrid_best.h5")
            callbacks = [
                EarlyStopping(monitor="val_accuracy", mode="max", patience=15, restore_best_weights=True, verbose=0),
                ModelCheckpoint(ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=0),
                TFKerasPruningCallback(trial, "val_accuracy"),
            ]

            history = model.fit(
                [self.X_cnn[idx_train], self.X_pose[idx_train]],
                y_cat[idx_train],
                validation_data=([self.X_cnn[idx_val], self.X_pose[idx_val]], y_cat[idx_val]),
                epochs=self.cfg.get("epochs", 250),
                batch_size=batch_size,
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=0,
                shuffle=True,
            )

            best_val_acc = float(np.max(history.history["val_accuracy"]))
            mlflow.log_params({"batch_size": batch_size, "epochs": self.cfg.get("epochs", 250)})
            for k, v in trial.params.items():
                mlflow.log_param(k, v)
            mlflow.log_metric("best_val_accuracy", best_val_acc)
            mlflow.log_metric("best_val_loss", float(np.min(history.history["val_loss"])))
            return best_val_acc


def retrain_and_register(X_cnn, X_pose, y, classes, cfg, random_state, experiment, best_params):
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="best_params_retrain"):
        idx_train, idx_val = train_test_split(
            np.arange(len(y)),
            test_size=0.2,
            stratify=y,
            random_state=random_state,
        )
        y_cat = to_categorical(y, len(classes))

        trial_stub = optuna.trial.FixedTrial(best_params)
        model = build_hybrid_model(trial_stub, X_pose.shape[1:], X_cnn.shape[1:], len(classes))
        batch_size = int(best_params.get("batch_size", cfg.get("batch_size", 8)))

        ckpt_dir = tempfile.mkdtemp()
        ckpt_path = os.path.join(ckpt_dir, "hybrid_best.h5")
        callbacks = [
            EarlyStopping(monitor="val_accuracy", mode="max", patience=18, restore_best_weights=True, verbose=0),
            ModelCheckpoint(ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=0),
        ]

        history = model.fit(
            [X_cnn[idx_train], X_pose[idx_train]],
            y_cat[idx_train],
            validation_data=([X_cnn[idx_val], X_pose[idx_val]], y_cat[idx_val]),
            epochs=cfg.get("epochs", 250),
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
            shuffle=True,
        )

        best_model = tf.keras.models.load_model(ckpt_path)
        os.makedirs("models", exist_ok=True)
        final_path = os.path.join("models", "tcn_hybrid_tuned.h5")
        best_model.save(final_path)

        best_val_acc = float(np.max(history.history["val_accuracy"]))
        mlflow.log_params(best_params)
        mlflow.log_param("pose_input_shape", str(X_pose.shape[1:]))
        mlflow.log_param("cnn_input_shape", str(X_cnn.shape[1:]))
        mlflow.log_param("num_classes", len(classes))
        mlflow.log_metric("best_val_accuracy", best_val_acc)
        mlflow.log_metric("best_val_loss", float(np.min(history.history["val_loss"])))
        mlflow.log_artifact(final_path)
        mlflow.keras.log_model(best_model, artifact_path="model", registered_model_name="Hybrid_TCN_Tuned")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    args = parser.parse_args()

    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    cfg = params["hybrid_pipeline"]
    ml_cfg = params.get("mlflow", {})
    mlflow.set_tracking_uri(ml_cfg.get("tracking_uri", "file:./mlruns"))
    if ml_cfg.get("enable_system_metrics", False):
        mlflow.enable_system_metrics_logging()

    random_state = params["base"]["random_state"]
    tf.keras.utils.set_random_seed(random_state)

    data_path = cfg["data_path"]
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Hybrid data path not found: {data_path}")

    X, y, classes = load_hybrid_data(data_path)
    if len(X) == 0:
        raise RuntimeError("No hybrid samples were loaded; check data_path.")

    cnn_dim = cfg["cnn_feature_dim"]
    X_pose = X[..., :-cnn_dim]
    X_cnn = X[..., -cnn_dim:]

    objective = HybridObjective(X_cnn, X_pose, y, classes, cfg, random_state, "Hybrid_TCN_Tuning")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=args.trials)

    retrain_and_register(X_cnn, X_pose, y, classes, cfg, random_state, "Hybrid_TCN_Tuning", study.best_trial.params)


if __name__ == "__main__":
    main()
