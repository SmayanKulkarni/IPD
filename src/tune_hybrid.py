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
    conv_filters = trial.suggest_int("conv_filters", 48, 160, step=16)
    kernel_size = trial.suggest_int("kernel_size", 3, 7, step=2)
    gru_units = trial.suggest_categorical("gru_units", [48, 64, 96, 128, 160])
    dropout = trial.suggest_float("dropout", 0.1, 0.6)
    l2_weight = trial.suggest_float("l2", 1e-6, 1e-3, log=True)
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)

    reg = l2(l2_weight)

    cnn_in = Input(shape=cnn_shape, name="cnn_input")
    x = Conv1D(conv_filters, kernel_size, padding="causal", dilation_rate=1, kernel_regularizer=reg)(cnn_in)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SpatialDropout1D(dropout)(x)
    x = Conv1D(conv_filters, kernel_size, padding="causal", dilation_rate=2, kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SpatialDropout1D(dropout)(x)
    x = GRU(gru_units, dropout=dropout)(x)
    x = Dense(max(gru_units // 2, 32), activation="relu", kernel_regularizer=reg)(x)
    x = Dropout(dropout)(x)

    pose_in = Input(shape=pose_shape, name="pose_input")
    y = GRU(gru_units, dropout=dropout)(pose_in)
    y = BatchNormalization()(y)
    y = Dense(max(gru_units // 2, 32), activation="relu", kernel_regularizer=reg)(y)
    y = Dropout(dropout)(y)

    out = Dense(num_classes, activation="softmax")(Concatenate()([x, y]))
    model = Model([cnn_in, pose_in], out)
    model.compile(optimizer=Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
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
            batch_size = trial.suggest_categorical("batch_size", [4, 8, 12, 16, 24, 32])

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
