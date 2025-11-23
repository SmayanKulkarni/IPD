import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, ReLU, SpatialDropout1D,
    GRU, Dense, Concatenate, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import tensorflow as tf

# --- PATH CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = "/home/smayan/Desktop/IPD"
DATA_DIR = os.path.join(PROJECT_ROOT, "Data_Normalized_Hybrid")
MODEL_PATH = os.path.join(BASE_DIR, "badminton_shot_classifier_tcn_gru_dual_temporal_pose_regularized.h5")

SEQ_LENGTH = 40
CNN_FEATURE_DIM = 128

def load_hybrid_data(data_dir):
    X, y = [], []
    shot_types = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    labels = {name: i for i, name in enumerate(shot_types)}
    for label, idx in labels.items():
        path = os.path.join(data_dir, label)
        for f in os.listdir(path):
            if f.endswith(".npz"):
                with np.load(os.path.join(path, f)) as data:
                    X.append(data["features"])
                    y.append(idx)
    return np.array(X), np.array(y), labels

def build_regularized_model(input_shape_pose, input_shape_cnn, num_classes):
    reg = l2(1e-4)

    # --- CNN Temporal Stream (TCN + GRU) ---
    cnn_input = Input(shape=input_shape_cnn, name="cnn_input")
    x = Conv1D(64, 3, padding="causal", dilation_rate=1, kernel_regularizer=reg)(cnn_input)
    x = BatchNormalization()(x); x = ReLU()(x); x = SpatialDropout1D(0.2)(x)
    x = Conv1D(64, 3, padding="causal", dilation_rate=2, kernel_regularizer=reg)(x)
    x = BatchNormalization()(x); x = ReLU()(x); x = SpatialDropout1D(0.2)(x)
    x = GRU(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3)(x)
    x = Dense(32, activation="relu", kernel_regularizer=reg)(x)

    # --- Pose Temporal Stream (GRU) ---
    pose_input = Input(shape=input_shape_pose, name="pose_input")
    y = GRU(64, return_sequences=False, dropout=0.4, recurrent_dropout=0.3)(pose_input)
    y = BatchNormalization()(y)
    y = Dense(32, activation="relu", kernel_regularizer=reg)(y)
    y = Dropout(0.3)(y)

    # --- Fusion ---
    fused = Concatenate()([x, y])
    fused = Dense(64, activation="relu", kernel_regularizer=reg)(fused)
    fused = Dropout(0.4)(fused)
    output = Dense(num_classes, activation="softmax")(fused)

    model = Model(inputs=[cnn_input, pose_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    X, y, labels = load_hybrid_data(DATA_DIR)
    if len(X) == 0:
        print(f"[ERROR] No data found in {DATA_DIR}")
        exit()

    print(f"[INFO] Classes: {labels}")
    y_cat = to_categorical(y, num_classes=len(labels))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y)

    timesteps, feature_dim = X_train.shape[1], X_train.shape[2]
    X_train_pose = X_train[..., :-CNN_FEATURE_DIM]
    X_train_cnn = X_train[..., -CNN_FEATURE_DIM:]
    X_test_pose = X_test[..., :-CNN_FEATURE_DIM]
    X_test_cnn = X_test[..., -CNN_FEATURE_DIM:]

    model = build_regularized_model(
        input_shape_pose=(timesteps, X_train_pose.shape[2]),
        input_shape_cnn=(timesteps, CNN_FEATURE_DIM),
        num_classes=len(labels)
    )
    model.summary()

    es = EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True)
    ckpt = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy")

    model.fit(
        [X_train_cnn, X_train_pose], y_train,
        epochs=150, batch_size=8,  # smaller batch helps generalization
        validation_data=([X_test_cnn, X_test_pose], y_test),
        callbacks=[es, ckpt])

    loss, acc = model.evaluate([X_test_cnn, X_test_pose], y_test)
    print(f"\n[RESULT] Test Accuracy: {acc*100:.2f}% | Loss: {loss:.4f}")
    print(f"[SAVED] Model → {MODEL_PATH}")
