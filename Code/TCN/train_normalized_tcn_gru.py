import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, Dropout, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- PATH CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = "/home/smayan/Desktop/IPD"
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data_Normalized_Hybrid')
MODEL_PATH = os.path.join(BASE_DIR, 'badminton_shot_classifier_tcn_gru.h5')
SEQ_LENGTH = 40

# --- LOAD DATA ---
def load_hybrid_data(data_dir):
    X, y = [], []
    shot_types = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    labels = {name: i for i, name in enumerate(shot_types)}

    for label, idx in labels.items():
        path = os.path.join(data_dir, label)
        for f in os.listdir(path):
            if f.endswith('.npz'):
                with np.load(os.path.join(path, f)) as data:
                    X.append(data['features'])
                    y.append(idx)
    return np.array(X), np.array(y), labels

# --- BUILD MODEL ---
def build_tcn_gru(input_shape, num_classes):
    model = Sequential([
        # --- Temporal Convolutional Stack (TCN-like) ---
        Conv1D(128, 3, dilation_rate=1, padding='causal'),
        BatchNormalization(), ReLU(),
        Conv1D(128, 3, dilation_rate=2, padding='causal'),
        BatchNormalization(), ReLU(),
        Conv1D(128, 3, dilation_rate=4, padding='causal'),
        BatchNormalization(), ReLU(),
        Dropout(0.3),

        # --- Temporal summarizer ---
        GRU(128, return_sequences=False),

        # --- Classifier ---
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- TRAINING LOOP ---
if __name__ == "__main__":
    X, y, labels = load_hybrid_data(DATA_DIR)
    if len(X) == 0:
        print(f"[ERROR] No data found in {DATA_DIR}")
        exit()

    print(f"[INFO] Classes: {labels}")
    y_cat = to_categorical(y, num_classes=len(labels))

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)
    timesteps, feature_dim = X_train.shape[1], X_train.shape[2]

    model = build_tcn_gru((timesteps, feature_dim), len(labels))
    model.summary()

    es = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    ckpt = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy')

    model.fit(X_train, y_train, epochs=100, batch_size=16,
              validation_data=(X_test, y_test), callbacks=[es, ckpt])

    loss, acc = model.evaluate(X_test, y_test)
    print(f"\n[RESULT] Test Accuracy: {acc*100:.2f}% | Loss: {loss:.4f}")
    print(f"[SAVED] Model → {MODEL_PATH}")
