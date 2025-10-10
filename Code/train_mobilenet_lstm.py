import numpy as np
import os
from sklearn.model_selection import train_test_split
# --- CHANGE: Import the new model builder ---
from badminton_utils3 import build_mobilenet_lstm_model, BadmintonDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

if __name__ == "__main__":
    # --- CONFIGURATION ---
    DATA_PATH = "/home/smayan/Desktop/IPD/Data"
    # --- CHANGE: New model name ---
    MODEL_NAME = "badminton_shot_classifier_mobilenet_lstm.h5"
    BATCH_SIZE = 32 # MobileNet is lighter, so you can often use a larger batch size

    print("--- Preparing Data for Generator ---")
    
    all_files = []
    all_labels = []
    shot_types = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    label_map = {label: num for num, label in enumerate(shot_types)}
    
    for shot_type, shot_type_index in label_map.items():
        shot_path = os.path.join(DATA_PATH, shot_type)
        for file_name in os.listdir(shot_path):
            if file_name.endswith(".npz") and "_window_" in file_name:
                all_files.append(os.path.join(shot_path, file_name))
                all_labels.append(shot_type_index)

    if not all_files:
        print("FATAL ERROR: No preprocessed window files found.")
        exit()

    X_train_files, X_test_files, y_train_labels, y_test_labels = train_test_split(
        all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    train_generator = BadmintonDataGenerator(X_train_files, y_train_labels, BATCH_SIZE, label_map)
    val_generator = BadmintonDataGenerator(X_test_files, y_test_labels, BATCH_SIZE, label_map)

    print(f"Data generators created successfully!")
    print(f"Training on {len(X_train_files)} samples, validating on {len(X_test_files)} samples.")

    # --- MODEL TRAINING ---
    NUM_CLASSES = len(label_map)
    INPUT_SHAPE = (50, 33, 3, 1)

    # --- CHANGE: Call the new model builder ---
    model = build_mobilenet_lstm_model(INPUT_SHAPE, NUM_CLASSES)
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("\n--- Starting MobileNetV2-LSTM Model Training ---")
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        callbacks=[early_stopping]
    )

    model.save(MODEL_NAME)
    print(f"\n✅ Model training complete. Saved as {MODEL_NAME}")