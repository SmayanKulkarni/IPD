import numpy as np
from badminton_utils2 import load_and_preprocess_data_for_training, build_lstm_model
from tensorflow.keras.callbacks import EarlyStopping

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Path to the pre-processed .npz files (the output of proprocessing.py)
    DATA_PATH = "/home/smayan/Desktop/IPD/Data"
    MODEL_NAME = "2_bigger_window_reduced_data_badminton_shot_classifier_v5.h5"

    print("--- Loading Pre-Windowed Data for Training ---")
    (X_train, X_test, y_train, y_test), label_map = load_and_preprocess_data_for_training(DATA_PATH)

    if X_train.shape[0] == 0:
        print("No data was loaded. Please run the preprocessing script first.")
    else:
        print(f"Data loaded successfully!")
        print(f"Found {len(label_map)} classes: {list(label_map.keys())}")
        print(f"Training with {len(X_train)} sequences and testing with {len(X_test)} sequences.")

        # --- MODEL TRAINING ---
        NUM_CLASSES = len(label_map)
        # Dynamically determine the input shape from the loaded data
        INPUT_SHAPE = (X_train.shape[1], X_train.shape[2])

        model = build_lstm_model(INPUT_SHAPE, NUM_CLASSES)
        model.summary()

        early_stopping = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)

        print("\n--- Starting Model Training ---")
        model.fit(
            X_train, y_train,
            epochs=45,
            batch_size= 8,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping]
        )

        model.save(MODEL_NAME)
        print(f"\n Model training complete. Saved as {MODEL_NAME}")