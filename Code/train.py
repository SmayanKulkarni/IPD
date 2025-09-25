import numpy as np
from badminton_utils import load_and_preprocess_data_for_training, build_lstm_model

if __name__ == "__main__":
    
    DATA_PATH = "/home/smayan/Desktop/IPD/Data" 
    SEQUENCE_LENGTH = 50
    MODEL_NAME = "badminton_shot_classifier.h5"

    print("--- MODE: Training shot classification model ---")
    
    print("Loading and preprocessing data...")
    (X_train, X_test, y_train, y_test), label_map = load_and_preprocess_data_for_training(
        DATA_PATH, 
        sequence_length=SEQUENCE_LENGTH
    )
    print(f"Found {len(label_map)} classes: {list(label_map.keys())}")

    NUM_CLASSES = len(label_map)
    INPUT_SHAPE = (SEQUENCE_LENGTH, 33 * 3)
    
    X_train = X_train.reshape(X_train.shape[0], SEQUENCE_LENGTH, -1)
    X_test = X_test.reshape(X_test.shape[0], SEQUENCE_LENGTH, -1)
    
    model = build_lstm_model(INPUT_SHAPE, NUM_CLASSES)
    print("\nModel Summary:")
    print(model.summary())
    
    print("\nStarting model training...")
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
    
    print(f"\nSaving model as {MODEL_NAME}...")
    model.save(MODEL_NAME)
    print("Model saved successfully.")
