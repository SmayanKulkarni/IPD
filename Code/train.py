# train_model.py
import numpy as np
from badminton_utils2 import load_and_preprocess_data_for_training, build_lstm_model

if __name__ == "__main__":
    DATA_PATH = "/home/smayan/Desktop/IPD/Data" 
    SEQUENCE_LENGTH = 50
    MODEL_NAME = "badminton_shot_classifier_v3_sliced.h5"
    NUM_FEATURES = 12 

    print("--- Training model on temporally sliced ANGULAR FEATURES ---")
    (X_train, X_test, y_train, y_test), label_map = load_and_preprocess_data_for_training(
        DATA_PATH, sequence_length=SEQUENCE_LENGTH
    )
    if X_train.shape[0] == 0:
        print("No data was loaded. Check your data path and preprocessing.")
    else:
        print(f"Found {len(label_map)} classes: {list(label_map.keys())}")
        NUM_CLASSES = len(label_map)
        input_shape = (X_train.shape[1], X_train.shape[2]) 
        
        model = build_lstm_model(input_shape, NUM_CLASSES)
        model.summary()
        
        model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))
        
        model.save(MODEL_NAME)
        print(f"\nModel saved as {MODEL_NAME}")