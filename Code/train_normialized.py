import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
NORMALIZED_DATA_DIR = os.path.join(PROJECT_ROOT, 'Data_Normalized')
MODEL_NAME = os.path.join(BASE_DIR, 'badminton_shot_classifier_normalized.h5')
SEQUENCE_LENGTH = 40

def load_normalized_data(data_dir):
    X = []
    y = []
    
    shot_types = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    labels = {label: num for num, label in enumerate(shot_types)}

    for shot_type, label_num in labels.items():
        shot_type_path = os.path.join(data_dir, shot_type)
        if os.path.isdir(shot_type_path):
            for file_name in os.listdir(shot_type_path):
                if not file_name.endswith('.npz'):
                    continue
                
                file_path = os.path.join(shot_type_path, file_name)
                
                # --- FIX: Load the .npz file and access the array by its key ---
                with np.load(file_path) as data:
                    keypoints = data['landmarks']
                    X.append(keypoints)
                    y.append(label_num)

    return np.array(X), np.array(y), labels

def build_model(input_shape, num_classes):
    """
    Builds and compiles the Conv1D + LSTM model.
    """
    model = Sequential([
        Conv1D(filters=128, kernel_size=4, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=3),
        Dropout(0.2),
        LSTM(128, return_sequences=True, activation='relu'),
        Dropout(0.5),
        LSTM(64, return_sequences=False, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model  

if __name__ == '__main__':
    # Load the NORMALIZED data
    X, y, labels = load_normalized_data(NORMALIZED_DATA_DIR)
    
    if len(X) == 0:
        print(f"Error: No data found in '{NORMALIZED_DATA_DIR}'.")
        print("Please run proprocessing_normalized.py first.")
    else:
        print(f"Labels found: {labels}")
        
        y_categorical = to_categorical(y, num_classes=len(labels))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y)
        
        # This line should now work correctly as X_train is 4D
        num_samples, timesteps, num_keypoints, num_coords = X_train.shape
        
        # Reshape data for LSTM: (samples, timesteps, features)
        X_train_reshaped = X_train.reshape(X_train.shape[0], timesteps, num_keypoints * num_coords)
        X_test_reshaped = X_test.reshape(X_test.shape[0], timesteps, num_keypoints * num_coords)

        input_shape = (timesteps, num_keypoints * num_coords)
        
        model = build_model(input_shape, len(labels))
        model.summary()
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(MODEL_NAME, save_best_only=True, monitor='val_accuracy')
        
        model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_data=(X_test_reshaped, y_test), callbacks=[early_stopping, model_checkpoint])
        
        print(f"Normalized model training complete. Model saved to '{MODEL_NAME}'")
