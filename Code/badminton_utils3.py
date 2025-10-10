import numpy as np
import os
import math
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Reshape, Resizing, TimeDistributed, Conv2D
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.utils import to_categorical, Sequence

def normalize_landmarks(landmarks):
    # ... (this function remains the same)
    if landmarks is None or len(landmarks) == 0:
        return None
    hip_center = (landmarks[:, 23] + landmarks[:, 24]) / 2
    return landmarks - hip_center[:, np.newaxis, :]

class BadmintonDataGenerator(Sequence):
    # ... (this class remains the same)
    def __init__(self, file_paths, labels, batch_size, label_map):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.label_map = label_map
        self.num_classes = len(label_map)

    def __len__(self):
        return math.ceil(len(self.file_paths) / self.batch_size)

    def __getitem__(self, idx):
        batch_files = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = []
        for file_path in batch_files:
            data = np.load(file_path)
            raw_landmarks = data['landmarks']
            normalized_landmarks = normalize_landmarks(raw_landmarks)
            if normalized_landmarks is not None:
                image_sequence = np.expand_dims(normalized_landmarks, axis=-1)
                batch_x.append(image_sequence)
        batch_y = to_categorical(batch_labels, num_classes=self.num_classes)
        return np.array(batch_x), batch_y

def build_efficientnet_lstm_model(input_shape, num_classes):
    # ... (this function remains the same)
    video_input = Input(shape=input_shape)
    frame_input = Input(shape=input_shape[1:])
    x = Conv2D(3, (1, 1), padding='same', activation='relu')(frame_input)
    x = Resizing(32, 32)(x)
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
    base_model.trainable = False
    frame_features = base_model(x)
    feature_extractor = Model(inputs=frame_input, outputs=frame_features, name="feature_extractor")
    encoded_frames = TimeDistributed(feature_extractor)(video_input)
    encoded_sequence = TimeDistributed(Reshape((-1,)))(encoded_frames)
    x = LSTM(128, return_sequences=True, activation='relu')(encoded_sequence)
    x = Dropout(0.2)(x)
    x = LSTM(256, return_sequences=False, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=video_input, outputs=output)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- NEW FUNCTION FOR MOBILENETV2 ---
def build_mobilenet_lstm_model(input_shape, num_classes):
    """
    Builds the MobileNetV2 + LSTM model.
    """
    video_input = Input(shape=input_shape)
    frame_input = Input(shape=input_shape[1:])
    
    x = Conv2D(3, (1, 1), padding='same', activation='relu')(frame_input)
    x = Resizing(32, 32)(x)
    
    # --- CHANGE: Use MobileNetV2 instead of EfficientNetB0 ---
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(32, 32, 3)
    )
    # --- END OF CHANGE ---
    
    base_model.trainable = False
    
    frame_features = base_model(x)
    feature_extractor = Model(inputs=frame_input, outputs=frame_features, name="feature_extractor")
    
    encoded_frames = TimeDistributed(feature_extractor)(video_input)
    encoded_sequence = TimeDistributed(Reshape((-1,)))(encoded_frames)
    
    x = LSTM(128, return_sequences=True, activation='relu')(encoded_sequence)
    x = Dropout(0.2)(x)
    x = LSTM(256, return_sequences=False, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=video_input, outputs=output)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model