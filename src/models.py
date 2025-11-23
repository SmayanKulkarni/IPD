import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, 
    GRU, SpatialDropout1D, Concatenate, ReLU
)
from tensorflow.keras.regularizers import l2

def build_lstm_pose(input_shape, num_classes):
    model = Sequential([
        Conv1D(128, 4, activation='relu', input_shape=input_shape),
        MaxPooling1D(3), Dropout(0.3),
        LSTM(128, return_sequences=True, activation='relu'), Dropout(0.4),
        BatchNormalization(),
        LSTM(64, activation='relu'), Dropout(0.3),
        Dense(64, activation='relu'), Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_tcn_hybrid(pose_shape, cnn_shape, num_classes):
    reg = l2(1e-4)
    cnn_in = Input(shape=cnn_shape)
    x = Conv1D(64, 3, padding="causal", dilation_rate=1, kernel_regularizer=reg)(cnn_in)
    x = BatchNormalization()(x); x = ReLU()(x); x = SpatialDropout1D(0.2)(x)
    x = Conv1D(64, 3, padding="causal", dilation_rate=2, kernel_regularizer=reg)(x)
    x = BatchNormalization()(x); x = ReLU()(x); x = SpatialDropout1D(0.2)(x)
    x = GRU(64, dropout=0.3)(x)
    x = Dense(32, activation="relu", kernel_regularizer=reg)(x)
    
    pose_in = Input(shape=pose_shape)
    y = GRU(64, dropout=0.4)(pose_in)
    y = BatchNormalization()(y)
    y = Dense(32, activation="relu", kernel_regularizer=reg)(y)
    y = Dropout(0.3)(y)
    
    out = Dense(num_classes, activation="softmax")(
        Dropout(0.4)(Dense(64, activation="relu", kernel_regularizer=reg)(Concatenate()([x, y])))
    )
    model = Model([cnn_in, pose_in], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model