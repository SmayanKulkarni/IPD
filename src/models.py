"""
Neural Network Architecture Definitions
========================================

Defines the deep learning model architectures for badminton shot classification.
Provides two model builders for different feature input types.

Architectures:
    1. build_lstm_pose(input_shape, num_classes)
       - Conv1D feature extractor: 128 filters, kernel_size=4
       - Stacked LSTM layers: 128 units → 64 units
       - Batch normalization for training stability
       - Dropout regularization (0.3-0.4) to prevent overfitting
       - Softmax classifier for multi-class output
       
    2. build_tcn_hybrid(pose_shape, cnn_shape, num_classes)
       - CNN Branch: Dilated causal Conv1D (TCN-style) + GRU
       - Pose Branch: GRU with batch normalization
       - Late fusion: Concatenation of branch outputs
       - L2 regularization (1e-4) on all kernels
       - Multi-input model for simultaneous pose+visual features

Design Rationale:
    - Conv1D captures local temporal patterns in pose sequences
    - LSTM/GRU models long-range temporal dependencies
    - Causal convolutions ensure no future information leakage
    - Dilated convolutions expand receptive field efficiently
    - Late fusion allows each modality to learn independently

Input/Output Specifications:
    Pose Model:
        Input: (batch, sequence_length, 99) - normalized pose features
        Output: (batch, num_classes) - shot type probabilities
    
    Hybrid Model:
        Inputs: [(batch, T, cnn_dim), (batch, T, 99)]
        Output: (batch, num_classes) - shot type probabilities

Dependencies:
    External: tensorflow, keras

Author: IPD Research Team  
Version: 1.0.0
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, 
    GRU, SpatialDropout1D, Concatenate, ReLU
)
from tensorflow.keras.regularizers import l2


def build_lstm_pose(input_shape, num_classes):
    """Build Conv1D + LSTM model for pose-based shot classification."""
    model = Sequential([
        Conv1D(filters=128, kernel_size=4, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=3),
        Dropout(0.3),
        
        LSTM(128, return_sequences=True, activation='relu'),
        Dropout(0.4),
        BatchNormalization(),
        
        LSTM(64, activation='relu'),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_tcn_hybrid(pose_shape, cnn_shape, num_classes):
    """Build hybrid TCN model with dual-input architecture for pose+CNN fusion."""
    reg = l2(1e-4)
    
    cnn_in = Input(shape=cnn_shape)
    x = Conv1D(64, 3, padding="causal", dilation_rate=1, kernel_regularizer=reg)(cnn_in)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SpatialDropout1D(0.2)(x)
    x = Conv1D(64, 3, padding="causal", dilation_rate=2, kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SpatialDropout1D(0.2)(x)
    x = GRU(64, dropout=0.3)(x)
    x = Dense(32, activation="relu", kernel_regularizer=reg)(x)
    
    pose_in = Input(shape=pose_shape)
    y = GRU(64, dropout=0.4)(pose_in)
    y = BatchNormalization()(y)
    y = Dense(32, activation="relu", kernel_regularizer=reg)(y)
    y = Dropout(0.3)(y)
    
    fused = Concatenate()([x, y])
    fused = Dense(64, activation="relu", kernel_regularizer=reg)(fused)
    fused = Dropout(0.4)(fused)
    out = Dense(num_classes, activation="softmax")(fused)
    
    model = Model([cnn_in, pose_in], out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy", 
        metrics=["accuracy"]
    )
    return model