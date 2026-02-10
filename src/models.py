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
    """Build hybrid TCN model with temporal self-attention.
    
    Architecture:
        - CNN Branch: Dilated causal TCN → Self-Attention → GRU
        - Pose Branch: Conv1D → Self-Attention → GRU
        - Fusion: Concatenate → Dense (softmax)
    
    Key features:
        - 4 TCN layers with dilations [1,2,4,8] for 31-frame receptive field
        - Residual connections for gradient flow
        - Temporal Multi-Head Self-Attention on both branches
        - Tuned Hyperparameters (Optuna): 92% Acc
    """
    from tensorflow.keras.layers import Add, MultiHeadAttention, LayerNormalization
    
    # Tuned Hyperparameters (Acc: 94.6% in tuning, 92% on test)
    reg = l2(7.6e-5)
    dropout_rate = 0.22
    tcn_filters = 80
    gru_units = 80
    attn_heads = 8
    attn_key_dim = 32
    fusion_units = 80
    
    # --- CNN/Visual Branch with Deep TCN + Attention ---
    cnn_in = Input(shape=cnn_shape, name="cnn_input")
    
    # Initial projection
    x = Conv1D(tcn_filters, 1, kernel_regularizer=reg)(cnn_in)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Deep TCN with residual connections: dilations 1, 2, 4, 8
    for dilation in [1, 2, 4, 8]:
        residual = x
        x = Conv1D(tcn_filters, 3, padding="causal", dilation_rate=dilation, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = SpatialDropout1D(dropout_rate)(x)
        # Residual connection
        x = Add()([x, residual])
    
    # Temporal Self-Attention: learn which frames are most important
    attn_out = MultiHeadAttention(num_heads=attn_heads, key_dim=attn_key_dim, dropout=0.2)(x, x)
    x = Add()([x, attn_out])  # Residual around attention
    x = LayerNormalization()(x)
    
    x = GRU(gru_units, dropout=dropout_rate)(x)
    x = Dense(max(gru_units // 2, 32), activation="relu", kernel_regularizer=reg)(x)
    x = Dropout(dropout_rate)(x)
    
    # --- Pose Branch with Conv1D + Attention + GRU ---
    pose_in = Input(shape=pose_shape, name="pose_input")
    
    # Local pattern extraction with Conv1D
    y = Conv1D(tcn_filters, 3, padding="causal", activation="relu", kernel_regularizer=reg)(pose_in)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(dropout_rate)(y)
    
    # Temporal Self-Attention on pose features
    pose_attn = MultiHeadAttention(num_heads=attn_heads, key_dim=attn_key_dim, dropout=0.2)(y, y)
    y = Add()([y, pose_attn])  # Residual around attention
    y = LayerNormalization()(y)
    
    y = GRU(gru_units, dropout=dropout_rate)(y)
    y = BatchNormalization()(y)
    y = Dense(max(gru_units // 2, 32), activation="relu", kernel_regularizer=reg)(y)
    y = Dropout(dropout_rate)(y)
    
    # --- Fusion Layer ---
    # Tuned fusion architecture
    fused = Concatenate()([x, y])
    fused = Dense(fusion_units, activation="relu", kernel_regularizer=reg)(fused)
    fused = Dropout(min(dropout_rate + 0.1, 0.5))(fused)
    fused = Dense(fusion_units // 2, activation="relu", kernel_regularizer=reg)(fused)
    fused = Dropout(dropout_rate)(fused)
    out = Dense(num_classes, activation="softmax")(fused)
    
    model = Model([cnn_in, pose_in], out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5.6e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), 
        metrics=["accuracy"]
    )
    return model