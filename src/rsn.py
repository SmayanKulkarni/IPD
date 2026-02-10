"""
Residual-Shuffle Network (RSN) for Feature Extraction.

A lightweight, efficient backbone combining:
- Residual Connections (skip connections for gradient flow)
- Channel Shuffling (efficient cross-group information mixing)

Based on ShuffleNet V2 principles with custom adaptations for
action recognition feature extraction.
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, Model


def channel_shuffle(x, groups):
    """
    Shuffle channels across groups.
    
    This operation rearranges channels so that information can flow
    between groups that were previously isolated. Essential for
    making grouped convolutions effective.
    
    Args:
        x: Input tensor (B, H, W, C)
        groups: Number of groups to shuffle across
    
    Returns:
        Tensor with shuffled channels
    """
    batch_size, height, width, channels = tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
    channels_per_group = channels // groups
    
    # Reshape to (B, H, W, groups, channels_per_group)
    x = tf.reshape(x, [batch_size, height, width, groups, channels_per_group])
    
    # Transpose to (B, H, W, channels_per_group, groups)
    x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
    
    # Flatten back to (B, H, W, C)
    x = tf.reshape(x, [batch_size, height, width, channels])
    
    return x


def shuffle_unit(x, out_channels, groups=4, stride=1, name_prefix="shuffle"):
    """
    Shuffle Unit - the core building block of RSN.
    
    For stride=1 (identity shortcut):
        - Split channels in half
        - Process one half through bottleneck
        - Concatenate and shuffle
    
    For stride=2 (downsampling):
        - Both branches downsample
        - Concatenate (doubles channels)
    
    Args:
        x: Input tensor
        out_channels: Output channels (will be split between branches)
        groups: Groups for channel shuffle
        stride: 1 for identity, 2 for downsampling
        name_prefix: Prefix for layer names
    
    Returns:
        Output tensor
    """
    in_channels = x.shape[-1]
    
    if stride == 1:
        # Split channels: left path (identity), right path (conv)
        split_channels = in_channels // 2
        branch_left = x[..., :split_channels]
        branch_right = x[..., split_channels:]
        
        # Right branch: Bottleneck with depthwise separable conv
        # 1x1 Conv -> DepthwiseConv -> 1x1 Conv
        branch_right = layers.Conv2D(
            split_channels, 1, padding='same', use_bias=False,
            name=f"{name_prefix}_1x1_1"
        )(branch_right)
        branch_right = layers.BatchNormalization(name=f"{name_prefix}_bn_1")(branch_right)
        branch_right = layers.ReLU(name=f"{name_prefix}_relu_1")(branch_right)
        
        branch_right = layers.DepthwiseConv2D(
            3, strides=1, padding='same', use_bias=False,
            name=f"{name_prefix}_dw"
        )(branch_right)
        branch_right = layers.BatchNormalization(name=f"{name_prefix}_bn_dw")(branch_right)
        
        branch_right = layers.Conv2D(
            split_channels, 1, padding='same', use_bias=False,
            name=f"{name_prefix}_1x1_2"
        )(branch_right)
        branch_right = layers.BatchNormalization(name=f"{name_prefix}_bn_2")(branch_right)
        branch_right = layers.ReLU(name=f"{name_prefix}_relu_2")(branch_right)
        
        # Concatenate
        x = layers.Concatenate(name=f"{name_prefix}_concat")([branch_left, branch_right])
        
    else:  # stride == 2: Downsampling
        # Left branch: DepthwiseConv (stride 2) -> 1x1
        branch_left = layers.DepthwiseConv2D(
            3, strides=2, padding='same', use_bias=False,
            name=f"{name_prefix}_left_dw"
        )(x)
        branch_left = layers.BatchNormalization(name=f"{name_prefix}_left_bn_dw")(branch_left)
        branch_left = layers.Conv2D(
            out_channels // 2, 1, padding='same', use_bias=False,
            name=f"{name_prefix}_left_1x1"
        )(branch_left)
        branch_left = layers.BatchNormalization(name=f"{name_prefix}_left_bn")(branch_left)
        branch_left = layers.ReLU(name=f"{name_prefix}_left_relu")(branch_left)
        
        # Right branch: 1x1 -> DepthwiseConv (stride 2) -> 1x1
        branch_right = layers.Conv2D(
            in_channels, 1, padding='same', use_bias=False,
            name=f"{name_prefix}_right_1x1_1"
        )(x)
        branch_right = layers.BatchNormalization(name=f"{name_prefix}_right_bn_1")(branch_right)
        branch_right = layers.ReLU(name=f"{name_prefix}_right_relu_1")(branch_right)
        
        branch_right = layers.DepthwiseConv2D(
            3, strides=2, padding='same', use_bias=False,
            name=f"{name_prefix}_right_dw"
        )(branch_right)
        branch_right = layers.BatchNormalization(name=f"{name_prefix}_right_bn_dw")(branch_right)
        
        branch_right = layers.Conv2D(
            out_channels // 2, 1, padding='same', use_bias=False,
            name=f"{name_prefix}_right_1x1_2"
        )(branch_right)
        branch_right = layers.BatchNormalization(name=f"{name_prefix}_right_bn_2")(branch_right)
        branch_right = layers.ReLU(name=f"{name_prefix}_right_relu_2")(branch_right)
        
        # Concatenate (channel expansion)
        x = layers.Concatenate(name=f"{name_prefix}_concat")([branch_left, branch_right])
    
    # Channel Shuffle
    x = layers.Lambda(
        lambda t: channel_shuffle(t, groups),
        name=f"{name_prefix}_shuffle"
    )(x)
    
    return x


def build_rsn(input_shape=(224, 224, 3), num_stages=4, base_channels=64, groups=4):
    """
    Build the Residual-Shuffle Network backbone.
    
    Architecture:
        - Stem: 3x3 Conv, stride 2 + MaxPool
        - Stage 1-4: Shuffle Units (increasing channels, decreasing resolution)
        - Global Average Pooling
    
    Args:
        input_shape: Input tensor shape (H, W, C)
        num_stages: Number of stages (default 4)
        base_channels: Starting channel count (doubles each stage)
        groups: Groups for channel shuffling
    
    Returns:
        Keras Model (backbone, outputs feature vector before final projection)
    """
    # Stage configs: (num_units, output_channels)
    # Channels double at each stage, resolution halves
    stage_configs = [
        (2, base_channels),       # Stage 1: 56x56 -> 28x28
        (3, base_channels * 2),   # Stage 2: 28x28 -> 14x14
        (3, base_channels * 4),   # Stage 3: 14x14 -> 7x7
        (2, base_channels * 8),   # Stage 4: 7x7 -> 4x4 (or smaller)
    ][:num_stages]
    
    inputs = layers.Input(shape=input_shape, name="rsn_input")
    
    # Stem: Initial conv + pooling
    x = layers.Conv2D(
        base_channels, 3, strides=2, padding='same', use_bias=False,
        name="stem_conv"
    )(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name="stem_pool")(x)
    
    # Build stages
    for stage_idx, (num_units, out_channels) in enumerate(stage_configs):
        stage_name = f"stage{stage_idx + 1}"
        
        # First unit with stride=2 for downsampling
        x = shuffle_unit(
            x, out_channels, groups=groups, stride=2,
            name_prefix=f"{stage_name}_unit0"
        )
        
        # Remaining units with stride=1
        for unit_idx in range(1, num_units):
            x = shuffle_unit(
                x, out_channels, groups=groups, stride=1,
                name_prefix=f"{stage_name}_unit{unit_idx}"
            )
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    
    model = Model(inputs=inputs, outputs=x, name="ResidualShuffleNet")
    return model


def build_rsn_feature_extractor(input_shape=(224, 224, 3), feature_dim=64, weights_path=None):
    """
    Build RSN with projection head for feature extraction.
    
    This wraps the backbone with a dense layer to project to the
    desired feature dimension, followed by L2 normalization.
    
    Args:
        input_shape: (H, W, C)
        feature_dim: Output feature dimension
        weights_path: Optional path to pretrained RSN weights (.h5 file)
                     These should be weights from train_rsn.py Stage 1
    
    Returns:
        Keras Model outputting normalized feature vectors
    """
    backbone = build_rsn(input_shape=input_shape)
    
    # Projection head
    x = backbone.output
    x = layers.Dense(feature_dim, activation='relu', name="projection")(x)
    x = layers.Lambda(
        lambda v: tf.math.l2_normalize(v, axis=1),
        name="l2_norm"
    )(x)
    
    model = Model(inputs=backbone.input, outputs=x, name="RSN_FeatureExtractor")
    
    # Load pretrained weights if provided
    if weights_path and os.path.exists(weights_path):
        print(f"📦 Loading pretrained RSN weights from: {weights_path}")
        try:
            # Try loading as full Keras model (from train_rsn.py)
            from tensorflow.keras.models import load_model
            pretrained_model = load_model(weights_path, compile=False)
            
            # Transfer weights by layer name (backbone layers only)
            pretrained_layers = {layer.name: layer for layer in pretrained_model.layers}
            transferred = 0
            for layer in model.layers:
                if layer.name in pretrained_layers:
                    try:
                        layer.set_weights(pretrained_layers[layer.name].get_weights())
                        transferred += 1
                    except Exception:
                        pass  # Shape mismatch, skip layer
            
            print(f"✅ Transferred weights for {transferred} layers")
        except Exception as e:
            print(f"⚠️ Could not load model, trying weights: {e}")
            try:
                model.load_weights(weights_path, by_name=True, skip_mismatch=True)
                print("✅ Pretrained weights loaded successfully")
            except Exception as e2:
                print(f"⚠️ Could not load weights: {e2}")
    elif weights_path:
        print(f"⚠️ Weights file not found: {weights_path}")
        print("   Using random initialization")
    
    model.trainable = False  # Freeze for feature extraction
    
    return model

