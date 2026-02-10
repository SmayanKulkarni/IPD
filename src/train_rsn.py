#!/usr/bin/env python3
"""
Stage 1: Train RSN End-to-End on Extracted Frame Images

This script trains the Residual-Shuffle Network (RSN) as a 6-class
frame classifier using pre-extracted images from 'data/rsn_frames'.
This is significantly faster than reading from video files.

Usage:
    python src/train_rsn.py [--epochs 50] [--batch-size 32]
"""

import os
import sys
import argparse
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

# Import custom RSN
sys.path.insert(0, os.path.dirname(__file__))
from rsn import build_rsn

def load_params():
    """Load configuration from params.yaml"""
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def build_rsn_classifier(input_shape, num_classes, dropout_rate=0.3):
    """Build RSN with classification head (trainable)."""
    backbone = build_rsn(input_shape=input_shape)
    backbone.trainable = True
    
    x = backbone.output
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=backbone.input, outputs=outputs, name="RSN_Classifier")
    return model

def main():
    parser = argparse.ArgumentParser(description="Train RSN on extracted frame images")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-path", type=str, default="data/rsn_frames")
    parser.add_argument("--output", type=str, default="models/rsn_pretrained.h5")
    args = parser.parse_args()
    
    params = load_params()
    img_size = params.get('hybrid_pipeline', {}).get('cnn_input_size', 224)
    
    if not os.path.isdir(args.data_path):
        print(f"❌ Data path not found: {args.data_path}")
        print("   Run 'python src/prepare_rsn_data.py' first!")
        sys.exit(1)
        
    print("=" * 60)
    print("🚀 RSN TRAINING (Fast Image Loader)")
    print("=" * 60)
    print(f"📂 Data: {args.data_path}")
    
    # Use standard Keras image loader - highly optimized
    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_path,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(img_size, img_size),
        batch_size=args.batch_size,
        label_mode='int'
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_path,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(img_size, img_size),
        batch_size=args.batch_size,
        label_mode='int'
    )
    
    class_names = train_ds.class_names
    print(f"📋 Classes: {class_names}")
    
    # Optimization: Prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    # Normalization layer (rescaling inputs from [0, 255] to [0, 1])
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    model = build_rsn_classifier(
        input_shape=(img_size, img_size, 3),
        num_classes=len(class_names)
    )
    
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    callbacks = [
        ModelCheckpoint(args.output, monitor='val_accuracy', save_best_only=True, save_weights_only=False, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]
    
    print(f"\n🏋️ Training RSN for {args.epochs} epochs...")
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    print(f"\n✅ Training Complete!")
    print(f"   Weights saved to: {args.output}")
    
    # Save classes
    class_file = args.output.replace('.h5', '_classes.txt')
    with open(class_file, 'w') as f:
        f.write('\n'.join(class_names))

if __name__ == "__main__":
    main()
