import os
import yaml
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf  # <--- Added import
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from dvclive.keras import DVCLiveCallback
# Local Import
from models import build_lstm_pose

def main():
    with open("params.yaml") as f: params = yaml.safe_load(f)
    cfg = params['pose_pipeline']
    
    # 1. Setup MLflow Experiment & System Monitoring
    mlflow.set_experiment("Pose_LSTM_Experiment")
    mlflow.enable_system_metrics_logging()
    
    # We disable auto-logging for models to manually log the BEST one later
    mlflow.tensorflow.autolog(log_models=False)

    with mlflow.start_run():
        # 2. Log Parameters
        mlflow.log_params(cfg)
        mlflow.log_params(params['mediapipe'])
        
        # 3. Load Data
        X, y = [], []
        if not os.path.exists(cfg['data_path']): 
            print(f"Data path {cfg['data_path']} not found.")
            return

        classes = sorted(os.listdir(cfg['data_path']))
        for i, cls in enumerate(classes):
            path = os.path.join(cfg['data_path'], cls)
            for f in os.listdir(path):
                X.append(np.load(os.path.join(path, f))['features'])
                y.append(i)
        
        if not X: 
            print("No data loaded.")
            return
            
        X_train, X_test, y_train, y_test = train_test_split(
            np.array(X), to_categorical(y, len(classes)), test_size=0.2, stratify=y
        )
        
        # 4. Build Model
        model = build_lstm_pose(X_train.shape[1:], len(classes))
        
        # 5. Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            # Saves the best model to the local path defined in params.yaml
            ModelCheckpoint(cfg['model_path'], save_best_only=True),
            DVCLiveCallback(save_dvc_exp=True)
        ]

        print("Starting Pose Training...")
        history = model.fit(
            X_train, y_train, validation_data=(X_test, y_test),
            epochs=cfg['epochs'], batch_size=cfg['batch_size'],
            callbacks=callbacks
        )
        
        best_val_acc = max(history.history['val_accuracy'])
        mlflow.log_metric("best_val_accuracy", best_val_acc)
        print(f"Training finished. Best Val Acc: {best_val_acc}")

        # 6. Log and Register the Best Model
        print("Logging and Registering Best Model to MLflow...")
        # Load the best model from the checkpoint file
        best_model = tf.keras.models.load_model(cfg['model_path'])
        
        mlflow.keras.log_model(
            best_model, 
            artifact_path="model", 
            registered_model_name="Pose_LSTM" # <--- Registers model in Registry
        )
        print("Model registered as 'Pose_LSTM'.")

if __name__ == "__main__": main()