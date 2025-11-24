import os
import yaml
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf # <--- Added import
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from dvclive.keras import DVCLiveCallback
# Local Import
from models import build_tcn_hybrid

def main():
    with open("params.yaml") as f: params = yaml.safe_load(f)
    cfg = params['hybrid_pipeline']
    
    # 1. Setup MLflow Experiment
    mlflow.set_experiment("Hybrid_TCN_Experiment")
    mlflow.enable_system_metrics_logging()
    
    # Disable auto model logging so we can do it manually for the best version
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
            
        X = np.array(X)
        
        # 4. Prepare Data (Split CNN vs Pose features)
        cnn_dim = cfg['cnn_feature_dim']
        X_pose = X[..., :-cnn_dim]
        X_cnn = X[..., -cnn_dim:]
        y_cat = to_categorical(y, len(classes))
        
        idx_train, idx_test = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y)
        
        # 5. Build Model
        model = build_tcn_hybrid(X_pose.shape[1:], X_cnn.shape[1:], len(classes))
        
        # 6. Callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ModelCheckpoint(cfg['model_path'], save_best_only=True),
            DVCLiveCallback(save_dvc_exp=True)
        ]

        print("Starting Hybrid Training...")
        history = model.fit(
            [X_cnn[idx_train], X_pose[idx_train]], y_cat[idx_train],
            validation_data=([X_cnn[idx_test], X_pose[idx_test]], y_cat[idx_test]),
            epochs=cfg['epochs'], batch_size=cfg['batch_size'],
            callbacks=callbacks
        )
        
        best_val_acc = max(history.history['val_accuracy'])
        mlflow.log_metric("best_val_accuracy", best_val_acc)
        print(f"Training finished. Best Val Acc: {best_val_acc}")

        # 7. Log and Register the Best Model
        print("Logging and Registering Best Model to MLflow...")
        best_model = tf.keras.models.load_model(cfg['model_path'])
        
        mlflow.keras.log_model(
            best_model, 
            artifact_path="model", 
            registered_model_name="Hybrid_TCN" # <--- Registers model in Registry
        )
        print("Model registered as 'Hybrid_TCN'.")

if __name__ == "__main__": main()