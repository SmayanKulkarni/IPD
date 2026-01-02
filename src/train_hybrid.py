import os
import random
import yaml
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from dvclive.keras import DVCLiveCallback
# Local Import
from models import build_tcn_hybrid
from mlflow_utils import MLflowRunManager

def main():
    with open("params.yaml") as f: params = yaml.safe_load(f)
    cfg = params['hybrid_pipeline']
    
    # Set random seeds for reproducibility
    seed = params['base']['random_state']
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # 1. Setup MLflow with Interactive Run Manager
    run_manager = MLflowRunManager("Hybrid_TCN_Experiment")
    mlflow.enable_system_metrics_logging()
    mlflow.tensorflow.autolog(log_models=False)

    # Start interactive run with prompts for name and description
    with run_manager.start_interactive_run(
        default_description="TCN-Hybrid pipeline training with pose + CNN fusion"
    ):
        # 2. Log Parameters (including nested configs)
        mlflow.log_params(cfg)
        mlflow.log_params(params['mediapipe'])
        mlflow.log_params(params['segment_rules'])
        mlflow.log_param("base.random_state", params['base']['random_state'])
        
        # 3. Load Data
        X, y = [], []
        if not os.path.exists(cfg['data_path']): 
            print(f"❌ Data path {cfg['data_path']} not found.")
            return

        classes = sorted([d for d in os.listdir(cfg['data_path']) 
                         if os.path.isdir(os.path.join(cfg['data_path'], d))])
        for i, cls in enumerate(classes):
            path = os.path.join(cfg['data_path'], cls)
            for f in os.listdir(path):
                if not f.endswith('.npz'):
                    continue
                X.append(np.load(os.path.join(path, f))['features'])
                y.append(i)
        
        if not X: 
            print("❌ No data loaded.")
            return
        
        X = np.array(X)
        y_cat = to_categorical(y, len(classes))
        
        # 4. Prepare Data (Split CNN vs Pose features)
        cnn_dim = cfg['cnn_feature_dim']
        X_pose = X[..., :-cnn_dim]
        X_cnn = X[..., -cnn_dim:]
        
        idx_train, idx_test = train_test_split(
            np.arange(len(X)), 
            test_size=0.2, 
            stratify=y,
            random_state=params['base']['random_state']
        )
        
        # Log dataset information
        run_manager.log_dataset_info(
            X_pose[idx_train], X_pose[idx_test], 
            y_cat[idx_train], y_cat[idx_test], 
            classes
        )
        
        # 5. Build Model
        model = build_tcn_hybrid(X_pose.shape[1:], X_cnn.shape[1:], len(classes))
        
        # Log model architecture
        run_manager.log_model_architecture(model)
        
        # 6. Ensure model directory exists
        os.makedirs(os.path.dirname(cfg['model_path']), exist_ok=True)
        
        # 7. Callbacks
        callbacks = [
            EarlyStopping(
                patience=15, 
                restore_best_weights=True,
                monitor='val_accuracy',
                verbose=1
            ),
            ModelCheckpoint(
                cfg['model_path'], 
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            ),
            DVCLiveCallback(save_dvc_exp=True)
        ]

        print("\n🚀 Starting Hybrid-TCN Training...")
        print(f"   Train samples: {len(idx_train)}")
        print(f"   Test samples: {len(idx_test)}")
        print(f"   Classes: {classes}")
        print(f"   Pose features: {X_pose.shape[1:]}")
        print(f"   CNN features: {X_cnn.shape[1:]}\n")
        
        history = model.fit(
            [X_cnn[idx_train], X_pose[idx_train]], y_cat[idx_train],
            validation_data=([X_cnn[idx_test], X_pose[idx_test]], y_cat[idx_test]),
            epochs=cfg['epochs'], 
            batch_size=cfg['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Log training artifacts and curves
        run_manager.log_training_artifacts(history, save_plots=True)
        
        print(f"\n✅ Training finished!")
        print(f"   Best Val Acc: {max(history.history['val_accuracy']):.4f}")

        # 8. Log and Register the Best Model
        print("\n📦 Logging and Registering Best Model to MLflow...")
        best_model = tf.keras.models.load_model(cfg['model_path'])
        
        # NOTE: Signature inference skipped for multi-input Keras models
        # MLflow's infer_signature() requires TensorSpec inputs, but multi-input
        # models use list inputs which are incompatible. This is a known limitation.
        # See: https://github.com/mlflow/mlflow/issues/4067
        mlflow.keras.log_model(
            best_model, 
            artifact_path="model", 
            registered_model_name="Hybrid_TCN"
        )
        print("✅ Model registered as 'Hybrid_TCN'.")

if __name__ == "__main__": main()
