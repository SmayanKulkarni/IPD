import argparse
import os
import yaml
import numpy as np
import json
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# Local Imports
from ksi import extract_ksi_features, calculate_ksi

def evaluate(pipeline_type):
    with open("params.yaml") as f: params = yaml.safe_load(f)
    cfg = params[f'{pipeline_type}_pipeline']
    
    # Ensure we log to the correct experiment
    exp_name = "Pose_LSTM_Experiment" if pipeline_type == 'pose' else "Hybrid_TCN_Experiment"
    mlflow.set_experiment(exp_name)

    # Start a new run for Evaluation
    with mlflow.start_run(run_name=f"Eval_{pipeline_type}"):
        
        # 1. Load Data
        X, y = [], []
        if not os.path.exists(cfg['data_path']): return
        classes = sorted(os.listdir(cfg['data_path']))
        for i, cls in enumerate(classes):
            path = os.path.join(cfg['data_path'], cls)
            for f in os.listdir(path):
                X.append(np.load(os.path.join(path, f))['features'])
                y.append(i)
        
        if not X: return
        X, y_cat = np.array(X), to_categorical(y, len(classes))
        _, X_test, _, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y, random_state=42)
        
        # 2. Load Model & Predict
        model = load_model(cfg['model_path'])
        
        if pipeline_type == 'hybrid':
            cnn_dim = cfg['cnn_feature_dim']
            X_pose, X_cnn = X_test[..., :-cnn_dim], X_test[..., -cnn_dim:]
            loss, acc = model.evaluate([X_cnn, X_pose], y_test, verbose=0)
            y_pred = np.argmax(model.predict([X_cnn, X_pose]), axis=1)
            sample_pose = X_pose
        else:
            loss, acc = model.evaluate(X_test, y_test, verbose=0)
            y_pred = np.argmax(model.predict(X_test), axis=1)
            sample_pose = X_test

        # 3. Calculate KSI
        avg_ksi = 0.0
        template_path = params['expert_pipeline']['output_path']
        if os.path.exists(template_path):
            templates = np.load(template_path)
            scores = []
            # Sample 50 items for speed
            for i in range(min(50, len(sample_pose))):
                cls = classes[np.argmax(y_test[i])]
                if cls in templates:
                    lms = sample_pose[i].reshape(-1, 33, 3)
                    user_ksi = np.array([extract_ksi_features(f) for f in lms])
                    scores.append(calculate_ksi(templates[cls], user_ksi, params['ksi']['weights'])['ksi_total'])
            avg_ksi = np.mean(scores) if scores else 0.0

        # 4. Logging
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("avg_ksi_score", avg_ksi)
        
        # Confusion Matrix
        cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
        plt.title(f"Confusion Matrix - {pipeline_type.upper()}")
        
        cm_path = f"dvclive/{pipeline_type}_confusion_matrix.png"
        os.makedirs("dvclive", exist_ok=True)
        plt.savefig(cm_path)
        plt.close()
        
        # Log Artifact to MLflow
        mlflow.log_artifact(cm_path)

        # Save Metrics for DVC
        metrics = {"accuracy": acc, "loss": loss, "avg_ksi": avg_ksi}
        with open(f"dvclive/{pipeline_type}_metrics.json", "w") as f:
            json.dump(metrics, f)

        print(f"{pipeline_type.upper()} EVAL | Acc: {acc:.4f} | KSI: {avg_ksi:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=['pose', 'hybrid'], required=True)
    evaluate(parser.parse_args().type)