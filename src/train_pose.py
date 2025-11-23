import os, yaml, numpy as np, mlflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from dvclive.keras import DVCLiveCallback
from models import build_lstm_pose

def main():
    with open("params.yaml") as f: params = yaml.safe_load(f)
    cfg = params['pose_pipeline']
    
    mlflow.set_experiment("Pose_LSTM")
    with mlflow.start_run():
        mlflow.log_params(cfg)
        X, y = [], []
        if not os.path.exists(cfg['data_path']): return

        classes = sorted(os.listdir(cfg['data_path']))
        for i, cls in enumerate(classes):
            path = os.path.join(cfg['data_path'], cls)
            for f in os.listdir(path):
                X.append(np.load(os.path.join(path, f))['features'])
                y.append(i)
        
        if not X: return
        X_train, X_test, y_train, y_test = train_test_split(
            np.array(X), to_categorical(y, len(classes)), test_size=0.2, stratify=y
        )
        
        model = build_lstm_pose(X_train.shape[1:], len(classes))
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(cfg['model_path'], save_best_only=True),
            DVCLiveCallback(save_dvc_exp=True)
        ]

        history = model.fit(
            X_train, y_train, validation_data=(X_test, y_test),
            epochs=cfg['epochs'], batch_size=cfg['batch_size'],
            callbacks=callbacks
        )
        mlflow.log_metric("val_acc", max(history.history['val_accuracy']))

if __name__ == "__main__": main()