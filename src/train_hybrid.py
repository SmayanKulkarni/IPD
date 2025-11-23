import os, yaml, numpy as np, mlflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from dvclive.keras import DVCLiveCallback
from models import build_tcn_hybrid

def main():
    with open("params.yaml") as f: params = yaml.safe_load(f)
    cfg = params['hybrid_pipeline']
    
    mlflow.set_experiment("Hybrid_TCN")
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
        X = np.array(X)
        
        cnn_dim = cfg['cnn_feature_dim']
        X_pose = X[..., :-cnn_dim]
        X_cnn = X[..., -cnn_dim:]
        y_cat = to_categorical(y, len(classes))
        
        idx_train, idx_test = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y)
        
        model = build_tcn_hybrid(X_pose.shape[1:], X_cnn.shape[1:], len(classes))
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ModelCheckpoint(cfg['model_path'], save_best_only=True),
            DVCLiveCallback(save_dvc_exp=True)
        ]

        model.fit(
            [X_cnn[idx_train], X_pose[idx_train]], y_cat[idx_train],
            validation_data=([X_cnn[idx_test], X_pose[idx_test]], y_cat[idx_test]),
            epochs=cfg['epochs'], batch_size=cfg['batch_size'],
            callbacks=callbacks
        )

if __name__ == "__main__": main()