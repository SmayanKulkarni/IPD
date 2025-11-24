import os, yaml, numpy as np
from tqdm import tqdm
from features import HybridFeatureExtractor

def main():
    with open("params.yaml") as f: params = yaml.safe_load(f)
    cfg = params['hybrid_pipeline']
    extractor = HybridFeatureExtractor(params['mediapipe'], cfg['cnn_feature_dim'])
    raw_dir, out_dir = params['base']['raw_data_path'], cfg['data_path']
    
    os.makedirs(out_dir, exist_ok=True)
    for cls in os.listdir(raw_dir):
        cls_in, cls_out = os.path.join(raw_dir, cls), os.path.join(out_dir, cls)
        if not os.path.isdir(cls_in): continue
        os.makedirs(cls_out, exist_ok=True)
        
        videos = [v for v in os.listdir(cls_in) if v.endswith(('.mp4', '.avi', '.mov'))]
        
        for vid in tqdm(videos, desc=f"Hybrid Prep {cls}"):
            # --- NEW: CHECK IF ALREADY PROCESSED ---
            first_window_path = os.path.join(cls_out, f"{vid[:-4]}_win_0.npz")
            if os.path.exists(first_window_path):
                continue
            # ---------------------------------------

            data, fps = extractor.extract(os.path.join(cls_in, vid), cfg['crop_config'])
            if data is None or len(data) < cfg['sequence_length']: continue
            
            for i in range(0, len(data) - cfg['sequence_length'] + 1, cfg['stride']):
                np.savez(os.path.join(cls_out, f"{vid[:-4]}_win_{i}.npz"),
                         features=data[i:i+cfg['sequence_length']], fps=fps)

if __name__ == "__main__": main()