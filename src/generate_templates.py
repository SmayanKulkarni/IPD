import os, yaml, numpy as np
from features import PoseFeatureExtractor
from ksi import extract_ksi_features, dynamic_time_warping

def main():
    with open("params.yaml") as f: params = yaml.safe_load(f)
    cfg, mp_cfg = params['expert_pipeline'], params['mediapipe']
    crop_cfg = params.get('crop', {'top': 0.0, 'bottom': 0.0, 'left': 0.0, 'right': 0.0})
    extractor = PoseFeatureExtractor(mp_cfg)
    
    templates = {}
    if not os.path.exists(cfg['raw_path']):
        print("No expert data found.")
        return

    for cls in os.listdir(cfg['raw_path']):
        cls_path = os.path.join(cfg['raw_path'], cls)
        if not os.path.isdir(cls_path): continue
        
        all_ksi = []
        for vid in os.listdir(cls_path):
            if not vid.endswith(('.mp4', '.avi')): continue
            lms = extractor.extract_full_sequence(os.path.join(cls_path, vid), crop_cfg)
            if lms is not None and len(lms) > 0:
                all_ksi.append(np.array([extract_ksi_features(f) for f in lms]))
        
        if all_ksi:
            ref = all_ksi[0]
            aligned = [ref] + [dynamic_time_warping(ref, s)[1] for s in all_ksi[1:]]
            templates[cls] = np.mean(aligned, axis=0)
            
    np.savez(cfg['output_path'], **templates)

if __name__ == "__main__": main()