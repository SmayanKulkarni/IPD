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
            # Use first video as reference
            ref = all_ksi[0]
            # DTW-align all other videos to reference
            aligned = [ref]
            for s in all_ksi[1:]:
                _, aligned_seq = dynamic_time_warping(ref, s)
                # Ensure aligned sequence has same length as reference
                if len(aligned_seq) != len(ref):
                    # Resample to match reference length
                    from scipy.interpolate import interp1d
                    old_indices = np.linspace(0, 1, len(aligned_seq))
                    new_indices = np.linspace(0, 1, len(ref))
                    resampled = np.array([interp1d(old_indices, aligned_seq[:, i], kind='linear')(new_indices) 
                                         for i in range(aligned_seq.shape[1])]).T
                    aligned.append(resampled)
                else:
                    aligned.append(aligned_seq)
            
            # Average all aligned sequences
            templates[cls] = np.mean(np.array(aligned), axis=0)
            print(f"✓ Generated template for '{cls}' from {len(all_ksi)} videos (shape: {templates[cls].shape})")
        else:
            print(f"⚠ No valid videos found for class '{cls}'")
            
    if templates:
        np.savez(cfg['output_path'], **templates)
        print(f"\n✅ Saved {len(templates)} expert templates to {cfg['output_path']}")
    else:
        print("\n❌ No templates generated - no expert videos found!")

if __name__ == "__main__": main()