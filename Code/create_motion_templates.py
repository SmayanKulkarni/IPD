# create_motion_templates.py
import numpy as np
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
from badminton_utils import dynamic_time_warping, normalize_landmarks

def create_templates(data_path, templates_path):
    if not os.path.exists(templates_path):
        os.makedirs(templates_path)

    shot_types = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

    for shot_type in shot_types:
        print(f"--- Creating template for: {shot_type} ---")
        shot_path = os.path.join(data_path, shot_type)
        
        all_sequences = []
        for file_name in os.listdir(shot_path):
            if file_name.endswith(".npz"):
                data = np.load(os.path.join(shot_path, file_name))
                raw_landmarks = data['landmarks']
                normalized = normalize_landmarks(raw_landmarks)
                if normalized is not None and len(normalized) > 20:
                    all_sequences.append(normalized)

        if len(all_sequences) < 2:
            print(f"Not enough sequences for {shot_type} to create a template, skipping.")
            continue

        lengths = [len(seq) for seq in all_sequences]
        median_length = int(np.median(lengths))
        ref_index = np.argmin([abs(l - median_length) for l in lengths])
        reference_seq = all_sequences[ref_index]
        
        aligned_sequences = [reference_seq]

        for i, seq in enumerate(tqdm(all_sequences, desc="Aligning sequences")):
            if i == ref_index: continue
            
            aligned_ref, aligned_seq_resampled = dynamic_time_warping(reference_seq, seq)
            
            if len(aligned_seq_resampled) != len(reference_seq):
                x_old = np.linspace(0, 1, len(aligned_seq_resampled))
                x_new = np.linspace(0, 1, len(reference_seq))
                f = interp1d(x_old, aligned_seq_resampled, axis=0, kind='linear', fill_value="extrapolate")
                aligned_seq_resampled = f(x_new)

            aligned_sequences.append(aligned_seq_resampled)
        
        motion_template = np.mean(aligned_sequences, axis=0)
        
        template_save_path = os.path.join(templates_path, f"{shot_type}_template.npy")
        np.save(template_save_path, motion_template)
        print(f"Saved template for {shot_type} to {template_save_path}")

if __name__ == "__main__":
    DATA_PATH = "/home/smayan/Desktop/IPD/Data"
    TEMPLATES_PATH = "motion_templates"
    create_templates(DATA_PATH, TEMPLATES_PATH)