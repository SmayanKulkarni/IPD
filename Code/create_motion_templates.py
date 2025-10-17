import numpy as np
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
# --- FIX: Import from your primary utility file ---
from badminton_utils2 import dynamic_time_warping 

# --- NEW: Define a base directory to make paths platform-independent ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def create_templates(data_path, templates_path):
    """
    Creates motion templates by aligning and averaging pre-normalized motion sequences.
    """
    if not os.path.exists(templates_path):
        os.makedirs(templates_path)

    shot_types = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

    for shot_type in shot_types:
        print(f"--- Creating template for: {shot_type} ---")
        shot_path = os.path.join(data_path, shot_type)
        
        all_sequences = []
        # --- FIX: Load already normalized sequences ---
        for file_name in os.listdir(shot_path):
            if file_name.endswith(".npz") and "_window_" in file_name:
                file_path = os.path.join(shot_path, file_name)
                with np.load(file_path) as data:
                    # The data is already normalized by the preprocessing script
                    sequences = data['landmarks']
                    all_sequences.append(sequences)

        if len(all_sequences) < 2:
            print(f"Not enough sequences for {shot_type} to create a template, skipping.")
            continue

        # Since all windows have the same length, we can use the first as reference
        reference_seq = all_sequences[0]
        aligned_sequences = [reference_seq]

        for i, seq in enumerate(tqdm(all_sequences[1:], desc="Aligning sequences")):
            # Align every other sequence to the reference sequence
            _, aligned_seq_resampled = dynamic_time_warping(reference_seq, seq)
            
            # Ensure consistent length after DTW (sometimes minor differences can occur)
            if len(aligned_seq_resampled) != len(reference_seq):
                x_old = np.linspace(0, 1, len(aligned_seq_resampled))
                x_new = np.linspace(0, 1, len(reference_seq))
                f = interp1d(x_old, aligned_seq_resampled, axis=0, kind='linear', fill_value="extrapolate")
                aligned_seq_resampled = f(x_new)

            aligned_sequences.append(aligned_seq_resampled)
        
        # Average all aligned sequences to create the final motion template
        motion_template = np.mean(aligned_sequences, axis=0)
        
        template_save_path = os.path.join(templates_path, f"{shot_type}_template.npy")
        np.save(template_save_path, motion_template)
        print(f"Saved template for {shot_type} to {template_save_path}")

if __name__ == "__main__":
    # --- FIX: Use os.path.join for platform-independent paths ---
    DATA_PATH = os.path.join(BASE_DIR, "Data_Normalized")
    TEMPLATES_PATH = os.path.join(BASE_DIR, "motion_templates")
    
    print(f"Reading normalized data from: {DATA_PATH}")
    print(f"Saving templates to: {TEMPLATES_PATH}")
    
    create_templates(DATA_PATH, TEMPLATES_PATH)
