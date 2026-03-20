"""Quick debug: check model input spec and test a prediction."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/tcn_hybrid_tuned.h5", compile=False)
model.summary()

print("\n--- Input layers ---")
for inp in model.inputs:
    print(f"  Name: {inp.name}, Shape: {inp.shape}")

print("\n--- Output layer ---")
print(f"  Name: {model.output.name}, Shape: {model.output.shape}")

# Load one sample
data = np.load("data/Data_Normalized_Hybrid/forehand_drive/002_win_0.npz")
feat = data['features']  # (30, 163)
print(f"\nSample features shape: {feat.shape}")
print(f"First 5 values of first frame features[:5]: {feat[0, :5]}")
print(f"Feature at col 64 (start of pose?): {feat[0, 64:67]}")

# Try both orderings
cnn = feat[:, :64][np.newaxis]   # (1, 30, 64)
pose = feat[:, 64:][np.newaxis]  # (1, 30, 99)

print(f"\nCNN input shape: {cnn.shape}, Pose input shape: {pose.shape}")

# Check model input names to determine order
input_names = [inp.name for inp in model.inputs]
print(f"Model input names: {input_names}")

# Try prediction with [cnn, pose]
pred1 = model.predict([cnn, pose], verbose=0)
print(f"\n[cnn, pose] prediction: {pred1[0]}")
print(f"  Predicted class: {np.argmax(pred1)}")

# Try prediction with [pose, cnn]
pred2 = model.predict([pose, cnn], verbose=0)
print(f"\n[pose, cnn] prediction: {pred2[0]}")
print(f"  Predicted class: {np.argmax(pred2)}")

# All data evaluation
print("\n--- Full dataset evaluation ---")
DATA_DIR = "data/Data_Normalized_Hybrid"
CLASSES = sorted([c for c in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, c))])

all_cnn, all_pose, all_labels = [], [], []
for label_idx, cls in enumerate(CLASSES):
    cls_dir = os.path.join(DATA_DIR, cls)
    for f in os.listdir(cls_dir):
        if not f.endswith('.npz'): continue
        d = np.load(os.path.join(cls_dir, f))
        feat = d['features']
        all_cnn.append(feat[:, :64])
        all_pose.append(feat[:, 64:])
        all_labels.append(label_idx)

all_cnn = np.array(all_cnn)
all_pose = np.array(all_pose)
all_labels = np.array(all_labels)

# Try both orderings on full data
preds_cp = model.predict([all_cnn, all_pose], verbose=0)
acc_cp = np.mean(np.argmax(preds_cp, axis=1) == all_labels)

preds_pc = model.predict([all_pose, all_cnn], verbose=0)
acc_pc = np.mean(np.argmax(preds_pc, axis=1) == all_labels)

print(f"[cnn, pose] accuracy: {acc_cp:.4f}")
print(f"[pose, cnn] accuracy: {acc_pc:.4f}")
print(f"Total samples: {len(all_labels)}")
print(f"Samples per class: {[int((all_labels==i).sum()) for i in range(len(CLASSES))]}")
