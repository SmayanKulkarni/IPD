"""
Evaluate the trained Attention-TCN model using the EXACT same
data loading and split as train_hybrid.py.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import yaml
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json

# --- Load params exactly like train_hybrid.py ---
with open("params.yaml") as f:
    params = yaml.safe_load(f)
cfg = params['hybrid_pipeline']
SEED = params['base']['random_state']  # 42
cnn_dim = cfg['cnn_feature_dim']       # 64
DATA_DIR = cfg['data_path']            # data/Data_Normalized_Hybrid

MODEL_PATH = "models/tcn_hybrid_tuned.h5"
OUTPUT_IMG = "Images/per_class_metrics_chart.png"

# --- Load data EXACTLY like train_hybrid.py ---
X, y = [], []
classes = sorted(os.listdir(DATA_DIR))
classes = [c for c in classes if os.path.isdir(os.path.join(DATA_DIR, c))]

for i, cls in enumerate(classes):
    path = os.path.join(DATA_DIR, cls)
    for f in sorted(os.listdir(path)):
        if not f.endswith('.npz'):
            continue
        X.append(np.load(os.path.join(path, f))['features'])
        y.append(i)

X = np.array(X)
y = np.array(y)
print(f"Classes: {classes}")
print(f"Total samples: {len(y)}, Feature shape: {X.shape}")

# Split features EXACTLY as train_hybrid.py does:
# X_pose = X[..., :-cnn_dim]   → first 99 columns (pose)
# X_cnn  = X[..., -cnn_dim:]   → last 64 columns (cnn)
X_pose = X[..., :-cnn_dim]
X_cnn  = X[..., -cnn_dim:]
print(f"Pose shape: {X_pose.shape}, CNN shape: {X_cnn.shape}")

# Reproduce exact train/test split from train_hybrid.py
idx_trainval, idx_test = train_test_split(
    np.arange(len(X)),
    test_size=0.2,
    stratify=y,
    random_state=SEED
)
print(f"Test set size: {len(idx_test)}")
print(f"Per-class test counts: {dict(zip(classes, [int((y[idx_test]==i).sum()) for i in range(len(classes))]))}")

# --- Load model and predict on TEST set ---
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded. Predicting on test set...")

# Model was trained with: model.fit([X_cnn, X_pose], y_cat)
preds = model.predict([X_cnn[idx_test], X_pose[idx_test]], verbose=0)
y_pred = np.argmax(preds, axis=1)
y_test = y[idx_test]

overall_acc = np.mean(y_pred == y_test)
print(f"\nTest set accuracy: {overall_acc:.4f}")

# --- Classification Report ---
report = classification_report(y_test, y_pred, target_names=classes, output_dict=True, zero_division=0)
print("\n" + classification_report(y_test, y_pred, target_names=classes, zero_division=0))

# --- Extract per-class metrics ---
precision_vals = [report[c]['precision'] for c in classes]
recall_vals = [report[c]['recall'] for c in classes]
f1_vals = [report[c]['f1-score'] for c in classes]
support_vals = [report[c]['support'] for c in classes]

# Save JSON
metrics_out = {
    "overall_accuracy": round(overall_acc, 4),
    "evaluation_type": "test_set",
    "split_seed": SEED,
    "test_size": 0.2,
    "per_class": {}
}
for i, c in enumerate(classes):
    metrics_out["per_class"][c] = {
        "precision": round(precision_vals[i], 4),
        "recall": round(recall_vals[i], 4),
        "f1_score": round(f1_vals[i], 4),
        "support": int(support_vals[i])
    }

with open("Images/per_class_metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)
print(f"Metrics saved to Images/per_class_metrics.json")

# --- Graph ---
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'sans-serif'

pretty_names = [c.replace('_', '\n').title() for c in classes]

x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 8))
colors = ['#4A90E2', '#50E3C2', '#F5A623']

r1 = ax.bar(x - width, precision_vals, width, label='Precision', color=colors[0], edgecolor='white', linewidth=1)
r2 = ax.bar(x, recall_vals, width, label='Recall', color=colors[1], edgecolor='white', linewidth=1)
r3 = ax.bar(x + width, f1_vals, width, label='F1-Score', color=colors[2], edgecolor='white', linewidth=1)

ax.set_ylabel('Score', fontsize=16, fontweight='bold', color='#333')
ax.set_title(f'Attention-TCN Per-Class Metrics (Test Accuracy: {overall_acc:.1%})',
             fontsize=18, fontweight='bold', pad=20, color='#1A1A1A')
ax.set_xticks(x)
ax.set_xticklabels(pretty_names, fontsize=12, fontweight='500', color='#444')
ax.set_ylim(0, 1.15)
ax.legend(loc='upper right', fontsize=14, framealpha=0.9, edgecolor='#DDD')

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, color='#555', fontweight='bold')

autolabel(r1); autolabel(r2); autolabel(r3)

for i, s in enumerate(support_vals):
    ax.text(i, -0.06, f'n={int(s)}', ha='center', va='top', fontsize=11, color='#888', style='italic')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#DDD')
ax.spines['bottom'].set_color('#DDD')

fig.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
print(f"Graph saved to {OUTPUT_IMG}")
