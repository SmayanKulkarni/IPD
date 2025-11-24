import argparse
import cv2
import yaml
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
# Local imports
from utils import normalize_sequence
from features import PoseFeatureExtractor

def load_config():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def visualize_2d(video_path, crop_config):
    print(f"\n--- 2D Visualization: {video_path} ---")
    print("Press 'q' to quit.")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Apply Crop
        h, w = frame.shape[:2]
        start_row = int(h * crop_config['top'])
        end_row = h - int(h * crop_config['bottom'])
        start_col = int(w * crop_config['left'])
        end_col = w - int(w * crop_config['right'])
        
        frame_cropped = frame[start_row:end_row, start_col:end_col]
        if frame_cropped.size == 0: continue

        # Detect
        image_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        # Draw
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame_cropped, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        # Draw crop box on original frame for context (optional visualization)
        # To keep it simple, we just show the cropped view which the model sees
        cv2.imshow('Badminton 2D View (Cropped)', frame_cropped)
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

def visualize_3d(video_path, crop_config, mp_config):
    print(f"\n--- 3D Visualization: {video_path} ---")
    print("Extracting landmarks...")
    
    # Use our unified Extractor logic
    extractor = PoseFeatureExtractor(mp_config)
    
    # Note: We need to temporarily patch the extraction logic to respect crop_config 
    # if extract_full_sequence doesn't support it natively yet.
    # For this visualizer, we'll implement a quick custom extraction loop that mirrors preprocessing.
    
    cap = cv2.VideoCapture(video_path)
    raw_landmarks = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        h, w = frame.shape[:2]
        frame = frame[
            int(h*crop_config['top']):h-int(h*crop_config['bottom']),
            int(w*crop_config['left']):w-int(w*crop_config['right'])
        ]
        if frame.size == 0: continue
        
        res = extractor.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_world_landmarks:
            lm = np.array([[l.x, l.y, l.z] for l in res.pose_world_landmarks.landmark])
            raw_landmarks.append(lm)
            
    cap.release()
    
    if not raw_landmarks:
        print("No landmarks found.")
        return

    print(f"Normalizing {len(raw_landmarks)} frames...")
    normalized_seq = normalize_sequence(raw_landmarks)

    # Animation Setup
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    pose_connections = mp.solutions.pose.POSE_CONNECTIONS

    def update(frame_idx):
        ax.clear()
        keypoints = normalized_seq[frame_idx]
        
        # Plot Joints
        ax.scatter(keypoints[:, 0], keypoints[:, 2], keypoints[:, 1], c='red', marker='o')

        # Plot Bones
        for conn in pose_connections:
            start, end = conn
            ax.plot([keypoints[start, 0], keypoints[end, 0]],
                    [keypoints[start, 2], keypoints[end, 2]],
                    [keypoints[start, 1], keypoints[end, 1]], 'b-')
        
        ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_zlabel('Y')
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
        ax.set_title(f"Frame {frame_idx}/{len(normalized_seq)}")
        ax.view_init(elev=20, azim=-60)

    print("Starting Animation Window...")
    ani = FuncAnimation(fig, update, frames=len(normalized_seq), interval=50)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--mode", choices=['2d', '3d', 'both'], default='both')
    args = parser.parse_args()

    params = load_config()
    # Use pose pipeline config as default for visualization
    crop_cfg = params['pose_pipeline']['crop_config']
    mp_cfg = params['mediapipe']

    if args.mode in ['2d', 'both']:
        visualize_2d(args.video, crop_cfg)
    
    if args.mode in ['3d', 'both']:
        visualize_3d(args.video, crop_cfg, mp_cfg)