import os
import argparse
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from badminton_utils2 import normalize_sequence

# --- CONFIGURATION ---
# Landmark connections from MediaPipe for drawing the skeleton
pose_connections = mp.solutions.pose.POSE_CONNECTIONS
# Define a consistent crop setting, same as in your preprocessing
CROP_CONFIG = {
    "top": 0.10, "bottom": 0.45, "left": 0.25, "right": 0.25
}

def extract_3d_landmarks_from_video(video_path, crop_config=None, min_detection_confidence=0.2, min_tracking_confidence=0.2):
    """
    Extracts 3D pose landmarks from a single video file.
    Returns a tuple of (landmarks_array, fps).
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False, model_complexity=2,
        min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence
    )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30  # Default FPS

    all_landmarks = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if crop_config:
            h, w, _ = frame.shape
            start_row, end_row = int(h * crop_config.get("top", 0.0)), h - int(h * crop_config.get("bottom", 0.0))
            start_col, end_col = int(w * crop_config.get("left", 0.0)), w - int(w * crop_config.get("right", 0.0))
            frame_cropped = frame[start_row:end_row, start_col:end_col]
        else:
            frame_cropped = frame

        if frame_cropped.size == 0: continue

        image_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            frame_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            all_landmarks.append(frame_landmarks)

    cap.release()
    pose.close()
    return np.array(all_landmarks) if all_landmarks else None, fps

def visualize_video(video_path):
    """
    Processes a single video to extract, normalize, and animate 3D landmarks.
    """
    print(f"Processing video: {video_path}")
    
    # 1. Extract landmarks from the video
    keypoints_sequence, fps = extract_3d_landmarks_from_video(video_path, crop_config=CROP_CONFIG)

    if keypoints_sequence is None or len(keypoints_sequence) == 0:
        print("Could not extract any landmarks from the video.")
        return
    
    print(f"Extracted {len(keypoints_sequence)} frames. Normalizing...")

    # 2. Normalize the entire sequence of landmarks
    normalized_keypoints = normalize_sequence(keypoints_sequence)
    
    print("Normalization complete. Starting visualization...")

    # 3. Set up the plot for animation
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_index):
        ax.clear()
        keypoints = normalized_keypoints[frame_index]
        
        # --- FIX ---
        # Removed the negative sign from the Y-coordinate (keypoints[:, 1])
        # The normalization already sets 'up' as the positive Y direction.
        # Plotting standard axes: X=width, Z=depth, Y=height
        ax.scatter(keypoints[:, 0], keypoints[:, 2], keypoints[:, 1], c='red', marker='o')

        for connection in pose_connections:
            start_idx, end_idx = connection
            # --- FIX ---
            # Also removed the negative sign here for the Y-coordinate
            ax.plot([keypoints[start_idx, 0], keypoints[end_idx, 0]],
                    [keypoints[start_idx, 2], keypoints[end_idx, 2]],
                    [keypoints[start_idx, 1], keypoints[end_idx, 1]], 'b-')
        
        # --- FIX ---
        # Updated axis labels for clarity
        ax.set_xlabel('X (Width)')
        ax.set_ylabel('Z (Depth)')
        ax.set_zlabel('Y (Height)')
        
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_title(f"Normalized Pose | Frame {frame_index + 1}/{len(normalized_keypoints)}")
        ax.view_init(elev=20, azim=-60)

    ani = FuncAnimation(fig, update, frames=len(normalized_keypoints), interval=1000/fps)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize normalized 3D pose landmarks from a video file.")
    parser.add_argument("video_path", type=str, help="Path to the video file you want to visualize.")
    
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: The file '{args.video_path}' was not found.")
    else:
        visualize_video(args.video_path)

