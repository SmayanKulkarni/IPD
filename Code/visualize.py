import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from config import CROP_CONFIG # <-- IMPORT THE CONFIG
import os # <-- Import the os module for path manipulation

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def visualize_keypoints_on_video(video_path, crop_config=None):
    """
    Crops video frames based on a config and displays pose landmarks in a 2D video window
    and a real-time 3D plot.
    """
    # Your confidence parameters are preserved
    pose = mp_pose.Pose(min_detection_confidence=0.2, min_tracking_confidence=0.2)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return
    
    print("Processing video... Press 'q' in the video window to quit.")

    # --- 3D PLOT SETUP ---
    plt.ion() # Turn on interactive mode for real-time plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # --- END PLOT SETUP ---

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Finished processing video.")
            break

        # This function's logic doesn't change; it just uses the config it's given.
        if crop_config:
            h, w, _ = frame.shape
            start_row = int(h * crop_config.get("top", 0.0))
            end_row = h - int(h * crop_config.get("bottom", 0.0))
            start_col = int(w * crop_config.get("left", 0.0))
            end_col = w - int(w * crop_config.get("right", 0.0))
            frame_cropped = frame[start_row:end_row, start_col:end_col]
        else:
            frame_cropped = frame

        # --- FIX: Corrected the typo from COLOR_BGR_RGB to COLOR_BGR2RGB ---
        image_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Draw 2D landmarks on the video frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        # --- UPDATE 3D PLOT ---
        if results.pose_world_landmarks:
            ax.clear() # Clear the previous frame's plot
            ax.set_xlim3d([-1.0, 1.0])
            ax.set_ylim3d([-1.0, 1.0])
            ax.set_zlim3d([-1.0, 1.0])
            ax.view_init(elev=10., azim=100) # Adjust camera angle

            landmarks = results.pose_world_landmarks.landmark
            
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = landmarks[start_idx]
                    end_point = landmarks[end_idx]
                    ax.plot([start_point.x, end_point.x],
                            [start_point.z, end_point.z],
                            [-start_point.y, -end_point.y],
                            'o-')

            plt.pause(0.001) 

        cv2.imshow('MediaPipe Pose Visualization', image_bgr)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    pose.close()
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()


if __name__ == '__main__':
 
    VIDEO_FILE = "/home/smayan/Desktop/IPD/Data/backhand_drive/backhand_drive (4).mp4" 
    
    filename = os.path.basename(VIDEO_FILE)
    filename_without_ext = os.path.splitext(filename)[0]

    active_crop_config = None
    if filename_without_ext.isdigit():
        print(f"--- Filename '{filename}' is numeric. Applying crop settings from config.py. ---")
        active_crop_config = CROP_CONFIG
    else:
        print(f"--- Filename '{filename}' is not numeric. Using full frame. ---")
        pass

    visualize_keypoints_on_video(VIDEO_FILE, crop_config=active_crop_config)

