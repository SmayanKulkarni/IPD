import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from config import CROP_CONFIG
import os

# Initialize MediaPipe components
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def visualize_keypoints_on_video(video_path, crop_config=None):
    """
    Crops video frames based on a config and displays both 2D and 3D pose landmarks.
    """
    pose = mp_pose.Pose(
        static_image_mode=False, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return
    
    print("Processing video... Press 'q' in the video window to quit.")

    # --- 3D PLOT SETUP ---
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Finished processing video.")
            break

        if crop_config:
            h, w, _ = frame.shape
            start_row = int(h * crop_config.get("top", 0.0))
            end_row = h - int(h * crop_config.get("bottom", 0.0))
            start_col = int(w * crop_config.get("left", 0.0))
            end_col = w - int(w * crop_config.get("right", 0.0))
            frame_cropped = frame[start_row:end_row, start_col:end_col]
        else:
            frame_cropped = frame

        image_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        # --- 3D PLOT UPDATE LOGIC (MANUAL VERSION) ---
        if results.pose_world_landmarks:
            ax.clear()
            ax.set_title('3D Pose Visualization')
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.set_xlim([-1.0, 1.0]); ax.set_ylim([-1.0, 1.0]); ax.set_zlim([-1.0, 1.0])

            landmarks = results.pose_world_landmarks.landmark
            plotted_landmarks = {}
            # Plot the points
            for idx, landmark in enumerate(landmarks):
                ax.scatter(landmark.x, landmark.y, landmark.z, c='red', marker='o', s=10)
                plotted_landmarks[idx] = (landmark.x, landmark.y, landmark.z)

            # Plot the connections
            if plotted_landmarks:
                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                        start_point = plotted_landmarks[start_idx]
                        end_point = plotted_landmarks[end_idx]
                        ax.plot([start_point[0], end_point[0]], 
                                [start_point[1], end_point[1]], 
                                [start_point[2], end_point[2]], 
                                'b')
            
            ax.invert_zaxis(); ax.invert_yaxis()
            plt.pause(0.001)
        # --------------------------------------------------

        cv2.imshow('MediaPipe Pose Visualization', image_bgr)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    pose.close()
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close(fig)

if __name__ == '__main__':
    VIDEO_FILE = "/home/smayan/Desktop/IPD/Data/forehand_clear/008.mp4"  
    filename = os.path.basename(VIDEO_FILE)
    filename_without_ext = os.path.splitext(filename)[0]
    active_crop_config = CROP_CONFIG if filename_without_ext.isdigit() else None
    
    if active_crop_config:
        print(f"--- Visualizing '{filename}' with cropping settings from config.py ---")
    else:
        print(f"--- Visualizing '{filename}' with full frame ---")
        
    visualize_keypoints_on_video(VIDEO_FILE, crop_config=active_crop_config)