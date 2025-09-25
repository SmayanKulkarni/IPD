import cv2
import mediapipe as mp
from config import CROP_CONFIG # <-- IMPORT THE CONFIG

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def visualize_keypoints_on_video(video_path, crop_config=None):
    """
    Crops video frames based on a config and displays pose landmarks.
    """
    pose = mp_pose.Pose(min_detection_confidence=0.2, min_tracking_confidence=0.2)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return
    
    print("Processing video... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
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

        cv2.imshow('MediaPipe Pose Visualization', image_bgr)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    pose.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    VIDEO_FILE = "/home/smayan/Desktop/IPD/Data/backhand_drive/002.mp4" 
    
    print(f"--- Visualizing with Crop Settings: {CROP_CONFIG} ---")
    visualize_keypoints_on_video(VIDEO_FILE, crop_config=CROP_CONFIG)
