import os
import numpy as np
import cv2
import mediapipe as mp

def extract_3d_landmarks_from_video(video_path, crop_config=None, target_width=720, min_detection_confidence=0.2, min_tracking_confidence=0.2):
    """
    Extracts 3D pose landmarks from a video file.

    Returns a tuple of (landmarks_array, fps).
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False, model_complexity=2,
        min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence
    )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 # Default FPS if not available

    all_landmarks = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if crop_config:
            h, w, _ = frame.shape
            start_row = int(h * crop_config.get("top", 0.0))
            end_row = h - int(h * crop_config.get("bottom", 0.0))
            start_col = int(w * crop_config.get("left", 0.0))
            end_col = w - int(w * crop_config.get("right", 0.0))
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


def preprocess_videos_with_sliding_window(data_path, sequence_length, stride, crop_config):
    """
    Finds videos in the data_path, extracts landmarks, and saves them
    as pre-sliced .npz files in the same directory.
    """
    print(f"Scanning for videos in: {data_path}")

    # Iterate over each shot type directory (e.g., 'clear', 'smash')
    for shot_type in os.listdir(data_path):
        shot_dir = os.path.join(data_path, shot_type)

        if not os.path.isdir(shot_dir):
            continue

        # Process each video file in the directory
        for file_name in os.listdir(shot_dir):
            if not file_name.lower().endswith(('.mp4', '.avi', '.mov')):
                continue

            video_path = os.path.join(shot_dir, file_name)
            print(f"\nProcessing video: {video_path}")

            # Extract landmarks and FPS from the video
            landmarks, fps = extract_3d_landmarks_from_video(video_path, crop_config=crop_config)

            if landmarks is None or len(landmarks) < sequence_length:
                print(f"  -> Skipped: Not enough frames ({len(landmarks) if landmarks is not None else 0})")
                continue

            # --- SLIDING WINDOW LOGIC ---
            window_count = 0
            for i in range(0, len(landmarks) - sequence_length + 1, stride):
                window = landmarks[i : i + sequence_length]
                
                # Define a unique name for each window file
                video_name_base = os.path.splitext(file_name)[0]
                output_filename = f"{video_name_base}_window_{window_count}.npz"
                output_path = os.path.join(shot_dir, output_filename) # Save in the same folder
                
                # Save the windowed landmarks and original FPS
                np.savez(output_path, landmarks=window, fps=fps)
                window_count += 1
            
            print(f"  -> Success: Saved {window_count} windows.")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # This is the single path to your data, containing subfolders for each shot type.
    DATA_PATH = "/home/smayan/Desktop/IPD/Data"
    SEQUENCE_LENGTH = 40
    STRIDE = 5
    CROP_CONFIG = {
    "top": 0.10, "bottom": 0.45, "left": 0.25, "right": 0.25
}

    print("--- Starting Preprocessing with Sliding Window ---")
    preprocess_videos_with_sliding_window(
        DATA_PATH,
        SEQUENCE_LENGTH,
        STRIDE,
        CROP_CONFIG
    )
    print("\n--- Preprocessing complete! ---")