import os
import yaml
import cv2
import numpy as np
import mediapipe as mp
import gc
from tqdm import tqdm
from collections import deque
from utils import normalize_pose, should_skip_crop, get_segment_bounds, resolve_crop_config_for_video

def get_pose_model(mp_config):
    """Helper to initialize MediaPipe Pose."""
    return mp.solutions.pose.Pose(
        static_image_mode=False, 
        model_complexity=mp_config['model_complexity'],
        min_detection_confidence=mp_config['min_detection_confidence'],
        min_tracking_confidence=mp_config['min_tracking_confidence']
    )

def process_video_streaming(video_path, output_dir, crop_config, mp_config, seq_len, stride, segment_rules=None):
    """
    Processes video frame-by-frame and saves windows immediately.
    Uses O(1) memory relative to video length.
    """
    filename = os.path.basename(video_path)
    file_id = os.path.splitext(filename)[0]
    
    # --- INCREMENTAL CHECK ---
    # If the first window exists, assume video is done.
    if os.path.exists(os.path.join(output_dir, f"{file_id}_win_0.npz")):
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine the segment bounds (start frame and number of frames)
    start_frame, tail_frames = get_segment_bounds(video_path, fps, total_frames, default_seconds=1.75, segment_cfg=segment_rules)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Decide whether to skip cropping for files with the (N) naming pattern
    skip_crop = should_skip_crop(filename)

    # Rolling buffer to hold exactly 'seq_len' frames
    window_buffer = deque(maxlen=seq_len)

    zeros_pose = np.zeros(99, dtype=np.float32)
    last_pose = None
    
    # Initialize MediaPipe (Local scope to ensure cleanup)
    pose = get_pose_model(mp_config)
    
    frame_idx = 0
    saved_count = 0
    
    try:
        while frame_idx < int(tail_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Crop (skip if filename matches the '(N)' pattern)
            if skip_crop:
                frame_cropped = frame
            else:
                h, w = frame.shape[:2]
                frame_cropped = frame[
                    int(h*crop_config['top']):h-int(h*crop_config['bottom']),
                    int(w*crop_config['left']):w-int(w*crop_config['right'])
                ]
                if frame_cropped.size == 0:
                    pose_vec = last_pose if last_pose is not None else zeros_pose
                    window_buffer.append(pose_vec)
                    if len(window_buffer) == seq_len and ((frame_idx - (seq_len - 1)) % stride == 0):
                        save_path = os.path.join(output_dir, f"{file_id}_win_{saved_count}.npz")
                        np.savez(save_path, features=np.array(window_buffer), fps=fps)
                        saved_count += 1
                    frame_idx += 1
                    del frame
                    del frame_cropped
                    continue

            # Process
            # Pass by reference to avoid copying large arrays
            image_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
            res = pose.process(image_rgb)
            
            # Cleanup heavy frame data immediately
            del frame
            del frame_cropped
            del image_rgb

            if res.pose_world_landmarks:
                lm = np.array([[l.x, l.y, l.z] for l in res.pose_world_landmarks.landmark], dtype=np.float32)
                pose_vec = normalize_pose(lm).flatten().astype(np.float32)
                last_pose = pose_vec
            else:
                pose_vec = last_pose if last_pose is not None else zeros_pose

            window_buffer.append(pose_vec)

            # Save windows on a fixed stride relative to the seeked start_frame
            if len(window_buffer) == seq_len and ((frame_idx - (seq_len - 1)) % stride == 0):
                save_path = os.path.join(output_dir, f"{file_id}_win_{saved_count}.npz")
                np.savez(save_path, features=np.array(window_buffer), fps=fps)
                saved_count += 1
            
            frame_idx += 1

    finally:
        # Ensure resources are freed
        cap.release()
        pose.close()
        del pose
        del window_buffer
        gc.collect()

def main():
    with open("params.yaml") as f: params = yaml.safe_load(f)
    cfg = params['pose_pipeline']
    mp_cfg = params['mediapipe']
    segment_rules = params.get('segment_rules', {})
    crop_overrides = params.get('crop_overrides', {})
    raw_dir = params['base']['raw_data_path']
    out_dir = cfg['data_path']
    
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(raw_dir): return

    for cls in os.listdir(raw_dir):
        cls_in = os.path.join(raw_dir, cls)
        cls_out = os.path.join(out_dir, cls)
        if not os.path.isdir(cls_in): continue
        os.makedirs(cls_out, exist_ok=True)
        
        videos = [v for v in os.listdir(cls_in) if v.lower().endswith(('.mp4', '.avi', '.mov', '.webm'))]
        
        for i, vid in enumerate(tqdm(videos, desc=f"Pose Prep {cls}")):
            video_path = os.path.join(cls_in, vid)
            crop_cfg = resolve_crop_config_for_video(video_path, cfg['crop_config'], crop_overrides)
            process_video_streaming(
                video_path,
                cls_out,
                crop_cfg,
                mp_cfg,
                cfg['sequence_length'],
                cfg['stride'],
                segment_rules
            )
            
            # Aggressive Garbage Collection every 10 videos to prevent memory creep
            if i % 10 == 0:
                gc.collect()

if __name__ == "__main__":
    main()