#!/usr/bin/env python3
"""
Prepare Data for RSN Training (Stage 1)

This script extracts frames from raw videos and saves them as images
in 'data/rsn_frames/'. This is done once to speed up training,
avoiding the need to open/decode video files repeatedly during training.

Output Structure:
data/rsn_frames/
    ├── backhand_drive/
    │   ├── video1_frame000.jpg
    │   ├── video1_frame002.jpg
    │   └── ...
    ├── forehand_clear/
    │   └── ...
    └── ...

Usage:
    python src/prepare_rsn_data.py [--frame-skip 2]
"""

import os
import sys
import argparse
import cv2
import yaml
import shutil
from glob import glob
from tqdm import tqdm

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def extract_frames(data_path, output_path, frame_skip=2, img_size=224, clean=False):
    params = load_params()
    # Override img_size from params if available
    img_size = params.get('hybrid_pipeline', {}).get('cnn_input_size', img_size)
    
    if clean and os.path.exists(output_path):
        print(f"🧹 Cleaning output directory: {output_path}")
        shutil.rmtree(output_path)
    
    os.makedirs(output_path, exist_ok=True)
    
    classes = sorted([d for d in os.listdir(data_path) 
                      if os.path.isdir(os.path.join(data_path, d))])
    
    print(f"🚀 Extracting frames for RSN training (Size: {img_size}x{img_size}, Skip: {frame_skip})")
    print(f"📂 Source: {data_path}")
    print(f"📂 Destination: {output_path}")
    
    total_extracted = 0
    
    for cls in classes:
        src_cls_dir = os.path.join(data_path, cls)
        dst_cls_dir = os.path.join(output_path, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)
        
        video_files = glob(os.path.join(src_cls_dir, "*.mp4")) + \
                      glob(os.path.join(src_cls_dir, "*.avi")) + \
                      glob(os.path.join(src_cls_dir, "*.mov"))
        
        for vid_path in tqdm(video_files, desc=f"   {cls}"):
            vid_name = os.path.splitext(os.path.basename(vid_path))[0]
            cap = cv2.VideoCapture(vid_path)
            frame_count = 0
            saved_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_skip == 0:
                    # Resize and save
                    frame = cv2.resize(frame, (img_size, img_size))
                    out_name = f"{vid_name}_f{frame_count:05d}.jpg"
                    out_path = os.path.join(dst_cls_dir, out_name)
                    cv2.imwrite(out_path, frame)
                    saved_count += 1
                
                frame_count += 1
            cap.release()
            total_extracted += saved_count
            
    print(f"\n✅ Extraction Complete! Total frames: {total_extracted}")
    print(f"📂 Data ready in: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/raw")
    parser.add_argument("--output-path", default="data/rsn_frames")
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--clean", action="store_true", help="Delete existing data first")
    args = parser.parse_args()
    
    extract_frames(args.data_path, args.output_path, args.frame_skip, clean=args.clean)
