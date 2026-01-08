#!/usr/bin/env python3
"""
Batch Video Visualizer and Reviewer
====================================
Visualizes all videos in a folder using visualize.py (2D/3D mode),
then prompts to keep or delete each video.

Usage:
    python src/batch_visualize_review.py --folder data/raw/forehand_lift --mode both
    python src/batch_visualize_review.py --folder data/raw/backhand_drive --mode 2d
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import List


def find_videos(folder: str) -> List[Path]:
    """Find all video files in the given folder."""
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"❌ Folder not found: {folder}")
        sys.exit(1)
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV'}
    videos = [f for f in folder_path.iterdir() 
              if f.is_file() and f.suffix in video_extensions]
    
    return sorted(videos)


def visualize_video(video_path: Path, mode: str, speed: float) -> bool:
    """
    Run visualize.py on a video.
    Returns True if visualization succeeded, False otherwise.
    """
    cmd = [
        sys.executable,  # Use current Python interpreter
        "src/research/visualize.py",
        "--video", str(video_path),
        "--mode", mode,
        "--speed", str(speed),
    ]
    
    print(f"\n{'='*70}")
    print(f"📹 Visualizing: {video_path.name}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n⚠️  Visualization interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error running visualize.py: {e}")
        return False


def prompt_keep_or_delete(video_path: Path) -> str:
    """
    Prompt user to keep, delete, or skip the video.
    Returns: 'keep', 'delete', or 'quit'
    """
    while True:
        print(f"\n📹 {video_path.name}")
        response = input("Keep this video? [y]es / [n]o (delete) / [s]kip / [q]uit: ").strip().lower()
        
        if response in ['y', 'yes', 'k', 'keep', '']:
            return 'keep'
        elif response in ['n', 'no', 'd', 'delete']:
            return 'delete'
        elif response in ['s', 'skip']:
            return 'skip'
        elif response in ['q', 'quit', 'exit']:
            return 'quit'
        else:
            print("❌ Invalid input. Please enter y/n/s/q")


def delete_video(video_path: Path) -> bool:
    """Delete a video file."""
    try:
        video_path.unlink()
        print(f"🗑️  Deleted: {video_path.name}")
        return True
    except Exception as e:
        print(f"❌ Failed to delete {video_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch visualize videos and prompt to keep or delete each one"
    )
    parser.add_argument(
        "--folder", 
        required=True,
        help="Folder containing videos to review"
    )
    parser.add_argument(
        "--mode", 
        choices=['2d', '3d', 'both'],
        default='both',
        help="Visualization mode (default: both)"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (e.g., 0.5 for half speed)"
    )
    parser.add_argument(
        "--auto-delete",
        action='store_true',
        help="Skip confirmation prompt for deletion (use with caution!)"
    )
    
    args = parser.parse_args()
    
    # Find all videos
    videos = find_videos(args.folder)
    
    if not videos:
        print(f"❌ No videos found in {args.folder}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"📁 Found {len(videos)} video(s) in {args.folder}")
    print(f"🎬 Visualization mode: {args.mode}")
    print(f"{'='*70}")
    
    # Stats
    kept = []
    deleted = []
    skipped = []
    
    # Process each video
    for i, video_path in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Processing: {video_path.name}")
        
        # Visualize
        success = visualize_video(video_path, args.mode, args.speed)
        
        if not success:
            print(f"⚠️  Visualization failed or was interrupted for {video_path.name}")
            # Still ask if user wants to delete
        
        # Prompt for action
        if args.auto_delete:
            # In auto-delete mode, just keep everything unless explicitly told to delete
            print(f"✅ Keeping: {video_path.name} (auto-delete mode)")
            kept.append(video_path.name)
            continue
        
        action = prompt_keep_or_delete(video_path)
        
        if action == 'keep':
            print(f"✅ Keeping: {video_path.name}")
            kept.append(video_path.name)
        
        elif action == 'delete':
            confirm = input(f"⚠️  Are you sure you want to DELETE {video_path.name}? [y/N]: ").strip().lower()
            if confirm in ['y', 'yes']:
                if delete_video(video_path):
                    deleted.append(video_path.name)
            else:
                print(f"↩️  Deletion cancelled. Keeping: {video_path.name}")
                kept.append(video_path.name)
        
        elif action == 'skip':
            print(f"⏭️  Skipped: {video_path.name}")
            skipped.append(video_path.name)
        
        elif action == 'quit':
            print("\n🛑 Quitting batch review...")
            break
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 REVIEW SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Kept:    {len(kept)} video(s)")
    print(f"🗑️  Deleted: {len(deleted)} video(s)")
    print(f"⏭️  Skipped: {len(skipped)} video(s)")
    print(f"📁 Total:   {len(videos)} video(s)")
    
    if deleted:
        print(f"\n🗑️  Deleted files:")
        for name in deleted:
            print(f"   - {name}")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
        sys.exit(0)
