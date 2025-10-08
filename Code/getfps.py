import cv2

def get_video_fps(video_path):
    """
    Opens a video file and returns its FPS.
    """
    # Create a VideoCapture object
    video_capture = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not video_capture.isOpened():
        print(f"Error: Could not open video file at '{video_path}'")
        return None
        
    # Get the FPS from the video's properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    
    # Release the video capture object
    video_capture.release()
    
    return fps

if __name__ == '__main__':
    # --- CHANGE THIS PATH ---
    VIDEO_FILE_PATH = "/home/smayan/Desktop/IPD/Data/backhand_net_shot/019.mp4" 
    
    video_fps = get_video_fps(VIDEO_FILE_PATH)
    
    if video_fps is not None:
        print("\n" + "="*30)
        print(f"🚀 The FPS of the video is: {video_fps:.2f}")
        print("="*30)