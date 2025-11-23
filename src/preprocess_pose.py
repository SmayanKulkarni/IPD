import os, yaml, cv2, numpy as np, mediapipe as mp
from tqdm import tqdm
from utils import normalize_pose

def main():
    with open("params.yaml") as f: params = yaml.safe_load(f)
    cfg, mp_cfg = params['pose_pipeline'], params['mediapipe']
    raw_dir, out_dir = params['base']['raw_data_path'], cfg['data_path']
    
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=mp_cfg['model_complexity'],
        min_detection_confidence=mp_cfg['min_detection_confidence']
    )
    
    os.makedirs(out_dir, exist_ok=True)
    for cls in os.listdir(raw_dir):
        cls_in, cls_out = os.path.join(raw_dir, cls), os.path.join(out_dir, cls)
        if not os.path.isdir(cls_in): continue
        os.makedirs(cls_out, exist_ok=True)
        
        for vid in tqdm(os.listdir(cls_in), desc=f"Pose Prep {cls}"):
            if not vid.endswith(('.mp4', '.avi', '.mov')): continue
            cap = cv2.VideoCapture(os.path.join(cls_in, vid))
            frames, fps = [], cap.get(cv2.CAP_PROP_FPS)
            while True:
                ret, frame = cap.read()
                if not ret: break
                h, w = frame.shape[:2]
                frame = frame[int(h*cfg['crop_config']['top']):h-int(h*cfg['crop_config']['bottom']),
                              int(w*cfg['crop_config']['left']):w-int(w*cfg['crop_config']['right'])]
                if frame.size == 0: continue
                
                res = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if res.pose_world_landmarks:
                    lm = np.array([[l.x, l.y, l.z] for l in res.pose_world_landmarks.landmark])
                    frames.append(normalize_pose(lm).flatten())
            
            if len(frames) >= cfg['sequence_length']:
                for i in range(0, len(frames)-cfg['sequence_length']+1, cfg['stride']):
                    np.savez(os.path.join(cls_out, f"{vid[:-4]}_win_{i}.npz"),
                             features=frames[i:i+cfg['sequence_length']], fps=fps)

if __name__ == "__main__": main()