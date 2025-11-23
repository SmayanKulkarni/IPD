import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
# Local import
from utils import normalize_pose

class HybridFeatureExtractor:
    def __init__(self, mp_config, cnn_dim=128):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, 
            model_complexity=mp_config['model_complexity'],
            min_detection_confidence=mp_config['min_detection_confidence'],
            min_tracking_confidence=mp_config['min_tracking_confidence']
        )
        
        # CNN Setup
        base_cnn = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        cnn_input = Input(shape=(base_cnn.output_shape[-1],))
        cnn_proj = Dense(cnn_dim, activation='relu')(cnn_input)
        self.cnn_model = Model(inputs=cnn_input, outputs=cnn_proj)
        self.base_cnn = base_cnn

    def extract(self, video_path, crop_config):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None, 30
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            h, w = frame.shape[:2]
            frame = frame[
                int(h*crop_config['top']):h-int(h*crop_config['bottom']),
                int(w*crop_config['left']):w-int(w*crop_config['right'])
            ]
            if frame.size == 0: continue

            # CNN Features
            img = cv2.resize(frame, (224, 224))
            img = preprocess_input(np.expand_dims(img[..., ::-1], axis=0))
            feat = self.cnn_model.predict(self.base_cnn.predict(img, verbose=0), verbose=0)[0]
            cnn_norm = feat / (np.linalg.norm(feat) + 1e-6)

            # Pose Features
            res = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_world_landmarks:
                lm = np.array([[l.x, l.y, l.z] for l in res.pose_world_landmarks.landmark])
                # Use Geometric Normalization (Compatible with KSI)
                pose_norm = normalize_pose(lm).flatten()
            else:
                pose_norm = np.zeros(99)

            frames.append(np.concatenate([pose_norm, cnn_norm]))
            
        cap.release()
        return np.array(frames), fps

class PoseFeatureExtractor:
    def __init__(self, mp_config):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, 
            model_complexity=mp_config['model_complexity'],
            min_detection_confidence=mp_config['min_detection_confidence'],
            min_tracking_confidence=mp_config['min_tracking_confidence']
        )

    def extract_full_sequence(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            res = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_world_landmarks:
                lm = np.array([[l.x, l.y, l.z] for l in res.pose_world_landmarks.landmark])
                frames.append(normalize_pose(lm))
        cap.release()
        return np.array(frames) if frames else None