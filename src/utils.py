import numpy as np

def normalize_pose(keypoints_3d):
    """
    Normalizes a single 3D pose (33, 3) to a standard, person-centric coordinate system.
    (Centering -> Alignment -> Scaling)
    """
    if keypoints_3d.shape != (33, 3): return keypoints_3d
    
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_HIP, RIGHT_HIP = 23, 24

    # 1. Centering
    hip_center = (keypoints_3d[LEFT_HIP] + keypoints_3d[RIGHT_HIP]) / 2.0
    centered = keypoints_3d - hip_center

    # 2. Alignment
    shoulder_center = (centered[LEFT_SHOULDER] + centered[RIGHT_SHOULDER]) / 2.0
    spine_len = np.linalg.norm(shoulder_center)
    
    if spine_len < 1e-6: return centered

    new_y = shoulder_center / spine_len
    right_shoulder_vec = centered[RIGHT_SHOULDER] - centered[LEFT_SHOULDER]
    
    proj = np.dot(right_shoulder_vec, new_y) * new_y
    new_x = right_shoulder_vec - proj
    
    if np.linalg.norm(new_x) < 1e-6:
         new_x = np.cross(new_y, [0, 1, 0]) if abs(new_y[0]) > 0.5 else np.cross(new_y, [1, 0, 0])

    new_x /= np.linalg.norm(new_x)
    new_z = np.cross(new_x, new_y)

    rotation = np.array([new_x, new_y, new_z])
    aligned = np.dot(centered, rotation.T)

    # 3. Scaling
    return aligned / spine_len

def normalize_sequence(keypoints_sequence):
    """Applies pose normalization to an entire sequence of frames."""
    return np.array([normalize_pose(frame) for frame in keypoints_sequence])