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


def should_skip_crop(video_path_or_name: str) -> bool:
    """Return True if the filename matches the pattern "name (N).ext".

    This matches filenames that contain a space followed by a parenthesized
    integer index before the extension, e.g. "backhand_drive (1).mp4".
    The check accepts either a full path or a bare filename.
    """
    import re, os
    name = os.path.basename(video_path_or_name)
    # Match 'something (123).ext' where the number is one or more digits
    return re.search(r"\s\(\d+\)\.[^.]+$", name) is not None


def extract_video_number(video_path_or_name: str) -> int | None:
    """Extract the leading numeric id from a filename/path.

    Examples:
      - "001.mp4" -> 1
      - "006_win_3.npz" -> 6
      - "12_some_name.mov" -> 12

    Returns None if no leading number exists.
    """
    import os
    import re

    name = os.path.basename(video_path_or_name)
    m = re.match(r"^(\d+)", name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def resolve_crop_config_for_video(
    video_path: str,
    base_crop_config: dict,
    crop_overrides: dict | None = None,
) -> dict:
    """Return crop_config with per-shot per-video overrides applied.

    - Only overrides keys present in the matching rule (currently just 'bottom').
    - Matches shot name by checking path components against crop_overrides keys.
    - Matches by leading video number in filename.
    - If multiple ranges match, the last matching range wins.
    """
    import copy
    import os

    effective = copy.deepcopy(base_crop_config) if base_crop_config is not None else {
        'top': 0.0,
        'bottom': 0.0,
        'left': 0.0,
        'right': 0.0,
    }

    if not crop_overrides:
        return effective

    video_num = extract_video_number(video_path)
    if video_num is None:
        return effective

    comps = set(os.path.normpath(video_path).split(os.path.sep))
    matched_shot = None
    for shot in crop_overrides.keys():
        if shot in comps:
            matched_shot = shot
            break

    if not matched_shot:
        return effective

    rules = crop_overrides.get(matched_shot) or []
    for rule in rules:
        try:
            start = int(rule.get('start'))
            end = int(rule.get('end'))
        except Exception:
            continue

        if start <= video_num <= end:
            for k, v in rule.items():
                if k in {'start', 'end'}:
                    continue
                effective[k] = v

    return effective


def get_tail_seconds_for_video(video_path: str, default: float = 1.75) -> float:
    """Return the tail window length in seconds for a given video path.

    Special-case: use 2.0 seconds for forehand_lift and forehand_clear shots.
    The check looks at any path component equal to those shot names so it
    works with full paths or relative paths.
    """
    import os
    p = os.path.normpath(video_path)
    comps = p.split(os.path.sep)
    shots_2s = {'forehand_lift', 'forehand_clear'}
    if any(c in shots_2s for c in comps):
        return 2.0
    return default


def get_segment_bounds(
    video_path: str,
    fps: float,
    total_frames: int,
    default_seconds: float = 1.75,
    segment_cfg: dict | None = None,
):
    """Return (start_frame, frame_count) for the segment to process for a given video.

    Config-driven rules (segment_cfg):
      - default_seconds: fallback window length (tail).
      - tail_seconds: window length for tail_shots.
      - tail_shots: list of shot folder names to use tail_seconds.
      - middle_shots: mapping of shot folder name -> seconds for middle window.

    If no config is provided, uses hardcoded defaults (tail, 1.75s).
    """
    import os

    if segment_cfg is None:
        segment_cfg = {}

    default_seconds = segment_cfg.get('default_seconds', default_seconds)
    tail_seconds = segment_cfg.get('tail_seconds', default_seconds)
    tail_shots = set(segment_cfg.get('tail_shots', []))
    middle_shots = segment_cfg.get('middle_shots', {})

    p = os.path.normpath(video_path)
    comps = p.split(os.path.sep)

    # Middle-shot rule
    for shot, secs in middle_shots.items():
        if shot in comps:
            frames = int(float(secs) * fps)
            center = total_frames // 2
            start = max(0, center - frames // 2)
            return start, frames

    # Tail-shot rule
    if any(shot in comps for shot in tail_shots):
        frames = int(float(tail_seconds) * fps)
        start = max(0, total_frames - frames)
        return start, frames

    # Default tail
    frames = int(float(default_seconds) * fps)
    start = max(0, total_frames - frames)
    return start, frames