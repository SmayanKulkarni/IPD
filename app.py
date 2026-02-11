import os
import sys
import json
import tempfile
import threading
from collections import deque
from typing import Any, Dict, List, Optional

# --- Determinism & CPU defaults (must be before TF import) ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import cv2
import numpy as np
import yaml
import tensorflow as tf
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from features import HybridFeatureExtractor
from ksi_v2 import EnhancedKSI
from natural_language_coach import generate_coaching_report
from utils import normalize_pose, resolve_crop_config_for_video, should_skip_crop
from evaluate_video import predict_shot_type_at_contact, _prepare_model_inputs


def _load_params(params_path: str) -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def _resolve_classes(params: dict) -> List[str]:
    cfg = params.get("hybrid_pipeline", {})
    data_path = cfg.get("data_path")
    if data_path and os.path.isdir(data_path):
        classes = sorted([
            d for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d))
        ])
        if classes:
            return classes

    classes_file = os.path.join(ROOT_DIR, "models", "rsn_pretrained_classes.txt")
    if os.path.exists(classes_file):
        with open(classes_file, "r") as f:
            classes = [line.strip() for line in f if line.strip()]
        if classes:
            return classes

    return [
        "backhand_drive",
        "backhand_net_shot",
        "forehand_clear",
        "forehand_drive",
        "forehand_lift",
        "forehand_net_shot",
    ]


def _apply_crop(frame: np.ndarray, crop_cfg: Optional[dict]) -> np.ndarray:
    if not crop_cfg:
        return frame

    h, w = frame.shape[:2]
    start_row = int(h * float(crop_cfg.get("top", 0.0)))
    end_row = h - int(h * float(crop_cfg.get("bottom", 0.0)))
    start_col = int(w * float(crop_cfg.get("left", 0.0)))
    end_col = w - int(w * float(crop_cfg.get("right", 0.0)))

    cropped = frame[start_row:end_row, start_col:end_col]
    return cropped if cropped.size else frame


def _select_expert_template(templates: Any, shot_class: str) -> Optional[np.ndarray]:
    template_key = shot_class
    if template_key not in templates.files:
        template_key = f"{shot_class}_variant1"
    if template_key not in templates.files:
        return None

    expert_template = templates[template_key]
    if expert_template.ndim == 2 and expert_template.shape[1] == 99:
        return expert_template.reshape(-1, 33, 3)
    return expert_template


def _ksi_to_dict(result) -> Dict[str, Any]:
    per_joint = list(result.per_joint_errors.values()) if result.per_joint_errors else []
    per_joint_sorted = sorted(per_joint, key=lambda x: x.mean_error, reverse=True)
    top_errors = [
        {
            "joint_name": e.joint_name,
            "mean_error": float(e.mean_error),
            "max_error": float(e.max_error),
            "std_error": float(e.std_error),
            "critical_frame": int(e.critical_frame),
            "critical_phase": e.critical_phase.value,
        }
        for e in per_joint_sorted[:5]
    ]

    return {
        "ksi_total": float(result.ksi_total),
        "ksi_weighted": float(result.ksi_weighted),
        "components": {k: float(v) for k, v in (result.components or {}).items()},
        "phase_scores": {k: float(v) for k, v in (result.phase_scores or {}).items()},
        "recommendations": list(result.recommendations or []),
        "top_joint_errors": top_errors,
    }


def _smooth_signal(signal, window_size=3):
    if len(signal) == 0:
        return signal
    alpha = 2.0 / (window_size + 1)
    smoothed = [signal[0]]
    for val in signal[1:]:
        smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
    return np.array(smoothed)


def _detect_contact_realtime(landmarks_buffer):
    if len(landmarks_buffer) < 3:
        return 0

    lm_array = np.array(landmarks_buffer)
    shoulder_pos = lm_array[:, 12, :2]
    elbow_pos = lm_array[:, 14, :2]
    wrist_pos = lm_array[:, 16, :2]

    shoulder_vel = np.linalg.norm(np.diff(shoulder_pos, axis=0), axis=1)
    elbow_vel = np.linalg.norm(np.diff(elbow_pos, axis=0), axis=1)
    wrist_vel = np.linalg.norm(np.diff(wrist_pos, axis=0), axis=1)

    composite_vel = 0.5 * wrist_vel + 0.3 * elbow_vel + 0.2 * shoulder_vel
    composite_vel_smooth = _smooth_signal(composite_vel, window_size=3)

    if len(composite_vel_smooth) > 1:
        acceleration = np.diff(composite_vel_smooth)
        contact_idx = np.argmax(np.abs(acceleration)) + 1
        return min(contact_idx, len(landmarks_buffer) - 1)

    return int(np.argmax(composite_vel))


class PredictionSmoother:
    def __init__(self, num_classes: int, ema_alpha: float = 0.6, history_len: int = 5):
        self.num_classes = num_classes
        self.ema_alpha = ema_alpha
        self.ema_probs = np.ones(num_classes) / num_classes
        self.prediction_history = deque(maxlen=history_len)

    def update(self, raw_probs: np.ndarray) -> tuple:
        self.ema_probs = self.ema_alpha * raw_probs + (1 - self.ema_alpha) * self.ema_probs
        predicted_class = int(np.argmax(raw_probs))
        self.prediction_history.append(predicted_class)
        from collections import Counter
        vote_counts = Counter(self.prediction_history)
        majority_class = vote_counts.most_common(1)[0][0]
        vote_confidence = vote_counts[majority_class] / len(self.prediction_history)
        return self.ema_probs, majority_class, vote_confidence


class ServiceState:
    def __init__(self):
        self.params = None
        self.cfg = None
        self.mp_cfg = None
        self.model = None
        self.templates = None
        self.classes: List[str] = []
        self.extractor: Optional[HybridFeatureExtractor] = None
        self.lock = threading.Lock()


state = ServiceState()
app = FastAPI(title="IPD KSI Inference", version="1.0.0")


@app.on_event("startup")
def _startup():
    params_path = os.path.join(ROOT_DIR, "params.yaml")
    state.params = _load_params(params_path)
    state.cfg = state.params["hybrid_pipeline"]
    state.mp_cfg = state.params["mediapipe"]

    model_path = os.path.join(ROOT_DIR, "models", "tcn_hybrid_tuned.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    templates_path = os.path.join(ROOT_DIR, "data", "expert_templates.npz")
    if not os.path.exists(templates_path):
        raise FileNotFoundError(f"Expert templates not found at {templates_path}")

    state.classes = _resolve_classes(state.params)

    state.extractor = HybridFeatureExtractor(
        mp_config=state.mp_cfg,
        cnn_dim=state.cfg["cnn_feature_dim"],
        cnn_input_size=state.cfg["cnn_input_size"],
        rsn_weights_path=state.cfg.get("rsn_pretrained_weights"),
    )

    state.model = tf.keras.models.load_model(model_path)
    state.templates = np.load(templates_path, allow_pickle=True)

    seq_len = int(state.cfg["sequence_length"])
    cnn_dim = int(state.cfg["cnn_feature_dim"])
    dummy_cnn = np.zeros((1, seq_len, cnn_dim), dtype=np.float32)
    dummy_pose = np.zeros((1, seq_len, 99), dtype=np.float32)
    for _ in range(2):
        _ = state.model.predict([dummy_cnn, dummy_pose], verbose=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index():
    html = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>IPD Realtime KSI Demo</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 24px; }
      video, canvas { width: 480px; height: 360px; border: 1px solid #ddd; }
      .row { display: flex; gap: 24px; align-items: flex-start; }
      pre { background: #f6f6f6; padding: 12px; width: 480px; height: 360px; overflow: auto; }
      button { padding: 8px 16px; }
    </style>
  </head>
  <body>
    <h2>IPD Realtime KSI Demo</h2>
    <div class="row">
      <div>
        <video id="video" autoplay muted></video>
        <div>
          <button id="start">Start</button>
          <button id="stop">Stop</button>
        </div>
      </div>
      <pre id="output">Waiting for data...</pre>
    </div>

    <script>
      const video = document.getElementById('video');
      const output = document.getElementById('output');
      const startBtn = document.getElementById('start');
      const stopBtn = document.getElementById('stop');
      let ws = null;
      let stream = null;
      let sending = false;

      async function start() {
        if (sending) return;
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/realtime');
        ws.onmessage = (evt) => { output.textContent = evt.data; };
        ws.onopen = () => { sending = true; sendFrames(); };
      }

      function stop() {
        sending = false;
        if (ws) ws.close();
        if (stream) stream.getTracks().forEach(t => t.stop());
      }

      async function sendFrames() {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 480;
        canvas.height = 360;
        while (sending) {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.6));
          if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(await blob.arrayBuffer());
          }
          await new Promise(r => setTimeout(r, 120));
        }
      }

      startBtn.onclick = start;
      stopBtn.onclick = stop;
    </script>
  </body>
</html>
    """
    return HTMLResponse(html)


def _extract_features_from_video(video_source: str) -> tuple:
    cfg = state.cfg
    seq_len = int(cfg["sequence_length"])
    cnn_dim = int(cfg["cnn_feature_dim"])

    base_crop = cfg.get("crop_config", {})
    overrides = state.params.get("crop_overrides", {})
    crop_cfg = resolve_crop_config_for_video(video_source, base_crop, overrides)
    if should_skip_crop(video_source):
        crop_cfg = None

    roi_cfg = cfg.get("cnn_roi") or {}

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {video_source}")

    window = deque(maxlen=seq_len)
    landmark_window = deque(maxlen=seq_len)
    valid_mask_window = deque(maxlen=seq_len)

    all_windows = []
    all_landmarks = []
    all_valid_ratios = []
    last_pose = None
    last_box = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if crop_cfg:
            frame = _apply_crop(frame, crop_cfg)

        res = state.extractor.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_count += 1

        if res.pose_landmarks:
            lm = np.array([[l.x, l.y, l.z] for l in res.pose_landmarks.landmark], dtype=np.float32)
            pose_flat = normalize_pose(lm).astype(np.float32).flatten()
            last_pose = pose_flat
            landmark_window.append(lm)
            valid_mask_window.append(1)
        else:
            zeros_pose = np.zeros(99, dtype=np.float32)
            pose_flat = last_pose if last_pose is not None else zeros_pose
            landmark_window.append(np.zeros((33, 3), dtype=np.float32))
            valid_mask_window.append(0)

        h, w = frame.shape[:2]
        box = state.extractor._compute_pose_roi_box(
            getattr(res, "pose_landmarks", None),
            w,
            h,
            roi_cfg,
            last_box=last_box,
        )
        last_box = box if box is not None else last_box
        roi_frame = state.extractor._crop_with_box(frame, box)

        img_size = cfg.get("cnn_input_size", 224)
        img = cv2.resize(roi_frame, (img_size, img_size))
        img = preprocess_input(np.expand_dims(img[..., ::-1], axis=0))
        cnn_feat = state.extractor.rgb_model.predict(img, verbose=0)[0].astype(np.float32)

        fused = np.concatenate([pose_flat, cnn_feat], axis=0)
        window.append(fused)

        if len(window) == seq_len:
            valid_ratio = sum(valid_mask_window) / float(seq_len)
            all_windows.append(np.array(list(window)))
            all_landmarks.append(np.array(list(landmark_window)))
            all_valid_ratios.append(valid_ratio)

    cap.release()

    if not all_windows:
        raise RuntimeError("No valid windows extracted from video")

    filtered_windows = []
    filtered_landmarks = []
    for win, lm, ratio in zip(all_windows, all_landmarks, all_valid_ratios):
        if ratio < 0.7:
            continue
        if not np.isfinite(win).all() or not np.isfinite(lm).all():
            continue
        if np.allclose(lm, 0):
            continue
        filtered_windows.append(win)
        filtered_landmarks.append(lm)

    if not filtered_windows:
        raise RuntimeError("All windows were filtered out due to low pose quality")

    return filtered_windows, filtered_landmarks, frame_count


def _infer_video_sync(video_path: str, skill_level: str) -> Dict[str, Any]:
    with state.lock:
        all_windows, all_landmarks, frame_count = _extract_features_from_video(video_path)
        all_predictions, best_prediction, best_class, contact_info = predict_shot_type_at_contact(
            all_windows,
            all_landmarks,
            state.model,
            state.classes,
            state.cfg["cnn_feature_dim"],
            "hybrid",
        )

        templates = state.templates
        expert_template = _select_expert_template(templates, best_class)
        if expert_template is None:
            raise RuntimeError(f"Template not found for class {best_class}")

        ksi_weights = state.params.get("ksi", {}).get("weights") or {
            "pose": 0.4,
            "velocity": 0.4,
            "acceleration": 0.2,
        }
        ksi_calc = EnhancedKSI()
        ksi_scores = []
        for user_lm in all_landmarks:
            result = ksi_calc.calculate(
                expert_landmarks=expert_template,
                user_landmarks=user_lm,
                weights=ksi_weights,
            )
            ksi_scores.append(result)

        contact_idx = contact_info.get("contact_window", 0)
        if ksi_scores and np.isfinite(ksi_scores[contact_idx].ksi_total) and ksi_scores[contact_idx].ksi_total > 0:
            ksi_result = ksi_scores[contact_idx]
            chosen_idx = contact_idx
            chosen_reason = "contact window"
        else:
            best_idx = int(np.argmax([r.ksi_total for r in ksi_scores])) if ksi_scores else 0
            ksi_result = ksi_scores[best_idx]
            chosen_idx = best_idx
            chosen_reason = "highest KSI"

        coaching_json = generate_coaching_report(
            ksi_result=ksi_result,
            shot_type_str=best_class,
            skill_level_str=skill_level,
            output_format="json",
            simplified=True,
        )

        return {
            "frames": frame_count,
            "prediction": best_prediction,
            "predictions": all_predictions[:10],
            "contact": contact_info,
            "ksi": _ksi_to_dict(ksi_result),
            "ksi_choice": {"index": chosen_idx, "reason": chosen_reason},
            "coaching": json.loads(coaching_json),
        }


@app.post("/infer/video")
async def infer_video(
    file: UploadFile = File(...),
    skill_level: str = Form("intermediate"),
):
    suffix = os.path.splitext(file.filename or "video.mp4")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = await run_in_threadpool(_infer_video_sync, tmp_path, skill_level)
        return JSONResponse(result)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


class RealtimeSession:
    def __init__(self):
        cfg = state.cfg
        self.seq_len = int(cfg["sequence_length"])
        self.cnn_dim = int(cfg["cnn_feature_dim"])
        self.roi_cfg = cfg.get("cnn_roi") or {}
        self.crop_cfg = cfg.get("crop_config") or None
        self.extractor = HybridFeatureExtractor(
            mp_config=state.mp_cfg,
            cnn_dim=self.cnn_dim,
            cnn_input_size=cfg.get("cnn_input_size", 224),
            rsn_weights_path=cfg.get("rsn_pretrained_weights"),
        )
        self.window = deque(maxlen=self.seq_len)
        self.landmarks_buffer = deque(maxlen=self.seq_len)
        self.last_pose = None
        self.last_box = None
        self.smoother = PredictionSmoother(len(state.classes), ema_alpha=0.6, history_len=5)
        self.frame_count = 0

    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        self.frame_count += 1
        if self.crop_cfg:
            frame = _apply_crop(frame, self.crop_cfg)

        res = self.extractor.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if res.pose_landmarks:
            lm = np.array([[l.x, l.y, l.z] for l in res.pose_landmarks.landmark], dtype=np.float32)
            pose_flat = normalize_pose(lm).astype(np.float32).flatten()
            self.last_pose = pose_flat
            self.landmarks_buffer.append(lm)
        else:
            zeros_pose = np.zeros(99, dtype=np.float32)
            pose_flat = self.last_pose if self.last_pose is not None else zeros_pose
            self.landmarks_buffer.append(np.zeros((33, 3), dtype=np.float32))

        h, w = frame.shape[:2]
        box = self.extractor._compute_pose_roi_box(
            getattr(res, "pose_landmarks", None),
            w,
            h,
            self.roi_cfg,
            last_box=self.last_box,
        )
        self.last_box = box if box is not None else self.last_box
        roi_frame = self.extractor._crop_with_box(frame, box)

        img_size = self.extractor.cnn_input_size
        img = cv2.resize(roi_frame, (img_size, img_size))
        img = preprocess_input(np.expand_dims(img[..., ::-1], axis=0))
        cnn_feat = self.extractor.rgb_model.predict(img, verbose=0)[0].astype(np.float32)

        fused = np.concatenate([pose_flat, cnn_feat], axis=0)
        self.window.append(fused)

        if len(self.window) < self.seq_len:
            return None

        x = np.asarray(self.window, dtype=np.float32)[None, ...]
        with state.lock:
            model_inputs = _prepare_model_inputs(state.model, x_fused=x, cnn_dim=self.cnn_dim)
            raw_probs = state.model.predict(model_inputs, verbose=0)[0]
        smoothed_probs, majority_class, vote_confidence = self.smoother.update(raw_probs)

        topk = 3
        top_idx = np.argsort(smoothed_probs)[::-1][:topk]
        top_scores = [
            {"class": state.classes[i], "confidence": float(smoothed_probs[i])}
            for i in top_idx
        ]

        predicted_class = state.classes[majority_class]
        predicted_conf = float(smoothed_probs[majority_class])

        ksi_payload = None
        coaching_payload = None

        expert_template = _select_expert_template(state.templates, predicted_class)
        if expert_template is not None and len(self.landmarks_buffer) == self.seq_len:
            _ = _detect_contact_realtime(self.landmarks_buffer)
            user_landmarks = np.array(self.landmarks_buffer)
            ksi_weights = state.params.get("ksi", {}).get("weights") or {
                "pose": 0.4,
                "velocity": 0.4,
                "acceleration": 0.2,
            }
            with state.lock:
                ksi_calc = EnhancedKSI()
                ksi_result = ksi_calc.calculate(
                    expert_landmarks=expert_template,
                    user_landmarks=user_landmarks,
                    weights=ksi_weights,
                )
            ksi_payload = _ksi_to_dict(ksi_result)
            coaching_json = generate_coaching_report(
                ksi_result=ksi_result,
                shot_type_str=predicted_class,
                skill_level_str="intermediate",
                output_format="json",
                simplified=True,
            )
            coaching_payload = json.loads(coaching_json)

        return {
            "frame": self.frame_count,
            "prediction": {
                "class": predicted_class,
                "confidence": predicted_conf,
                "vote_confidence": float(vote_confidence),
                "topk": top_scores,
            },
            "ksi": ksi_payload,
            "coaching": coaching_payload,
        }


@app.websocket("/ws/realtime")
async def realtime_ws(websocket: WebSocket):
    await websocket.accept()
    session = RealtimeSession()

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"] is not None:
                data = message["bytes"]
                npbuf = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                result = await run_in_threadpool(session.process_frame, frame)
                if result is not None:
                    await websocket.send_text(json.dumps(result))
            elif "text" in message and message["text"] is not None:
                if message["text"].strip().lower() == "reset":
                    session = RealtimeSession()
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
