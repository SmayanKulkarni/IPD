---
title: IPD KSI Inference
emoji: "🏸"
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
---

# IPD KSI Inference (FastAPI + WebSocket)

This Space exposes:
- POST /infer/video: video upload → shot class, KSI numeric, KSI natural language coaching
- WS /ws/realtime: realtime webcam stream → per-frame shot class + KSI + coaching

## Endpoints
- GET /: simple webcam demo page
- GET /health: health check
- POST /infer/video (multipart/form-data)
  - file: video file
  - skill_level: beginner | intermediate | advanced | expert

## Notes
- Uses tuned model at models/tcn_hybrid_tuned.h5
- Uses templates at data/expert_templates.npz
- CPU-only deterministic inference

## Files
- app.py: FastAPI server
- Dockerfile: Space runtime
- requirements.space.txt: minimal runtime dependencies
