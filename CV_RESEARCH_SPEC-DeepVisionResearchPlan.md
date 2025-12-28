# CV Research Specification — Deep Vision Research Plan (Badminton Skill Correction v2.1)

## Objective
Elevate the badminton shot correction pipeline with SOTA action-quality assessment (AQA) ideas—temporal transformers and pose GCNs with ranking losses—while keeping fast, pose-only defaults. Target COCO/MediaPipe landmarks plus existing expert templates; increase sensitivity to timing and kinetic-chain errors on a single GPU.

## Core Losses
- Score regression (MSE): Match predicted score to ground truth.
- Pairwise ranking hinge: Enforce ordering between better/worse clips.
  - Formula: L_rank = max(0, m - (s_hat_i - s_hat_j))
  - Role: Stabilizes ordering; adds sensitivity to meaningful errors.
- Phase classification (CE): Teach the model to recognize 5 swing phases.
- Consistency loss (L2): Keep pose and optional RGB features aligned.
- Kinetic-chain delay penalty: Penalize out-of-order hip→shoulder→elbow→wrist peaks.

### Expanded math snippets (ready for implementation)
- Total loss: \(\mathcal{L} = \lambda_{reg}\,\mathcal{L}_{MSE} + \lambda_{rank}\,\mathcal{L}_{hinge} + \lambda_{phase}\,\mathcal{L}_{CE} + \lambda_{cons}\,\lVert f_{pose}-f_{rgb}\rVert_2^2 + \lambda_{delay}\,\mathcal{L}_{delay}\).
- Hinge ranking: \(\mathcal{L}_{hinge} = \max(0,\ m - (\hat{s}_i - \hat{s}_j))\); margin \(m\) (e.g., 0.05–0.1) encourages a clear gap between better and weaker clips.
- Phase CE: \(\mathcal{L}_{CE} = -\sum_{t} y^{phase}_t \log p^{phase}_t\); teaches 5-phase recognition and stabilizes attention.
- Consistency: \(\mathcal{L}_{cons} = \lVert f_{pose} - f_{rgb} \rVert_2^2\); keeps pose and RGB embeddings aligned when RGB is enabled.
- Kinetic-chain delay: \(\mathcal{L}_{delay} = \sum_{(p\to d)} |\Delta t_{p\to d} - \Delta t^{*}_{p\to d}|\); enforces proximal-to-distal timing (hip→shoulder→elbow→wrist).
- Temporal attention (per frame): \(Attn(Q,K,V) = softmax\big(\tfrac{QK^T}{\sqrt{d}}\big) V\); phase tokens bias attention toward critical regions (acceleration/contact).
- Bootstrap CI (score): sample K times, \(CI_{95} = [\text{perc}_{2.5}(\hat{s}), \text{perc}_{97.5}(\hat{s})]\); use width to calibrate feedback strength.

## Architecture
- Backbone: Pose ST-GCN/2s-AGCN (Kinetics init); optional lightweight Video-Swin-T gated by quality.
- Neck: Temporal transformer (TALLFormer-style) over pose tokens + 5 learnable phase tokens; cross-attention for pose/RGB fusion when enabled.
- Heads:
  - Score head (MSE + ranking)
  - Phase head (CE)
  - Timing head (peak offsets for joints)
  - Joint-severity head (per-joint error severity)

## Data & Augmentation
- Sources: Internal badminton + AQA-7/MTL-AQA for warm-start.
- Splits: 80/10/10, stratified by shot type.
- Windows: Contact-centered, 32–64 frames, with contact ±16–24 frames.
- Augs: Temporal jitter (±2–3 frames), speed (0.9–1.1×), handedness flip with joint remap, light Gaussian keypoint noise, Mixup/RandAug on trajectories.
- Inputs: Pose 33×3; optional RGB 224×224 @ 16–32 frames.

## Experiments & Ablations
- Metrics: Spearman’s ρ, MAE, phase F1, joint localization accuracy, latency/FPS.
- Ablations: pose-only vs pose+RGB; ranking on/off; phase tokens on/off; contact-centered vs fixed windows; delay loss on/off; bootstrap CI sizes (100/500/1000).
- Goal: +0.03–0.05 Spearman’s ρ vs current KSI + NL coach; tighter confidence intervals.

## Compute & Deployment
- Train: Single 24–32GB GPU; FP16 + grad accumulation (2–4); batch ≈16 pose-only or ≈8 pose+RGB; 30–50 epochs post warm-start.
- Inference: Pose-only default; trigger RGB only if quality <0.6 or user requests deep review. Export ONNX/TensorRT for transformer-lite head if latency-critical. Favor QAT-friendly ops for INT8.

## Actionable Pipeline Upgrades
- Ranking supervision with expert/user pairs.
- Contact-centered windows to focus on high-salience frames.
- Phase-token transformer for phase-aware weighting.
- Kinetic-chain timing head + delay loss for sequencing.
- Joint-severity head to replace heuristics.
- Pose-quality gating for attention and RGB triggers.
- Bootstrap-aware calibration to adjust feedback strength.
- Latency-aware dual path with early exit.
- Curriculum finetuning: AQA warm-start → badminton with ranking+phase → optional RGB fusion.
- Few-shot personalization with decayed stats and bounded bias.
- Safety/guardrails: OOD pose checks, re-record prompts.

## Open Inputs Needed
- GPU spec (e.g., single 24GB?) to lock batch/window sizes.
- Whether to stay pose-only or allow conditional RGB.
- Target latency (CPU-only vs GPU real-time).
