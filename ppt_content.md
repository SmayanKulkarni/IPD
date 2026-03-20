# Smashifix PPT — Slide-by-Slide Suggested Changes

Below are specific, slide-by-slide changes based on the actual content of `FinalPPTnew.pdf` and the latest research paper / codebase.

---

## Slide 1: Title Slide
**Current:** SMASHFIX: An Application for Badminton Posture Correction Using Deep Learning
**Change:** Fix the title spelling to **"SMASHIFIX"** (with an 'I') to match the paper.

---

## Slide 2: Introduction
**Current content is fine.** No major changes required — the introduction aligns with the paper.

---

## Slides 3–5: Literature Survey
**Current content is fine.** The 9 papers summarized are consistent with the paper's literature review.

---

## Slide 6: Research Gaps ⚠️ UPDATE NEEDED
**Current Gaps listed:**
1. Focus is mostly on pose estimation or movement classification with limited real-time corrective feedback.
2. Existing systems (e.g., yoga apps) are unsuitable for complex, fast-paced sports like badminton.
3. Lack of complete teaching aids that systematically improve novice players' skills.
4. Limited mobile accessibility and flexibility.
5. Models are limited to a small number of shot types.
6. Performance/latency issues due to processing bottlenecks.

**Suggested Update — Add Gap #7:**
> 7. **Spatial Tracking Instability:** Existing dynamic bounding box methods based on joint extrema frequently suffer from temporal jitter, occlusion-induced failures, and inconsistent player scaling, leading to noisy downstream feature extraction that compromises forensic analysis.

This aligns with the new research gap added to the paper (Section 1.2).

---

## Slide 7: Problem Statement
**No changes.** The user explicitly said no changes to the problem statement.

---

## Slide 8: System Architecture & Workflow
**No major changes needed.** The flowchart (Input Video → Preprocessing → Feature Extraction → Temporal Analysis → Prediction) is consistent with the paper.

---

## Slide 9: Model Architecture ⚠️ UPDATE NEEDED
**Current content mentions:**
- "Visual Stream: **MobileNetV2** extracts 64D features from a cropped ROI."

**Change:** Replace **MobileNetV2** with **Residual-Shuffle Network (RSN)**. The paper and codebase use a custom RSN backbone, not MobileNetV2.

**Updated bullet:**
> Visual Stream: Self-trained **Residual-Shuffle Network (RSN)** extracts 64D visual features from the configuration-cropped ROI via channel shuffle operations and residual connections.

Also update the architecture description to mention:
- **Attention-TCN** (not just generic TCN) with 4 dilated causal Conv1D layers (dilations 1, 2, 4, 8)
- **Multi-Head Self-Attention** (8 heads, key dim 32) with residual connections
- Optimized hyperparameters: F=80 filters, G=80 GRU units, dropout=0.22, label smoothing=0.1

---

## Slide 10: Testing and Validation
**No major changes.** DTW alignment and quality-weighted sequencing are consistent with the KSI algorithm in the paper.

---

## Slide 11: Results and Output ⚠️ UPDATE NEEDED
**Current content mentions:**
- "Hybrid Feature Extraction (Self-trained RSN 64-D + Pose 99-D)"

**This is correct and consistent with the paper.** However, ensure the following is reflected:
- **Configuration-Driven Static Cropping** feeds the RSN, not dynamic joint-based bounding boxes.
- The feature fusion produces a **163-D hybrid vector** (99D pose + 64D visual).

**Suggested addition to the slide:**
> Preprocessing uses hierarchical, configuration-driven static spatial cropping (fractional bounding boxes per shot type) instead of dynamic joint-based tracking — eliminating temporal jitter.

---

## Slide 12: Relevant Dataset
**Current content:**
- Source: BWF Indonesia Open videos

**Change:** The paper states the dataset is sourced from the **"Badminton Stroke Video Dataset" from Kaggle** (Chang, 2024), augmented with proprietary HD footage. Update the source:
> **Source:** Badminton Stroke Video Dataset (Kaggle) + proprietary high-definition footage. Total size: 1.32 GB.

The 6 classes listed (Forehand drive, lift, net shot, clear; Backhand net shot, drive) are correct.

---

## Slides 13–14: Demo & Comparison
**Placeholders** — No text changes needed. Fill with actual demo content.

---

## Slides 15–16: Keypoints Extraction and Detection
**No changes needed.** The MediaPipe Pose skeleton (33 keypoints) and RSN Grad-CAM activation maps shown are consistent with the paper's Fig. 2 (Hybrid Feature Extraction Pipeline).

---

## Slide 17: Result Analysis ⚠️ UPDATE NEEDED
**Currently a placeholder.** Add the following quantitative results from the paper:

| Model Architecture | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Pose-LSTM (Real-time) | 0.81 | 0.79 | 0.78 | 0.78 |
| **Attention-TCN (Proposed)** | **0.86** | **0.85** | **0.84** | **0.85** |
| RSN (Visual only) | 0.74 | 0.72 | 0.71 | 0.71 |

**Additional results to add:**
- Attention-TCN peak validation accuracy: **94.70%** at epoch 165 (193 epochs total)
- Pose-LSTM peak validation accuracy: **90.44%** at epoch 59 (71 epochs total)
- Optuna hyperparameter optimization: **98.92%** validation accuracy at trial 13 (46 trials)

---

## Slide 18: Conclusion & Future Scope ⚠️ UPDATE NEEDED

**Current Conclusion:**
> Smashifix bridges the gap between amateur practice and professional analysis, achieving up to 98.92% validation accuracy.

**Updated Conclusion:**
> Smashifix bridges the gap between amateur practice and professional-grade biomechanical analysis through a novel dual-pipeline architecture:
> - **Pose-LSTM:** Real-time on-court feedback with ~50ms latency
> - **Attention-TCN + RSN:** Forensic analysis with 86% classification accuracy across 6 stroke types
> - **Configuration-driven static cropping** eliminates temporal jitter from dynamic bounding boxes
> - **KSI scoring** (Sakoe-Chiba DTW, 32 hierarchical features, phase-aware attention) translated into actionable drills by a Natural Language Coach
> - Bayesian optimization via Optuna yielded **98.92% peak validation accuracy**

**Current Future Scope:**
> Focus on edge-device deployment and multi-player tracking.

**Updated Future Scope:**
> - **Unified Dynamic Tracking:** Self-adjusting bounding box logic that maintains temporal stability without manual per-video configuration
> - **Multi-person Tracking:** Extend to doubles matches and crowded environments
> - **Edge Deployment:** Quantized models for on-device smartphone execution
> - **Dense Surface Modeling:** Deeper muscle activation and torso rotation analysis
> - **Deep Metric Learning:** Capture subtler elite motion nuances beyond hand-engineered features

---

## Slide 19: References ⚠️ UPDATE NEEDED
**Current:** Lists 9 references.

**Ensure the following key references from the paper are included:**
1. Ma et al. (2018) — ShuffleNet V2 (basis for RSN backbone)
2. Vaswani et al. (2017) — Attention is All You Need (Multi-Head Self-Attention)
3. Akiba et al. (2019) — Optuna (hyperparameter optimization)
4. Selvaraju et al. (2017) — Grad-CAM (visual explanations)
5. Sakoe & Chiba (1978) — DTW algorithm
6. Chang (2024) — Badminton Stroke Video Dataset (Kaggle)

---

## Summary of Critical Changes

| Slide | Change | Priority |
|---|---|---|
| 1 | Fix spelling: SMASHFIX → SMASHIFIX | Low |
| 6 | Add Research Gap #7 (spatial tracking instability) | High |
| 9 | Replace MobileNetV2 → RSN; add Attention-TCN details | **Critical** |
| 11 | Add note about static config-driven cropping | Medium |
| 12 | Fix dataset source: Kaggle, not BWF Indonesia Open | High |
| 17 | Fill in quantitative results table + training stats | **Critical** |
| 18 | Expand conclusion & future scope with full details | High |
| 19 | Add missing key references | Medium |
