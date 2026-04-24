---
marp: false
---

**Slide 1: KSI Computation Flow (How the score is built)**

**Title:** KSI Formula Pipeline

**Content (copy-paste):**
- Step 1: Normalize pose to remove camera/body-scale bias  
  $$c=\frac{P_{LH}+P_{RH}}{2},\quad \tilde{P}=P-c,\quad \bar{P}=\frac{P'}{l}$$

- Step 2: Build fused frame-level representation  
  $$x_t=\left[\mathrm{vec}(\bar{P}_t)\parallel c_t\right]\in\mathbb{R}^{163}$$

- Step 3: Contact-aware shot decision (most informative instant)  
  $$A=\|\Delta v\|_2,\quad \text{Class}=\arg\max_c\;p(c\mid X_{\text{peak\_acc}})$$

- Step 4: Align expert and user sequences with DTW  
  $$D_{i,j}=\|F_i^E-F_j^U\|_2+\min\{D_{i-1,j},D_{i,j-1},D_{i-1,j-1}\}$$

- Step 5: Give higher importance to key stroke phases  
  $$\alpha_k=w_{\text{phase}}\!\left(0.5+0.5\exp\!\left[-0.5\left(\frac{k-\text{center}}{\text{width}}\right)^2\right]\right)$$

**One-line takeaway:** KSI first normalizes and aligns motion, then evaluates similarity with phase-aware weighting.

---

**Slide 2: Final KSI Score and Practical Use**

**Title:** Final KSI Score and Why It Matters

**Content (copy-paste):**
- Pose similarity (shape/biomechanics match)  
  $$S_{\text{pose}}=\frac{\sum_k \alpha_k\cdot \frac{F_k^E\cdot F_k^U}{\|F_k^E\|\|F_k^U\|}}{\sum_k \alpha_k}$$

- Velocity similarity (timing + speed profile match)  
  $$S_{\text{vel}}=\frac{\sum_k \alpha_k\cdot \cos_{\text{vel},k}\cdot \exp\!\left(-0.1\|V_k^E-V_k^U\|^2\right)}{\sum_k \alpha_k}$$

- Final weighted score  
  $$\mathrm{KSI}_{\text{total}}=w_pS_{\text{pose}}+w_vS_{\text{vel}}+w_aS_{\text{acc}}$$

- Interpretation:
  - Higher KSI -> user stroke is closer to expert mechanics
  - Lower KSI -> larger deviation in posture, timing, or acceleration

- Practical use in coaching:
  - Rank performance quality objectively
  - Identify weak phases/joints for correction
  - Generate targeted drill recommendations
  - Track improvement across sessions

**One-line takeaway:** KSI converts motion quality into a single interpretable score for feedback and progress tracking.

**Slide 3: KSI -> Natural Language Feedback + Exact Fault Frame Detection**

**Title:** From KSI Scores to Actionable Coaching Feedback

**Content (copy-paste):**
- **Step 1: Compute per-feature error over time (after DTW alignment)**  
  $$e_{k,t}=|F^E_{k,t}-F^U_{k,t}|$$
  where $k$ is biomechanical feature/joint metric and $t$ is aligned frame index.

- **Step 2: Find exact fault frame using phase-weighted error**  
  $$t_k^*=\arg\max_t\big(\alpha_t\cdot e_{k,t}\big)$$
  - $t_k^*$ gives the most critical frame for each fault (e.g., elbow drop, spine lean).
  - DTW path mapping links aligned index back to original user frame/timestamp.

- **Step 3: Rank faults by severity and coaching priority**  
  - High mean/max error + critical phase (acceleration/contact) -> top priority issue.
  - Timing faults use peak-velocity offset (early/late swing) for temporal feedback.

- **Step 4: Convert faults to natural-language coaching**  
  - Structured mapping: **Issue -> probable cause -> correction cue -> drill**.
  - Example style:  
    - “Your peak speed occurs late at contact.”  
    - “Fix: start forward swing earlier.”  
    - “Drill: shadow swings with contact-timing count.”

- **Step 5: Personalized output to user**  
  - Skill-adaptive wording (beginner/intermediate/advanced/expert)
  - Top 3 critical faults + exact frame references + corrective drills

**One-line takeaway:** KSI not only scores motion quality, it pinpoints *where* the fault happens and turns it into clear, personalized coaching language.



Absolutely. Here is the **logic-first**, detailed explanation of expert template generation in your pipeline, without code details.

## 1) Why Expert Templates Exist
KSI needs a “gold standard” motion to compare against.  
A single expert video is usually too noisy or style-specific, so the system builds a **robust class template** (for each shot type) from multiple expert examples.

Goal:
- represent ideal biomechanics for a shot class
- reduce recording noise and individual style bias
- keep timing and preprocessing consistent with your training/inference pipeline

---

## 2) Inputs and Desired Outputs

### Inputs
- Multiple expert videos per shot class (forehand clear, backhand net shot, etc.)
- Same preprocessing rules used elsewhere (segmentation, crop logic, fps handling)

### Outputs
For each class, the generator stores:
- **Main template** (reference motion)
- **Variants** (style alternatives)
- **Contact-phase template** (critical impact zone)
- Metadata (quality and fps stats)

This goes into expert_templates.npz, produced via generate_templates.py.

---

## 3) Core Logic Pipeline (Conceptual)

## Stage A: Make all expert samples comparable
Each expert video may differ in:
- camera framing
- speed (fps + execution tempo)
- motion duration

So first, the pipeline normalizes input conditions:
1. Same segmenting policy (focus on meaningful shot interval)
2. Same crop policy (including overrides)
3. Pose extraction in a consistent format
4. Temporal normalization (resample to common fps)

Why this matters:
- if one sample is 60 fps and another 24 fps, direct frame-to-frame comparison is invalid
- if one clip includes extra idle frames, it corrupts alignment

---

## Stage B: Represent motion in two levels
The system keeps **two representations**:

1. **Raw normalized landmarks** (the actual biomechanical sequence)
- this is what becomes the final stored template

2. **Enhanced feature sequence** (compact motion descriptors)
- used for alignment and quality evaluation

Logic:
- landmarks preserve full geometric information for KSI downstream
- enhanced features make similarity and DTW more stable/semantic

---

## Stage C: Filter poor-quality expert samples
Not every “expert” clip is equally useful (blur, jitter, poor pose detection, low motion).

The pipeline computes a quality score based on:
- smoothness of motion
- feature stability
- presence of real movement (not near-static)

Then it removes weak samples and keeps higher-confidence ones.

Why this matters:
- poor inputs cause template drift
- weighted averaging only works when outliers are controlled

---

## Stage D: Choose the best reference sequence
To align multiple sequences, one must act as the anchor.

The system computes pairwise sequence distances (DTW-based) among retained expert videos, then picks the sequence with the **smallest total distance to all others**.

Interpretation:
- this is the “most central” example
- avoids anchoring on an unusual expert style

---

## Stage E: Align all expert motions in time (DTW)
Experts execute the same shot at different speeds.  
DTW solves this by matching semantically similar moments even if frame indices differ.

Logic:
- align each sample to the chosen reference timeline
- preserve biomechanical event correspondence (loading, acceleration, contact, follow-through)
- align both feature trajectories and landmarks consistently

Why this matters:
- averaging unaligned sequences causes temporal smearing
- DTW gives phase-coherent averaging

---

## Stage F: Build the final template by weighted averaging
After alignment, all sequences lie on a common timeline.  
Now compute weighted average landmarks, where better-quality samples get higher influence.

Result:
- stable, less noisy “canonical” class motion
- quality-aware fusion rather than simple mean

---

## Stage G: Store useful variants and contact-specific template
Besides main template:
- variant1/2/3 capture style diversity across experts
- contact template captures the most critical phase around impact

Why contact template matters:
- many coaching-critical errors happen near impact timing and body configuration
- allows sharper KSI feedback for decisive frames

---

## 4) Why this is better than naive averaging
Naive averaging fails because:
- sequences are out of phase
- fps and duration differ
- outliers dominate

Your pipeline fixes this by:
- consistent preprocessing
- fps normalization
- quality filtering
- central-reference selection
- DTW temporal alignment
- quality-weighted aggregation
- phase-focused template extraction

---

## 5) How this connects to KSI scoring logic
During KSI evaluation:
- user motion is compared against these expert templates
- since templates are already denoised/aligned/representative, similarity scores are more trustworthy
- per-phase and per-feature fault localization becomes more meaningful

In short: **template quality directly controls feedback quality**.  
If template generation is strong, KSI can confidently say what is wrong, where, and how severe.

---

If you want, I can next give you a **presentation-ready “architecture narration” script** (90 seconds) explaining this exact logic in spoken form for viva/ppt delivery.


Perfect, purely logic:

When KSI compares a user stroke to expert templates, it follows this reasoning pipeline:

1. Standardize both motions first  
- User and expert motions are put into the same normalized coordinate space.  
- This removes body position/camera differences so comparison is fair.

2. Focus on meaningful part of the stroke  
- The system identifies stroke phases and centers analysis around the critical action window (especially around contact/impact).  
- This avoids scoring irrelevant frames too heavily.

3. Convert movement into biomechanical descriptors  
- Instead of comparing raw points directly, both sequences are represented as interpretable motion features (joint geometry, body alignment, rotation, etc.).  
- This makes the comparison biomechanically meaningful.

4. Align timing differences using DTW  
- Expert and user rarely move at the same speed.  
- DTW aligns “equivalent movement moments” even when one sequence is faster/slower.  
- So the system compares same action intent, not same frame index.

5. Apply phase importance weighting  
- Critical phases (acceleration/contact) are weighted more than low-impact phases.  
- This improves coaching relevance and reduces noise.

6. Compute similarity across multiple motion aspects  
- Structural posture similarity  
- Velocity profile similarity  
- Acceleration/power similarity  
- (and smoothness diagnostics)  
These are combined into a final KSI score.

7. Localize faults, not just score overall quality  
- For each biomechanical feature, the system tracks where deviation is highest after alignment.  
- This identifies the exact critical moments and phases where user form diverges from expert form.

8. Translate comparison into coaching meaning  
- Final output is not only “good/bad score.”  
- It includes what is wrong, when it happens, and which correction should be prioritized.

In one line: KSI compares user vs expert as aligned, phase-aware biomechanical sequences, not raw frame-to-frame coordinates.