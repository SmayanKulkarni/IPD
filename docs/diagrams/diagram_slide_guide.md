# Diagram-to-Slide Mapping Guide

- **Slide 1 – Context**: High-level overview using C4 Context (docs/diagrams/c4_context.md). Audience learns actors and external systems.
- **Slide 2 – Containers**: System subsystems and storage using C4 Container (docs/diagrams/c4_container.md).
- **Slide 3 – Components**: Key modules and their relationships (docs/diagrams/c4_component.md).
- **Slide 4 – Pipelines DFD**: End-to-end data flow from raw videos to models and evaluation (docs/diagrams/dfd_pipelines.md).
- **Slide 5 – Sequence (Evaluate)**: Step-by-step single-video evaluation (docs/diagrams/sequence_evaluate_video.md).
- **Slide 6 – Sequence (Demo Coaching)**: Data-based coaching generation (docs/diagrams/sequence_demo_coaching.md).
- **Slide 7 – State Machine**: Shot phases and transitions (docs/diagrams/state_phase_segmentation.md).
- **Slide 8 – Storage Map**: Directories, artifacts, and schemas (docs/diagrams/storage_map.md).
- **Slide 9 – Risks & Contracts**: Summarize key assumptions (e.g., raw_landmarks availability) and mitigations.

Tips:
- Keep text minimal; let diagrams lead.
- Use consistent shot-type examples across slides.
- Link to code references when presenting (e.g., [src/evaluate_video.py](../../src/evaluate_video.py)).
