# Course requirements checklist (ECE 4990 final project)

## Deliverables (one zip on Canvas, 11:59 AM Dec 8 2026)
- [ ] **Paper**: CVPR format, LaTeX source + PDF, at least 9 pages without references, own figures.
  Sections: Abstract, Introduction, Related Work, Proposed Approach, Experiments, Conclusion (with each member's contribution), References.
- [ ] **Slides**: .pptx for a 15-minute talk by all members; slide 1 has the GitHub link and full names.
- [ ] **GitHub repo**: code, clear description, dataset links, everything needed to run it. Broken links are penalised.

## What the project must show
| Requirement | How we cover it |
|---|---|
| Own novelty (architecture or loss change, not hyperparameter tuning) | Instance-aware loss (mask-edge term + per-object scale-invariant term); decoder change (mask-edge map as extra input through a zero-init conv) |
| State-of-the-art baseline | Depth Anything V2 Small: pretrained (zero-shot) and fine-tuned with its original SiLog loss |
| At least 2 datasets | NYU Depth V2 (train/test) + iBims-1 (zero-shot; DIODE indoor as backup); DA-2K as a bonus |
| Metrics defined as equations | AbsRel, SqRel, RMSE, RMSE-log, log10, δ1–δ3, depth-boundary errors, flying-pixel % |
| Architecture, loss, inputs/outputs described | DA-V2 encoder (DINOv2 ViT-S) + DPT decoder; SiLog + our terms |
| Hyperparameter analysis | Learning rate × batch size sweep, loss weight λ |
| Technical depth in the Q&A | Every member can explain the model, the loss and the results |
