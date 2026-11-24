# Plan

## Method
Fine-tune Depth Anything V2 Small (indoor metric checkpoint) on NYU Depth V2.

- **Baselines**: B0 pretrained zero-shot; B1 fine-tuned with SiLog (same settings as ours).
- **Novelty 1, instance-aware loss**: SiLog + λ·(mask-edge gradient term + per-object scale-invariant term).
  The edge term only uses pixels where raw depth is valid on both sides of the edge.
- **Novelty 2, decoder change** (after the loss results): mask-edge map → zero-init conv → added to the decoder.
  The loss-only model needs no masks at test time; the decoder version does.
- **Masks**: YOLO11s-seg (usable on any image) vs NYU ground-truth instances (upper bound).
  GT instance ids restart per class, so an object is a (label, instance) pair.

## Data decisions
- NYU 795 train / 654 test. A scene-disjoint validation split (~80–100 images) is held out from train;
  all tuning uses validation, the test set is evaluated once at the end.
- Training target (raw vs filled depth) is decided by experiment H1.
- Main test numbers on filled depth (comparable with the literature), also reported on raw depth.
  Boundary metrics come from iBims-1 and raw NYU pixels, not filled depth.
- Cross-dataset results are reported both metric and per-image scale-aligned.
- Resolution is never reduced to suit the laptop: training uses DA-V2's native 518×518 crop, and eval uses the
  full border-free 625×468 image. Memory is handled with fp16 + memory-efficient attention, or by moving the
  run to the T4. The only crop is the 6–8 px white border of the NYU labeled images, which is invalid data.
- Runtime numbers for the paper are measured on the T4 (the laptop GPU throttles when hot).

## Work sections
| | Section | Contents | Depends on |
|---|---|---|---|
| A | Data | Extract filled/raw depth, instances, scene ids from the .mat; validation split; NYU dataset class; data zip for Colab | — |
| B | Training | `train_nyu.py`: single GPU, fp16 + grad scaler, memory-efficient attention, resume, one folder per run | A |
| C | Masks | YOLO11s-seg masks for all 1449 images; GT instance masks; mask-edge maps | C1 none, rest A |
| D | Loss | Edge + per-object scale-invariant terms | code with B, train after A + C |
| E | Decoder | Mask-edge input branch | B + loss results |
| F | Metrics | Boundary accuracy, flying-pixel %, scale-aligned metrics | — |
| G | 2nd dataset | iBims-1 loader + zero-shot eval | — |
| H | Experiments | H1 raw vs filled · H2 loss ablations + λ + GT masks · H3 LR × batch · H4 decoder · H5 3 seeds of B1 and full · H6 final test | B, D, E |
| I | 3D | Point-cloud renders, before/after, failure cases | — |
| J | Tracking demo (optional) | YOLO tracking + depth on a room video, people in 3D | — |
| K | Paper | CVPR LaTeX; Results written after H6 | — |
| L | Slides | 15-minute talk | K |
| M | Repo | README, setup, Colab notebook, clean-clone check | ongoing |

## Order
```
GATE: novelty sign-off · repo · Overleaf · iBims check · Colab notebook + T4 benchmark
  A → B → H1 ─┬─→ H3 LR×batch sweep ............... [Colab T4]
  D coded ──┘ └─→ H2 loss ablations + λ ........... [laptop]
  C masks ──────────┘        └─→ E → H4 decoder ... [laptop]
                                      └─→ H5 seeds [Colab] → H6 final test
Any time, no training needed: C1 · F · G · I · J · K drafts · M
```
Critical path: A → B → H1 → D → H2 → E → H4 → H5 → H6 → Results.

## Compute (laptop RTX A3000 6 GB + one Colab T4 16 GB)
- At most two runs at once, one per GPU.
- fp16 + grad scaler on both (T4 has no bf16); loss computed in fp32.
- One experiment group per machine (never split a comparison across GPUs).
- Runs are resumable (checkpoint each epoch to Drive); one `runs/<name>/` folder per run; one script aggregates.
- Keep only best + last checkpoint per run.

## Decision points
1. After H1: raw or filled training target for all later runs.
   **Decided 2026-11-23: filled** (val AbsRel, mean of last 5 epochs: 0.0910 filled vs 0.0944 raw).
2. After the T4 benchmark: swap machine roles if the T4 is much slower.
   **Outcome: the T4 is about 2x faster** than the throttled laptop (46 s vs 92 s per epoch at bs 4).
3. After H2: keep the best λ and go to E; if flat, judge on boundary metrics, consider dropping the scale term.
   **Outcome 2026-11-23:** AbsRel is flat (all within ±0.0005 of the λ=0 control), so H2 is judged on boundary
   metrics. The edge term raises object-boundary F1 0.149 → 0.164 and cuts flying pixels 0.593 → 0.547; the object
   term adds nothing on boundaries. YOLO masks match GT masks. Next: λ_edge sweep (H2b: 0.25, 1, 2).
   Loading masks changes the random crops, so H2 compares against `h2_silog_yolo_l0` (masks loaded, λ = 0).
   **Decided 2026-11-24 after H2b: λ_edge = 1, λ_obj = 0, YOLO masks** is the method for H4–H6. It sits in the
   middle of the boundary plateau (λ 0.5–2) at +0.001 AbsRel; the object term is dropped (no boundary effect).
4. If E fails: loss-only is the main method, E reported as a negative ablation.
5. After H6: experiments frozen, only writing.
