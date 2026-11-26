# RGB-D Scene: Instance-Aware Depth for Single-Image Indoor 3D Scene Parsing

**Group RGB-D Scene.** ECE 4990 Computer Vision final project, Cal Poly Pomona, Fall 2026.

From one RGB photo of a room we build a **labeled 3D scene**: every object and person becomes its own 3D point
cloud with a position and distance. Depth comes from **[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)**
(Small), fine-tuned on **NYU Depth V2** with an **instance-aware loss**: YOLO11-seg object masks mark where depth
has to break cleanly, which cuts the "flying pixels" that smear objects into the background in 3D. The same masks
then split the 3D scene into objects.

```
RGB image ──► Depth Anything V2 (fine-tuned, edge loss) ──► metric depth  (RGB + D = RGB-D)
     │                    ▲ training only                         │
     └─► YOLO11-seg ──────┴── instance masks + classes ───────────┴──► labeled 3D scene (.ply + per-object .json)
```

- Requirements checklist: [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md)
- Plan, work sections and order: [docs/PLAN.md](docs/PLAN.md)

## Status
| Part | State |
|---|---|
| Environment, checkpoints, setup script, Colab notebook | ✅ |
| NYU extracted from the .mat (filled + raw depth, instances, scenes) | ✅ |
| Scene-disjoint split: 703 train / 92 val / 654 test | ✅ |
| YOLO11s-seg masks for all 1449 NYU images | ✅ |
| iBims-1 downloaded | ✅ |
| Inference + point clouds, NYU eval, DA-2K eval | ✅ |
| Fine-tuning script (`tools/train_nyu.py`), resumable, laptop + Colab | ✅ |
| H1: baseline fine-tune, filled vs raw target (filled wins) | ✅ |
| Instance-aware loss (`tools/losses.py`: edge-step + per-object terms) | ✅ |
| H2: loss ablations + λ=0 control, YOLO vs GT masks | ✅ |
| H2b: λ_edge sweep (0, 0.25, 0.5, 1, 2) on the laptop | ✅ |
| H3: lr × batch sweep (3 × 3) | ✅ |
| Boundary + flying-pixel metrics (`tools/boundary.py`) | ✅ |
| Method frozen: λ_edge = 1, λ_obj = 0, YOLO masks | ✅ |
| iBims-1 loader + eval with official metrics (`tools/eval_ibims.py`) | ✅ |
| H4: decoder change (`tools/edge_decoder.py`, `--decoder-edges`): no effect, negative ablation | ✅ |
| H5: 3 seeds of control and method | ✅ |
| iBims-1 zero-shot results (seed 0, Colab) | ✅ |
| Labeled 3D scene (`tools/scene.py`): objects and people as separate point clouds with distance and size | ✅ |
| Flying pixels per object class (`tools/flying_by_class.py`) | ✅ |
| Novel-view renders (`tools/novel_view.py`): GT, zero-shot, control and method point clouds from rotated cameras, flying points in red | ✅ |
| Person tracking video demo | ⬜ |
| Final test numbers (H6): NYU test + iBims-1, 3 seeds | ✅ |
| Fixed-epoch rerun (H7): last-epoch weights, 3 seeds, NYU test + iBims-1 | ✅ |
| TUM RGB-D person tracking demo | ⬜ |
| Paper, slides | ⬜ |

## Baseline results (ViT-S, no fine-tuning)
| Benchmark | AbsRel | RMSE (m) | δ1 |
|---|---|---|---|
| NYU test (654), filled depth | 0.186 | 0.590 | 0.740 |
| NYU test (654), raw depth | 0.181 | 0.542 | 0.752 |
| NYU val (92), filled depth | 0.198 | 0.733 | 0.719 |
| DA-2K (relative model) | **95.16 %** of 2068 pairs correct (paper: 95.3 %) | | |

NYU uses the zero-shot Hypersim metric model, the Eigen crop and 0.001–10 m. Most of its error is global scale:
per-image scale alignment drops AbsRel from about 0.18 to 0.07.

## Final results (H7: last-epoch weights; NYU test, 654 images, and iBims-1; 3 seeds)
The headline table. Each run is the weights after epoch 20, with no checkpoint selection, so the seeds are
compared at the same point in training. Mean ± std over seeds; p as in H6 below. `results/h7_test.csv`, built by
`tools/h6_table.py --final` after `runs/h7_queue.sh`.

| Benchmark | Metric | Zero-shot | Control (SiLog) | Method (SiLog + edge) | p |
|---|---|---|---|---|---|
| NYU test, filled | AbsRel ↓ | 0.186 | **0.092 ± 0.001** | **0.092 ± 0.000** | 0.63 |
| | RMSE ↓ | 0.590 | **0.335 ± 0.002** | 0.336 ± 0.001 | 0.004 |
| | δ1 ↑ | 0.740 | **0.926 ± 0.001** | **0.926 ± 0.000** | 0.009 |
| | Boundary F1 ↑ | **0.175** | 0.111 ± 0.001 | 0.123 ± 0.001 | <0.001 |
| | Object boundary F1 ↑ | **0.221** | 0.149 ± 0.001 | 0.157 ± 0.002 | <0.001 |
| | Flying ↓ | **0.489** | 0.603 ± 0.003 | 0.554 ± 0.003 | <0.001 |
| NYU test, raw | AbsRel ↓ | 0.181 | **0.086 ± 0.001** | **0.086 ± 0.000** | 0.36 |
| | RMSE ↓ | 0.542 | **0.294 ± 0.002** | **0.294 ± 0.001** | 0.80 |
| | δ1 ↑ | 0.752 | 0.939 ± 0.001 | **0.939 ± 0.001** | 0.21 |
| iBims-1 | AbsRel ↓ | 0.162 | **0.148 ± 0.000** | **0.148 ± 0.001** | 0.51 |
| | Scale-aligned AbsRel ↓ | **0.069** | 0.073 ± 0.000 | 0.073 ± 0.000 | 0.34 |
| | DBE accuracy ↓ | 1.93 | **1.91 ± 0.02** | 2.23 ± 0.04 | <0.001 |
| | DBE completeness ↓ | **9.79** | 22.8 ± 1.5 | 21.9 ± 1.5 | 0.10 |
| | Boundary F1 ↑ | **0.143** | 0.052 ± 0.001 | 0.047 ± 0.000 | 0.07 |
| | Flying ↓ | **0.451** | 0.564 ± 0.002 | 0.539 ± 0.001 | <0.001 |

Without checkpoint selection the seed std drops 3-8x, and the edge loss has clear effects at no depth cost. On NYU
it raises boundary F1 by 0.011 (about 10 seed stds), object-boundary F1 by 0.009, and cuts flying pixels by 0.049.
On iBims-1 (unseen) it still cuts flying pixels (−0.025), but its edges are less precisely placed (DBE accuracy,
boundary F1). The best-epoch selection used in H5/H6 picked different epochs per seed (11 to 19), which hid the
boundary gain in seed noise. Zero-shot keeps the best boundaries everywhere; fine-tuning trades them for metric
accuracy, and the edge loss wins back part of that trade on NYU.

## H6: best-val-epoch checkpoints (NYU test and iBims-1; 3 seeds)
Mean ± std over seeds; zero-shot is one model. p: Wilcoxon signed-rank over images, method vs control, on each
image's seed-averaged score. It ignores seed variance, so read it together with the std. `results/h6_test.csv`,
built by `tools/h6_table.py`.

| Benchmark | Metric | Zero-shot | Control (SiLog) | Method (SiLog + edge) | p |
|---|---|---|---|---|---|
| NYU test, filled | AbsRel ↓ | 0.186 | **0.091 ± 0.001** | **0.091 ± 0.000** | 0.007 |
| | RMSE ↓ | 0.590 | **0.336 ± 0.002** | 0.338 ± 0.002 | <0.001 |
| | δ1 ↑ | 0.740 | **0.928 ± 0.001** | 0.927 ± 0.001 | <0.001 |
| | Boundary F1 ↑ | **0.175** | 0.108 ± 0.007 | 0.112 ± 0.007 | <0.001 |
| | Object boundary F1 ↑ | **0.221** | 0.145 ± 0.008 | 0.145 ± 0.008 | 0.35 |
| | Flying ↓ | **0.489** | 0.604 ± 0.011 | 0.573 ± 0.009 | <0.001 |
| NYU test, raw | AbsRel ↓ | 0.181 | **0.085 ± 0.001** | **0.085 ± 0.000** | 0.06 |
| | RMSE ↓ | 0.542 | **0.295 ± 0.002** | 0.296 ± 0.002 | 0.003 |
| | δ1 ↑ | 0.752 | **0.941 ± 0.001** | 0.940 ± 0.000 | 0.07 |
| iBims-1 | AbsRel ↓ | 0.162 | **0.151 ± 0.000** | 0.152 ± 0.002 | 0.55 |
| | Scale-aligned AbsRel ↓ | **0.069** | 0.073 ± 0.001 | 0.074 ± 0.000 | 0.08 |
| | DBE accuracy ↓ | 1.93 | **1.89 ± 0.02** | 2.23 ± 0.06 | <0.001 |
| | DBE completeness ↓ | **9.79** | 23.8 ± 1.3 | 24.8 ± 2.1 | 0.01 |
| | Boundary F1 ↑ | **0.143** | 0.052 ± 0.003 | 0.044 ± 0.004 | 0.001 |
| | Flying ↓ | **0.451** | 0.568 ± 0.010 | 0.557 ± 0.007 | 0.005 |

Fine-tuning halves NYU AbsRel and helps iBims-1 metric depth a little, but costs boundary quality on both. The edge
loss cuts NYU flying pixels by 0.031, about 3 seed stds, at no measurable depth cost; that is the one effect larger
than the seed spread. Its boundary-F1 gain on NYU (+0.003) is inside the seed std, and on iBims-1 its edges are
worse (DBE accuracy, F1). Raw-GT boundary metrics equal the filled ones (boundaries are always scored on filled
depth) and are left out.

## Fine-tuning results so far (NYU val, 92 images, one seed)
Boundary metrics (`tools/boundary.py`): `bf1` is the scale-invariant depth-boundary F1 of Depth Pro, `bf1_obj` the
same at object boundaries only, `flying` the share of pixels at depth jumps predicted in the gap (points that float
in the point cloud). Ground truth is filled depth.

**H2, instance-aware loss** (Colab T4, bs 4, lr 5e-6, filled target; `results/h2_val.csv`)

| Run | λ_edge | λ_obj | Masks | AbsRel ↓ | AbsRel, last 5 ↓ | bf1_obj ↑ | flying ↓ |
|---|---|---|---|---|---|---|---|
| Zero-shot | – | – | – | 0.1975 | – | 0.233 | 0.472 |
| `h2_silog_yolo_l0` (control) | 0 | 0 | YOLO | 0.0889 | 0.0914 | 0.149 | 0.593 |
| `h2_edge_yolo_l0.5` | 0.5 | 0 | YOLO | 0.0903 | 0.0919 | 0.164 | **0.547** |
| `h2_obj_yolo_l0.5` | 0 | 0.5 | YOLO | **0.0882** | 0.0911 | 0.148 | 0.587 |
| `h2_both_yolo_l0.5` | 0.5 | 0.5 | YOLO | 0.0895 | 0.0911 | **0.168** | 0.554 |
| `h2_both_gt_l0.5` | 0.5 | 0.5 | GT | 0.0893 | 0.0917 | **0.168** | 0.568 |

Fine-tuning halves AbsRel but blurs object boundaries (bf1_obj 0.233 → 0.149). The edge term wins back part of that
at no AbsRel cost; the object term does not act on boundaries; YOLO masks match GT masks.

**H2b, edge-loss weight** (laptop, λ_obj 0, YOLO masks, own λ=0 control; `results/h2b_val.csv`)

| λ_edge | AbsRel ↓ | AbsRel, last 5 ↓ | bf1 ↑ | bf1_obj ↑ | flying ↓ |
|---|---|---|---|---|---|
| 0 | 0.0885 | 0.0900 | 0.119 | 0.150 | 0.588 |
| 0.25 | **0.0881** | **0.0900** | 0.128 | 0.159 | 0.574 |
| 0.5 | 0.0887 | 0.0903 | 0.129 | 0.159 | 0.553 |
| 1.0 | 0.0891 | 0.0910 | 0.130 | 0.155 | 0.556 |
| 2.0 | 0.0897 | 0.0912 | **0.133** | **0.160** | **0.546** |

The λ=0.5 gain repeats on a second machine (Colab: bf1_obj +0.015, flying −0.047; laptop: +0.008, −0.034).
Raising λ trades a little AbsRel (+0.001 at λ=2) for sharper boundaries and fewer flying pixels.

**H3, lr × batch on SiLog** (laptop RTX A3000, filled target; `results/h3_val.csv`). Val AbsRel, mean of last 5 epochs:

| lr \ batch | 2 | 4 | 8 |
|---|---|---|---|
| 2e-6 | 0.0914 | 0.0910 | 0.0931 |
| 5e-6 | 0.0941 | **0.0902** | **0.0902** |
| 1e-5 | 0.0974 | 0.0948 | 0.0920 |

lr 5e-6 (the official recipe) is best; at high lr a larger batch matters most. Across the nine settings, bf1_obj
ranges 0.149–0.160 and flying 0.572–0.606.

**H5, seeds** (laptop, λ_edge 1 vs the λ 0 control of the same seed; seed 0 is H2b; `results/h5_val.csv`)

| Seed | AbsRel, control → method | bf1 | bf1_obj | flying |
|---|---|---|---|---|
| 0 | 0.0885 → 0.0891 | 0.119 → 0.130 | 0.150 → 0.155 | 0.588 → 0.556 |
| 1 | 0.0880 → 0.0880 | 0.127 → 0.118 | 0.160 → 0.143 | 0.579 → 0.570 |
| 2 | 0.0898 → 0.0902 | 0.115 → 0.127 | 0.146 → 0.155 | 0.601 → 0.562 |
| Paired change, mean ± sd | +0.0003 ± 0.0003 | +0.004 ± 0.012 | −0.001 ± 0.014 | **−0.026 ± 0.016** |

The edge loss lowers flying pixels on every seed; its effect on boundary F1 is within seed noise. The seed-1 pair has
best checkpoints from different epochs (11 vs 19), so checkpoint selection adds to that spread.

**H4, decoder change** (Colab T4, seed 0, all four on one machine; `results/h4_val.csv`)

| Run | Decoder | λ_edge | AbsRel ↓ | AbsRel, last 5 ↓ | bf1 ↑ | bf1_obj ↑ | flying ↓ |
|---|---|---|---|---|---|---|---|
| `h2_silog_yolo_l0` | no | 0 | 0.0889 | 0.0914 | 0.119 | 0.149 | 0.593 |
| `h2_edge_yolo_l1.0` | no | 1 | 0.0902 | 0.0924 | 0.130 | 0.161 | **0.546** |
| `h4_dec_yolo_l0` | yes | 0 | 0.0891 | 0.0914 | 0.119 | 0.150 | 0.593 |
| `h4_dec_yolo_l1.0` | yes | 1 | 0.0906 | 0.0921 | 0.131 | 0.161 | 0.557 |

The mask-edge decoder input changes nothing; the decoder-only run ends at the control's training loss (0.0629).

**iBims-1, zero-shot** (100 images, ≤ 10 m, seed 0 models from Colab; `results/ibims_colab.csv`)

| Model | AbsRel ↓ | AbsRel, scale-aligned ↓ | bf1 ↑ | flying ↓ | DBE acc (px) ↓ | DBE comp (px) ↓ |
|---|---|---|---|---|---|---|
| Zero-shot | 0.162 | **0.069** | **0.143** | **0.451** | 1.93 | **9.8** |
| Fine-tuned control | 0.148 | 0.072 | 0.052 | 0.572 | **1.85** | 23.8 |
| Edge loss (λ 1) | 0.148 | 0.072 | 0.042 | 0.549 | 2.26 | 23.9 |
| Decoder only | 0.148 | 0.072 | 0.052 | 0.571 | 1.85 | 25.2 |
| Decoder + edge loss | **0.147** | 0.072 | 0.042 | 0.556 | 2.24 | 23.6 |

NYU fine-tuning improves metric depth on unseen rooms but costs most of the boundary quality; the edge loss recovers
some flying pixels, not boundary F1.

**Flying pixels per object class** (NYU val, YOLO11s-seg classes, 3 seeds pooled per group;
`results/nyu/flying_by_class_val.csv`). Lower is better.

| Class | Objects | Zero-shot | Control | Edge loss (λ 1) |
|---|---|---|---|---|
| Background | – | 0.468 | 0.596 | 0.581 |
| Chair | 146 | 0.492 | 0.627 | 0.596 |
| Dining table | 31 | 0.492 | 0.611 | 0.594 |
| Bed | 26 | 0.415 | 0.621 | **0.520** |
| Person | 40 | 0.452 | 0.655 | 0.682 |
| Couch | 21 | 0.485 | 0.527 | 0.501 |

The edge loss helps furniture most (bed −0.10). People are the class that fine-tuning damages most, and the edge
loss does not help them: NYU has few people to learn from.

**Novel views** (`results/novel_view/<id>_views.jpg`, seed-0 runs). Each scene is re-rendered from virtual cameras
rotated about its centre, with flying points in red. Flying fraction per image:

| NYU val image | Zero-shot | Control | Edge loss (λ 1) |
|---|---|---|---|
| 000500 (bathroom, fixed example) | 0.226 | 0.504 | 0.465 |
| 000921 (bedroom) | 0.309 | 0.453 | **0.169** |
| 000920 (bedroom) | 0.587 | 0.695 | 0.496 |
| 000162 | 0.316 | 0.446 | 0.291 |

000921, 000920 and 000162 are the three val images where the edge loss lowers flying most (`--pick 3`), so they
show the best case, not the average. In 000921 the control smears the bed's near edge into a sheet of points
between bed and floor; the method mostly removes it.

## Setup
**Local**
```bash
conda env create -f environment.yml && conda activate depth3d
bash scripts/setup.sh        # code (pinned), checkpoints, DA-2K, iBims-1, NYU -> images, YOLO masks
```
Datasets go to `data/` unless `DATA_ROOT` is set. Re-running `setup.sh` skips finished steps.

**Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/csgomez25/instance-aware-depth/blob/main/colab/run_on_colab.ipynb)
opens [`colab/run_on_colab.ipynb`](colab/run_on_colab.ipynb); pick a T4 runtime.

Data for Colab comes from a ~1.9 GB bundle, `iad_data.tar`:
- **Teammates:** open the team's shared `iad_data.tar` in Google Drive ("Shared with me"), then right-click →
  **Organize → Add shortcut → My Drive**. The notebook restores from it in about a minute. It is view-only:
  you can read it but not change it, and your checkpoints and results go to your own `MyDrive/instance-aware-depth/`.
- **No shortcut:** the first session downloads and prepares everything (~10 min) and saves your own copy.
- **Bundle owner:** build it with `bash scripts/pack_data.sh`, upload to `MyDrive/instance-aware-depth/`,
  and share the file (not the folder) as **Viewer**.

## Usage (run from `tools/`)
```bash
python infer.py path/to/image.png                  # depth png + .npy + coloured .ply -> results/infer/
python eval_nyu.py --split val --gt filled         # tuning; --split test only for final numbers
python scene.py 000500 --ckpt ../runs/<run>/best.pth   # labeled 3D scene: .ply (photo + per-object colours), objects .json, overview .png -> results/scene/
python scene.py room.jpg                               # any photo; YOLO11s-seg finds the objects and people
python flying_by_class.py --split val --group zero-shot=zero-shot --group method=<run>,<run>   # flying pixels per class
python novel_view.py 000921 --orbit                    # rotated-camera renders (+ turntable mp4) -> results/novel_view/
python novel_view.py --pick 3 --split val              # the images where the edge loss changes flying most
python h6_table.py                                     # H6 table (best-val checkpoints) -> results/h6_test.csv
python h6_table.py --final                             # H7 table (last-epoch checkpoints) -> results/h7_test.csv
python eval_da2k.py                                # DA-2K accuracy per scene -> results/da2k/
python eval_ibims.py --ckpt ../runs/<run>/best.pth --tag <run>   # iBims-1 zero-shot, official metrics -> results/ibims/

python train_nyu.py --name h1_silog_filled --train-gt filled --bs 4 --epochs 20   # fine-tune -> runs/<name>/
python train_nyu.py --name h2_both_yolo --masks yolo11s-seg --lambda-edge 0.5 --lambda-obj 0.5   # instance-aware loss
python train_nyu.py --name h4_dec_yolo_l1.0 --masks yolo11s-seg --decoder-edges --lambda-edge 1   # decoder variant
python eval_nyu.py --split test --ckpt ../runs/h1_silog_filled/best.pth --tag h1_silog_filled
```
Decoder-variant checkpoints are recognised automatically and take instance masks at test time: NYU eval reads
`--masks` (default `yolo11s-seg`), iBims-1 eval runs YOLO11s-seg on each image.
Training follows the official Depth Anything V2 recipe (AdamW, decoder lr ×10, poly decay, 518×518 crops, flips)
with fp16 and memory-efficient attention so it fits one GPU (batch 4 ≈ 2 GB, batch 8 ≈ 3.4 GB).
The best epoch is chosen on the val split. Each run writes `config.json`, `log.csv` and `best.pth`;
re-running the same command resumes after an interruption. Set `RUNS_DIR` to store runs elsewhere
(the Colab notebook uses Google Drive).

## Data notes
- NYU comes from the official labeled `.mat` (`tools/prepare_nyu.py`): 640×480 RGB, filled and raw depth (mm),
  894-class labels, per-class instance ids, scene names.
- The labeled RGB has a 6–8 px white border. The model only sees the region `[6:474, 7:632]`; feeding the border
  costs about 0.03 AbsRel.
- Validation is 92 images from 27 whole training scenes (`tools/make_val_split.py`, seed 0), so it shares
  no scene with train or test.

## Repository layout
```
tools/          our code: data prep, masks, dataset, losses, training, inference, evaluation
scripts/        setup.sh, pack_data.sh
colab/          Colab notebook
data/splits/    NYU split lists (data itself is downloaded, not committed)
docs/           requirements checklist and plan
results/        small CSV summaries of results
```

## Datasets
- NYU Depth V2 (labeled): https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
- DA-2K: https://huggingface.co/datasets/depth-anything/DA-2K
- iBims-1: https://www.asg.ed.tum.de/lmf/ibims1/ (core set via rsync, see `scripts/setup.sh`)

## Team
**RGB-D Scene**. Members: _to be added._

## Acknowledgements
Built on Depth Anything V2 (Yang et al., NeurIPS 2024) and Ultralytics YOLO11.
