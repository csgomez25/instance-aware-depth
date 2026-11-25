# Instance-Aware Depth for Single-Image 3D Reconstruction

ECE 4990 Computer Vision final project, Cal Poly Pomona, Fall 2026.

We fine-tune **[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)** (Small) on
**NYU Depth V2** with an **instance-aware loss**: object masks from YOLO11-seg guide where depth has to be
sharp (object boundaries) and keep each object's depth consistent. Sharper boundaries mean fewer
"flying pixels" when the depth map is lifted to a 3D point cloud.

```
RGB image ──► Depth Anything V2 ──► metric depth ──► 3D point cloud (.ply)
     │                ▲
     └─► YOLO11-seg ──┘  masks → instance-aware loss (training) / edge input (decoder variant)
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
| Final test numbers (H6) | ⬜ |
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
_Group name and members to be added._

## Acknowledgements
Built on Depth Anything V2 (Yang et al., NeurIPS 2024) and Ultralytics YOLO11.
