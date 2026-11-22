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
| Fine-tuning script | ⬜ |
| Instance-aware loss, decoder change | ⬜ |
| iBims-1 eval, boundary + flying-pixel metrics | ⬜ |
| Experiments, paper, slides | ⬜ |

## Baseline results (ViT-S, no fine-tuning)
| Benchmark | AbsRel | RMSE (m) | δ1 |
|---|---|---|---|
| NYU test (654), filled depth | 0.186 | 0.590 | 0.740 |
| NYU test (654), raw depth | 0.181 | 0.542 | 0.752 |
| NYU val (92), filled depth | 0.198 | 0.733 | 0.719 |
| DA-2K (relative model) | **95.16 %** of 2068 pairs correct (paper: 95.3 %) | | |

NYU uses the zero-shot Hypersim metric model, the Eigen crop and 0.001–10 m. Most of its error is global scale:
per-image scale alignment drops AbsRel from about 0.18 to 0.07.

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
```

## Data notes
- NYU comes from the official labeled `.mat` (`tools/prepare_nyu.py`): 640×480 RGB, filled and raw depth (mm),
  894-class labels, per-class instance ids, scene names.
- The labeled RGB has a 6–8 px white border. The model only sees the region `[6:474, 7:632]`; feeding the border
  costs about 0.03 AbsRel.
- Validation is 92 images from 27 whole training scenes (`tools/make_val_split.py`, seed 0), so it shares
  no scene with train or test.

## Repository layout
```
tools/          our code: data prep, masks, inference, evaluation (training to come)
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
