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
| Environment, checkpoints, DA-2K, NYU | ✅ |
| Inference + point clouds (`tools/infer.py`) | ✅ |
| NYU eval (`tools/eval_nyu.py`), DA-2K eval (`tools/eval_da2k.py`) | ✅ |
| NYU data preparation from the .mat, validation split | ⬜ |
| Fine-tuning script | ⬜ |
| Instance-aware loss, decoder change | ⬜ |
| iBims-1 eval, boundary + flying-pixel metrics | ⬜ |
| Experiments, paper, slides | ⬜ |

## Baseline results (ViT-S, no fine-tuning)
| Benchmark | Result |
|---|---|
| DA-2K (relative model) | **95.16 %** of 2068 pairs (paper: 95.3 %) |
| NYU test, 654 images, raw depth, zero-shot Hypersim metric model | AbsRel 0.181 · RMSE 0.542 m · δ1 0.752 · 86 ms/image |

Most of the zero-shot NYU error is global scale: per-image scale alignment drops AbsRel from 0.177 to 0.071.

## Setup
```bash
conda env create -f environment.yml
conda activate depth3d
bash scripts/setup.sh        # upstream code (pinned), checkpoints, DA-2K, NYU .mat
```

## Usage
```bash
python tools/infer.py path/to/image.png      # depth png + .npy + coloured .ply in results/infer/
python tools/eval_nyu.py                      # NYU test metrics -> results/nyu/
python tools/eval_da2k.py                     # DA-2K accuracy per scene -> results/da2k/
```

## Repository layout
```
tools/          our code (inference, evaluation; training to come)
scripts/        setup
data/splits/    NYU 795/654 split lists (data itself is downloaded, not committed)
docs/           requirements checklist and plan
results/        small CSV summaries of results
```

## Datasets
- NYU Depth V2 (labeled): https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
- DA-2K: https://huggingface.co/datasets/depth-anything/DA-2K
- iBims-1 (planned second dataset)

## Team
_Group name and members to be added._

## Acknowledgements
Built on Depth Anything V2 (Yang et al., NeurIPS 2024) and Ultralytics YOLO11.
