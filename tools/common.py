"""Shared paths, model loading, and NYU helpers.

Paths can be overridden with environment variables so the same code runs on a
laptop, on Colab, or in a teammate's clone:
    DATA_ROOT  datasets          (default: <repo>/data)
    CKPT_DIR   model checkpoints (default: <repo>/checkpoints)

The relative and metric models live in two different copies of the
`depth_anything_v2` package inside the official repo, so the loader puts the
right one on sys.path before importing.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "Depth-Anything-V2"
DATA = Path(os.environ.get("DATA_ROOT", ROOT / "data"))
CKPT = Path(os.environ.get("CKPT_DIR", ROOT / "checkpoints"))
RESULTS = ROOT / "results"
SPLITS = ROOT / "data" / "splits"  # committed with the code, independent of DATA_ROOT
NYU_DIR = DATA / "nyu"              # produced by tools/prepare_nyu.py

# NYU Depth V2 (labeled set), 640x480, RGB and depth registered.
NYU_H, NYU_W = 480, 640
NYU_FX, NYU_FY, NYU_CX, NYU_CY = 518.8579, 519.4696, 325.5824, 253.7362
NYU_DEPTH_SCALE = 1000.0                 # png uint16 millimetres -> metres
NYU_EIGEN_CROP = (45, 471, 41, 601)      # top, bottom, left, right (standard eval crop)
# The labeled-set RGB has a constant white border (rows 0-5 / 474-479, cols 0-6 / 632-639).
# Feeding it to the model costs ~0.03 AbsRel, so the model only ever sees this region.
NYU_VALID_CROP = (6, 474, 7, 632)

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _import_dpt(metric: bool):
    pkg_root = REPO / "metric_depth" if metric else REPO
    sys.path.insert(0, str(pkg_root))
    from depth_anything_v2.dpt import DepthAnythingV2
    return DepthAnythingV2


def load_relative(encoder="vits"):
    DepthAnythingV2 = _import_dpt(metric=False)
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    model.load_state_dict(torch.load(CKPT / f"depth_anything_v2_{encoder}.pth", map_location="cpu"))
    return model.to(DEVICE).eval()


def load_metric(encoder="vits", max_depth=20.0, ckpt=None):
    DepthAnythingV2 = _import_dpt(metric=True)
    model = DepthAnythingV2(**{**MODEL_CONFIGS[encoder], "max_depth": max_depth})
    path = ckpt or CKPT / f"depth_anything_v2_metric_hypersim_{encoder}.pth"
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state.get("model", state))
    return model.to(DEVICE).eval()


def read_split(name):
    """Image ids ('000001', ...) for 'train' (scene-disjoint, val removed), 'val', 'test' (654) or 'train_full' (795)."""
    return [l.strip() for l in (SPLITS / f"nyu_{name}.txt").read_text().splitlines() if l.strip()]


def nyu_rgb_path(img_id):
    return NYU_DIR / "rgb" / f"{img_id}.png"


def read_nyu_depth(img_id, kind="filled"):
    """Depth in metres, 480x640. kind='filled' (dense) or 'raw' (0 = missing)."""
    d = cv2.imread(str(NYU_DIR / f"depth_{kind}" / f"{img_id}.png"), cv2.IMREAD_UNCHANGED)
    return d.astype(np.float32) / NYU_DEPTH_SCALE


def predict_nyu(model, bgr):
    """Run the model on the border-free region of a 640x480 NYU image; returns a 480x640 map (0 on the border)."""
    t, b, l, r = NYU_VALID_CROP
    pred = np.zeros(bgr.shape[:2], np.float32)
    pred[t:b, l:r] = model.infer_image(bgr[t:b, l:r])
    return pred


def depth_to_points(depth, fx, fy, cx, cy):
    """Back-project an HxW metric depth map to an (H*W, 3) array in camera coords."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    return np.stack([x, y, depth], axis=-1).reshape(-1, 3)
