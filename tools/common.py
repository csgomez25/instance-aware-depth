"""Shared paths, model loading, and NYU constants.

The relative and metric models live in two different copies of the
`depth_anything_v2` package inside the official repo, so the loader puts the
right one on sys.path before importing.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "Depth-Anything-V2"
CKPT = ROOT / "checkpoints"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
NYU_DIR = Path(os.environ.get("NYU_DIR", DATA / "nyu"))  # contains rgb/, depth/, masks/
NYU_SPLITS = DATA / "splits"

# This NYU copy stores RGB at 625x468 (already centre-cropped) but depth at the
# full 640x480. Centre-cropping depth by (6, 7) aligns it with RGB; a sweep of
# scale-aligned log error over offsets puts the optimum within 1 px of this.
NYU_CROP_TOP, NYU_CROP_LEFT = 6, 7
NYU_H, NYU_W = 468, 625
NYU_DEPTH_SCALE = 5000.0  # uint16 png -> metres

# Kinect intrinsics (defined at 640x480), principal point shifted into the 625x468 crop.
NYU_FX, NYU_FY = 518.8579, 519.4696
NYU_CX, NYU_CY = 325.5824 - NYU_CROP_LEFT, 253.7362 - NYU_CROP_TOP

# Eigen evaluation crop (45:471, 41:601 at 640x480), in the 625x468 frame.
NYU_EIGEN_CROP = (45 - NYU_CROP_TOP, 471 - NYU_CROP_TOP, 41 - NYU_CROP_LEFT, 601 - NYU_CROP_LEFT)


def read_nyu_depth(path):
    """Load a NYU depth png in metres, cropped to align with the 625x468 RGB."""
    import cv2
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / NYU_DEPTH_SCALE
    t, l = NYU_CROP_TOP, NYU_CROP_LEFT
    return d[t:t + NYU_H, l:l + NYU_W]

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
    """Return [(rgb_path, depth_path)] for the NYU 'train' (795) or 'test' (654) split."""
    fname = {"train": "nyu_train.txt", "test": "nyu_test.txt"}[name]
    pairs = []
    for line in (NYU_SPLITS / fname).read_text().splitlines():
        if line.strip():
            rgb, _mask, depth = line.split("\t")
            pairs.append((NYU_DIR / rgb, NYU_DIR / depth))
    return pairs


def depth_to_points(depth, fx, fy, cx, cy):
    """Back-project an HxW metric depth map to an (H*W, 3) array in camera coords."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    return np.stack([x, y, depth], axis=-1).reshape(-1, 3)
