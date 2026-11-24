"""iBims-1 core set (Koch et al., ECCVW 2018): 100 indoor 640x480 images with dense laser-scan depth.

Files under $DATA_ROOT/ibims/ibims1_core_raw/ (see its readme.txt):
    rgb/, depth/ (uint16, metres = value / 65535 * 50), edges/ (binary, 86 images have them),
    mask_invalid/, mask_transp/ (255 = usable), mask_{wall,floor,table}/ (plane id per pixel, 0 = none)
    with <name>.txt holding one row per plane: A, B, C, D, nx, ny, nz; calib/<name>.txt: fx, fy, cx, cy.
"""
from dataclasses import dataclass

import cv2
import numpy as np

import common as C

IBIMS_DIR = C.DATA / "ibims" / "ibims1_core_raw"
IBIMS_DEPTH_SCALE = 50.0 / 65535.0
PLANE_TYPES = ("wall", "floor", "table")


@dataclass
class Plane:
    mask: np.ndarray    # HxW bool
    normal: np.ndarray  # (3,) GT unit normal, in the official evaluation's (x, depth, -y) frame


@dataclass
class IbimsSample:
    name: str
    bgr: np.ndarray
    depth: np.ndarray         # HxW metres, 0 = no measurement
    usable: np.ndarray        # HxW bool: measured, not masked invalid or transparent
    edges: np.ndarray | None  # HxW bool GT depth edges, None when the image has none
    calib: tuple              # fx, fy, cx, cy
    planes: list


def ibims_names():
    return [l.strip() for l in (IBIMS_DIR / "imagelist.txt").read_text().splitlines() if l.strip()]


def _png(sub, name):
    return cv2.imread(str(IBIMS_DIR / sub / f"{name}.png"), cv2.IMREAD_UNCHANGED)


def _planes(name):
    planes = []
    for kind in PLANE_TYPES:
        txt = IBIMS_DIR / f"mask_{kind}" / f"{name}.txt"
        if not txt.exists():
            continue
        ids = _png(f"mask_{kind}", name)
        paras = np.loadtxt(txt, delimiter=",", ndmin=2)
        planes += [Plane(ids == j + 1, paras[j, 4:7]) for j in range(len(paras))]
    return planes


def read_ibims(name):
    depth = _png("depth", name).astype(np.float32) * IBIMS_DEPTH_SCALE
    usable = (_png("mask_invalid", name) > 0) & (_png("mask_transp", name) > 0) & (depth > 0)
    edges = _png("edges", name) if (IBIMS_DIR / "edges" / f"{name}.png").exists() else None
    calib = tuple(float(v) for v in (IBIMS_DIR / "calib" / f"{name}.txt").read_text().strip().split(","))
    return IbimsSample(name, _png("rgb", name)[..., :3].copy(), depth, usable,
                       None if edges is None else edges > 0, calib, _planes(name))
