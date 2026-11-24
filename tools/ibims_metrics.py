"""iBims-1 geometric metrics, ported from the official evaluate_ibims_error_metrics.py (v0.9).

Same definitions and constants as the official script; numpy replaces sklearn PCA and scipy's distance transform
(cv2 exact L2 transform). The Canny step is skimage's, as in the official Python script, whose DBE the authors note
differs slightly from their Matlab numbers.
    dbe_acc / dbe_com  depth boundary error: chamfer distance (px) from predicted to GT edges / GT to predicted
    dde_0 / dde_m / dde_p  directed depth error at 3 m: share correct / predicted too close / too far
    pe_fla (cm) / pe_ori (deg)  planarity error: spread of the fitted plane / angle to the GT normal
"""
import math

import cv2
import numpy as np
from skimage import feature

DBE_MAX_DIST = 10.0
DDE_THRESHOLD = 3.0
PLANE_MIN_FRACTION = 0.05


def _edt(binary):
    """Euclidean distance of every pixel to the nearest True pixel."""
    return cv2.distanceTransform((~binary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def depth_boundary_error(gt_edges, pred):
    if gt_edges is None or not gt_edges.any():
        return np.nan, np.nan
    p = pred.astype(np.float32).copy()
    p[p <= 0] = np.nan
    p = (p - np.nanmin(p)) / (np.nanmax(p) - np.nanmin(p))
    est = feature.canny(p, sigma=np.sqrt(2), low_threshold=0.1, high_threshold=0.2)
    d_gt, d_est = _edt(gt_edges), _edt(est)
    near = est & (d_gt < DBE_MAX_DIST)
    if not near.any():
        return DBE_MAX_DIST, DBE_MAX_DIST
    return d_gt[near].mean(), d_est[gt_edges].mean()


def directed_depth_error(gt, pred):
    """gt, pred: 1-D arrays of valid pixels."""
    diff = (pred <= DDE_THRESHOLD).astype(int) - (gt <= DDE_THRESHOLD).astype(int)
    return np.mean(diff == 0), np.mean(diff == 1), np.mean(diff == -1)


def planarity_errors(gt, pred, valid, planes, calib):
    """One (pe_fla, pe_ori) per GT plane; NaN for planes under 5 % of the image."""
    fx, fy, cx, cy = calib
    h, w = gt.shape
    c, r = np.meshgrid(np.arange(1, w + 1), np.arange(1, h + 1))  # 1-based, as in the official script
    out = []
    for plane in planes:
        m = plane.mask & valid
        if m.sum() / (640.0 * 480.0) < PLANE_MIN_FRACTION:
            out.append((np.nan, np.nan))
            continue
        d = pred[m] * np.median(gt[m]) / np.median(pred[m])
        pts = np.stack([(c[m] - cx) * d / fx, d, -(r[m] - cy) * d / fy], axis=1)
        centred = pts - pts.mean(axis=0)
        normal = np.linalg.svd(centred, full_matrices=False)[2][2]
        flat = np.std(centred @ normal) * 100.0
        n_gt = plane.normal
        if normal @ n_gt < 0:
            normal = -normal
        ori = math.degrees(math.atan2(np.linalg.norm(np.cross(n_gt, normal)), n_gt @ normal))
        out.append((flat, ori))
    return out
