"""Boundary-quality metrics for a predicted depth map against NYU ground truth.

bf1        scale-invariant depth-boundary F1 (Depth Pro, Bochkovskii et al. 2024): a neighbouring pixel pair is an
           occluding contour when the far/near depth ratio exceeds t; F1 of predicted vs GT contours, mean over
           t in 1.05..1.25. Exact pixel-pair match, no tolerance band.
bf1_obj    the same, counting only pairs that straddle an NYU instance boundary (two different objects, or object
           and background). This is where the instance-aware loss acts.
flying     fraction of pixels near a GT depth discontinuity (far/near > 1.25 within 7x7) whose GT lies on the near
           or far surface but whose predicted depth falls in the gap between them. Back-projection moves a pixel along its camera ray, so these are exactly the
           points that float between foreground and background in the point cloud.

All three ignore global scale. Ground truth is the filled depth: raw depth has holes along occlusion edges.
"""
import cv2
import numpy as np

THRESHOLDS = np.linspace(1.05, 1.25, 10)
FLY_WINDOW = 7      # px, neighbourhood that defines the near and far surface
FLY_JUMP = 1.25     # far/near ratio that counts as a discontinuity
FLY_TOL = 1.05      # a prediction within 5 % of either surface is not flying


def _pair_ratios(d):
    """far/near depth ratio for every right and down neighbour pair."""
    right = np.maximum(d[:, 1:], d[:, :-1]) / np.minimum(d[:, 1:], d[:, :-1])
    down = np.maximum(d[1:], d[:-1]) / np.minimum(d[1:], d[:-1])
    return right, down


def _pair_mask(m):
    """Pairs whose two pixels are both True in m."""
    return m[:, 1:] & m[:, :-1], m[1:] & m[:-1]


def _f1(pred_edges, gt_edges):
    tp = sum((p & g).sum() for p, g in zip(pred_edges, gt_edges))
    n_pred = sum(p.sum() for p in pred_edges)
    n_gt = sum(g.sum() for g in gt_edges)
    if n_gt == 0:
        return np.nan
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt
    return 2 * precision * recall / (precision + recall) if tp else 0.0


def boundary_f1(pred, gt, valid, pair_filter=None):
    pr, gr = _pair_ratios(pred), _pair_ratios(gt)
    keep = _pair_mask(valid)
    if pair_filter is not None:
        keep = tuple(k & f for k, f in zip(keep, pair_filter))
    if not any(k.any() for k in keep):
        return np.nan
    scores = [_f1([(r > t) & k for r, k in zip(pr, keep)], [(r > t) & k for r, k in zip(gr, keep)])
              for t in THRESHOLDS]
    return np.nanmean(scores) if not np.all(np.isnan(scores)) else np.nan


def instance_crossings(instances):
    """Pairs whose two pixels belong to different instances (0 = background counts as one region)."""
    return instances[:, 1:] != instances[:, :-1], instances[1:] != instances[:-1]


def flying_fraction(pred, gt, valid):
    kernel = np.ones((FLY_WINDOW, FLY_WINDOW), np.uint8)
    near = cv2.erode(np.where(valid, gt, 1e6).astype(np.float32), kernel)
    far = cv2.dilate(np.where(valid, gt, 0).astype(np.float32), kernel)
    in_gap = lambda d: (d > FLY_TOL * near) & (d < far / FLY_TOL)
    # GT pixels that are themselves between the surfaces (slopes, hole filling) have no right answer here
    band = valid & (far > FLY_JUMP * near) & ~in_gap(gt)
    if not band.any():
        return np.nan
    return in_gap(pred)[band].mean()


def boundary_metrics(pred, gt, valid, instances=None):
    """pred, gt: HxW metres; valid: HxW bool; instances: HxW int ids, or None to skip bf1_obj.
    pred is median-scaled to gt first."""
    pred = np.where(valid, pred, 1.0)
    gt = np.where(valid, gt, 1.0)
    pred = pred * np.median(gt[valid] / pred[valid])
    out = {"bf1": boundary_f1(pred, gt, valid)}
    if instances is not None:
        out["bf1_obj"] = boundary_f1(pred, gt, valid, instance_crossings(instances))
    out["flying"] = flying_fraction(pred, gt, valid)
    return out
