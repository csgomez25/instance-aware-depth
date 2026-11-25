"""Metric-depth evaluation on NYU Depth V2 (Eigen crop, 0.001-10 m).

    python tools/eval_nyu.py [--split test|val] [--gt filled|raw] [--ckpt path] [--tag name] [--limit N]

Writes results/nyu/<tag>_<split>_<gt>_per_image.csv and appends a summary row to results/nyu/summary.csv.
Boundary metrics (bf1, bf1_obj, flying; see boundary.py) always use the filled depth.
Use --split val while tuning; the test split is for final numbers only.
"""
import argparse
import time

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import common as C
from boundary import boundary_metrics


def metrics(pred, gt):
    thresh = np.maximum(gt / pred, pred / gt)
    return {
        "abs_rel": np.mean(np.abs(gt - pred) / gt),
        "sq_rel": np.mean((gt - pred) ** 2 / gt),
        "rmse": np.sqrt(np.mean((gt - pred) ** 2)),
        "rmse_log": np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2)),
        "log10": np.mean(np.abs(np.log10(gt) - np.log10(pred))),
        "d1": np.mean(thresh < 1.25),
        "d2": np.mean(thresh < 1.25 ** 2),
        "d3": np.mean(thresh < 1.25 ** 3),
    }


def eval_mask(gt):
    t, b, l, r = C.NYU_EIGEN_CROP
    valid = np.zeros_like(gt, dtype=bool)
    valid[t:b, l:r] = True
    return valid & (gt > 1e-3) & (gt < 10)


@torch.no_grad()
def evaluate(model, ids, gts=("filled", "raw"), masks=None):
    """Mean metrics over `ids` for each ground-truth kind, one prediction per image (same protocol as main()).
    masks: instance-mask source for the decoder variant (see common.read_nyu_masks); ignored by other models."""
    rows = {g: [] for g in gts}
    for img_id in ids:
        pred = C.predict_nyu(model, cv2.imread(str(C.nyu_rgb_path(img_id))), model_masks(model, img_id, masks))
        for g in gts:
            gt = C.read_nyu_depth(img_id, g)
            valid = eval_mask(gt)
            rows[g].append(metrics(np.clip(pred[valid], 1e-3, 10), gt[valid]))
    return {g: pd.DataFrame(r).mean().to_dict() for g, r in rows.items()}


def model_masks(model, img_id, masks):
    """The instance map the model takes as input, or None for models that take only the image."""
    return C.read_nyu_masks(img_id, masks) if getattr(model, "needs_instances", False) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--gt", default="filled", choices=["filled", "raw"])
    ap.add_argument("--encoder", default="vits")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max-depth", type=float, default=20.0)
    ap.add_argument("--tag", default="pretrained_hypersim_vits")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--masks", default="yolo11s-seg", help="mask input of the decoder variant: 'gt' or a mask folder")
    args = ap.parse_args()

    model = C.load_metric(args.encoder, args.max_depth, args.ckpt)
    ids = C.read_split(args.split)[: args.limit]
    rows = []

    C.predict_nyu(model, cv2.imread(str(C.nyu_rgb_path(ids[0]))), model_masks(model, ids[0], args.masks))  # warm-up (cuDNN autotune)

    for img_id in tqdm(ids):
        bgr = cv2.imread(str(C.nyu_rgb_path(img_id)))
        gt = C.read_nyu_depth(img_id, args.gt)
        inst = model_masks(model, img_id, args.masks)
        if C.DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = C.predict_nyu(model, bgr, inst)
        if C.DEVICE == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000

        valid = eval_mask(gt)
        p = np.clip(pred, 1e-3, 10)
        filled = gt if args.gt == "filled" else C.read_nyu_depth(img_id, "filled")
        rows.append({"image": img_id, "ms": ms, **metrics(p[valid], gt[valid]),
                     **boundary_metrics(p, filled, eval_mask(filled), C.read_nyu_instances(img_id))})

    df = pd.DataFrame(rows)
    out = C.RESULTS / "nyu"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{args.tag}_{args.split}_{args.gt}_per_image.csv", index=False)

    summary = {"tag": args.tag, "split": args.split, "gt": args.gt, "n": len(df),
               **df.drop(columns="image").mean().round(4).to_dict()}
    sp = out / "summary.csv"
    prev = [pd.read_csv(sp)] if sp.exists() else []
    pd.concat(prev + [pd.DataFrame([summary])], ignore_index=True).to_csv(sp, index=False)
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
