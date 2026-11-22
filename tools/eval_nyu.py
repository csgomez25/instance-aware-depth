"""Metric-depth evaluation on the NYU Depth V2 test split (654 images, Eigen crop, 0.001-10 m).

    python tools/eval_nyu.py [--ckpt path] [--tag name] [--limit N]

Writes results/nyu/<tag>_per_image.csv and appends a summary row to results/nyu/summary.csv.
"""
import argparse
import time

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import common as C


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="vits")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max-depth", type=float, default=20.0)
    ap.add_argument("--tag", default="pretrained_hypersim_vits")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model = C.load_metric(args.encoder, args.max_depth, args.ckpt)
    pairs = C.read_split("test")[: args.limit]
    t, b, l, r = C.NYU_EIGEN_CROP
    rows = []

    # Warm-up so the first timing isn't cuDNN autotuning.
    model.infer_image(cv2.imread(str(pairs[0][0])))

    for rgb_p, depth_p in tqdm(pairs):
        bgr = cv2.imread(str(rgb_p))
        gt = C.read_nyu_depth(depth_p)
        if C.DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = model.infer_image(bgr)
        if C.DEVICE == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000

        valid = np.zeros_like(gt, dtype=bool)
        valid[t:b, l:r] = True
        valid &= (gt > 1e-3) & (gt < 10)
        p = np.clip(pred[valid], 1e-3, 10)
        rows.append({"image": rgb_p.name, "ms": ms, **metrics(p, gt[valid])})

    df = pd.DataFrame(rows)
    out = C.RESULTS / "nyu"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{args.tag}_per_image.csv", index=False)

    summary = {"tag": args.tag, "n": len(df), **df.drop(columns="image").mean().round(4).to_dict()}
    sp = out / "summary.csv"
    prev = [pd.read_csv(sp)] if sp.exists() else []
    pd.concat(prev + [pd.DataFrame([summary])], ignore_index=True).to_csv(sp, index=False)
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
