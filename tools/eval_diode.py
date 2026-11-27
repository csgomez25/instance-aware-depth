"""Zero-shot metric-depth evaluation on DIODE indoor val (325 images, laser-scanned, never seen in training).

    python eval_diode.py [--ckpt path] [--tag name] [--limit N]

Per image: scale (median GT / median prediction; >1 means the model predicts the room too small), depth metrics in
metres, the same after per-image median scaling (sa_*), and our bf1 / flying
(boundary.py). Valid pixels: DIODE's depth mask and 1e-3 < depth <= 10 m (ZoeDepth's DIODE-indoor protocol).
Data: data/diode/val/indoors/<scene>/<scan>/<name>{.png,_depth.npy,_depth_mask.npy}, from
http://diode-dataset.s3.amazonaws.com/val.tar.gz (only val/indoors is needed).
Writes results/diode/<tag>_per_image.csv and appends a summary row to results/diode/summary.csv.
"""
import argparse

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import common as C
from boundary import boundary_metrics
from eval_nyu import metrics

DIODE = C.DATA / "diode" / "val" / "indoors"
MAX_EVAL_DEPTH = 10.0


def diode_images():
    return sorted(DIODE.glob("*/*/*.png"))


def read_diode(png):
    depth = np.load(str(png).replace(".png", "_depth.npy"))[..., 0]
    mask = np.load(str(png).replace(".png", "_depth_mask.npy")) > 0
    return cv2.imread(str(png)), depth, mask & (depth > 1e-3) & (depth <= MAX_EVAL_DEPTH)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", default="pretrained_hypersim_vits")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model = C.load_metric(ckpt=args.ckpt)
    rows = []
    for png in tqdm(diode_images()[: args.limit]):
        bgr, gt, valid = read_diode(png)
        pred = np.clip(model.infer_image(bgr), 1e-3, None)
        gt_v, pred_v = gt[valid], pred[valid]
        scale = np.median(gt_v) / np.median(pred_v)
        rows.append({"image": png.stem, "scale": scale, **metrics(pred_v, gt_v),
                     **{f"sa_{k}": v for k, v in metrics(pred_v * scale, gt_v).items()},
                     **boundary_metrics(pred, gt, valid)})

    df = pd.DataFrame(rows)
    out = C.RESULTS / "diode"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{args.tag}_per_image.csv", index=False)
    summary = {"tag": args.tag, "n": len(df), **df.drop(columns="image").mean().round(4).to_dict()}
    sp = out / "summary.csv"
    prev = [pd.read_csv(sp)] if sp.exists() else []
    pd.concat(prev + [pd.DataFrame([summary])], ignore_index=True).to_csv(sp, index=False)
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
