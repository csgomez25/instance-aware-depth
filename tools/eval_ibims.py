"""Zero-shot metric-depth evaluation on iBims-1 (100 images, never seen in training).

    python tools/eval_ibims.py [--ckpt path] [--tag name] [--limit N] [--max-eval-depth 10]

Per image: the usual depth metrics in metres, the same after per-image median scaling (sa_*), the official iBims
geometric metrics (dbe_*, dde_*, pe_*; see ibims_metrics.py) and our bf1 / flying (see boundary.py; no bf1_obj,
iBims has no instance labels). Pixels beyond --max-eval-depth are ignored (10 m, as in ZoeDepth's iBims protocol;
the laser scans reach 50 m through windows and doorways).
Writes results/ibims/<tag>_per_image.csv and appends a summary row to results/ibims/summary.csv.
"""
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import common as C
from boundary import boundary_metrics
from eval_nyu import metrics
from ibims_data import ibims_names, read_ibims
from ibims_metrics import depth_boundary_error, directed_depth_error, planarity_errors


def evaluate_sample(s, pred, max_eval_depth):
    pred = np.clip(pred, 1e-3, None)
    valid = s.usable & (s.depth <= max_eval_depth)
    gt_v, pred_v = s.depth[valid], pred[valid]
    scale = np.median(gt_v) / np.median(pred_v)
    dbe_acc, dbe_com = depth_boundary_error(s.edges, pred)
    dde_0, dde_m, dde_p = directed_depth_error(gt_v, pred_v)
    planes = planarity_errors(s.depth, pred, valid, s.planes, s.calib)
    row = {"image": s.name, **metrics(pred_v, gt_v),
           **{f"sa_{k}": v for k, v in metrics(pred_v * scale, gt_v).items()},
           "dbe_acc": dbe_acc, "dbe_com": dbe_com, "dde_0": dde_0, "dde_m": dde_m, "dde_p": dde_p,
           **boundary_metrics(pred, s.depth, valid)}
    return row, planes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="vits")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max-depth", type=float, default=20.0, help="output range of the model head")
    ap.add_argument("--max-eval-depth", type=float, default=10.0)
    ap.add_argument("--tag", default="pretrained_hypersim_vits")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--yolo", default="yolo11s-seg.pt", help="mask model for the decoder variant")
    args = ap.parse_args()

    model = C.load_metric(args.encoder, args.max_depth, args.ckpt)
    if getattr(model, "needs_instances", False):  # decoder variant: iBims has no instance labels, so run YOLO
        from make_masks import load_yolo, yolo_instances
        yolo = load_yolo(args.yolo)
        predict = lambda bgr: model.infer_image(bgr, yolo_instances(yolo, bgr))
    else:
        predict = model.infer_image
    rows, planes = [], []
    for name in tqdm(ibims_names()[: args.limit]):
        s = read_ibims(name)
        row, p = evaluate_sample(s, predict(s.bgr), args.max_eval_depth)
        rows.append({**row, "n_planes": sum(not np.isnan(f) for f, _ in p)})
        planes += p

    df = pd.DataFrame(rows)
    out = C.RESULTS / "ibims"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{args.tag}_per_image.csv", index=False)

    pe = np.array(planes, dtype=float).reshape(-1, 2)
    summary = {"tag": args.tag, "n": len(df), "max_eval_depth": args.max_eval_depth,
               **df.drop(columns=["image", "n_planes"]).mean().round(4).to_dict(),
               "pe_fla": round(np.nanmean(pe[:, 0]), 4), "pe_ori": round(np.nanmean(pe[:, 1]), 4),
               "n_planes": int(np.sum(~np.isnan(pe[:, 0])))}
    sp = out / "summary.csv"
    prev = [pd.read_csv(sp)] if sp.exists() else []
    pd.concat(prev + [pd.DataFrame([summary])], ignore_index=True).to_csv(sp, index=False)
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
