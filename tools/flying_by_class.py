"""Flying pixels per object class on NYU: which objects smear into the background in 3D.

    python flying_by_class.py --split val \\
        --group zero-shot=zero-shot \\
        --group control=h2b_edge_yolo_l0_laptop,h5_edge_yolo_l0_s1_laptop,h5_edge_yolo_l0_s2_laptop \\
        --group method=h2b_edge_yolo_l1.0_laptop,h5_edge_yolo_l1.0_s1_laptop,h5_edge_yolo_l1.0_s2_laptop

A group pools the pixels of its runs (e.g. 3 seeds); 'zero-shot' is the pretrained model. Scored pixels are the
`flying` band of boundary.py (filled GT, median-scaled prediction). Each band pixel counts for every YOLO11s-seg
object whose mask, grown by half the flying window, contains it, and for 'background' when it touches none, so an
object owns the depth edge on both sides of its outline. Classes come from the stored YOLO masks used in training.
Writes results/nyu/flying_by_class_<split>.csv: class, group, objects, band_px, flying_px, flying.
"""
import argparse
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import common as C
from boundary import FLY_WINDOW, flying_maps, median_scaled
from eval_nyu import eval_mask, model_masks
from scene import nyu_masks


def class_masks(inst, info):
    """{class: HxW bool} of every object of that class, each grown by half the flying window."""
    kernel = np.ones((FLY_WINDOW, FLY_WINDOW), np.uint8)
    out = defaultdict(lambda: np.zeros(inst.shape, bool))
    for o in info:
        out[o["name"]] |= cv2.dilate((inst == o["k"]).astype(np.uint8), kernel) > 0
    return dict(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--group", action="append", required=True, help="name=run1,run2,... ('zero-shot' = pretrained)")
    ap.add_argument("--masks", default="yolo11s-seg")
    args = ap.parse_args()

    groups = {g.split("=")[0]: g.split("=")[1].split(",") for g in args.group}
    ids = C.read_split(args.split)
    gts = {i: C.read_nyu_depth(i, "filled") for i in ids}
    masks = {i: nyu_masks(i, args.masks) for i in ids}
    counts = defaultdict(lambda: np.zeros(2, np.int64))  # (class, group) -> band_px, flying_px
    n_objects = defaultdict(int)
    for i in ids:
        for o in masks[i][1]:
            n_objects[o["name"]] += 1

    for gname, runs in groups.items():
        for run in runs:
            model = C.load_metric(ckpt=None if run == "zero-shot" else C.RUNS / run / "best.pth")
            for i in tqdm(ids, desc=f"{gname}: {run}"):
                inst, info = masks[i]
                pred = C.predict_nyu(model, cv2.imread(str(C.nyu_rgb_path(i))), model_masks(model, i, args.masks))
                valid = eval_mask(gts[i])
                band, fly = flying_maps(*median_scaled(np.clip(pred, 1e-3, 10), gts[i], valid), valid)
                cm = class_masks(inst, info)
                touched = np.zeros(inst.shape, bool)
                for cls, m in cm.items():
                    counts[(cls, gname)] += (int((band & m).sum()), int((fly & m).sum()))
                    touched |= m
                counts[("background", gname)] += (int((band & ~touched).sum()), int((fly & ~touched).sum()))

    rows = [{"class": cls, "group": g, "objects": n_objects.get(cls, 0), "band_px": int(b), "flying_px": int(f),
             "flying": round(f / b, 4) if b else np.nan} for (cls, g), (b, f) in counts.items()]
    df = pd.DataFrame(rows).sort_values(["band_px", "group"], ascending=[False, True])
    out = C.RESULTS / "nyu" / f"flying_by_class_{args.split}.csv"
    df.to_csv(out, index=False)
    wide = df.pivot(index="class", columns="group", values="flying")
    wide.insert(0, "objects", df.groupby("class").objects.first())
    wide.insert(1, "band_px", df.groupby("class").band_px.max())
    print(wide.sort_values("band_px", ascending=False).head(20).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
