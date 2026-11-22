"""Extract the NYU Depth V2 labeled set (.mat) into plain image files.

    python tools/prepare_nyu.py [--mat PATH] [--out DIR]

Defaults: reads $DATA_ROOT/nyu_mat/nyu_depth_v2_labeled.mat, writes $DATA_ROOT/nyu/.
All outputs are 640x480 and pixel-aligned (RGB and depth are registered in the .mat):

    rgb/000001.png            RGB
    depth_filled/000001.png   uint16 millimetres, hole-filled (dense)
    depth_raw/000001.png      uint16 millimetres, raw Kinect, 0 = missing
    labels/000001.png         uint16 class id (894 classes, 0 = unlabelled)
    instances/000001.png      uint8 instance id; restarts per class, so an object is (label, instance)
    meta.csv                  id, scene, scene_type

Ids are 1-based .mat indices, matching data/splits/*.txt. Re-running skips finished images.
"""
import argparse
import csv
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

import common as C


def mat_str(f, ref):
    return "".join(map(chr, f[ref][:].ravel()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default=str(C.DATA / "nyu_mat" / "nyu_depth_v2_labeled.mat"))
    ap.add_argument("--out", default=str(C.NYU_DIR))
    args = ap.parse_args()

    out = Path(args.out)
    dirs = {k: out / k for k in ("rgb", "depth_filled", "depth_raw", "labels", "instances")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    f = h5py.File(args.mat, "r")
    n = f["images"].shape[0]
    png = [cv2.IMWRITE_PNG_COMPRESSION, 1]

    with open(out / "meta.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "scene", "scene_type"])
        for i in range(n):
            w.writerow([f"{i + 1:06d}", mat_str(f, f["scenes"][0][i]), mat_str(f, f["sceneTypes"][0][i])])

    for i in tqdm(range(n), desc="NYU"):
        name = f"{i + 1:06d}.png"
        if (dirs["instances"] / name).exists():
            continue
        # .mat arrays are stored transposed (W, H); transpose back to (H, W).
        rgb = f["images"][i].transpose(2, 1, 0)
        cv2.imwrite(str(dirs["rgb"] / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), png)
        for key, d in (("depths", "depth_filled"), ("rawDepths", "depth_raw")):
            depth_mm = np.round(f[key][i].T * C.NYU_DEPTH_SCALE).astype(np.uint16)
            cv2.imwrite(str(dirs[d] / name), depth_mm, png)
        cv2.imwrite(str(dirs["labels"] / name), f["labels"][i].T.astype(np.uint16), png)
        cv2.imwrite(str(dirs["instances"] / name), f["instances"][i].T.astype(np.uint8), png)

    print(f"Wrote {n} images to {out}")


if __name__ == "__main__":
    main()
