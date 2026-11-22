"""DA-2K relative-depth ordering accuracy with the relative (disparity) model.

    python tools/eval_da2k.py [--encoder vits] [--tag name]

The relative model outputs disparity (larger = closer). DA-2K always marks point1
as the closer point, so a pair is correct when disp[point1] > disp[point2].
Writes results/da2k/<tag>_pairs.csv, <tag>_by_scene.csv and prints overall accuracy.
"""
import argparse
import json
import time

import cv2
import pandas as pd
import torch
from tqdm import tqdm

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="vits")
    ap.add_argument("--tag", default="relative_vits")
    ap.add_argument("--root", default=str(C.DATA / "DA-2K" / "DA-2K"))
    args = ap.parse_args()

    from pathlib import Path
    root = Path(args.root)
    ann = json.loads((root / "annotations.json").read_text())
    model = C.load_relative(args.encoder)

    rows = []
    for rel_path, pairs in tqdm(ann.items()):
        bgr = cv2.imread(str(root / rel_path))
        if C.DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        disp = model.infer_image(bgr)
        if C.DEVICE == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        scene = Path(rel_path).parts[-2]
        for a in pairs:
            (h1, w1), (h2, w2) = a["point1"], a["point2"]
            closer_first = a["closer_point"] == "point1"
            pred_first = disp[h1, w1] > disp[h2, w2]
            rows.append({"image": rel_path, "scene": scene, "ms": ms, "correct": pred_first == closer_first})

    df = pd.DataFrame(rows)
    out = C.RESULTS / "da2k"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{args.tag}_pairs.csv", index=False)
    by_scene = df.groupby("scene").agg(pairs=("correct", "size"), accuracy=("correct", "mean")).reset_index()
    by_scene.to_csv(out / f"{args.tag}_by_scene.csv", index=False)
    print(by_scene.to_string(index=False))
    print(f"\nOverall: {df.correct.mean()*100:.2f}% over {len(df)} pairs, "
          f"{df.drop_duplicates('image').ms.mean():.1f} ms/image")


if __name__ == "__main__":
    main()
