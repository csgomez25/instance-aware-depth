"""Run YOLO11-seg on all NYU images and save one instance map per image.

    python tools/make_masks.py [--model yolo11s-seg.pt] [--conf 0.25]

Writes $DATA_ROOT/nyu/<model>/<id>.png (uint8, 0 = background, k = k-th instance) and
<model>.jsonl with one line per image: {"id", "instances": [{"k", "cls", "name", "conf"}]}.
Overlaps: masks are painted largest first, so smaller objects in front (a cup on a
table) keep their own pixels. Re-running skips finished images.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import common as C


def load_yolo(name="yolo11s-seg.pt"):
    weights = C.CKPT / name
    if not weights.exists():  # ultralytics downloads by name into the current directory
        C.CKPT.mkdir(parents=True, exist_ok=True)
        # move, not rename: CKPT_DIR may be on another filesystem (Google Drive on Colab)
        shutil.move(YOLO(name).ckpt_path, weights)
    return YOLO(str(weights))


def instance_map(r):
    """One YOLO result -> (HxW uint8 map, 0 = background, k = k-th instance; per-instance info), largest painted first."""
    inst = np.zeros(r.orig_shape, np.uint8)
    info = []
    if r.masks is not None:
        masks = r.masks.data.cpu().numpy() > 0.5
        order = np.argsort(-masks.reshape(len(masks), -1).sum(1))[:255]
        for k, j in enumerate(order, start=1):
            inst[masks[j]] = k
            c = int(r.boxes.cls[j])
            info.append({"k": k, "cls": c, "name": r.names[c], "conf": round(float(r.boxes.conf[j]), 3)})
    return inst, info


def yolo_instances(model, bgr, conf=0.25):
    """Instance map for one BGR image, made exactly as the NYU mask folders are."""
    r = model.predict(bgr, conf=conf, retina_masks=True, verbose=False, device=0 if C.DEVICE == "cuda" else "cpu")[0]
    return instance_map(r)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s-seg.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    name = Path(args.model).stem
    out = C.NYU_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    meta_path = C.NYU_DIR / f"{name}.jsonl"
    done = set()
    if meta_path.exists():
        done = {json.loads(l)["id"] for l in meta_path.read_text().splitlines() if l.strip()}

    ids = sorted(p.stem for p in (C.NYU_DIR / "rgb").glob("*.png"))
    todo = [i for i in ids if i not in done or not (out / f"{i}.png").exists()]
    model = load_yolo(args.model)

    with open(meta_path, "a") as meta:
        for start in tqdm(range(0, len(todo), args.batch), desc=name):
            chunk = todo[start:start + args.batch]
            results = model.predict([str(C.nyu_rgb_path(i)) for i in chunk], conf=args.conf,
                                    retina_masks=True, verbose=False, device=0 if C.DEVICE == "cuda" else "cpu")
            for img_id, r in zip(chunk, results):
                inst, info = instance_map(r)
                cv2.imwrite(str(out / f"{img_id}.png"), inst)
                meta.write(json.dumps({"id": img_id, "instances": info}) + "\n")

    print(f"Masks for {len(ids)} images in {out}")


if __name__ == "__main__":
    main()
