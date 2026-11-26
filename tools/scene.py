"""Single-image RGB-D scene: one RGB photo -> metric depth -> labeled 3D scene.

    python scene.py 000500 [--ckpt ../runs/<run>/best.pth]      # NYU image id: NYU intrinsics, stored YOLO masks
    python scene.py room.jpg [--fx 520 --fy 520]                 # any photo: YOLO11s-seg runs on it

Writes to results/scene/ (or --outdir), per image <stem>:
    <stem>_rgb.ply        point cloud in the photo's colours
    <stem>_labels.ply     the same points, one colour per object, background grey
    <stem>_objects.json   one entry per object: class, confidence, point count, distance (median depth),
                          3D centre and size (5th-95th percentile extent, so stray flying points do not inflate it)
    <stem>_overview.png   photo | depth | objects with "class distance" labels
Camera coordinates: x right, y down, z forward (metres). Objects smaller than --min-points are left out.
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import open3d as o3d

import common as C

PALETTE = (np.array(matplotlib.colormaps["tab20"].colors)[:, ::-1] * 255).astype(np.uint8)  # BGR
BACKGROUND_BGR = np.array([150, 150, 150], np.uint8)


def nyu_masks(img_id, masks="yolo11s-seg"):
    """Stored instance map and per-instance info for a NYU image (the masks used in training)."""
    inst = C.read_nyu_masks(img_id, masks)
    for line in (C.NYU_DIR / f"{masks}.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec["id"] == img_id:
            return inst, rec["instances"]
    raise KeyError(f"{img_id} not in {masks}.jsonl")


def scene_objects(depth, inst, info, intrinsics, min_points):
    """Per-object 3D summary. depth: HxW metres (0 = no prediction); inst: HxW ids; info: [{k, name, conf}]."""
    fx, fy, cx, cy = intrinsics
    v, u = np.nonzero(depth > 0)
    z = depth[v, u]
    pts = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], 1)
    ids = inst[v, u]
    objects = []
    for o in info:
        p = pts[ids == o["k"]]
        if len(p) < min_points:
            continue
        lo, hi = np.percentile(p, 5, axis=0), np.percentile(p, 95, axis=0)
        objects.append({"id": o["k"], "class": o["name"], "conf": o["conf"], "n_points": len(p),
                        "distance_m": round(float(np.median(p[:, 2])), 3),
                        "center_m": np.round((lo + hi) / 2, 3).tolist(), "size_m": np.round(hi - lo, 3).tolist()})
    return objects, pts, (v, u)


def label_colors(inst, kept_ids):
    colors = np.tile(BACKGROUND_BGR, (*inst.shape, 1))
    for i, k in enumerate(kept_ids):
        colors[inst == k] = PALETTE[i % len(PALETTE)]
    return colors


def write_ply(path, pts, bgr):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(bgr[:, ::-1] / 255.0)
    o3d.io.write_point_cloud(str(path), pcd)


def overview(bgr, depth, inst, objects, labels_bgr):
    d = depth[depth > 0]
    norm = np.clip((depth - d.min()) / (d.max() - d.min() + 1e-8), 0, 1)
    depth_vis = (matplotlib.colormaps["Spectral"](norm)[..., :3][..., ::-1] * 255).astype(np.uint8)
    depth_vis[depth <= 0] = 0
    obj = cv2.addWeighted(bgr, 0.4, labels_bgr, 0.6, 0)
    for o in objects:
        m = (inst == o["id"]).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(obj, cnts, -1, (255, 255, 255), 1)
        ys, xs = np.nonzero(m)
        text = f"{o['class']} {o['distance_m']:.1f} m"
        org = (int(xs.mean()) - 4 * len(text), int(ys.mean()))
        cv2.putText(obj, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(obj, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.hstack([bgr, depth_vis, obj])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="NYU image ids (e.g. 000500) or image paths")
    ap.add_argument("--ckpt", default=None, help="fine-tuned run (default: zero-shot Hypersim model)")
    ap.add_argument("--masks", default="yolo11s-seg", help="stored NYU mask folder used for NYU ids")
    ap.add_argument("--fx", type=float, default=C.NYU_FX, help="focal length for non-NYU photos (default: NYU Kinect)")
    ap.add_argument("--fy", type=float, default=C.NYU_FY)
    ap.add_argument("--min-points", type=int, default=200)
    ap.add_argument("--outdir", default=str(C.RESULTS / "scene"))
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    model = C.load_metric(ckpt=args.ckpt)
    yolo = None
    for name in args.images:
        is_nyu = name.isdigit() and C.nyu_rgb_path(name).exists()
        if is_nyu:
            bgr = cv2.imread(str(C.nyu_rgb_path(name)))
            inst, info = nyu_masks(name, args.masks)
            depth = C.predict_nyu(model, bgr, inst if getattr(model, "needs_instances", False) else None)
            intrinsics = (C.NYU_FX, C.NYU_FY, C.NYU_CX, C.NYU_CY)
        else:
            from make_masks import instance_map, load_yolo
            yolo = yolo or load_yolo()
            bgr = cv2.imread(name)
            r = yolo.predict(bgr, conf=0.25, retina_masks=True, verbose=False, device=0 if C.DEVICE == "cuda" else "cpu")[0]
            inst, info = instance_map(r)
            inst = inst.astype(np.int32)
            depth = model.infer_image(bgr, inst) if getattr(model, "needs_instances", False) else model.infer_image(bgr)
            h, w = bgr.shape[:2]
            intrinsics = (args.fx, args.fy, w / 2, h / 2)

        objects, pts, (v, u) = scene_objects(depth, inst, info, intrinsics, args.min_points)
        labels = label_colors(inst, [o["id"] for o in objects])
        stem = Path(name).stem
        write_ply(out / f"{stem}_rgb.ply", pts, bgr[v, u])
        write_ply(out / f"{stem}_labels.ply", pts, labels[v, u])
        (out / f"{stem}_objects.json").write_text(json.dumps(
            {"image": name, "ckpt": args.ckpt or "zero-shot", "intrinsics": intrinsics, "objects": objects}, indent=1))
        cv2.imwrite(str(out / f"{stem}_overview.png"), overview(bgr, depth, inst, objects, labels))
        summary = ", ".join(f"{o['class']} {o['distance_m']:.1f} m" for o in sorted(objects, key=lambda o: o["distance_m"]))
        print(f"{stem}: {len(objects)} objects ({summary}) -> {out}")


if __name__ == "__main__":
    main()
