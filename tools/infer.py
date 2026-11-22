"""Run relative + metric depth on images; save depth visualisations and coloured .ply point clouds.

    python tools/infer.py IMG [IMG ...] [--fx F --fy F]   # default focal length: NYU Kinect
"""
import argparse
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import open3d as o3d

import common as C


def colorize(d):
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return (matplotlib.colormaps["Spectral_r"](d)[..., :3][..., ::-1] * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--encoder", default="vits")
    ap.add_argument("--fx", type=float, default=C.NYU_FX)
    ap.add_argument("--fy", type=float, default=C.NYU_FY)
    ap.add_argument("--outdir", default=str(C.RESULTS / "infer"))
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    metric = C.load_metric(args.encoder)

    for p in args.images:
        bgr = cv2.imread(p)
        # NYU labeled images: drop the white border and shift the principal point to match.
        is_nyu = bgr.shape[:2] == (C.NYU_H, C.NYU_W)
        if is_nyu:
            t, b, l, r = C.NYU_VALID_CROP
            bgr = bgr[t:b, l:r]
        h, w = bgr.shape[:2]
        t0 = time.perf_counter()
        depth = metric.infer_image(bgr)
        dt = time.perf_counter() - t0

        stem = Path(p).stem
        np.save(out / f"{stem}_depth.npy", depth)
        cv2.imwrite(str(out / f"{stem}_depth.png"), np.hstack([bgr, colorize(-depth)]))

        # Principal point: NYU's (shifted by the border crop) for NYU images, else the image centre.
        cx, cy = (C.NYU_CX - l, C.NYU_CY - t) if is_nyu else (w / 2, h / 2)
        pts = C.depth_to_points(depth, args.fx, args.fy, cx, cy)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        o3d.io.write_point_cloud(str(out / f"{stem}.ply"), pcd)
        print(f"{stem}: {w}x{h}, depth {depth.min():.2f}-{depth.max():.2f} m, {dt*1000:.0f} ms -> {out}")


if __name__ == "__main__":
    main()
