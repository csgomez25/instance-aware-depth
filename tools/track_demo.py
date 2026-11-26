"""People in 3D from one RGB camera: YOLO tracking + metric depth on TUM RGB-D sequences with walking people.

    python track_demo.py walking_xyz                     # demo video (method model) + people depth scores
    python track_demo.py walking_xyz walking_halfsphere sitting_xyz --stride 2

Per frame, YOLO11s-seg + ByteTrack find people and keep an ID per person. Each person's distance is the median
predicted depth inside their mask; their 3D position comes from the mask centre and that distance. The Kinect depth
of TUM is ground truth: scored per model (zero-shot, control, method; last-epoch H7 seed-0 weights) are
    abs_rel         whole frame, metric (no scale alignment), Kinect depth 0.1-10 m
    person_abs_rel  the same over people's pixels only
    dist_err_m      |predicted - Kinect| median distance per person per frame, metres
    person_flying   flying fraction (boundary.py) in the band around people's outlines
Writes results/tracking/<seq>_<video model>.mp4 (photo with IDs | predicted depth | top-down map: filled dots =
predicted positions with trails, rings = Kinect) and appends results/tracking/tum_people.csv.
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm

import common as C
from boundary import FLY_WINDOW, flying_maps
from make_masks import load_yolo

TUM = C.DATA / "tum"
TUM_FR3 = (535.4, 539.2, 320.1, 247.6)  # fx, fy, cx, cy of the freiburg3 Kinect
TUM_DEPTH_SCALE = 5000.0
MODELS = {"zero-shot": None,
          "control": "h7_edge_yolo_l0_s0_laptop/final.pth",
          "method": "h7_edge_yolo_l1.0_s0_laptop/final.pth"}
PALETTE = (np.array(matplotlib.colormaps["tab10"].colors)[:, ::-1] * 255).astype(int).tolist()  # BGR
MAP_SIZE, MAP_RANGE = 480, 6.0  # top-down panel: px, metres shown in x (-3..3) and z (0..6)


def frames(seq_dir, max_dt=0.02):
    """(rgb path, depth path) pairs matched by nearest timestamp; frames without depth within max_dt are dropped."""
    read = lambda f: [l.split() for l in (seq_dir / f).read_text().splitlines() if not l.startswith("#")]
    rgb, depth = read("rgb.txt"), read("depth.txt")
    dt = np.array([float(t) for t, _ in depth])
    out = []
    for t, f in rgb:
        k = np.abs(dt - float(t)).argmin()
        if abs(dt[k] - float(t)) < max_dt:
            out.append((seq_dir / f, seq_dir / depth[k][1]))
    return out


def people(yolo, bgr):
    """[(track id, HxW bool mask)] of the people YOLO tracks in this frame."""
    r = yolo.track(bgr, persist=True, classes=[0], conf=0.3, retina_masks=True, verbose=False,
                   tracker="bytetrack.yaml", device=0 if C.DEVICE == "cuda" else "cpu")[0]
    if r.masks is None or r.boxes.id is None:
        return []
    masks = r.masks.data.cpu().numpy() > 0.5
    return [(int(i), m) for i, m in zip(r.boxes.id.int().tolist(), masks)]


def position(mask, depth):
    """(x, z) in metres of a person: mask centre column back-projected at the median depth over the mask."""
    d = depth[mask & (depth > 0)]
    if d.size < 50:
        return None
    z = float(np.median(d))
    u = float(np.nonzero(mask)[1].mean())
    return (u - TUM_FR3[2]) * z / TUM_FR3[0], z


def score(pred, gt, persons):
    valid = (gt > 0.1) & (gt < 10)
    rel = np.abs(pred - gt) / np.where(valid, gt, 1)
    out = {"abs_rel": rel[valid].mean()}
    if persons:
        body = np.any([m for _, m in persons], axis=0)
        near = cv2.dilate(body.astype(np.uint8), np.ones((FLY_WINDOW, FLY_WINDOW), np.uint8)) > 0
        band, fly = flying_maps(pred, np.where(valid, gt, 1.0), valid)
        out["person_abs_rel"] = rel[valid & body].mean() if (valid & body).any() else np.nan
        out["person_flying"] = (fly & near).sum() / max((band & near).sum(), 1) if (band & near).any() else np.nan
        errs = [abs(p[1] - g[1]) for _, m in persons
                if (p := position(m, pred)) is not None and (g := position(m, gt)) is not None]
        out["dist_err_m"] = np.mean(errs) if errs else np.nan
    return out


def to_map(x, z):
    s = MAP_SIZE / MAP_RANGE
    return int(MAP_SIZE / 2 + x * s), int(MAP_SIZE - z * s)


def top_down(trails, current):
    """Bird's-eye panel: camera at the bottom centre, 1 m grid, people as dots with trails, Kinect as rings."""
    img = np.full((MAP_SIZE, MAP_SIZE, 3), 245, np.uint8)
    for m in range(1, int(MAP_RANGE)):
        cv2.line(img, to_map(-3, m), to_map(3, m), (220, 220, 220), 1)
        cv2.putText(img, f"{m} m", (4, to_map(0, m)[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    half_fov = np.arctan(TUM_FR3[2] / TUM_FR3[0])
    for sgn in (-1, 1):
        cv2.line(img, to_map(0, 0), to_map(sgn * MAP_RANGE * np.tan(half_fov), MAP_RANGE), (190, 190, 190), 1)
    for tid, pts in trails.items():
        col = PALETTE[tid % len(PALETTE)]
        for a, b in zip(pts[-60:], pts[-59:]):
            cv2.line(img, to_map(*a), to_map(*b), col, 2, cv2.LINE_AA)
    for tid, (p, g) in current.items():
        col = PALETTE[tid % len(PALETTE)]
        if g is not None:
            cv2.circle(img, to_map(*g), 9, col, 2, cv2.LINE_AA)
        cv2.circle(img, to_map(*p), 6, col, -1, cv2.LINE_AA)
        cv2.putText(img, f"ID {tid}", (to_map(*p)[0] + 9, to_map(*p)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    col, 1, cv2.LINE_AA)
    cv2.putText(img, "top view: dot = predicted, ring = Kinect", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (80, 80, 80), 1, cv2.LINE_AA)
    return img


def annotate(bgr, pred, persons, positions):
    photo = bgr.copy()
    for tid, m in persons:
        col = np.array(PALETTE[tid % len(PALETTE)], np.uint8)
        photo[m] = (0.5 * photo[m] + 0.5 * col).astype(np.uint8)
        if tid in positions:
            ys, xs = np.nonzero(m)
            text = f"ID {tid}  {positions[tid][0][1]:.1f} m"
            org = (int(xs.min()), max(int(ys.min()) - 6, 14))
            cv2.putText(photo, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(photo, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    norm = np.clip(pred / MAP_RANGE, 0, 1)
    depth = (matplotlib.colormaps["Spectral"](norm)[..., :3][..., ::-1] * 255).astype(np.uint8)
    return photo, depth


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequences", nargs="+", help="TUM freiburg3 sequence names, e.g. walking_xyz")
    ap.add_argument("--video-model", default="method", choices=list(MODELS))
    ap.add_argument("--stride", type=int, default=1, help="use every n-th frame")
    ap.add_argument("--limit", type=int, default=None, help="first n frames only (quick test)")
    ap.add_argument("--outdir", default=str(C.RESULTS / "tracking"))
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    models = {n: C.load_metric(ckpt=None if r is None else C.RUNS / r) for n, r in MODELS.items()}
    rows = []
    for seq in args.sequences:
        pairs = frames(TUM / f"rgbd_dataset_freiburg3_{seq}")[::args.stride][:args.limit]
        yolo = load_yolo()  # fresh tracker state per sequence
        video = cv2.VideoWriter(str(out / f"{seq}_{args.video_model}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                30 / args.stride, (2 * C.NYU_W + MAP_SIZE, C.NYU_H))
        scores = {n: [] for n in models}
        trails, ids = {}, set()
        for rgb_path, depth_path in tqdm(pairs, desc=seq):
            bgr = cv2.imread(str(rgb_path))
            gt = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / TUM_DEPTH_SCALE
            persons = people(yolo, bgr)
            ids |= {t for t, _ in persons}
            for name, model in models.items():
                pred = model.infer_image(bgr)
                scores[name].append(score(pred, gt, persons))
                if name == args.video_model:
                    current = {}
                    for tid, m in persons:
                        if (p := position(m, pred)) is not None:
                            current[tid] = (p, position(m, gt))
                            trails.setdefault(tid, []).append(p)
                    photo, depth = annotate(bgr, pred, persons, current)
                    video.write(np.hstack([photo, depth, top_down(trails, current)]))
        video.release()
        for name, s in scores.items():
            df = pd.DataFrame(s)
            rows.append({"sequence": seq, "model": name, "frames": len(df), "people_ids": len(ids),
                         "frames_with_people": int(df.get("dist_err_m", pd.Series(dtype=float)).notna().sum()),
                         **df.mean().round(4).to_dict()})
        print(pd.DataFrame(rows[-len(models):]).to_string(index=False))

    path = out / "tum_people.csv"
    df = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old[~old.set_index(["sequence", "model"]).index.isin(df.set_index(["sequence", "model"]).index)], df])
    df.to_csv(path, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
