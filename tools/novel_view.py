"""Novel-view renders of single-image point clouds: what flying pixels look like once the camera moves.

    python novel_view.py 000500 001123                       # NYU ids -> one grid per image
    python novel_view.py --pick 4 --split val                # images where the edge loss changes flying most
    python novel_view.py 000500 --orbit                      # also a side-by-side turntable video for slides

Rows are scenes from GT filled depth, the zero-shot model, the control (SiLog) and the method (SiLog + edge);
columns are virtual cameras rotated about the scene centre (--views "yaw,pitch" in degrees). Points that the
flying metric counts as predicted inside a depth gap are drawn red, so the smear between foreground and
background is visible. Points are splatted with a z-buffer into a frame the size of the photo, NYU intrinsics.
Writes results/novel_view/<id>_views.jpg (and <id>_orbit.mp4).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import common as C
from boundary import flying_maps, median_scaled
from eval_nyu import eval_mask, model_masks

MODELS = {"zero-shot": None,
          "control": "h2b_edge_yolo_l0_laptop",
          "method": "h2b_edge_yolo_l1.0_laptop"}
FLYING_BGR = np.array([40, 40, 230], np.uint8)
K = (C.NYU_FX, C.NYU_FY, C.NYU_CX, C.NYU_CY)


def backproject(depth, mask):
    fx, fy, cx, cy = K
    v, u = np.nonzero(mask)
    z = depth[v, u]
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], 1), (v, u)


def rotation(yaw, pitch):
    a, b = np.radians(yaw), np.radians(pitch)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return rx @ ry


def render(pts, bgr, pivot, yaw, pitch, shape, splat=2):
    """Orbit the camera about `pivot` by yaw/pitch and z-buffer splat the coloured points into an HxW image."""
    h, w = shape
    fx, fy, cx, cy = K
    p = (pts - pivot) @ rotation(yaw, pitch).T + pivot
    keep = p[:, 2] > 0.1
    p, col = p[keep], bgr[keep]
    u = np.round(p[:, 0] * fx / p[:, 2] + cx).astype(int)
    v = np.round(p[:, 1] * fy / p[:, 2] + cy).astype(int)
    offs = [(dv, du) for dv in range(-splat, splat + 1) for du in range(-splat, splat + 1)]
    uu = np.concatenate([u + du for _, du in offs])
    vv = np.concatenate([v + dv for dv, _ in offs])
    zz, cc = np.tile(p[:, 2], len(offs)), np.tile(col, (len(offs), 1))
    inside = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
    idx, zz, cc = (vv * w + uu)[inside], zz[inside], cc[inside]
    order = np.argsort(zz, kind="stable")
    first = np.unique(idx[order], return_index=True)[1]  # nearest point per pixel
    img = np.full((h * w, 3), 255, np.uint8)
    img[idx[order][first]] = cc[order][first]
    return img.reshape(h, w, 3)


def scene_clouds(img_id, models):
    """{row name: (points, colours, flying fraction)} for GT and every model, all on the same valid pixels."""
    bgr = cv2.imread(str(C.nyu_rgb_path(img_id)))
    gt = C.read_nyu_depth(img_id, "filled")
    valid = eval_mask(gt)
    pts, (v, u) = backproject(gt, valid)
    rows = {"GT (filled)": (pts, bgr[v, u], None)}
    for name, model in models.items():
        pred = C.predict_nyu(model, bgr, model_masks(model, img_id, "yolo11s-seg"))
        band, fly = flying_maps(*median_scaled(np.clip(pred, 1e-3, 10), gt, valid), valid)
        pts, (v, u) = backproject(pred, valid)
        col = bgr[v, u].copy()
        col[fly[v, u]] = FLYING_BGR
        rows[name] = (pts, col, fly.sum() / max(band.sum(), 1))
    return rows, gt[valid]


def label(img, text):
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def grid(rows, pivot, views, shape):
    out = []
    for name, (pts, col, fly) in rows.items():
        tiles = []
        for yaw, pitch in views:
            text = f"{name}  yaw {yaw:+.0f} pitch {pitch:+.0f}"
            if fly is not None and (yaw, pitch) == views[0]:
                text += f"  flying {fly:.2f}"
            tiles.append(label(render(pts, col, pivot, yaw, pitch, shape), text))
        out.append(np.hstack(tiles))
    return np.vstack(out)


def orbit(rows, pivot, shape, path, names=("control", "method"), frames=90, amplitude=50):
    """Side-by-side turntable: yaw swings -amplitude..+amplitude and back."""
    h, w = shape
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w * len(names), h))
    for t in range(frames):
        yaw = amplitude * np.sin(2 * np.pi * t / frames)
        vw.write(np.hstack([label(render(*rows[n][:2], pivot, yaw, 10, shape), n) for n in names]))
    vw.release()


def pick(ids, models, n):
    """Images where the method's flying fraction drops most below the control's."""
    gain = []
    for i in tqdm(ids, desc="pick"):
        bgr, gt = cv2.imread(str(C.nyu_rgb_path(i))), C.read_nyu_depth(i, "filled")
        valid = eval_mask(gt)
        fly = {}
        for name in ("control", "method"):
            m = models[name]
            pred = C.predict_nyu(m, bgr, model_masks(m, i, "yolo11s-seg"))
            band, f = flying_maps(*median_scaled(np.clip(pred, 1e-3, 10), gt, valid), valid)
            fly[name] = f.sum() / max(band.sum(), 1)
        gain.append(fly["control"] - fly["method"])
    return [ids[k] for k in np.argsort(gain)[::-1][:n]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="NYU image ids")
    ap.add_argument("--pick", type=int, default=0, help="also render the N images where the edge loss helps most")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--views", nargs="+", default=["0,0", "35,0", "-25,0", "0,50"], help="yaw,pitch in degrees")
    ap.add_argument("--orbit", action="store_true", help="also write a control-vs-method turntable mp4")
    ap.add_argument("--outdir", default=str(C.RESULTS / "novel_view"))
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    models = {n: C.load_metric(ckpt=None if r is None else C.RUNS / r / "best.pth") for n, r in MODELS.items()}
    ids = list(args.images)
    if args.pick:
        picked = pick(C.read_split(args.split), models, args.pick)
        print("picked:", " ".join(picked))
        ids += [i for i in picked if i not in ids]
    views = [tuple(float(x) for x in v.split(",")) for v in args.views]
    for i in ids:
        rows, gt_valid = scene_clouds(i, models)
        pivot = np.array([0.0, 0.0, float(np.median(gt_valid))])
        shape = (C.NYU_H, C.NYU_W)
        cv2.imwrite(str(out / f"{i}_views.jpg"), grid(rows, pivot, views, shape), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if args.orbit:
            orbit(rows, pivot, shape, out / f"{i}_orbit.mp4")
        fly = ", ".join(f"{n} {r[2]:.3f}" for n, r in rows.items() if r[2] is not None)
        print(f"{i}: flying {fly} -> {out}")


if __name__ == "__main__":
    main()
