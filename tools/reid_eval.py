"""Identity-level tracker evaluation on TUM RGB-D: do the trackers keep the two actors apart?

    ~/venvs/reid/bin/python reid_eval.py detect      # cache YOLO people + safe tracklets, write crop sheets
    (hand-label the sheets into results/tracking/identity_labels.csv: sequence,tracklet,person)
    ~/venvs/reid/bin/python reid_eval.py score       # every tracker vs the labels -> results/tracking/identity.csv

Runs in a separate venv with boxmot (pip install boxmot==26.0.0 into a --system-site-packages venv of the main
env) so the main environment stays as in the paper. All trackers get the same cached YOLO11s-seg person boxes
(conf 0.3, as in track_demo.py), so they differ only in association.

Ground truth: detections are first linked into tracklets by a cautious IoU rule (link to the previous frame only
when the match is mutual, IoU > 0.5 and no rival above 0.3), so a tracklet should hold one person. Tracklets of at
least MIN_LEN frames are labelled by hand (A, B, or x for a false or partial detection) from crop sheets showing
their first, middle and last crop; shorter tracklets are left out of the scores. Per tracker:
    ids        track IDs used on labelled people (2 is ideal)
    idf1       identity F1 (Ristani et al. 2016) of the best one-to-one match of tracks to people
    switches   times a person's track ID changes between consecutive tracked frames
    impure     share of labelled detections in a track whose majority person is someone else (wrong merges)
    coverage   share of labelled detections that got a track
"""
import argparse

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

import common as C
from make_masks import load_yolo
from track_demo import TRACKERS, TUM, frames, osnet_tracker

SEQUENCES = ["walking_xyz", "walking_halfsphere", "sitting_xyz"]
CACHE = C.RUNS / "reid"
OUT = C.RESULTS / "tracking"
LABELS = OUT / "identity_labels.csv"
MIN_LEN = 10
ULTRALYTICS = ["bytetrack", "botsort", "botsort+reid long"]  # track_demo.py / tracker_compare.py trackers


def iou(a, b):
    """Pairwise IoU of (n, 4) and (m, 4) xyxy boxes."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], axis=1)
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def safe_tracklets(dets):
    """Tracklet index per detection; links only unambiguous frame-to-frame matches."""
    out, prev, nxt = [], None, 0
    for d in dets:
        ids = np.full(len(d), -1)
        if prev is not None and len(d) and len(prev[0]):
            m = iou(d[:, :4], prev[0][:, :4])
            for i in range(len(d)):
                j = m[i].argmax()
                rivals = np.r_[np.delete(m[i], j), np.delete(m[:, j], i)]
                if m[i, j] > 0.5 and m[:, j].argmax() == i and (rivals.size == 0 or rivals.max() < 0.3):
                    ids[i] = prev[1][j]
        for i in np.nonzero(ids < 0)[0]:
            ids[i], nxt = nxt, nxt + 1
        out.append(ids)
        prev = (d, ids)
    return out


def detect(seq):
    yolo = load_yolo()
    dets = []
    for rgb, _ in tqdm(frames(TUM / f"rgbd_dataset_freiburg3_{seq}"), desc=seq):
        r = yolo.predict(cv2.imread(str(rgb)), classes=[0], conf=0.3, verbose=False,
                         device=0 if C.DEVICE == "cuda" else "cpu")[0]
        dets.append(np.c_[r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), np.zeros(len(r.boxes))])
    rows = [(f, i, k, *d[i, :5]) for f, (d, ks) in enumerate(zip(dets, safe_tracklets(dets)))
            for i, k in enumerate(ks)]
    df = pd.DataFrame(rows, columns=["frame", "det", "tracklet", "x1", "y1", "x2", "y2", "conf"])
    df.to_csv(CACHE / f"{seq}_dets.csv", index=False)
    return df


def sheet(seq, df, n=10):
    """One row per tracklet of >= MIN_LEN frames: n evenly spaced crops, each with its frame number."""
    paths = frames(TUM / f"rgbd_dataset_freiburg3_{seq}")
    groups = [g for _, g in df.groupby("tracklet") if len(g) >= MIN_LEN]
    rows = []
    for g in groups:
        crops = []
        for _, r in g.iloc[np.linspace(0, len(g) - 1, n).round().astype(int)].iterrows():
            img = cv2.imread(str(paths[int(r.frame)][0]))
            x1, y1, x2, y2 = (int(v) for v in (r.x1, r.y1, r.x2, r.y2))
            crop = cv2.copyMakeBorder(cv2.resize(img[max(y1, 0):y2, max(x1, 0):x2], (60, 120)), 14, 0, 1, 1,
                                      cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(crop, str(int(r.frame)), (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
            crops.append(crop)
        label = np.full((134, 90, 3), 255, np.uint8)
        cv2.putText(label, f"#{int(g.tracklet.iloc[0])}", (4, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(label, f"{len(g)} fr", (4, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        rows.append(np.hstack([label] + crops))
    cv2.imwrite(str(CACHE / f"{seq}_tracklets.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{seq}: {len(groups)} tracklets >= {MIN_LEN} frames, "
          f"{sum(len(g) for g in groups) / len(df):.0%} of detections -> {CACHE / f'{seq}_tracklets.jpg'}")


def main_sheets(seqs):
    for seq in seqs:
        sheet(seq, pd.read_csv(CACHE / f"{seq}_dets.csv"))


def trackers():
    """name -> factory of a fresh boxmot tracker."""
    from boxmot import ReIDConfig, create_tracker
    osnet = lambda: ReIDConfig("osnet_x1_0_msmt17", device="cuda:0", precision="fp16")
    return {"bytetrack": lambda: create_tracker("bytetrack"),
            "botsort": lambda: create_tracker("botsort", use_embeddings=False),
            "botsort+osnet": lambda: create_tracker("botsort", reid=osnet()),
            "botsort+osnet long": lambda: create_tracker("botsort", reid=osnet(), max_lost_frames=300),
            # appearance matching without the IoU gate, so people who leave view are re-identified
            **{f"botsort+osnet reid {a}": (lambda a=a: osnet_tracker(a)) for a in (0.25, 0.35, 0.45)},
            "deepocsort+osnet": lambda: create_tracker("deepocsort", reid=osnet()),
            "strongsort+osnet": lambda: create_tracker("strongsort", reid=osnet()),
            **{f"ultralytics {n}": n for n in ULTRALYTICS}}


def run_ultralytics(tracker, seq, df):
    """Ultralytics tracker (track_demo.py) on its own YOLO pass; output boxes matched to cached ones by IoU > 0.8."""
    yolo, ids = load_yolo(), np.full(len(df), -1)
    by_frame = dict(list(df.groupby("frame")))
    for f, (rgb, _) in enumerate(tqdm(frames(TUM / f"rgbd_dataset_freiburg3_{seq}"), leave=False)):
        r = yolo.track(cv2.imread(str(rgb)), persist=True, classes=[0], conf=0.3, verbose=False,
                       tracker=TRACKERS[tracker], device=0 if C.DEVICE == "cuda" else "cpu")[0]
        g = by_frame.get(f)
        if g is None or r.boxes.id is None:
            continue
        m = iou(r.boxes.xyxy.cpu().numpy(), g[["x1", "y1", "x2", "y2"]].to_numpy())
        for i, tid in enumerate(r.boxes.id.int().tolist()):
            if m[i].max() > 0.8:
                ids[g.index[m[i].argmax()]] = tid
    return ids


def run_tracker(make, seq, df):
    """Track ID per cached detection (-1 if the tracker did not output it)."""
    if isinstance(make, str):
        return run_ultralytics(make, seq, df)
    tracker, ids = make(), np.full(len(df), -1)
    by_frame = dict(list(df.groupby("frame")))
    for f, (rgb, _) in enumerate(tqdm(frames(TUM / f"rgbd_dataset_freiburg3_{seq}"), leave=False)):
        g = by_frame.get(f)
        det = (g[["x1", "y1", "x2", "y2", "conf"]].to_numpy(np.float32) if g is not None
               else np.empty((0, 5), np.float32))
        out = tracker.update(np.c_[det, np.zeros(len(det), np.float32)], cv2.imread(str(rgb)))
        if g is not None and len(out):
            ids[g.index[out[:, 7].astype(int)]] = out[:, 4].astype(int)
    return ids


def identity_scores(person, track, frame):
    """Scores for labelled detections: person label, track ID (-1 = untracked) and frame per detection."""
    tracked = track >= 0
    ct = pd.crosstab(track[tracked], person[tracked])
    r, c = linear_sum_assignment(-ct.to_numpy())
    idtp = ct.to_numpy()[r, c].sum()
    switches = 0
    for p in np.unique(person):
        seq = track[(person == p) & tracked][np.argsort(frame[(person == p) & tracked], kind="stable")]
        switches += int((seq[1:] != seq[:-1]).sum())
    return {"ids": len(ct), "idf1": round(float(2 * idtp / (tracked.sum() + len(person))), 3), "switches": switches,
            "impure": round(float(1 - ct.max(axis=1).sum() / tracked.sum()), 3),
            "coverage": round(float(tracked.mean()), 3)}


def score(names, sequences):
    labels = pd.read_csv(LABELS)
    makers, rows = trackers(), []
    for seq in [s for s in sequences if s in set(labels.sequence)]:
        df = pd.read_csv(CACHE / f"{seq}_dets.csv")
        lab = labels[(labels.sequence == seq) & labels.person.isin(["A", "B"])].set_index("tracklet").person
        keep = df.tracklet.isin(lab.index).to_numpy()
        for name in names:
            ids = run_tracker(makers[name], seq, df)
            rows.append({"sequence": seq, "tracker": name, "labelled_dets": int(keep.sum()),
                         **identity_scores(df.tracklet[keep].map(lab).to_numpy(), ids[keep],
                                           df.frame[keep].to_numpy())})
            print(rows[-1])
    out, path = pd.DataFrame(rows), OUT / "identity.csv"
    if path.exists():  # replace only the (sequence, tracker) rows scored now
        old = pd.read_csv(path)
        key = lambda d: d.sequence + "|" + d.tracker
        out = pd.concat([old[~key(old).isin(key(out))], out], ignore_index=True)
    out.to_csv(path, index=False)
    print(out.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["detect", "sheets", "score"])
    ap.add_argument("--sequences", nargs="+", default=SEQUENCES)
    ap.add_argument("--trackers", nargs="+", default=None)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.step == "detect":
        for seq in args.sequences:
            sheet(seq, detect(seq))
    elif args.step == "sheets":
        main_sheets(args.sequences)
    else:
        score(args.trackers or list(trackers()), args.sequences)


if __name__ == "__main__":
    main()
