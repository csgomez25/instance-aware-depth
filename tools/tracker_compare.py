"""Compare person trackers on the TUM RGB-D sequences: how many IDs do the 2 actors end up with?

    python tracker_compare.py                     # all trackers x walking_xyz, walking_halfsphere, sitting_xyz

Each sequence shows the same 2 people, so the ideal tracker gives 2 IDs. Without identity labels we report
    ids            distinct track IDs (lower is better, 2 is ideal)
    top2_share     share of person detections carried by the 2 longest tracks (higher is better, 1 is ideal)
    mean_track_s   mean track length in seconds (30 fps)
A tracker can also lower the ID count by wrongly giving one person's ID to the other; check the videos
(track_demo.py --tracker ...) before trusting a low count. Writes results/tracking/trackers.csv.
"""
import argparse

import cv2
import pandas as pd
from tqdm import tqdm

import common as C
from make_masks import load_yolo
from track_demo import TRACKERS, TUM, frames

SEQUENCES = ["walking_xyz", "walking_halfsphere", "sitting_xyz"]


def track_ids(seq, tracker):
    """Per-frame track IDs of the people in a sequence."""
    yolo = load_yolo()
    ids = []
    for t, (rgb, _) in enumerate(tqdm(frames(TUM / f"rgbd_dataset_freiburg3_{seq}"), desc=f"{seq} {tracker}")):
        r = yolo.track(cv2.imread(str(rgb)), persist=True, classes=[0], conf=0.3, verbose=False,
                       tracker=TRACKERS[tracker], device=0 if C.DEVICE == "cuda" else "cpu")[0]
        if r.boxes.id is not None:
            ids += [(t, int(i)) for i in r.boxes.id.int().tolist()]
    return pd.DataFrame(ids, columns=["frame", "id"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trackers", nargs="+", default=list(TRACKERS), choices=list(TRACKERS))
    ap.add_argument("--sequences", nargs="+", default=SEQUENCES)
    args = ap.parse_args()
    rows = []
    for seq in args.sequences:
        for tr in args.trackers:
            df = track_ids(seq, tr)
            lengths = df.groupby("id").size().sort_values(ascending=False)
            rows.append({"sequence": seq, "tracker": tr, "ids": len(lengths),
                         "top2_share": round(lengths.iloc[:2].sum() / len(df), 3),
                         "mean_track_s": round(lengths.mean() / 30, 1)})
            print(rows[-1])
    out = C.RESULTS / "tracking" / "trackers.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
