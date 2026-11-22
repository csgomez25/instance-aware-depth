"""Make the scene-disjoint validation split from the official NYU train set.

    python tools/make_val_split.py [--target 90] [--seed 0]

Reads data/splits/nyu_train_full.txt (795 ids) and $DATA_ROOT/nyu/meta.csv, moves whole
scenes (chosen at random with a fixed seed) into validation until it holds about
--target images, and writes data/splits/nyu_train.txt and nyu_val.txt. Deterministic:
the same seed always gives the same split. The outputs are committed to git.
"""
import argparse
import csv
import random
from collections import Counter, defaultdict

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train_full = C.read_split("train_full")
    test = set(C.read_split("test"))
    with open(C.NYU_DIR / "meta.csv") as fh:
        meta = {r["id"]: r for r in csv.DictReader(fh)}

    by_scene = defaultdict(list)
    for i in train_full:
        by_scene[meta[i]["scene"]].append(i)

    test_scenes = {meta[i]["scene"] for i in test}
    assert not test_scenes & set(by_scene), "official train/test splits share a scene"

    scenes = sorted(by_scene)
    random.Random(args.seed).shuffle(scenes)
    val = []
    for s in scenes:
        if len(val) >= args.target:
            break
        val += by_scene[s]
    val_set = set(val)
    train = [i for i in train_full if i not in val_set]

    (C.SPLITS / "nyu_val.txt").write_text("\n".join(sorted(val)) + "\n")
    (C.SPLITS / "nyu_train.txt").write_text("\n".join(train) + "\n")

    val_scenes = {meta[i]["scene"] for i in val}
    print(f"train {len(train)} | val {len(val)} images from {len(val_scenes)} scenes | test {len(test)}")
    print("val scene types:", dict(Counter(meta[i]["scene_type"] for i in val).most_common()))


if __name__ == "__main__":
    main()
