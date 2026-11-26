"""H6 final table: mean ± std over 3 seeds, plus a paired per-image test of method vs control.

    python h6_table.py          # best-val-epoch checkpoints (runs/h6_queue.sh) -> results/h6_test.csv
    python h6_table.py --final  # last-epoch checkpoints of the H7 reruns (runs/h7_queue.sh) -> results/h7_test.csv

Each run's score is its mean over the images. A group's score is the mean (± std) of its runs' scores. The p-value
is a two-sided Wilcoxon signed-rank test over images, on each image's score averaged over the 3 seeds. It ignores
seed-to-seed variance, so trust a difference only when it is also larger than the seed std.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import common as C

GROUPS = {"h6": {"zero-shot": ["h6_zero-shot"],
                 "control": ["h6_h2b_edge_yolo_l0_laptop", "h6_h5_edge_yolo_l0_s1_laptop", "h6_h5_edge_yolo_l0_s2_laptop"],
                 "method": ["h6_h2b_edge_yolo_l1.0_laptop", "h6_h5_edge_yolo_l1.0_s1_laptop",
                            "h6_h5_edge_yolo_l1.0_s2_laptop"]},
          "h7": {"zero-shot": ["h6_zero-shot"],
                 "control": [f"h7_edge_yolo_l0_s{s}_laptop_final" for s in range(3)],
                 "method": [f"h7_edge_yolo_l1.0_s{s}_laptop_final" for s in range(3)]}}
# Boundary metrics always use filled GT (eval_nyu.py), so the raw-GT row only adds the depth metrics.
BENCHMARKS = {"NYU test, filled GT": ("nyu", "_test_filled_per_image.csv", ["abs_rel", "rmse", "d1", "bf1", "bf1_obj", "flying"]),
              "NYU test, raw GT": ("nyu", "_test_raw_per_image.csv", ["abs_rel", "rmse", "d1"]),
              "iBims-1": ("ibims", "_per_image.csv",
                          ["abs_rel", "rmse", "d1", "sa_abs_rel", "dbe_acc", "dbe_com", "bf1", "flying"])}
LOWER_IS_BETTER = {"abs_rel", "rmse", "sa_abs_rel", "dbe_acc", "dbe_com", "flying"}


def per_image(sub, suffix, tag):
    df = pd.read_csv(C.RESULTS / sub / f"{tag}{suffix}")
    return df.set_index(df.columns[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", action="store_true", help="H7 last-epoch checkpoints instead of H6 best-val ones")
    name = "h7" if ap.parse_args().final else "h6"
    groups = GROUPS[name]
    rows = []
    for bench, (sub, suffix, metrics) in BENCHMARKS.items():
        data = {g: [per_image(sub, suffix, r) for r in runs] for g, runs in groups.items()}
        for m in metrics:
            row = {"benchmark": bench, "metric": m, "better": "lower" if m in LOWER_IS_BETTER else "higher"}
            for g, dfs in data.items():
                scores = [d[m].mean() for d in dfs]
                row[g] = np.mean(scores)
                row[f"{g}_std"] = np.std(scores, ddof=1) if len(scores) > 1 else np.nan
            ctrl = pd.concat([d[m] for d in data["control"]], axis=1).mean(axis=1)
            meth = pd.concat([d[m] for d in data["method"]], axis=1).mean(axis=1)
            both = pd.concat([ctrl, meth], axis=1).dropna()
            row["p_method_vs_control"] = wilcoxon(both.iloc[:, 0], both.iloc[:, 1]).pvalue
            rows.append(row)
    df = pd.DataFrame(rows)
    out = C.RESULTS / f"{name}_test.csv"
    df.round(4).to_csv(out, index=False)

    show = df[["benchmark", "metric", "better"]].copy()
    for g in groups:
        show[g] = [f"{v:.4f}" if np.isnan(s) else f"{v:.4f} ± {s:.4f}" for v, s in zip(df[g], df[f"{g}_std"])]
    show["p"] = df.p_method_vs_control.map(lambda p: f"{p:.2g}")
    print(show.to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
