"""WiSE-FT table: zero-shot -> fine-tuned weight blends for the control and the method, 3 seeds each.

    python wise_table.py        # after runs/wise_queue.sh

alpha 0 is the zero-shot model, alpha 1 the H7 last-epoch weights, the rest are tools/wise_ft.py blends.
Scores are mean ± std over seeds on NYU val (to choose alpha), NYU test and iBims-1. The p-value compares
method and control at the same alpha (Wilcoxon over images, seed-averaged; read it with the seed std).
Writes results/wise_ft.csv.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import common as C

ALPHAS = [0, 0.25, 0.5, 0.75, 1]
LAMBDAS = {"control": "0", "method": "1.0"}
BENCHMARKS = {"NYU val": ("nyu", "_val_filled_per_image.csv", ["abs_rel", "d1", "bf1", "flying"]),
              "NYU test": ("nyu", "_test_filled_per_image.csv", ["abs_rel", "d1", "bf1", "bf1_obj", "flying"]),
              "iBims-1": ("ibims", "_per_image.csv", ["abs_rel", "sa_abs_rel", "d1", "dbe_acc", "bf1", "flying"])}
ZERO_SHOT = {"NYU val": "pretrained_hypersim_vits", "NYU test": "h6_zero-shot", "iBims-1": "h6_zero-shot"}


def tags(group, alpha, bench):
    if alpha == 0:
        return [ZERO_SHOT[bench]]
    run = lambda s: f"h7_edge_yolo_l{LAMBDAS[group]}_s{s}_laptop"
    return [f"{run(s)}_final" if alpha == 1 else f"wise_{run(s)}_a{alpha:g}" for s in range(3)]


def main():
    rows = []
    for bench, (sub, suffix, metrics) in BENCHMARKS.items():
        for a in ALPHAS:
            data = {g: [pd.read_csv(C.RESULTS / sub / f"{t}{suffix}").set_index("image") for t in tags(g, a, bench)]
                    for g in LAMBDAS}
            for m in metrics:
                row = {"benchmark": bench, "alpha": a, "metric": m}
                for g, dfs in data.items():
                    s = [d[m].mean() for d in dfs]
                    row[g], row[f"{g}_std"] = np.mean(s), (np.std(s, ddof=1) if len(s) > 1 else 0.0)
                if 0 < a:
                    avg = lambda g: pd.concat([d[m] for d in data[g]], axis=1).mean(axis=1)
                    both = pd.concat([avg("control"), avg("method")], axis=1).dropna()
                    row["p"] = wilcoxon(both.iloc[:, 0], both.iloc[:, 1]).pvalue
                rows.append(row)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(C.RESULTS / "wise_ft.csv", index=False)
    for bench in BENCHMARKS:
        sub = df[df.benchmark == bench]
        print(f"\n{bench}")
        for g in LAMBDAS:
            t = sub.pivot(index="alpha", columns="metric", values=g)
            sd = sub.pivot(index="alpha", columns="metric", values=f"{g}_std")
            print(f"  {g}\n" + (t.round(4).astype(str) + " ±" + sd.round(4).astype(str)).to_string())
    print(f"-> {C.RESULTS / 'wise_ft.csv'}")


if __name__ == "__main__":
    main()
