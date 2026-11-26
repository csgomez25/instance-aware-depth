"""Weight-space blend of the zero-shot and a fine-tuned model (WiSE-FT, Wortsman et al., CVPR 2022).

    python wise_ft.py ../runs/h7_edge_yolo_l1.0_s0_laptop/final.pth --alphas 0.25 0.5 0.75

theta = (1 - alpha) * zero-shot + alpha * fine-tuned, so alpha 0 is the pretrained Hypersim model and alpha 1 the
fine-tuned one. Each blend is a new, separate model: runs/wise_<run>_a<alpha>/final.pth (+ source.json with the
inputs' sha256). Inputs are only read, an existing blend is never overwritten, and blends are made read-only.
Score a blend like any run: eval_nyu.py / eval_ibims.py --ckpt runs/wise_<run>_a<alpha>/final.pth
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

import common as C


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_state(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state.get("model", state)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("finetuned", help="fine-tuned checkpoint, e.g. ../runs/<run>/final.pth")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--zero-shot", default=str(C.CKPT / "depth_anything_v2_metric_hypersim_vits.pth"))
    args = ap.parse_args()

    ft_path, zs_path = Path(args.finetuned).resolve(), Path(args.zero_shot).resolve()
    ft, zs = load_state(ft_path), load_state(zs_path)
    if ft.keys() != zs.keys():
        raise SystemExit(f"architectures differ: {sorted(ft.keys() ^ zs.keys())[:5]}")
    run = ft_path.parent.name
    for a in args.alphas:
        out_dir = C.RUNS / f"wise_{run}_a{a:g}"
        out = out_dir / "final.pth"
        if out.exists():
            print(f"{out} exists, kept as is")
            continue
        out_dir.mkdir(parents=True)
        blend = {k: (1 - a) * zs[k].float() + a * ft[k].float() if zs[k].is_floating_point() else ft[k]
                 for k in ft}
        blend = {k: v.to(ft[k].dtype) for k, v in blend.items()}
        torch.save({"model": blend, "alpha": a}, out)
        (out_dir / "source.json").write_text(json.dumps(
            {"alpha": a, "formula": "(1 - alpha) * zero_shot + alpha * finetuned",
             "zero_shot": str(zs_path), "zero_shot_sha256": sha256(zs_path),
             "finetuned": str(ft_path), "finetuned_sha256": sha256(ft_path)}, indent=1))
        out.chmod(0o444)
        print(f"alpha {a:g} -> {out}")


if __name__ == "__main__":
    main()
