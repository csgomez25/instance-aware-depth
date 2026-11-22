"""Fine-tune Depth Anything V2 (metric, indoor) on NYU Depth V2 with one GPU.

    python train_nyu.py --name b1_silog_filled [--train-gt filled|raw] [--epochs 20] [--bs 4] [--lr 5e-6]

Recipe follows the official metric_depth/train.py: AdamW (decoder lr = 10x encoder lr,
weight decay 0.01), poly learning-rate decay (power 0.9) per step, 518x518 random crops,
random horizontal flips. Differences, all for running on one 6-16 GB GPU: fp16 autocast
with a gradient scaler (the loss is computed in fp32), and memory-efficient attention.

Each run lives in $RUNS_DIR/<name>/:
    config.json   arguments
    log.csv       one row per epoch: train loss, lr, time, val metrics on filled and raw depth
    best.pth      weights with the lowest val AbsRel (filled depth); use with eval_nyu.py --ckpt
    last.pth      weights + optimizer state for resuming (deleted when the run finishes)
    done.json     written when the run finishes: best epoch and val AbsRel
Re-running the same command resumes from last.pth, so a Colab disconnect costs at most one epoch.
Model selection uses the val split only; the test split is never touched here.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import common as C
from attention import use_memory_efficient_attention
from eval_nyu import evaluate
from losses import build_loss
from nyu_data import NYUDepth

# Arguments that change the experiment; resuming with different values is refused.
EXPERIMENT_KEYS = ["encoder", "train_gt", "epochs", "bs", "lr", "wd", "size", "min_depth", "max_depth",
                   "model_max_depth", "seed", "masks", "amp", "train_split", "val_split"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run folder name under $RUNS_DIR")
    ap.add_argument("--encoder", default="vits")
    ap.add_argument("--train-gt", default="filled", choices=["filled", "raw"], help="depth used as the loss target")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6, help="encoder lr; decoder uses 10x")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--size", type=int, default=518, help="training crop (DA-V2 native: 518)")
    ap.add_argument("--min-depth", type=float, default=1e-3)
    ap.add_argument("--max-depth", type=float, default=10.0, help="loss ignores depth outside min..max (NYU: 10 m)")
    ap.add_argument("--model-max-depth", type=float, default=20.0, help="output range of the pretrained head (Hypersim: 20 m)")
    ap.add_argument("--masks", default=None, help="None, 'gt', or a mask folder such as 'yolo11s-seg'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", dest="amp", action="store_false", help="train in fp32")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--max-steps", type=int, default=None, help="stop each epoch after N steps (smoke tests)")
    return ap.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_atomic(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)  # never leaves a half-written checkpoint (matters on Google Drive)


def main():
    args = parse_args()
    run = C.RUNS / args.name
    run.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k not in ("workers", "max_steps")}
    cfg_path = run / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        changed = {k: (old.get(k), config[k]) for k in EXPERIMENT_KEYS if old.get(k) != config[k]}
        if changed:
            raise SystemExit(f"{run} already holds a run with different settings {changed}; pick a new --name")
    cfg_path.write_text(json.dumps(config, indent=2))
    seed_everything(args.seed)

    model = C.load_metric(args.encoder, args.model_max_depth)
    use_memory_efficient_attention()
    loss_fn = build_loss(args)

    train_set = NYUDepth(args.train_split, size=args.size, masks=args.masks, augment=True)
    g = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, drop_last=True, generator=g,
                        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    val_ids = C.read_split(args.val_split)

    enc = [p for n, p in model.named_parameters() if "pretrained" in n]
    dec = [p for n, p in model.named_parameters() if "pretrained" not in n]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr}, {"params": dec, "lr": args.lr * 10}],
                            betas=(0.9, 0.999), weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    steps_per_epoch = min(len(loader), args.max_steps or len(loader))
    total_steps = args.epochs * steps_per_epoch

    start_epoch, best = 0, float("inf")
    last_path, log_path, done_path = run / "last.pth", run / "log.csv", run / "done.json"
    if done_path.exists():
        print(f"{args.name} already finished: {done_path.read_text().strip()}")
        return
    if last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"], ck["best"]
        torch.set_rng_state(ck["rng"]["torch"])
        np.random.set_state(ck["rng"]["numpy"])
        random.setstate(ck["rng"]["python"])
        g.set_state(ck["rng"]["loader"])
        print(f"Resuming {args.name} after epoch {start_epoch} (best val AbsRel {best:.4f})")
    if start_epoch >= args.epochs:
        print(f"{args.name} already finished {args.epochs} epochs; best checkpoint: {run / 'best.pth'}")
        return

    print(f"{args.name}: {len(train_set)} train / {len(val_ids)} val images, {steps_per_epoch} steps/epoch, "
          f"bs {args.bs}, lr {args.lr:g}, target {args.train_gt}, amp {args.amp}, device {C.DEVICE}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        torch.cuda.reset_peak_memory_stats()
        t0, losses, skipped = time.time(), [], 0
        for i, batch in enumerate(loader):
            if i >= steps_per_epoch:
                break
            step = epoch * steps_per_epoch + i
            lr = args.lr * (1 - step / total_steps) ** 0.9
            opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = lr, lr * 10

            batch = {k: v.to(C.DEVICE, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                pred = model(batch["image"])
            loss, parts = loss_fn(pred, batch)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        train_time = time.time() - t0

        model.eval()
        t1 = time.time()
        val = evaluate(model, val_ids)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)) if losses else float("nan"),
               "lr": lr, "skipped_steps": skipped, "train_s": round(train_time, 1),
               "val_s": round(time.time() - t1, 1),
               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if C.DEVICE == "cuda" else 0}
        for gname, m in val.items():
            row.update({f"val_{gname}_{k}": round(v, 4) for k, v in m.items()})
        pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

        score = val["filled"]["abs_rel"]
        if score < best:
            best = score
            save_atomic({"model": model.state_dict(), "epoch": epoch + 1, "val": val}, run / "best.pth")
        save_atomic({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": epoch + 1, "best": best,
                     "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                             "python": random.getstate(), "loader": g.get_state()}}, last_path)
        print(f"epoch {epoch + 1}/{args.epochs}  loss {row['train_loss']:.4f}  "
              f"val AbsRel {score:.4f} (raw {val['raw']['abs_rel']:.4f})  d1 {val['filled']['d1']:.4f}  "
              f"best {best:.4f}  [{train_time:.0f}s train, {row['val_s']:.0f}s val, {row['peak_mem_gb']} GB]")

    # Finished: the resume state (~300 MB with optimizer) is no longer needed; keep best.pth + log.
    best_epoch = int(torch.load(run / "best.pth", map_location="cpu", weights_only=False)["epoch"])
    done_path.write_text(json.dumps({"epochs": args.epochs, "best_epoch": best_epoch, "best_val_abs_rel": round(best, 4)}))
    last_path.unlink()
    print(f"Done. Best val AbsRel {best:.4f} at epoch {best_epoch} -> {run / 'best.pth'}")


if __name__ == "__main__":
    main()
