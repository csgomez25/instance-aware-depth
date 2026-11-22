"""Training losses. `build_loss(args)` returns loss(pred, batch) -> (total, {name: value}).

    total = SiLog(pred, target)
          + lambda_edge * EdgeStep(pred, raw depth, instances)
          + lambda_obj  * ObjectSI(pred, target, instances)

SiLog      the scale-invariant log loss of Depth Anything V2 (lambda = 0.5) on the training
           target (filled or raw depth), masked to min_depth..max_depth.
EdgeStep   (ours) across instance boundaries, the predicted log-depth step between two pixels
           must match the measured one. Pairs (p, p + d) along x and y, for each offset d, that
           lie in different instances; only pairs where *raw* depth is valid on both sides count,
           so the term never learns from hole-filled values. L1 on log differences.
ObjectSI   (ours) a fully scale-invariant log loss (lambda = 1: the standard deviation of the
           log error) inside each instance separately, averaged over instances with at least
           `min_obj_px` valid pixels, so each object's shape is right even when the scale is off.

SiLog with lambda = 0.5 is only partly scale-invariant; that is how the model learns metric
scale. EdgeStep and ObjectSI are fully scale-invariant (multiplying the prediction by any
constant leaves them unchanged), so they only shape the depth map and never fight SiLog over
global scale. Instance id 0 (background / unlabelled) is a region like any other for
EdgeStep, and is skipped by ObjectSI.
"""
import torch


def silog(pred, target, valid, lambd=0.5):
    d = torch.log(target[valid]) - torch.log(pred[valid])
    return torch.sqrt(torch.clamp((d ** 2).mean() - lambd * d.mean() ** 2, min=1e-12))


def edge_step(log_pred, log_raw, raw_valid, inst, offsets=(1, 3, 6)):
    """Mean |predicted log-step - measured log-step| over cross-instance pixel pairs."""
    total, count = log_pred.new_zeros(()), 0
    for d in offsets:
        for dim in (1, 2):  # y, x (tensors are B x H x W)
            a = slice(None, -d)
            b = slice(d, None)
            sl_a = (slice(None), a, slice(None)) if dim == 1 else (slice(None), slice(None), a)
            sl_b = (slice(None), b, slice(None)) if dim == 1 else (slice(None), slice(None), b)
            pair = (inst[sl_a] != inst[sl_b]) & raw_valid[sl_a] & raw_valid[sl_b]
            if pair.any():
                step_pred = log_pred[sl_a] - log_pred[sl_b]
                step_gt = log_raw[sl_a] - log_raw[sl_b]
                total = total + (step_pred - step_gt).abs()[pair].sum()
                count += int(pair.sum())
    return total / count if count else total, count


def object_si(log_pred, log_target, valid, inst, lambd=1.0, min_obj_px=100):
    """Scale-invariant log loss inside each instance (lambda=1: std of the log error), averaged over instances."""
    B = inst.shape[0]
    n_ids = int(inst.max()) + 1
    # Unique id per (image, instance); background (0) and invalid pixels are dropped.
    keep = valid & (inst > 0)
    if not keep.any():
        return log_pred.new_zeros(()), 0
    ids = (inst + n_ids * torch.arange(B, device=inst.device).view(B, 1, 1))[keep]
    d = (log_target - log_pred)[keep]
    n = torch.bincount(ids, minlength=B * n_ids).float()
    mean = torch.zeros_like(n).index_add_(0, ids, d) / n.clamp(min=1)
    # Two-pass (centred) variance: E[d^2] - E[d]^2 cancels badly in float32 when the scale is off.
    var = torch.zeros_like(n).index_add_(0, ids, (d - mean[ids]) ** 2) / n.clamp(min=1)
    big = n >= min_obj_px
    if not big.any():
        return log_pred.new_zeros(()), 0
    # SI loss = sqrt(E[d^2] - lambda E[d]^2) = sqrt(var + (1 - lambda) mean^2)
    per_obj = torch.sqrt(torch.clamp(var[big] + (1 - lambd) * mean[big] ** 2, min=1e-12))
    return per_obj.mean(), int(big.sum())


def build_loss(args):
    lam_edge = getattr(args, "lambda_edge", 0.0)
    lam_obj = getattr(args, "lambda_obj", 0.0)
    offsets = tuple(getattr(args, "edge_offsets", (1, 3, 6)))
    if (lam_edge or lam_obj) and not getattr(args, "masks", None):
        raise ValueError("--lambda-edge / --lambda-obj need --masks (gt or a mask folder)")

    def loss_fn(pred, batch):
        pred = pred.float().clamp(min=1e-3)  # fp16 sigmoid can underflow to 0 -> log(0)
        target = batch[f"depth_{args.train_gt}"]
        valid = (target > args.min_depth) & (target < args.max_depth)
        base = silog(pred, target, valid)
        total, parts = base, {"silog": base.item()}
        if lam_edge or lam_obj:
            log_pred = torch.log(pred)
            inst = batch["instances"]
            if lam_edge:
                raw = batch["depth_raw"]
                raw_valid = (raw > args.min_depth) & (raw < args.max_depth)
                e, n_pairs = edge_step(log_pred, torch.log(raw.clamp(min=1e-3)), raw_valid, inst, offsets)
                total = total + lam_edge * e
                parts.update(edge=e.item(), edge_pairs=n_pairs)
            if lam_obj:
                o, n_obj = object_si(log_pred, torch.log(target.clamp(min=1e-3)), valid, inst)
                total = total + lam_obj * o
                parts.update(obj=o.item(), objects=n_obj)
        return total, parts

    return loss_fn
