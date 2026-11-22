"""Training losses. `build_loss(args)` returns loss(pred, batch) -> (total, {name: value}).

silog: the scale-invariant log loss used by Depth Anything V2 (lambda = 0.5), on the
chosen training target (filled or raw depth), masked to min_depth..max_depth.
The instance-aware terms (section D of the plan) are added here.
"""
import torch


def silog(pred, target, valid, lambd=0.5):
    d = torch.log(target[valid]) - torch.log(pred[valid])
    return torch.sqrt(torch.clamp((d ** 2).mean() - lambd * d.mean() ** 2, min=1e-12))


def build_loss(args):
    def loss_fn(pred, batch):
        pred = pred.float().clamp(min=1e-3)  # fp16 sigmoid can underflow to 0 -> log(0)
        target = batch[f"depth_{args.train_gt}"]
        valid = (target > args.min_depth) & (target < args.max_depth)
        base = silog(pred, target, valid)
        return base, {"silog": base.item()}

    return loss_fn
