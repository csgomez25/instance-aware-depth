"""Decoder variant (H4): the instance-mask boundary map as an extra decoder input.

    instances (B x H x W ids) -> mask_edges -> conv3x3(1->16) -> ReLU -> conv3x3(16->32, zero-init)
                              -> added to the full-resolution decoder features right before output_conv2

output_conv2 is the last block of the DPT head (conv3x3 -> ReLU -> conv1x1 -> sigmoid), so the edge features
act where the depth map is formed at full resolution. The last conv starts at zero, so before training the model
is exactly the pretrained one. Unlike the loss-only method, this model needs instance masks at test time.
Checkpoints hold `edge_branch.*` weights; common.load_metric builds this class when it sees them.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common as C

DepthAnythingV2 = C._import_dpt(metric=True)


def mask_edges(inst, width=3):
    """B x H x W instance ids -> B x 1 x H x W float, 1 within `width` px of a change of instance id."""
    inst = inst.long()
    edge = torch.zeros_like(inst, dtype=torch.bool)
    dy = inst[:, 1:] != inst[:, :-1]
    dx = inst[:, :, 1:] != inst[:, :, :-1]
    edge[:, 1:] |= dy
    edge[:, :-1] |= dy
    edge[:, :, 1:] |= dx
    edge[:, :, :-1] |= dx
    edge = edge[:, None].float()
    return F.max_pool2d(edge, width, stride=1, padding=width // 2) if width > 1 else edge


class EdgeDepthAnythingV2(DepthAnythingV2):
    needs_instances = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        c = self.depth_head.scratch.output_conv1.out_channels
        self.edge_branch = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(True), nn.Conv2d(16, c, 3, padding=1))
        nn.init.zeros_(self.edge_branch[2].weight)
        nn.init.zeros_(self.edge_branch[2].bias)
        self._edge_feat = None
        # The DPT head has no input for extra features; a pre-hook adds them to output_conv2's input.
        self.depth_head.scratch.output_conv2.register_forward_pre_hook(self._add_edges)

    def _add_edges(self, module, args):
        return (args[0] + self._edge_feat.to(args[0].dtype),)

    def forward(self, x, inst):
        self._edge_feat = self.edge_branch(mask_edges(inst).to(x.dtype))
        try:
            return super().forward(x)
        finally:
            self._edge_feat = None

    @torch.no_grad()
    def infer_image(self, raw_image, inst, input_size=518):
        image, (h, w) = self.image2tensor(raw_image, input_size)
        inst = cv2.resize(inst.astype(np.int32), image.shape[-1:-3:-1], interpolation=cv2.INTER_NEAREST)
        depth = self.forward(image, torch.from_numpy(inst)[None].to(image.device))
        return F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0].cpu().numpy()
