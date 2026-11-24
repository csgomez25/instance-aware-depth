"""NYU Depth V2 training dataset, following the Depth Anything V2 recipe.

Each sample is the border-free 625x468 region of a labeled NYU image, resized so the
short side is 518 (multiple of 14, cubic for RGB, nearest for depth/masks), normalised
with ImageNet statistics, then randomly cropped to 518x518 and randomly flipped.
Filled depth, raw depth and (optionally) an instance map are cropped identically,
so any loss can use any of them.
"""
import math
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import common as C

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def resized_shape(h, w, short=518, multiple=14):
    """DA-V2 'lower_bound' resize: short side >= `short`, both sides multiples of `multiple`."""
    scale = short / min(h, w)
    def fit(x):
        y = round(x * scale / multiple) * multiple
        return y if y >= short else math.ceil(x * scale / multiple) * multiple
    return fit(h), fit(w)


class NYUDepth(Dataset):
    def __init__(self, split="train", size=518, masks=None, augment=True):
        """masks: None, 'gt' (NYU instances; object = (label, instance)), or a mask folder name such as 'yolo11s-seg'."""
        self.ids = C.read_split(split)
        self.size = size
        self.masks = masks
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def _instances(self, img_id):
        if self.masks is None:
            return None
        if self.masks == "gt":
            return C.read_nyu_instances(img_id)
        return cv2.imread(str(C.NYU_DIR / self.masks / f"{img_id}.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)

    def __getitem__(self, i):
        img_id = self.ids[i]
        t, b, l, r = C.NYU_VALID_CROP
        rgb = cv2.cvtColor(cv2.imread(str(C.nyu_rgb_path(img_id))), cv2.COLOR_BGR2RGB)[t:b, l:r]
        maps = {"depth_filled": C.read_nyu_depth(img_id, "filled")[t:b, l:r],
                "depth_raw": C.read_nyu_depth(img_id, "raw")[t:b, l:r]}
        inst = self._instances(img_id)
        if inst is not None:
            maps["instances"] = inst[t:b, l:r]

        h, w = resized_shape(*rgb.shape[:2], short=self.size)
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_CUBIC)
        maps = {k: cv2.resize(v, (w, h), interpolation=cv2.INTER_NEAREST) for k, v in maps.items()}

        if self.augment:
            y0, x0 = random.randint(0, h - self.size), random.randint(0, w - self.size)
            flip = random.random() < 0.5
        else:
            y0, x0, flip = (h - self.size) // 2, (w - self.size) // 2, False
        crop = lambda a: a[y0:y0 + self.size, x0:x0 + self.size]
        rgb = crop(rgb)
        maps = {k: crop(v) for k, v in maps.items()}
        if flip:
            rgb = rgb[:, ::-1]
            maps = {k: v[:, ::-1] for k, v in maps.items()}

        img = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
        sample = {"image": torch.from_numpy(np.ascontiguousarray(img)), "id": img_id}
        for k, v in maps.items():
            sample[k] = torch.from_numpy(np.ascontiguousarray(v))
        return sample
