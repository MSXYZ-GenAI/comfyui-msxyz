# Video TAA + DLAA utilities
# MSXYZ

import torch


LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)


def rgb_luma(x: torch.Tensor) -> torch.Tensor:
    r, g, b = LUMA_WEIGHTS
    return r * x[:, 0:1] + g * x[:, 1:2] + b * x[:, 2:3]


def clamp01(value):
    return max(0.0, min(float(value), 1.0))