# Video TAA + DLAA utilities
# MSYNTRIX


import torch


LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)


def rgb_luma(image: torch.Tensor) -> torch.Tensor:
    r, g, b = LUMA_WEIGHTS
    return r * image[:, 0:1] + g * image[:, 1:2] + b * image[:, 2:3]


def clamp01(value) -> float:
    return max(0.0, min(float(value), 1.0))