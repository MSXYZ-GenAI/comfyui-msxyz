# Video TAA + DLAA temporal state
# MSXYZ

import torch
import torch.nn.functional as F

try:
    from .utils import rgb_luma
except ImportError:
    from utils import rgb_luma


class TAAState:
    def __init__(
        self,
        variance_gamma=1.5,
        edge_guard_strength=2.0,
        edge_guard_min=0.25,
        edge_guard_max=1.0,
    ):
        self.history = None
        self.frame_id = 0

        self.variance_gamma = variance_gamma
        self.edge_guard_strength = edge_guard_strength
        self.edge_guard_min = edge_guard_min
        self.edge_guard_max = edge_guard_max

    def reset(self):
        self.history = None
        self.frame_id = 0

    @staticmethod
    def smoothstep(x: torch.Tensor) -> torch.Tensor:
        return x * x * (3.0 - 2.0 * x)

    def update(self, frame, alpha, sensitivity):
        if self.history is None or self.history.shape != frame.shape:
            self.history = frame.detach().clone().to(frame.device)
            return frame

        local_mean = F.avg_pool2d(frame, kernel_size=3, stride=1, padding=1)
        local_sq_mean = F.avg_pool2d(frame * frame, kernel_size=3, stride=1, padding=1)
        local_var = (local_sq_mean - local_mean * local_mean).clamp(min=0.0)
        local_std = torch.sqrt(local_var + 1e-6)

        local_min = local_mean - local_std * self.variance_gamma
        local_max = local_mean + local_std * self.variance_gamma

        history_clipped = torch.maximum(torch.minimum(self.history, local_max), local_min)

        diff = torch.abs(frame - history_clipped).mean(dim=1, keepdim=True)
        raw_diff = torch.abs(frame - self.history).mean(dim=1, keepdim=True)

        disocclusion = ((raw_diff - sensitivity * 2.0) / (sensitivity + 1e-6)).clamp(0.0, 1.0)
        disocclusion = self.smoothstep(disocclusion)

        gray = rgb_luma(frame)

        edge_x = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
        edge_y = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])

        edge_x = F.pad(edge_x, (0, 1, 0, 0))
        edge_y = F.pad(edge_y, (0, 0, 0, 1))

        edge_strength = (edge_x + edge_y).clamp(0.0, 1.0)
        edge_guard = (
            self.edge_guard_max - edge_strength * self.edge_guard_strength
        ).clamp(self.edge_guard_min, self.edge_guard_max)

        motion_soft = ((diff - sensitivity) / (sensitivity + 1e-6)).clamp(0.0, 1.0)
        motion_soft = self.smoothstep(motion_soft)

        dynamic_alpha = alpha * (1.0 - motion_soft) * edge_guard

        confidence = torch.exp(-diff * 10.0)
        confidence = confidence * (1.0 - disocclusion)
        confidence = confidence.clamp(0.15, 1.0)

        dynamic_alpha = dynamic_alpha * confidence

        reject_strength = ((diff - sensitivity * 1.5) / (sensitivity + 1e-6)).clamp(0.0, 1.0)
        reject_strength = torch.maximum(reject_strength, disocclusion)
        reject_strength = self.smoothstep(reject_strength)

        history_clipped = torch.lerp(history_clipped, frame, reject_strength)

        out = torch.lerp(frame, history_clipped, dynamic_alpha)
        self.history = out.detach()

        return out