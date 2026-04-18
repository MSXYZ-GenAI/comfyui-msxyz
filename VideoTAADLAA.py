# Created by MSXYZ (AI-assisted)
# Temporal Anti-Aliasing (TAA) + A Lightweight DLAA-style Sharpening
# v0.1.1 - Temporal stability improvements, ghosting reduction

import torch
import torch.nn.functional as F
import logging
import comfy.model_management as mm
import cv2
import numpy as np

logger = logging.getLogger("VideoTAADLAA")


# TEMPORAL STATE
class _TAAState:
    """
    TAA (Zamansal Kenar Yumuşatma) için önceki karelerin geçmişini tutar.
    """

    def __init__(self):
        self.history: torch.Tensor | None = None
        self.prev_frame: torch.Tensor | None = None
        self.frame_id: int = 0

    def reset(self):
        self.history = None
        self.prev_frame = None
        self.frame_id = 0

    # Yardımcılar
    @staticmethod
    def _optical_flow(prev: torch.Tensor, curr: torch.Tensor) -> torch.Tensor:
        """
        CPU üzerinde akış (hareket) hesaplar ve düzeltme tensörü döndürür.
        Boyut: (1, H, W, 2)
        """
        def to_gray_u8(t: torch.Tensor) -> np.ndarray:
            arr = t[0].permute(1, 2, 0).detach().cpu().numpy()
            return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        prev_gray = to_gray_u8(prev)
        curr_gray = to_gray_u8(curr)

        # Calculate flow
        flow_np = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )  # (H, W, 2) — raw pixel offsets

        H, W = flow_np.shape[:2]
        flow = torch.from_numpy(flow_np).to(curr.device).float()  # (H, W, 2)

        # Since the grid ranges from -1 to 1 (a total distance of 2),
        flow[..., 0] /= (W / 2.0)
        flow[..., 1] /= (H / 2.0)

        return flow.unsqueeze(0)  # (1, H, W, 2)

    def update(
        self,
        frame: torch.Tensor,    # (1, 3, H, W)  float32
        alpha: float,           # base temporal blend weight
        sensitivity: float,     # motion sensitivity
    ) -> torch.Tensor:

        if self.history is None:
            self.history = frame.clone()
            self.prev_frame = frame.clone()
            self.frame_id += 1
            return frame

        B, C, H, W = frame.shape

        # highlight mask
        luma = (
            frame[:, 0:1] * 0.299
            + frame[:, 1:2] * 0.587
            + frame[:, 2:3] * 0.114
        )
        highlight_mask = torch.sigmoid((luma - 0.75) * 15.0)  # (1,1,H,W)

        # Neighbourhood clamping
        local_min = -F.max_pool2d(-frame, 3, 1, 1)
        local_max = F.max_pool2d(frame, 3, 1, 1)
        history_clamped = torch.minimum(torch.maximum(self.history, local_min), local_max)

        # Optical-flow warp of history
        flow = self._optical_flow(self.prev_frame, frame)  # (1, H, W, 2)

        base_grid_y, base_grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=frame.device),
            torch.linspace(-1, 1, W, device=frame.device),
            indexing="ij",
        )
        base_grid = torch.stack((base_grid_x, base_grid_y), dim=-1)  # (H, W, 2)
        sample_grid = (base_grid.unsqueeze(0) + flow).clamp(-1, 1)    # (1, H, W, 2)

        warped_history = F.grid_sample(
            self.history,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )  # (1, 3, H, W)

        # Occlusion
        diff_flow = (frame - warped_history).abs().mean(dim=1, keepdim=True)
        occ_mask = torch.sigmoid((diff_flow - 0.08) * 20.0)
        history_mix = torch.lerp(warped_history, history_clamped, occ_mask)

        # Motion-reactive
        motion = (frame - self.prev_frame).abs().mean(dim=1, keepdim=True)
        reactive = torch.sigmoid(((motion / max(sensitivity, 0.001)) - 0.3) * 10.0)

        temporal_weight = (
            alpha
            * (1.0 - reactive * 0.9)
            * (1.0 - highlight_mask * 0.85)
        )  # per-pixel [0, alpha]

        out = torch.lerp(frame, history_mix, temporal_weight)

        self.history = out.detach()
        self.prev_frame = frame.detach()
        self.frame_id += 1

        return out


# DLAA-style sharpening
def _dlaa_sharpen(rgb: torch.Tensor, strength: float) -> torch.Tensor:
    """
    DLAA detay geri kazanımına benzer sonuçlar veren, sistemi yormayan 
    keskinleştirme işlemi. Herhangi bir çözünürlükte, yapay zeka modeli 
    (öğrenilmiş ağırlıklar) kullanmadan çalışır.
    """
    blurred = F.avg_pool2d(rgb, kernel_size=3, stride=1, padding=1)
    detail = rgb - blurred           # high-frequency residual
    sharpened = rgb + detail * 0.5   # amplify edges
    return torch.lerp(rgb, sharpened, strength).clamp(0.0, 1.0)


# COMFYUI NODE
class VideoTAADLAA:
    def __init__(self):
        self.taa = _TAAState()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":            ("IMAGE",),
                "taa_alpha":         ("FLOAT",   {"default": 0.60, "min": 0.00, "max": 0.95, "step": 0.01}),
                "motion_sensitivity":("FLOAT",   {"default": 0.05, "min": 0.01, "max": 0.50, "step": 0.01}),
                "dlaa_strength":     ("FLOAT",   {"default": 0.60, "min": 0.00, "max": 1.00, "step": 0.05}),
                "reset_history":     ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "CustomPostProcess"

    def execute(
        self,
        images: torch.Tensor,
        taa_alpha: float,
        motion_sensitivity: float,
        dlaa_strength: float,
        reset_history: bool = True,
    ) -> tuple[torch.Tensor]:

        device = mm.get_torch_device()

        if reset_history:
            self.taa.reset()

        B, H, W, C = images.shape
        out_frames: list[torch.Tensor] = []

        with torch.no_grad():
            for i in range(B):
                # images are (B, H, W, C) — convert to (1, C, H, W)
                rgb = images[i : i + 1].to(device).permute(0, 3, 1, 2).float()

                # process the first 3 channels
                rgb_3 = rgb[:, :3]

                # 1. TAA
                taa_out = self.taa.update(rgb_3, taa_alpha, motion_sensitivity)
                rgb_3 = taa_out

                # 2. DLAA sharpening
                if dlaa_strength > 0.0:
                    rgb_3 = _dlaa_sharpen(rgb_3, dlaa_strength)

                # Alpha channel RGBA input
                if C == 4:
                    alpha_ch = rgb[:, 3:4]
                    result = torch.cat([rgb_3, alpha_ch], dim=1)
                else:
                    result = rgb_3

                # Back to (1, H, W, C)
                out_frames.append(result.permute(0, 2, 3, 1).cpu())

        return (torch.cat(out_frames, dim=0),)


NODE_CLASS_MAPPINGS = {
    "VideoTAADLAA": VideoTAADLAA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoTAADLAA": "🎮 Video TAA + DLAA Anti-Aliasing",
}