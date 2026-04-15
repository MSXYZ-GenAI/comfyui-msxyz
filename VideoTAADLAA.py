# Created By MSXYZ and Claude Opus 4.6
# GPU başında ciddi işler döndürüyoruz. 🚀
# TAA (Temporal Anti-Aliasing) + DLAA (Deep Learning Anti-Aliasing) Adaptation
# v0.1.1 - Fixed: Batch Size validation and Range Error

import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import logging

logger = logging.getLogger("VideoTAADLAA")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

class _DLAANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1, bias=False),
        )
        self.register_buffer("sobel_x", torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("sobel_y", torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("jitter_offsets", torch.tensor([
            [0.25, 0.25], [-0.25, -0.25], [-0.25, 0.25], [0.25, -0.25]
        ], dtype=torch.float32))
        self._init_weights()

    def _init_weights(self):
        sharpen = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32)
        convs = [m for m in self.conv if isinstance(m, nn.Conv2d)]
        nn.init.dirac_(convs[0].weight)
        nn.init.dirac_(convs[1].weight)
        nn.init.zeros_(convs[2].weight)
        with torch.no_grad():
            for i in range(3):
                for j in range(3):
                    convs[2].weight[i, j] = sharpen * 0.1

    def forward(self, x):
        return torch.clamp(x + self.conv(x) * 0.15, 0.0, 1.0)

class _TAAState:
    def __init__(self):
        self.history: torch.Tensor | None = None
        self.frame_count: int = 0

    def reset(self):
        self.history = None
        self.frame_count = 0

    def update(self, frame: torch.Tensor, alpha: float, motion_threshold: float) -> torch.Tensor:
        if self.history is None or self.history.shape != frame.shape:
            self.history = frame.clone()
            self.frame_count += 1
            return frame
        
        diff = torch.abs(frame - self.history).mean(dim=1, keepdim=True)
        motion_mask = torch.clamp(diff * motion_threshold * 10.0, 0.0, 1.0)
        dynamic_alpha = alpha * (1.0 - motion_mask)
        
        out = dynamic_alpha * self.history + (1.0 - dynamic_alpha) * frame
        self.history = out.detach()
        self.frame_count += 1
        return out

class VideoTAADLAA:
    def __init__(self):
        self._lock = threading.Lock()
        self._net_cache: dict[str, _DLAANet] = {}
        self._taa = _TAAState()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "taa_strength":    ("FLOAT", {"default": 0.7,  "min": 0.0, "max": 1.0,  "step": 0.05}),
                "taa_alpha":       ("FLOAT", {"default": 0.8,  "min": 0.0, "max": 0.99, "step": 0.01}),
                "motion_sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "jitter_scale":    ("FLOAT", {"default": 0.5,  "min": 0.0, "max": 2.0,  "step": 0.1}),
                "dlaa_strength":   ("FLOAT", {"default": 0.6,  "min": 0.0, "max": 1.0,  "step": 0.05}),
                "edge_threshold":  ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0,  "step": 0.01}),
                "blur_radius":     ("INT",   {"default": 1,    "min": 0,   "max": 5,    "step": 1}), # Min 0 yapıldı
                "reset_history":   ("BOOLEAN", {"default": False}),
                "batch_size":      ("INT",   {"default": 0,    "min": 0,   "max": 64,   "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "CustomPostProcess"

    def _get_net(self, device: torch.device) -> _DLAANet:
        key = str(device)
        if key not in self._net_cache:
            self._net_cache[key] = _DLAANet().to(device).eval()
        return self._net_cache[key]

    def _jitter(self, x: torch.Tensor, idx: int, scale: float, net: _DLAANet) -> torch.Tensor:
        if scale <= 1e-4: return x
        off = net.jitter_offsets[idx % 4]
        B, C, H, W = x.shape
        theta = torch.eye(2, 3, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        theta[:, 0, 2] = off[0] * scale / W
        theta[:, 1, 2] = off[1] * scale / H
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode="bicubic", padding_mode="reflection", align_corners=False)

    def _edge_aa(self, x: torch.Tensor, thr: float, blur: int, net: _DLAANet) -> torch.Tensor:
        if blur <= 0: return x
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        sx = F.conv2d(gray, net.sobel_x, padding=1)
        sy = F.conv2d(gray, net.sobel_y, padding=1)
        mask = (torch.sqrt(sx * sx + sy * sy + 1e-6) > thr).float()
        k = blur * 2 + 1
        blurred = F.avg_pool2d(F.pad(x, [blur] * 4, mode="reflect"), k, stride=1)
        return x * (1.0 - mask) + blurred * mask

    def execute(self, images, taa_strength, taa_alpha, motion_sensitivity, jitter_scale,
                dlaa_strength, edge_threshold, blur_radius, reset_history, batch_size):

        with self._lock:
            if reset_history:
                self._taa.reset()

            device = images.device
            net = self._get_net(device)
            B, H, W, C = images.shape
            
            # --- Kritik Hata Düzeltme: Batch Size Hesabı ---
            working_batch_size = batch_size
            if working_batch_size <= 0:
                cost = H * W * C * 4 * 14
                free, _ = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0,0)
                working_batch_size = max(1, min(16, int((free * 0.6) // cost))) if free > 0 else 1
            
            # Range hatasını önlemek için batch_size'ın en az 1 olduğundan emin oluyoruz
            working_batch_size = max(1, working_batch_size)
            # ----------------------------------------------

            images = images.to(device)
            has_alpha = (C == 4)
            outputs = []

            with torch.no_grad():
                for i in range(0, B, working_batch_size):
                    chunk = images[i:i + working_batch_size]
                    img = chunk.permute(0, 3, 1, 2).float()

                    img_rgb = img[:, :3]
                    img_alpha = img[:, 3:4] if has_alpha else None
                    results_rgb = []

                    for f_idx in range(img_rgb.shape[0]):
                        f = img_rgb[f_idx:f_idx + 1]

                        f = self._jitter(f, self._taa.frame_count, jitter_scale, net)
                        f = self._edge_aa(f, edge_threshold, blur_radius, net)
                        
                        taa_out = self._taa.update(f, taa_alpha, motion_sensitivity)
                        f = torch.lerp(f, taa_out, taa_strength)

                        if dlaa_strength > 0.0:
                            f = torch.lerp(f, net(f), dlaa_strength)

                        results_rgb.append(f.clamp(0.0, 1.0))

                    batch_rgb = torch.cat(results_rgb, dim=0)
                    batch_out = torch.cat([batch_rgb, img_alpha], dim=1) if has_alpha else batch_rgb
                    outputs.append(batch_out.permute(0, 2, 3, 1).cpu())
                    
                    logger.info(f"İşlendi: {min(i + working_batch_size, B)} / {B}")

            return (torch.cat(outputs, dim=0),)

NODE_CLASS_MAPPINGS = {"VideoTAADLAA": VideoTAADLAA}
NODE_DISPLAY_NAME_MAPPINGS = {"VideoTAADLAA": "🎮 Video TAA + DLAA Anti-Aliasing"}