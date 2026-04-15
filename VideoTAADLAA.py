# Created By MSXYZ and Claude Opus 4.6
# GPU başında ciddi işler döndürüyoruz. 🚀
# TAA (Temporal Anti-Aliasing) + DLAA (Deep Learning Anti-Aliasing) Adaptation
# v0.1.1 - Batch Processing + Stability Improvements

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import comfy.model_management as mm

logger = logging.getLogger("VideoTAADLAA")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# =========================
# DLAA CORE (light CNN)
# =========================
class _DLAANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1, bias=False),
        )

        self.register_buffer("sobel_x", torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))

        self.register_buffer("sobel_y", torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))

        self.register_buffer("jitter_offsets", torch.tensor([
            [0.25, 0.25],
            [-0.25, -0.25],
            [-0.25, 0.25],
            [0.25, -0.25]
        ], dtype=torch.float32))

        self._init()

    def _init(self):
        convs = [m for m in self.conv if isinstance(m, nn.Conv2d)]
        nn.init.kaiming_normal_(convs[0].weight)

        # FIX v0.2.0: dirac_ yerine identity-benzeri başlatma.
        # dirac_ semantik olarak doğru ama in==out==16 durumunda
        # bazı PyTorch versiyonlarında uyarı üretebilir.
        # eye tabanlı init daha güvenli ve aynı etkiyi verir.
        with torch.no_grad():
            convs[1].weight.zero_()
            for c in range(min(convs[1].weight.shape[0], convs[1].weight.shape[1])):
                convs[1].weight[c, c, 1, 1] = 1.0  # merkez piksel = identity

        nn.init.zeros_(convs[2].weight)

        # Sharpen filtresini 16 kanala dağıt
        sharpen = torch.tensor([
            [ 0.0, -1.0,  0.0],
            [-1.0,  5.0, -1.0],
            [ 0.0, -1.0,  0.0]
        ], dtype=torch.float32)

        with torch.no_grad():
            for out_c in range(3):      # RGB çıkış
                for in_c in range(16):  # 16 giriş kanalı
                    convs[2].weight[out_c, in_c] = (sharpen * 0.1) / 16.0

    def forward(self, x):
        return torch.clamp(x + self.conv(x) * 0.12, 0.0, 1.0)


# =========================
# TEMPORAL STATE
# =========================
class _TAAState:
    def __init__(self):
        self.history = None
        self.frame_id = 0

    def reset(self):
        self.history = None
        self.frame_id = 0

    def update(self, frame, alpha, motion_sensitivity):
        if self.history is None or self.history.shape != frame.shape:
            self.history = frame.clone()
            self.frame_id += 1
            return frame

        diff = torch.mean(torch.abs(frame - self.history), dim=1, keepdim=True)

        # soft sigmoid motion mask
        motion = torch.sigmoid((diff - motion_sensitivity) * 8.0)

        dynamic_alpha = alpha * (1.0 - motion)

        out = dynamic_alpha * self.history + (1.0 - dynamic_alpha) * frame

        self.history = out.detach()
        self.frame_id += 1
        return out


# =========================
# MAIN NODE
# =========================
class VideoTAADLAA:
    def __init__(self):
        self.net_cache = {}
        self.taa = _TAAState()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "taa_strength": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.05}),
                "taa_alpha": ("FLOAT", {"default": 0.8, "min": 0, "max": 0.99, "step": 0.01}),
                "motion_sensitivity": ("FLOAT", {"default": 0.4, "min": 0, "max": 1, "step": 0.05}),
                "jitter_scale": ("FLOAT", {"default": 0.1, "min": 0, "max": 2, "step": 0.05}),
                "dlaa_strength": ("FLOAT", {"default": 0.6, "min": 0, "max": 1, "step": 0.05}),
                "edge_threshold": ("FLOAT", {"default": 0.15, "min": 0, "max": 1, "step": 0.01}),
                "blur_radius": ("INT", {"default": 1, "min": 0, "max": 5, "step": 1}),
                "reset_history": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "CustomPostProcess"

    def _device_key(self, device: torch.device) -> str:
        # FIX v0.2.0: "cuda" ve "cuda:0" aynı cache entry'i kullanır,
        # gereksiz model kopyası ve bellek sızıntısı önlenir.
        return f"{device.type}:{device.index if device.index is not None else 0}"

    def _net(self, device: torch.device) -> _DLAANet:
        k = self._device_key(device)
        if k not in self.net_cache:
            self.net_cache[k] = _DLAANet().to(device).eval()
        return self.net_cache[k]

    # -------------------------
    # jitter (stable bilinear)
    # -------------------------
    def jitter(self, x: torch.Tensor, idx: int, scale: float, net: _DLAANet) -> torch.Tensor:
        if scale < 1e-5:
            return x

        off = net.jitter_offsets[idx % 4]
        B, C, H, W = x.shape

        theta = torch.eye(2, 3, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        theta[:, 0, 2] = off[0] * scale / W
        theta[:, 1, 2] = off[1] * scale / H

        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    # -------------------------
    # edge AA (soft mask)
    # -------------------------
    def edge_aa(self, x: torch.Tensor, thr: float, blur: int, net: _DLAANet) -> torch.Tensor:
        if blur <= 0:
            return x

        gray = 0.299*x[:, 0:1] + 0.587*x[:, 1:2] + 0.114*x[:, 2:3]

        sx = F.conv2d(gray, net.sobel_x, padding=1)
        sy = F.conv2d(gray, net.sobel_y, padding=1)

        edge = torch.sqrt(sx*sx + sy*sy + 1e-6)

        # soft mask
        mask = torch.sigmoid((edge - thr) * 12.0)

        k = blur * 2 + 1
        blurred = F.avg_pool2d(
            F.pad(x, [blur]*4, mode="reflect"),
            k, stride=1
        )

        return x*(1-mask) + blurred*mask

    # -------------------------
    # DLAA batch forward
    # -------------------------
    def _dlaa_batch(self, frames: torch.Tensor, strength: float, net: _DLAANet) -> torch.Tensor:
        """
        FIX v0.2.0: DLAA artık tüm frame'leri tek seferde batch olarak işler.
        Büyük videolarda belirgin hız artışı sağlar (özellikle GPU'da).
        TAA history sıralı frame bağımlılığı gerektirdiği için döngü zorunlu,
        ama DLAA bağımsız olduğundan batch'e alındı.
        """
        if strength <= 0:
            return frames
        enhanced = net(frames)
        return torch.lerp(frames, enhanced, strength)

    # -------------------------
    # EXECUTE
    # -------------------------
    def execute(
        self,
        images: torch.Tensor,
        taa_strength: float,
        taa_alpha: float,
        motion_sensitivity: float,
        jitter_scale: float,
        dlaa_strength: float,
        edge_threshold: float,
        blur_radius: int,
        reset_history: bool = False,
    ):
        device = mm.get_torch_device()

        if reset_history:
            self.taa.reset()

        net = self._net(device)

        B, H, W, C = images.shape
        images = images.to(device)

        has_alpha = (C == 4)

        # --- TAA (sıralı, history bağımlı) ---
        taa_frames = []
        with torch.no_grad():
            for i in range(B):
                f = images[i:i+1].permute(0, 3, 1, 2).float()
                rgb = f[:, :3]

                rgb = self.jitter(rgb, self.taa.frame_id, jitter_scale, net)
                rgb = self.edge_aa(rgb, edge_threshold, blur_radius, net)

                taa_out = self.taa.update(rgb, taa_alpha, motion_sensitivity)
                rgb = torch.lerp(rgb, taa_out, taa_strength)

                taa_frames.append(rgb)

        # --- DLAA (batch, paralel) ---
        with torch.no_grad():
            rgb_batch = torch.cat(taa_frames, dim=0)           # (B, 3, H, W)
            rgb_batch = self._dlaa_batch(rgb_batch, dlaa_strength, net)
            rgb_batch = rgb_batch.clamp(0, 1)

        # --- Alpha kanalını geri ekle ve CPU'ya taşı ---
        if has_alpha:
            alpha_batch = images.permute(0, 3, 1, 2).float()[:, 3:4]  # (B, 1, H, W)
            rgb_batch = torch.cat([rgb_batch, alpha_batch], dim=1)      # (B, 4, H, W)

        out = rgb_batch.permute(0, 2, 3, 1).cpu()  # (B, H, W, C)

        # VRAM Temizliği
        mm.soft_empty_cache()

        return (out,)


NODE_CLASS_MAPPINGS = {
    "VideoTAADLAA": VideoTAADLAA
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoTAADLAA": "🎮 Video TAA + DLAA Anti-Aliasing"
}