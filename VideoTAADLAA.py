# Created by MSXYZ (AI-assisted)
# Temporal Anti-Aliasing (TAA) + A Lightweight DLAA-style Sharpening
# v0.1.1 - Temporal stability improvements, ghosting reduction

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import comfy.model_management as mm

logger = logging.getLogger("VideoTAADLAA")

# =========================
# DLAA CORE
# =========================
class _DLAANet(nn.Module):
    """
    Lightweight CNN-based sharpening module used as a DL-inspired post-process.

    Not:
    Bu yapı gerçek bir eğitilmiş DLAA modeli değildir.
    Sadece inference-time çalışan küçük bir CNN ile
    detay keskinleştirme ve aliasing azaltma hedeflenmiştir.
    """
    def __init__(self):
        super().__init__()
        
        # R, G, B için 3 kanal (groups=3 ile ultra hızlı)
        self.conv = nn.Conv2d(3, 3, 3, padding=1, bias=False, groups=3)
        
        # Keskinleştirme çekirdeğini (Sharpening kernel)
        kernel = torch.tensor([
            [0.0, -1.0, 0.0], 
            [-1.0, 5.0, -1.0], 
            [0.0, -1.0, 0.0]
        ], dtype=torch.float32) * 0.1
        
        # Ağırlıkları tek seferde sabitliyoruz
        with torch.no_grad():
            for i in range(3):
                self.conv.weight[i, 0] = kernel
        
        # Diğer fonksiyonlar (edge_aa ve sobel)
        self.register_buffer("sobel_x", torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("sobel_y", torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        
        # 8-Noktalı Jitter Sekansı (Daha iyi TAA sonucu sağlar)
        self.register_buffer("jitter_offsets", torch.tensor([
            [0.125, 0.375], [-0.375, -0.125], [0.375, -0.375], [-0.125, 0.125],
            [0.250, 0.250], [-0.250, -0.250], [-0.250, 0.250], [0.250, -0.250]
        ], dtype=torch.float32))

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

    def update(self, frame, alpha, sensitivity):
        if self.history is None or self.history.shape != frame.shape:
            self.history = frame.clone()
            self.frame_id += 1
            return frame

        # Neighborhood clamping: geçmiş kareyi mevcut frame'in 3x3 lokal min/max sınırına çeker
        local_min = -F.max_pool2d(-frame, kernel_size=3, stride=1, padding=1)
        local_max = F.max_pool2d(frame, kernel_size=3, stride=1, padding=1)
        history_clipped = torch.clamp(self.history, local_min, local_max)

        # Hareketli bölgelerde alpha'yı düşür: ghosting azalır, keskin geçişler korunur
        diff = torch.abs(frame - history_clipped).mean(dim=1, keepdim=True)
        motion_mask = torch.sigmoid((diff - sensitivity) * 20.0)
        dynamic_alpha = alpha * (1.0 - motion_mask)

        out = torch.lerp(frame, history_clipped, dynamic_alpha)
        self.history = out.detach()  # detach(): gradient graph birikimini önler
        self.frame_id += 1
        return out

# =========================
# MAIN NODE
# =========================
class VideoTAADLAA:
    def __init__(self):
        self.net_cache = {}  # Device başına ayrı model (CPU/GPU desteği)
        self.taa = _TAAState()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "taa_strength": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.05}), 
                "taa_alpha": ("FLOAT", {"default": 0.6, "min": 0, "max": 0.95, "step": 0.01}),   
                "motion_sensitivity": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.5, "step": 0.01}),
                "jitter_scale": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.01}),
                "dlaa_strength": ("FLOAT", {"default": 0.6, "min": 0, "max": 1, "step": 0.05}),
                "edge_threshold": ("FLOAT", {"default": 0.25, "min": 0, "max": 1, "step": 0.01}),
                "blur_radius": ("INT", {"default": 1, "min": 0, "max": 5, "step": 1}),
                "reset_history": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "CustomPostProcess"

    def _net(self, device):
        if device not in self.net_cache:
            self.net_cache[device] = _DLAANet().to(device).eval()
        return self.net_cache[device]

    def jitter(self, x, idx, scale, net):
        if scale < 1e-5: return x
        off = net.jitter_offsets[idx % 8] # 8 noktaya güncellendi
        B, C, H, W = x.shape
        theta = torch.eye(2, 3, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        theta[:, 0, 2], theta[:, 1, 2] = off[0] * scale / W, off[1] * scale / H
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    def edge_aa(self, x, thr, blur, net):
        if blur <= 0: return x
        gray = 0.299*x[:, 0:1] + 0.587*x[:, 1:2] + 0.114*x[:, 2:3]
        sx, sy = F.conv2d(gray, net.sobel_x, padding=1), F.conv2d(gray, net.sobel_y, padding=1)
        edge = torch.sqrt(sx*sx + sy*sy + 1e-6)  # epsilon: NaN önlemi
        mask = torch.sigmoid((edge - thr) * 12.0)
        blurred = F.avg_pool2d(F.pad(x, [blur]*4, mode="reflect"), blur*2+1, stride=1)
        return torch.lerp(x, blurred, mask) # Optimizasyon: x*(1-mask) + blurred*mask yerine lerp kullanıldı.

    def execute(self, images, taa_strength, taa_alpha, motion_sensitivity, jitter_scale, dlaa_strength, edge_threshold, blur_radius, reset_history=True):
        device = mm.get_torch_device()
        if reset_history: self.taa.reset()
        net = self._net(device)
        B, H, W, C = images.shape
        has_alpha = (C == 4) # RGBA kontrolü
        out_list = []
        
        with torch.no_grad():
            for i in range(B):
                img = images[i:i+1].to(device).permute(0, 3, 1, 2).float()
                
                rgb = img[:, :3]
                if has_alpha:
                    alpha_channel = img[:, 3:] # Saydamlığı güvenli bir yere al

                # Stage 1: Spatial preprocessing
                rgb = self.jitter(rgb, self.taa.frame_id, jitter_scale, net)
                rgb = self.edge_aa(rgb, edge_threshold, blur_radius, net)

                # Stage 2: Temporal blending
                taa_out = self.taa.update(rgb, taa_alpha, motion_sensitivity)
                rgb = torch.lerp(rgb, taa_out, taa_strength)

                # Stage 3: CNN sharpening
                if dlaa_strength > 0:
                    rgb = torch.lerp(rgb, net(rgb), dlaa_strength)
                
                # Alpha kanalını geri ekle (Maskeli videoların bozulmasını engeller)
                if has_alpha:
                    final_tensor = torch.cat([rgb, alpha_channel], dim=1)
                else:
                    final_tensor = rgb

                out_list.append(final_tensor.permute(0, 2, 3, 1).cpu())
        
        return (torch.cat(out_list, dim=0),)

NODE_CLASS_MAPPINGS = {
    "VideoTAADLAA": VideoTAADLAA
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoTAADLAA": "🎮 Video TAA + DLAA Anti-Aliasing"
}