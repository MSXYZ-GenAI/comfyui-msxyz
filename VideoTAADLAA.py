# Created By MSXYZ and Claude Opus 4.6
# GPU başında ciddi işler döndürüyoruz. 🚀
# TAA (Temporal Anti-Aliasing) + DLAA (Deep Learning Anti-Aliasing) Adaptation
# v0.1.1

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Yardımcı: Hafif DLAA Konvolüsyon
# ---------------------------------------------------------------------------
class _DLAANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(16, 3, 3, padding=1, bias=False)
        self._init_weights()

    def _init_weights(self):
        sharpen = torch.tensor(
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32
        )
        nn.init.dirac_(self.conv1.weight)
        with torch.no_grad():
            for i in range(self.conv3.weight.shape[0]):
                for j in range(self.conv3.weight.shape[1]):
                    self.conv3.weight[i, j] = sharpen * 0.1
        nn.init.dirac_(self.conv2.weight)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return torch.clamp(residual + x * 0.15, 0.0, 1.0)


# ---------------------------------------------------------------------------
# TAA Geçmiş Kare Yöneticisi
# ---------------------------------------------------------------------------
class _TAAHistory:
    def __init__(self):
        self.history: torch.Tensor | None = None

    def reset(self):
        self.history = None

    def update(self, frame: torch.Tensor, alpha: float) -> torch.Tensor:
        if self.history is None or self.history.shape != frame.shape:
            self.history = frame.clone()
            return frame
        blended = alpha * self.history + (1.0 - alpha) * frame
        self.history = blended.detach().clone()
        return blended


# ---------------------------------------------------------------------------
# Ana Node
# ---------------------------------------------------------------------------
class VideoTAADLAA:
    _taa_history = _TAAHistory()
    _dlaa_net: _DLAANet | None = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "taa_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
                "taa_history_alpha": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.5, "step": 0.01}),
                "jitter_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1}),
                "dlaa_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "edge_threshold": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blur_radius": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
                # ARTIK VARSAYILAN TRUE: Her yeni render temiz bir başlangıç yapar.
                "reset_history": ("BOOLEAN", {"default": True}), 
                "batch_size": ("INT", {"default": 4, "min": 0, "max": 32, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_taa_dlaa"
    CATEGORY = "CustomPostProcess"

    # -----------------------------------------------------------------------
    def _auto_batch_size(self, images):
        if not torch.cuda.is_available():
            return 1
        total_vram = torch.cuda.get_device_properties(0).total_memory
        free_vram = total_vram - torch.cuda.memory_allocated(0)
        B, H, W, C = images.shape
        bytes_per_frame = H * W * C * 4 * 12  # float32 * buffer katsayısı
        safe_batch = max(1, int((free_vram * 0.6) // bytes_per_frame))
        return min(safe_batch, 16)

    # -----------------------------------------------------------------------
    def _process_frame(self, frame, net, device,
                       taa_strength, taa_history_alpha,
                       jitter_scale, dlaa_strength,
                       edge_threshold, blur_radius):
        """Tek bir mini-batch kareyi işle."""
        img = frame.permute(0, 3, 1, 2).float()

        has_alpha = img.shape[1] == 4
        if has_alpha:
            alpha_ch = img[:, 3:4]
            img_rgb = img[:, :3]
        else:
            img_rgb = img

        # Adım 1: Jitter
        jittered = self._apply_jitter(img_rgb, jitter_scale)

        # Adım 2: Edge AA
        edge_aa = self._edge_aa(jittered, edge_threshold, blur_radius)

        # Adım 3: TAA
        if taa_strength > 0.0:
            taa_out = VideoTAADLAA._taa_history.update(edge_aa, taa_history_alpha)
            taa_out = (1.0 - taa_strength) * edge_aa + taa_strength * taa_out
        else:
            taa_out = edge_aa

        taa_out = torch.clamp(taa_out, 0.0, 1.0)

        # Adım 4: DLAA
        if dlaa_strength > 0.0:
            dlaa_out = net(taa_out)
            result_rgb = (1.0 - dlaa_strength) * taa_out + dlaa_strength * dlaa_out
        else:
            result_rgb = taa_out

        result_rgb = torch.clamp(result_rgb, 0.0, 1.0)

        if has_alpha:
            result = torch.cat([result_rgb, alpha_ch], dim=1)
        else:
            result = result_rgb

        return result.permute(0, 2, 3, 1)

    # -----------------------------------------------------------------------
    def apply_taa_dlaa(self, images, taa_strength, taa_history_alpha,
                       jitter_scale, dlaa_strength, edge_threshold,
                       blur_radius, reset_history, batch_size):

        if reset_history:
            VideoTAADLAA._taa_history.reset()

        device = images.device

        # DLAA ağı lazy init
        if VideoTAADLAA._dlaa_net is None:
            VideoTAADLAA._dlaa_net = _DLAANet()
        net = VideoTAADLAA._dlaa_net.to(device).eval()

        # batch_size=0 → otomatik
        if batch_size == 0:
            batch_size = self._auto_batch_size(images)
            print(f"[VideoTAADLAA] Auto batch size: {batch_size}")

        total_frames = images.shape[0]
        print(f"[VideoTAADLAA] Total frames: {total_frames} | Batch size: {batch_size}")

        results = []

        with torch.no_grad():
            for start in range(0, total_frames, batch_size):
                end = min(start + batch_size, total_frames)
                mini_batch = images[start:end].to(device)

                processed = self._process_frame(
                    mini_batch, net, device,
                    taa_strength, taa_history_alpha,
                    jitter_scale, dlaa_strength,
                    edge_threshold, blur_radius
                )

                # İşlenince CPU'ya at → VRAM boşalt
                results.append(processed.cpu())

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(f"[VideoTAADLAA] Processed frames {start+1}-{end} / {total_frames}")

        final = torch.cat(results, dim=0).to(device)
        return (final,)

    # -----------------------------------------------------------------------
    @staticmethod
    def _apply_jitter(x: torch.Tensor, scale: float) -> torch.Tensor:
        if scale == 0.0:
            return x
        B, C, H, W = x.shape
        jx = (0.5 / W) * scale
        jy = (0.5 / H) * scale
        theta = torch.tensor(
            [[1.0, 0.0, jx], [0.0, 1.0, jy]], dtype=torch.float32, device=x.device
        ).unsqueeze(0).expand(B, -1, -1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    # -----------------------------------------------------------------------
    @staticmethod
    def _edge_aa(img_rgb: torch.Tensor, edge_threshold: float, blur_radius: int) -> torch.Tensor:
        gray = (
            0.2989 * img_rgb[:, 0:1]
            + 0.5870 * img_rgb[:, 1:2]
            + 0.1140 * img_rgb[:, 2:3]
        )
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3).to(img_rgb.device)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3).to(img_rgb.device)

        edges = torch.sqrt(
            F.conv2d(gray, sobel_x, padding=1) ** 2
            + F.conv2d(gray, sobel_y, padding=1) ** 2
            + 1e-6
        )
        mask = torch.clamp((edges > edge_threshold).float(), 0.0, 1.0)
        ks = blur_radius * 2 + 1
        padded = F.pad(img_rgb, [blur_radius] * 4, mode="replicate")
        blurred = F.avg_pool2d(padded, kernel_size=ks, stride=1, padding=0)
        return img_rgb * (1.0 - mask) + blurred * mask


# ---------------------------------------------------------------------------
# ComfyUI kayıt
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "VideoTAADLAA": VideoTAADLAA,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoTAADLAA": "🎮 Video TAA + DLAA Anti-Aliasing",
}
