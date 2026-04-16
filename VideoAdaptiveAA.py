# Created By MSXYZ and Claude 3 Opus
# Memory & Performance Optimized

import torch
import torch.nn.functional as F


class VideoAdaptiveAA:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "edge_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blur_radius": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_aa"
    CATEGORY = "CustomPostProcess"

    def apply_aa(self, images, strength, edge_threshold, blur_radius):
        # ComfyUI Image format: [B, H, W, C] -> Torch format: [B, C, H, W]
        img = images.permute(0, 3, 1, 2)

        # --- BUG FIX #3: Alpha kanal desteği ---
        # Eğer görüntü RGBA ise (4 kanal), alpha kanalını ayır ve sadece RGB üzerinde işlem yap.
        has_alpha = img.shape[1] == 4
        if has_alpha:
            alpha = img[:, 3:4, :, :]  # Alpha kanalını ayır [B, 1, H, W]
            img_rgb = img[:, :3, :, :]  # Sadece RGB [B, 3, H, W]
        else:
            img_rgb = img

        # 1. Grayscale dönüşümü (Kenar tespiti için)
        grayscale = (
            0.2989 * img_rgb[:, 0:1, :, :]
            + 0.5870 * img_rgb[:, 1:2, :, :]
            + 0.1140 * img_rgb[:, 2:3, :, :]
        )

        # 2. Sobel Filtresi ile kenar tespiti
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3).to(img.device)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3).to(img.device)

        edges_x = F.conv2d(grayscale, sobel_x, padding=1)
        edges_y = F.conv2d(grayscale, sobel_y, padding=1)

        # NaN hatalarını önlemek için epsilon ekliyoruz
        edges = torch.sqrt(edges_x**2 + edges_y**2 + 1e-6)

        # 3. Kenar maskesi oluşturma
        mask = (edges > edge_threshold).float()
        mask = mask * strength
        # strength > 1 olsa bile renk patlaması olmaması için clamp
        mask = torch.clamp(mask, min=0.0, max=1.0)

        # 4. Bulanıklaştırma (Anti-aliasing etkisi)
        # --- BUG FIX #1: Sıfır-padding yerine replicate padding kullan ---
        # avg_pool2d varsayılan olarak sıfırla doldurur → kenar kararmasi!
        # Çözüm: önce replicate padding, sonra padding=0 ile pool.
        kernel_size = blur_radius * 2 + 1
        img_padded = F.pad(img_rgb, (blur_radius, blur_radius, blur_radius, blur_radius), mode="replicate")
        blurred = F.avg_pool2d(img_padded, kernel_size=kernel_size, stride=1, padding=0)

        # 5. Adaptive Blending: Sadece kenarları bulanık olanla değiştir
        # Formül: Orijinal * (1 - mask) + Bulanık * mask
        result_rgb = img_rgb * (1.0 - mask) + blurred * mask

        # --- BUG FIX #2: Çıkış değerlerini [0, 1] aralığına hapset ---
        result_rgb = torch.clamp(result_rgb, min=0.0, max=1.0)

        # Alpha kanalı varsa geri birleştir (alpha kanalına AA uygulanmaz)
        if has_alpha:
            result = torch.cat([result_rgb, alpha], dim=1)
        else:
            result = result_rgb

        # Formatı geri çevir [B, C, H, W] -> [B, H, W, C]
        result = result.permute(0, 2, 3, 1)

        return (result,)

NODE_CLASS_MAPPINGS = {
    "VideoAdaptiveAA": VideoAdaptiveAA
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoAdaptiveAA": "🚀 Video Adaptive Anti-Aliasing"
}