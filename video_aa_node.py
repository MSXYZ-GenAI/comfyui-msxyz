import torch
import torch.nn.functional as F
import numpy as np

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
        
        # 1. Grayscale dönüşümü (Kenar tespiti için)
        grayscale = 0.2989 * img[:, 0:1, :, :] + 0.5870 * img[:, 1:2, :, :] + 0.1140 * img[:, 2:3, :, :]
        
        # 2. Sobel Filtresi ile kenar tespiti
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(img.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(img.device)
        
        edges_x = F.conv2d(grayscale, sobel_x, padding=1)
        edges_y = F.conv2d(grayscale, sobel_y, padding=1)
        edges = torch.sqrt(edges_x**2 + edges_y**2)
        
        # 3. Kenar maskesi oluşturma
        mask = (edges > edge_threshold).float()
        mask = mask * strength
        
        # 4. Bulanıklaştırma (Anti-aliasing etkisi)
        kernel_size = blur_radius * 2 + 1
        blurred = F.avg_pool2d(img, kernel_size=kernel_size, stride=1, padding=blur_radius)
        
        # 5. Adaptive Blending: Sadece kenarları bulanık olanla değiştir
        # Formül: Orijinal * (1 - mask) + Bulanık * mask
        result = img * (1.0 - mask) + blurred * mask
        
        # Formatı geri çevir [B, C, H, W] -> [B, H, W, C]
        result = result.permute(0, 2, 3, 1)
        
        return (result,)

NODE_CLASS_MAPPINGS = {
    "VideoAdaptiveAA": VideoAdaptiveAA
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoAdaptiveAA": "🚀 Video Adaptive Anti-Aliasing"
}
