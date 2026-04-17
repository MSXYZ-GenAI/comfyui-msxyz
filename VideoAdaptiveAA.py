# Created by MSXYZ (AI-assisted)
# Lightweight Adaptive Anti-Aliasing Node
# Optimized for memory-efficient frame processing (ComfyUI pipeline)

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
        # Convert ComfyUI tensor format [B, H, W, C] to PyTorch [B, C, H, W]
        img = images.permute(0, 3, 1, 2)

        # Alpha channel handling (RGBA support)
        # If input contains alpha channel, process RGB only and preserve alpha
        has_alpha = img.shape[1] == 4
        if has_alpha:
            alpha = img[:, 3:4, :, :]  # Alpha kanalını ayır [B, 1, H, W]
            img_rgb = img[:, :3, :, :]  # Sadece RGB [B, 3, H, W]
        else:
            img_rgb = img

       # Convert RGB to luminance (grayscale) for edge detection
        grayscale = (
            0.2989 * img_rgb[:, 0:1, :, :]
            + 0.5870 * img_rgb[:, 1:2, :, :]
            + 0.1140 * img_rgb[:, 2:3, :, :]
        )

        # Sobel operators for gradient-based edge magnitude estimation
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3).to(img.device)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3).to(img.device)

        edges_x = F.conv2d(grayscale, sobel_x, padding=1)
        edges_y = F.conv2d(grayscale, sobel_y, padding=1)

        # Add epsilon to prevent numerical instability (NaN in sqrt)
        edges = torch.sqrt(edges_x**2 + edges_y**2 + 1e-6)

        # Generate binary edge mask and apply strength scaling
        mask = (edges > edge_threshold).float()
        mask = mask * strength
        # strength > 1 olsa bile renk patlaması olmaması için clamp
        mask = torch.clamp(mask, min=0.0, max=1.0)

        # 4. Bulanıklaştırma (Anti-aliasing etkisi)
        # Edge-preserving blur using replicate padding
        # Avoid zero-padding artifacts that can cause border darkening
        # Symmetric blur kernel size based on radius
        kernel_size = blur_radius * 2 + 1
        img_padded = F.pad(img_rgb, (blur_radius, blur_radius, blur_radius, blur_radius), mode="replicate")
        blurred = F.avg_pool2d(img_padded, kernel_size=kernel_size, stride=1, padding=0)

        # Edge-aware blending: interpolates between original and blurred image
        # result = original * (1 - mask) + blurred * mask
        result_rgb = img_rgb * (1.0 - mask) + blurred * mask

        # Clamp output to valid image range [0, 1]
        result_rgb = torch.clamp(result_rgb, min=0.0, max=1.0)

        # Reattach alpha channel if present (alpha is preserved, not processed)
        if has_alpha:
            result = torch.cat([result_rgb, alpha], dim=1)
        else:
            result = result_rgb

        # Convert back to ComfyUI format [B, H, W, C]
        result = result.permute(0, 2, 3, 1)

        return (result,)

NODE_CLASS_MAPPINGS = {
    "VideoAdaptiveAA": VideoAdaptiveAA
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoAdaptiveAA": "🚀 Video Adaptive Anti-Aliasing"
}