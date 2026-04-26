# Video TAA + DLAA
# v0.1.0
# AI-assisted implementation

import torch
import torch.nn.functional as F

class VideoAdaptiveAA:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "edge_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blur_radius": ("INT", {"default": 1, "min": 0, "max": 3, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_aa"
    CATEGORY = "CustomPostProcess"

    def apply_aa(self, images, strength, edge_threshold, blur_radius):
        img = images.permute(0, 3, 1, 2)
        
        if strength <= 0 or blur_radius <= 0:
            return (images,)

        has_alpha = img.shape[1] == 4
        if has_alpha:
            alpha = img[:, 3:4, :, :]
            img_rgb = img[:, :3, :, :]
        else:
            img_rgb = img

        img_gray = (
            0.2 * img_rgb[:, 0:1, :, :]
            + 0.7 * img_rgb[:, 1:2, :, :]
            + 0.1 * img_rgb[:, 2:3, :, :]
        )
        
        get_sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device
        ).view(1, 1, 3, 3)

        get_sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=img.dtype, device=img.device
        ).view(1, 1, 3, 3)

        edges_x = F.conv2d(img_gray, get_sobel_x, padding=1)
        edges_y = F.conv2d(img_gray, get_sobel_y, padding=1)

        edges = torch.sqrt(edges_x**2 + edges_y**2 + 1e-6)

        fade_width = 0.1
        mask = torch.clamp((edges - edge_threshold) / fade_width, 0.0, 1.0)
        mask = torch.clamp(mask * strength, 0.0, 1.0)

        kernel_size = blur_radius * 2 + 1
        img_padded = F.pad(
            img_rgb,
            (blur_radius, blur_radius, blur_radius, blur_radius),
            mode="replicate",
        )
        blurred = F.avg_pool2d(img_padded, kernel_size=kernel_size, stride=1, padding=0)

        result_rgb = img_rgb * (1.0 - mask) + blurred * mask
        result_rgb = torch.clamp(result_rgb, 0.0, 1.0)

        if has_alpha:
            result = torch.cat([result_rgb, alpha], dim=1)
        else:
            result = result_rgb

        result = result.permute(0, 2, 3, 1)

        return (result,)


NODE_CLASS_MAPPINGS = {
    "VideoAdaptiveAA": VideoAdaptiveAA
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoAdaptiveAA": "✨ Video Adaptive Anti-Aliasing"
}