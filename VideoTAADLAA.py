# Created by MSXYZ (AI-assisted)
# Temporal Anti-Aliasing (TAA) + Lightweight DLAA-style Sharpening
# v0.1.1 - Artifact reduction + high-frequency noise fix

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import comfy.model_management as mm

logger = logging.getLogger("VideoTAADLAA")

class _DLAANet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 3-layer CNN post-process filter
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1, bias=False),
        )
        
        self.register_buffer("sobel_x", torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("sobel_y", torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("jitter_offsets", torch.tensor([[0.25, 0.25], [-0.25, -0.25], [-0.25, 0.25], [0.25, -0.25]], dtype=torch.float32))
        self._init()

    def _init(self):
        convs = [m for m in self.conv if isinstance(m, nn.Conv2d)]
        
        # Layer 1: RGB Identity
        with torch.no_grad():
            convs[0].weight.zero_()
            for c in range(3):
                convs[0].weight[c, c, 1, 1] = 1.0
                
        # Layer 2: Linear passthrough
        with torch.no_grad():
            convs[1].weight.zero_()
            for c in range(min(convs[1].weight.shape[0], convs[1].weight.shape[1])):
                convs[1].weight[c, c, 1, 1] = 1.0
        
        # Layer 3: Controlled sharpening kernel to prevent dots/artifacts
        nn.init.zeros_(convs[2].weight)
        # Smoother laplacian kernel (Artifact resistant)
        sharpen = torch.tensor([[-0.25, -0.5, -0.25], [-0.5, 4.0, -0.5], [-0.25, -0.5, -0.25]], dtype=torch.float32)
        with torch.no_grad():
            for out_c in range(3):
                convs[2].weight[out_c, out_c] = sharpen * 0.22

    def forward(self, x):
        # Balanced sharpen intensity
        return torch.clamp(x + self.conv(x) * 0.55, 0.0, 1.0)


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

        # Neighborhood clamping for stability
        local_min = -F.max_pool2d(-frame, kernel_size=3, stride=1, padding=1)
        local_max = F.max_pool2d(frame, kernel_size=3, stride=1, padding=1)
        history_clipped = torch.clamp(self.history, local_min, local_max)

        # Motion masking logic
        diff = torch.abs(frame - history_clipped).mean(dim=1, keepdim=True)
        motion_mask = torch.sigmoid((diff - sensitivity) * 45.0)
        dynamic_alpha = alpha * (1.0 - motion_mask)

        out = torch.lerp(frame, history_clipped, dynamic_alpha)
        self.history = out.detach()
        self.frame_id += 1
        return out

class VideoTAADLAA:
    def __init__(self):
        self.net_cache = {}
        self.taa = _TAAState()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "taa_strength": ("FLOAT", {"default": 0.55, "min": 0, "max": 1, "step": 0.05}), 
                "taa_alpha": ("FLOAT", {"default": 0.45, "min": 0, "max": 0.9, "step": 0.01}),   
                "motion_sensitivity": ("FLOAT", {"default": 0.07, "min": 0.01, "max": 0.5, "step": 0.01}),
                "jitter_scale": ("FLOAT", {"default": 0.06, "min": 0, "max": 0.08, "step": 0.01}),
                "dlaa_strength": ("FLOAT", {"default": 0.45, "min": 0, "max": 1, "step": 0.05}),
                "edge_threshold": ("FLOAT", {"default": 0.22, "min": 0, "max": 0.5, "step": 0.01}),
                "blur_radius": ("INT", {"default": 0, "min": 0, "max": 5, "step": 1}),
                "reset_history": ("BOOLEAN", {"default": False}),
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
        off = net.jitter_offsets[idx % 4]
        B, C, H, W = x.shape
        
        theta = torch.eye(2, 3, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        theta[:, 0, 2], theta[:, 1, 2] = off[0] * scale / W, off[1] * scale / H
        
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    def edge_aa(self, x, thr, blur, net):
        if blur <= 0: return x
        
        # Grayscale for edge detection
        gray = 0.299*x[:, 0:1] + 0.587*x[:, 1:2] + 0.114*x[:, 2:3]
        sx, sy = F.conv2d(gray, net.sobel_x, padding=1), F.conv2d(gray, net.sobel_y, padding=1)
        edge = torch.sqrt(sx*sx + sy*sy + 1e-6)
        
        mask = torch.sigmoid((edge - thr) * 8-10)
        blurred = F.avg_pool2d(F.pad(x, [blur]*4, mode="reflect"), blur*2+1, stride=1)
        
        return x*(1-mask) + blurred*mask

    def execute(self, images, taa_strength, taa_alpha, motion_sensitivity, jitter_scale, dlaa_strength, edge_threshold, blur_radius, reset_history=False):
        device = mm.get_torch_device()
        if reset_history: self.taa.reset()
        net = self._net(device)
        B, H, W, C = images.shape
        out_list = []
        
        with torch.no_grad():
            for i in range(B):
                # NCHW conversion
                img = images[i:i+1].to(device).permute(0, 3, 1, 2).float()
                rgb = img[:, :3]

                # 1. Spatial Preprocessing
                rgb = self.jitter(rgb, self.taa.frame_id, jitter_scale, net)
                rgb = self.edge_aa(rgb, edge_threshold, blur_radius, net)

                # 2. Temporal Pass
                taa_out = self.taa.update(rgb, taa_alpha, motion_sensitivity)
                rgb = torch.lerp(rgb, taa_out, taa_strength)

                # 3. Post-Sharpen
                if dlaa_strength > 0:
                    rgb = torch.lerp(rgb, net(rgb), dlaa_strength)
                
                out_list.append(rgb.permute(0, 2, 3, 1).cpu())
                
                if i % 10 == 0: mm.soft_empty_cache()
        
        return (torch.cat(out_list, dim=0),)

NODE_CLASS_MAPPINGS = {"VideoTAADLAA": VideoTAADLAA}
NODE_DISPLAY_NAME_MAPPINGS = {"VideoTAADLAA": "🎮 Video TAA + DLAA Anti-Aliasing"}