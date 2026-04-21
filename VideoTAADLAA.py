# Created by MSXYZ (AI-assisted)
# Temporal (TAA) + Lightweight DLAA-style Anti-Aliasing
# v0.1.1 - Active Weights, MSE Monitoring & Pixel Integrity

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
try:
    import comfy.model_management as mm
except:
    mm = None

logger = logging.getLogger("VideoTAADLAA")


class _DLAANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("sobel_x", torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("sobel_y", torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("jitter_offsets", torch.tensor([[0.25, 0.25], [-0.25, -0.25], [-0.25, 0.25], [0.25, -0.25]], dtype=torch.float32))

        self.extract_feature = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.refiner = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
        )
        self.reconstructor = nn.Conv2d(32, 3, 3, padding=1, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.extract_feature.weight)
        for m in self.refiner.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.zeros_(self.reconstructor.weight)

    def forward(self, x):
        features = F.leaky_relu(self.extract_feature(x), 0.2)
        refined = self.refiner(features)
        residual = self.reconstructor(refined)
        return x + residual


class _TAAState:
    def __init__(self):
        self.history = None
        self.frame_id = 0

    def reset(self):
        self.history = None
        self.frame_id = 0

    def update(self, frame, alpha, sensitivity):
        if self.history is None or self.history.shape != frame.shape:
            self.history = frame.detach().clone().to(frame.device)
            return frame

        # Neighborhood clamping
        local_min = -F.max_pool2d(-frame, kernel_size=3, stride=1, padding=1)
        local_max = F.max_pool2d(frame, kernel_size=3, stride=1, padding=1)
        history_clipped = torch.maximum(torch.minimum(self.history, local_max), local_min)

        diff = torch.abs(frame - history_clipped).mean(dim=1, keepdim=True)
        motion_mask = torch.sigmoid((diff - sensitivity) * 45.0)
        dynamic_alpha = alpha * (1.0 - motion_mask)

        out = torch.lerp(frame, history_clipped, dynamic_alpha)
        self.history = out.detach()
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
                "taa_strength": ("FLOAT", {"default": 0.50, "min": 0, "max": 1, "step": 0.05}),
                "taa_alpha": ("FLOAT", {"default": 0.20, "min": 0, "max": 0.9, "step": 0.01}),
                "motion_sensitivity": ("FLOAT", {"default": 0.1, "min": 0, "max": 0.3, "step": 0.01}),
                "jitter_scale": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.01}),
                "dlaa_strength": ("FLOAT", {"default": 0.40, "min": 0, "max": 1, "step": 0.05}),
                "edge_threshold": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 0.3, "step": 0.01}),
                "blur_radius": ("INT", {"default": 0, "min": 0, "max": 3, "step": 1}),
                "reset_history": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "CustomPostProcess"

    def _net(self, device):
        key = str(device)
        if key not in self.net_cache:
            net = _DLAANet().to(device)
            
            base_path = os.path.dirname(os.path.realpath(__file__))
            path = os.path.join(base_path, "DLAANet.pth")
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"[DLAA] Model file not found: {path}")

            state_dict = torch.load(path, map_location=device, weights_only=True)
            net.load_state_dict(state_dict)

            print(f"\033[92m[DLAA] Model loaded successfully: {path}\033[0m")
            
            net.eval()
            self.net_cache[key] = net
            
        return self.net_cache[key]

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
        
        gray = 0.2126*x[:,0:1] + 0.7152*x[:,1:2] + 0.0722*x[:,2:3]
        sx, sy = F.conv2d(gray, net.sobel_x, padding=1), F.conv2d(gray, net.sobel_y, padding=1)
        edge = torch.sqrt(sx*sx + sy*sy + 1e-6)
        
        mask = torch.sigmoid((edge - thr) * 8.0)
        blurred = F.avg_pool2d(F.pad(x, [blur]*4, mode="reflect"), blur*2+1, stride=1)
        
        return x*(1-mask) + blurred*mask

    def execute(self, images, taa_strength, taa_alpha, motion_sensitivity,
                jitter_scale, dlaa_strength, edge_threshold,
                blur_radius, reset_history=True):

        device = mm.get_torch_device() if mm else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if reset_history:
            self.taa.reset()
            torch.cuda.empty_cache()

        net = self._net(device)

        B, H, W, C = images.shape
        out_list = []

        with torch.no_grad():
            for i in range(B):
                # convert to NCHW for PyTorch
                img = images[i:i+1].to(device).permute(0, 3, 1, 2).float()
                rgb = img[:, :3]

                fid = self.taa.frame_id
                self.taa.frame_id = (fid + 1) % 4
                
                rgb = self.jitter(rgb, fid, jitter_scale, net)
                rgb = self.edge_aa(rgb, edge_threshold, blur_radius, net)

                taa_out = self.taa.update(rgb, taa_alpha, motion_sensitivity)
                rgb = torch.lerp(rgb, taa_out, taa_strength)
                
                
                if dlaa_strength > 0:

                    refined_output = torch.clamp(net(rgb), 0.0, 1.0)
                    diff_check = torch.mean((refined_output - rgb) ** 2).item()
                    if i == 0: 
                        print(f"\033[92m[DLAA] Model Delta (MSE): \033[93m{diff_check:.6f}\033[0m")
                        
                    # residual = refined_output - rgb
                    # luma_res = 0.2126*residual[:,0:1] + 0.7152*residual[:,1:2] + 0.0722*residual[:,2:3]
                    # rgb = rgb + (luma_res * dlaa_strength * 2.5)
                    # rgb = rgb * (1.0 + dlaa_strength * 0.4)
                    # rgb = torch.clamp(rgb, 0.0, 1.0)
                    # rgb = torch.pow(rgb, 0.98)
                    
                    rgb = torch.lerp(rgb, refined_output, dlaa_strength)
                    rgb = torch.clamp(rgb, 0.0, 1.0)

                out_list.append(rgb.permute(0,2,3,1).cpu())

                if i % 10 == 0 and mm is not None:
                    mm.soft_empty_cache()

        return (torch.cat(out_list, dim=0),)

NODE_CLASS_MAPPINGS = {
    "VideoTAADLAA": VideoTAADLAA
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoTAADLAA": "🎮 Video TAA + DLAA Anti-Aliasing"
}