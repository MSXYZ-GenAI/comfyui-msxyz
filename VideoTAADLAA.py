# Created by MSXYZ (AI-assisted)
# Temporal (TAA) + DLAA Anti-Aliasing Inference
# v0.1.1 - Smart Neural Pass

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


# Model Inference
class DLAANet(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer("sobel_x",
            torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3))
        self.register_buffer("sobel_y",
            torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3))
        self.register_buffer("jitter_offsets",
            torch.tensor([[0.25,0.25],[-0.25,-0.25],[-0.25,0.25],[0.25,-0.25]], dtype=torch.float32))

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, 3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.reconstructor = nn.Conv2d(32, 3, 3, padding=1, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.zeros_(self.reconstructor.weight)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b  = self.bottleneck(e2)
        d  = self.dec(torch.cat([b, e1], dim=1))
        r  = self.reconstructor(d)
        return x + r


class TAAState:
    def __init__(self):
        self.history  = None
        self.frame_id = 0

    def reset(self):
        self.history  = None
        self.frame_id = 0

    def update(self, frame, alpha, sensitivity):
        if self.history is None or self.history.shape != frame.shape:
            self.history = frame.detach().clone().to(frame.device)
            return frame

        local_min = -F.max_pool2d(-frame, kernel_size=3, stride=1, padding=1)
        local_max =  F.max_pool2d( frame, kernel_size=3, stride=1, padding=1)
        history_clipped = torch.maximum(torch.minimum(self.history, local_max), local_min)

        diff          = torch.abs(frame - history_clipped).mean(dim=1, keepdim=True)
        motion_mask   = torch.sigmoid((diff - sensitivity) * 15.0)
        dynamic_alpha = alpha * (1.0 - motion_mask)

        out = torch.lerp(frame, history_clipped, dynamic_alpha)
        self.history = out.detach()
        return out

# Sharpen helper
def _unsharp_mask(x: torch.Tensor, strength: float) -> torch.Tensor:
    if strength < 1e-4:
        return x
    blurred = F.avg_pool2d(F.pad(x, [2, 2, 2, 2], mode="reflect"), 5, stride=1)
    return torch.clamp(x + (x - blurred) * strength, 0.0, 1.0)


# ComfyUI
class VideoTAADLAA:
    def __init__(self):
        self.net_cache = {}
        self.taa = TAAState()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images"            : ("IMAGE",),
                "taa_strength"      : ("FLOAT", {"default": 0.45, "min": 0, "max": 1, "step": 0.05}),
                "dlaa_strength"     : ("FLOAT", {"default": 0.65, "min": 0, "max": 1, "step": 0.05}),
                "sharpen_strength"  : ("FLOAT", {"default": 0.15, "min": 0, "max": 2, "step": 0.05}),
                "motion_sensitivity": ("FLOAT", {"default": 0.08, "min": 0, "max": 0.3, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION     = "execute"
    CATEGORY     = "CustomPostProcess"

    def _net(self, device):
        key = str(device)
        if key not in self.net_cache:
            net = DLAANet().to(device)
            net = net.to(memory_format=torch.channels_last)

            base_path = os.path.dirname(os.path.realpath(__file__))
            path      = os.path.join(base_path, "DLAANet.pth")

            if not os.path.exists(path):
                raise FileNotFoundError(f"[DLAA] Model file not found: {path}")

            state_dict = torch.load(path, map_location=device)
            net.load_state_dict(state_dict)
            net.eval()

            n_params = sum(p.numel() for p in net.parameters())
            print(f"\033[92m[DLAA] Model loaded: {path}  ({n_params:,} params)\033[0m")

            self.net_cache[key] = net

        return self.net_cache[key]

    def _tiled_forward(self, net, x: torch.Tensor, tile_size: int = 512, overlap: int = 32) -> torch.Tensor:
        
        # Splits x into overlapping tiles to reduce VRAM usage
        B, C, H, W = x.shape

        # Small image
        if H <= tile_size and W <= tile_size:
            return torch.clamp(net(x), 0.0, 1.0)

        step   = tile_size - overlap * 2
        out    = torch.zeros_like(x)
        weight = torch.zeros(B, 1, H, W, device=x.device)

        y0 = 0
        while y0 < H:
            y1    = min(y0 + tile_size, H)
            y0_c  = max(0, y1 - tile_size)   # always full size

            x0 = 0
            while x0 < W:
                x1    = min(x0 + tile_size, W)
                x0_c  = max(0, x1 - tile_size)

                tile         = x[:, :, y0_c:y1, x0_c:x1]
                refined_tile = torch.clamp(net(tile), 0.0, 1.0)

                # Smooth blend weight
                th, tw = refined_tile.shape[2], refined_tile.shape[3]
                w_map  = torch.ones(1, 1, th, tw, device=x.device)
                ramp   = min(overlap, th // 4, tw // 4)

                if ramp > 0:
                    for k in range(ramp):
                        v = (k + 1) / (ramp + 1)
                        w_map[:, :, k, :]      = torch.clamp(w_map[:, :, k, :],      max=v)
                        w_map[:, :, th-1-k, :] = torch.clamp(w_map[:, :, th-1-k, :], max=v)
                        w_map[:, :, :, k]      = torch.clamp(w_map[:, :, :, k],      max=v)
                        w_map[:, :, :, tw-1-k] = torch.clamp(w_map[:, :, :, tw-1-k], max=v)

                out[:, :, y0_c:y1, x0_c:x1]    += refined_tile * w_map
                weight[:, :, y0_c:y1, x0_c:x1] += w_map

                if x1 == W:
                    break
                x0 += step

            if y1 == H:
                break
            y0 += step

        return torch.clamp(out / weight.clamp(min=1e-6), 0.0, 1.0)

    def _jitter(self, x, idx, scale, net):
        if scale < 1e-5:
            return x
        off  = net.jitter_offsets[idx % 4]
        B, C, H, W = x.shape
        theta = torch.eye(2, 3, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        theta[:, 0, 2] = off[0] * scale / W
        theta[:, 1, 2] = off[1] * scale / H
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    def _edge_aa(self, x, thr, blur, net):
        if blur <= 0:
            return x
        gray = 0.2126*x[:,0:1] + 0.7152*x[:,1:2] + 0.0722*x[:,2:3]
        sx   = F.conv2d(gray, net.sobel_x, padding=1)
        sy   = F.conv2d(gray, net.sobel_y, padding=1)
        edge = torch.sqrt(sx*sx + sy*sy + 1e-6)
        mask = torch.sigmoid((edge - thr) * 8.0)
        blurred = F.avg_pool2d(F.pad(x, [blur]*4, mode="reflect"), blur*2+1, stride=1)
        return x*(1.0 - mask) + blurred*mask

    def execute(self, images, taa_strength, motion_sensitivity, dlaa_strength, sharpen_strength):

        device = mm.get_torch_device() if mm else \
                 torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Fixed Settings
        taa_alpha      = 0.10
        jitter_scale   = 0.20
        edge_threshold = 0.15
        
        # IMAGE or VIDEO
        B, H, W, C = images.shape
        is_single_image = (B == 1)

        if is_single_image:
            blur_radius   = 0
            reset_history = True
        else:
            blur_radius   = 1
            reset_history = False
            
        if reset_history:
            self.taa.reset()

        net        = self._net(device)
        B, H, W, C = images.shape
        out_list   = []

        # Auto tile size for VRAM
        if mm is not None:
            try:
                vram_mb = torch.cuda.get_device_properties(device).total_memory // (1024 * 1024)
            except:
                vram_mb = 8192
        else:
            vram_mb = 8192

        if   vram_mb <= 8192:  tile_size = 512
        elif vram_mb <= 16384: tile_size = 1024
        else:                  tile_size = 2048

        # Log tiling
        if H > tile_size or W > tile_size:
            step    = tile_size - 64
            n_tiles = ((H + step - 1) // step) * ((W + step - 1) // step)
            print(f"\033[93m[DLAA] Tiled mode: {H}x{W} → ~{n_tiles} tiles ({tile_size}px, VRAM:{vram_mb}MB)\033[0m")
        else:
            print(f"\033[92m[DLAA] Single-pass mode: {H}x{W}\033[0m")

        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda")
        ):
            for i in range(B):
                img = images[i:i+1].to(device).permute(0, 3, 1, 2)
                rgb = img[:, :3]
                rgb = rgb.contiguous(memory_format=torch.channels_last).float()

                # jitter
                fid = self.taa.frame_id
                self.taa.frame_id = (fid + 1) % 4
                rgb = self._jitter(rgb, fid, jitter_scale, net)

                # Edge-based blur
                rgb = self._edge_aa(rgb, edge_threshold, blur_radius, net)

                # TAA temporal accumulation
                taa_out = self.taa.update(rgb, taa_alpha, motion_sensitivity)
                rgb     = torch.lerp(rgb, taa_out, taa_strength)

                # DLAA Smart Neural Pass
                if dlaa_strength > 0:
                    # AI Processing
                    try:
                        refined = self._tiled_forward(net, rgb, tile_size=tile_size, overlap=32)
                    except RuntimeError as OOMRec:
                        if "out of memory" in str(OOMRec).lower():
                            print("\033[91m[DLAA] OOM detected, retrying with smaller tiles...\033[0m")
                            tile_try = tile_size // 2

                            for _ in range(3):
                                try:
                                    refined = self._tiled_forward(net, rgb, tile_size=tile_try, overlap=32)
                                    break # Exit if successful
                                except RuntimeError as OOMRecInner:
                                    if "out of memory" in str(OOMRecInner).lower():
                                        print(f"\033[91m[DLAA] OOM → retry with {tile_try//2}px\033[0m")
                                        if torch.cuda.is_available():
                                            torch.cuda.empty_cache()
                                        tile_try = max(256, tile_try // 2)
                                    else:
                                        raise OOMRecInner
                            else:
                                raise RuntimeError("DLAA failed even after retries")
                        else:
                            raise OOMRec

                    
                    # Smart Luma
                    raw_luma = torch.mean(rgb) 
                    new_luma = torch.mean(refined)

                    luma_boost = (raw_luma / (new_luma + 1e-6)).clamp(1.0, 1.35) 
                    refined = (refined * luma_boost).clamp(0.0, 1.0) 
                    
                    # Smart Clarity
                    low_freq = F.avg_pool2d(F.pad(refined, [5, 5, 5, 5], mode="reflect"), 11, stride=1)
                    details = refined - low_freq
                    detail_std = torch.std(details)
                    
                    auto_clarity_gain = (0.06 / (detail_std + 1e-6)).clamp(0.0, 0.6) 
                    refined = refined + (details * auto_clarity_gain)

                    # Final result
                    mse = torch.mean((refined - rgb) ** 2).item()
                    if i == 0:
                        print(f"\033[92m[DLAA] MSE delta: \033[93m{mse:.6f}\033[0m")

                    # Blend the original image
                    rgb = torch.lerp(rgb, refined, dlaa_strength)
                    rgb = torch.clamp(rgb, 0.0, 1.0)

                # Post-sharpening
                if sharpen_strength > 0:
                    rgb = _unsharp_mask(rgb, sharpen_strength)

                out_list.append(rgb.permute(0, 2, 3, 1).cpu())

                if mm is not None and i > 0 and i % 50 == 0:
                    mm.soft_empty_cache()

        return (torch.cat(out_list, dim=0),)


NODE_CLASS_MAPPINGS        = {"VideoTAADLAA": VideoTAADLAA}
NODE_DISPLAY_NAME_MAPPINGS = {"VideoTAADLAA": "🎮 Video TAA + DLAA Anti-Aliasing"}