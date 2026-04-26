# Video TAA + DLAA
# v0.1.2
# AI-assisted implementation


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from safetensors.torch import load_file

try:
    import comfy.model_management as mm
except ImportError:
    mm = None
logger = logging.getLogger("VideoTAADLAA")


class DLAANet(nn.Module):
    """Small refinement network used by the DLAA pass."""
    def __init__(self):
        super().__init__()

        base_channels = 192

        # Sobel kernels
        self.register_buffer("sobel_x", torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3))
        self.register_buffer("sobel_y", torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3))
        
        # subpixel offsets
        self.register_buffer(
            "jitter_offsets",
            torch.tensor([
                [ 0.0000, -0.1667],
                [-0.2500,  0.1667],
                [ 0.2500, -0.3889],
                [-0.3750, -0.0556],
                [ 0.1250,  0.2778],
                [-0.1250, -0.2778],
                [ 0.3750,  0.0556],
                [-0.4375,  0.3889],
                [ 0.0625, -0.4630],
                [-0.3125,  0.1296],
                [ 0.1875, -0.2963],
                [-0.4375, -0.0370],
                [ 0.3125,  0.2407],
                [-0.0625, -0.2037],
                [ 0.4375,  0.0185],
                [-0.4688,  0.3148],
            ], dtype=torch.float32)
        )

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(12, base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(12, base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # multi-scale context
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1, dilation=1, bias=False),
            nn.GroupNorm(12, base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(12, base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels, 3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(12, base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels, 3, padding=1, dilation=1, bias=False),
            nn.GroupNorm(12, base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.dec = nn.Sequential(
            # skip connection
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(12, base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.reconstructor = nn.Conv2d(64, 3, 3, padding=1, bias=False)
        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="leaky_relu")
            elif isinstance(layer, nn.GroupNorm):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

        nn.init.zeros_(self.reconstructor.weight)

    def forward(self, x):
        input_img = x

        feat_low  = self.enc1(input_img)
        feat_high = self.enc2(feat_low)

        context = self.bottleneck(feat_high)

        decoded = self.dec(torch.cat([context, feat_low], dim=1))
        residual = self.reconstructor(decoded)

        return input_img + residual


class TAAState:
    # TAA state
    def __init__(self,
        variance_gamma=1.5,
        edge_guard_strength=2.0,
        edge_guard_min=0.25,
        edge_guard_max=1.0
    ):
        self.history  = None
        self.frame_id = 0

        self.variance_gamma      = variance_gamma
        self.edge_guard_strength = edge_guard_strength
        self.edge_guard_min      = edge_guard_min
        self.edge_guard_max      = edge_guard_max

    def reset(self):
        self.history  = None
        self.frame_id = 0
        
    @staticmethod
    def smoothstep(x: torch.Tensor) -> torch.Tensor:
        return x * x * (3.0 - 2.0 * x)

    def update(self, frame, alpha, sensitivity):
        if self.history is None or self.history.shape != frame.shape:
            self.history = frame.detach().clone().to(frame.device)
            return frame
            
        local_mean = F.avg_pool2d(frame, kernel_size=3, stride=1, padding=1)
        local_sq_mean = F.avg_pool2d(frame * frame, kernel_size=3, stride=1, padding=1)
        local_var = (local_sq_mean - local_mean * local_mean).clamp(min=0.0)
        local_std = torch.sqrt(local_var + 1e-6)
        
        local_min = local_mean - local_std * self.variance_gamma
        local_max = local_mean + local_std * self.variance_gamma

        history_clipped = torch.maximum(torch.minimum(self.history, local_max), local_min)

        diff = torch.abs(frame - history_clipped).mean(dim=1, keepdim=True)
        
        # detect disocclusion
        raw_diff = torch.abs(frame - self.history).mean(dim=1, keepdim=True)

        disocclusion = ((raw_diff - sensitivity * 2.0) / (sensitivity + 1e-6)).clamp(0.0, 1.0)
        disocclusion    = self.smoothstep(disocclusion)
        
        gray = 0.299 * frame[:, 0:1] + 0.587 * frame[:, 1:2] + 0.114 * frame[:, 2:3]

        edge_x = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
        edge_y = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])

        edge_x = F.pad(edge_x, (0, 1, 0, 0))
        edge_y = F.pad(edge_y, (0, 0, 0, 1))

        edge_strength = (edge_x + edge_y).clamp(0.0, 1.0)
        edge_guard = (
            self.edge_guard_max - edge_strength * self.edge_guard_strength
        ).clamp(self.edge_guard_min, self.edge_guard_max)

        motion_soft = ((diff - sensitivity) / (sensitivity + 1e-6)).clamp(0.0, 1.0)
        motion_soft     = self.smoothstep(motion_soft)

        dynamic_alpha = alpha * (1.0 - motion_soft) * edge_guard
        
        confidence = torch.exp(-diff * 10.0)
        confidence = confidence * (1.0 - disocclusion)
        confidence = confidence.clamp(0.15, 1.0)
        
        dynamic_alpha = dynamic_alpha * confidence
        
        reject_strength = ((diff - sensitivity * 1.5) / (sensitivity + 1e-6)).clamp(0.0, 1.0)
        reject_strength = torch.maximum(reject_strength, disocclusion)
        reject_strength = self.smoothstep(reject_strength)

        history_clipped = torch.lerp(history_clipped, frame, reject_strength)

        out = torch.lerp(frame, history_clipped, dynamic_alpha)
        self.history = out.detach()
        return out
        
        
class VideoTAADLAA:
    def __init__(self):
        self.net_cache = {}
        self.taa       = TAAState()
        
        self._prev_dlaa_output = None

        # TAA
        self.taa_alpha      = 0.10
        self.jitter_scale   = 0.20
        self.edge_threshold = 0.15

        # DLAA blend
        self.dlaa_blend_scale = 0.85

        # Highlight / tone
        self.tone_curve_bias      = 0.6
        self.highlight_pre_blend  = 0.15
        self.highlight_post_blend = 0.08
        self.highlight_threshold  = 0.85
        self.highlight_slope      = 12.0

        # Detail
        self.detail_base_scale = 9.0
        self.detail_ref_scale  = 0.02
        self.detail_min_scale  = 6.0
        self.detail_max_scale  = 12.0
        self.detail_min_gain   = 0.10
        self.detail_max_gain   = 0.26
        self.detail_edge_boost = 0.35
        self.detail_highlight_suppression = 0.5

        # Edge sharpening
        self.edge_sharp_threshold = 0.08
        self.edge_sharp_slope     = 12.0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images"            : ("IMAGE",),
                "preset": (["Auto", "Balanced", "Detail", "Smooth"],),
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION     = "execute"
    CATEGORY     = "CustomPostProcess"

    def _net(self, device):
        key = str(device)
        if key not in self.net_cache:
            net = DLAANet().to(device)

            base_path = os.path.dirname(os.path.realpath(__file__))

            safetensors_path = os.path.join(base_path, "DLAANet.safetensors")
            pth_path         = os.path.join(base_path, "DLAANet.pth")

            if os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path, device=str(device))
                logger.info("Loaded DLAANet.safetensors")
            elif os.path.exists(pth_path):
                state_dict = torch.load(pth_path, map_location=device)
                logger.info("Loaded DLAANet.pth")
            else:
                raise FileNotFoundError(
                    f"[DLAA] Model not found. Expected: {safetensors_path} or {pth_path}"
                )
            
            if "jitter_offsets" in state_dict:
                del state_dict["jitter_offsets"]

            net.load_state_dict(state_dict, strict=False)
            net.eval()
            
            n_params = sum(p.numel() for p in net.parameters())
            logger.info(f"[DLAA] Model parameters: {n_params/1e6:.2f}M")

            self.net_cache[key] = net

        return self.net_cache[key]

    def _tiled_forward(self, net, x: torch.Tensor, tile_size: int = 512, overlap: int = 32) -> torch.Tensor:
        
        B, C, H, W = x.shape

        if H <= tile_size and W <= tile_size:
            return torch.clamp(net(x), 0.0, 1.0)

        step   = tile_size - overlap * 2
        out    = torch.zeros_like(x)
        weight = torch.zeros(B, 1, H, W, device=x.device)

        y0 = 0
        while y0 < H:
            y1    = min(y0 + tile_size, H)
            y0_c  = max(0, y1 - tile_size)   # clamp tile start

            x0 = 0
            while x0 < W:
                x1    = min(x0 + tile_size, W)
                x0_c  = max(0, x1 - tile_size)

                tile         = x[:, :, y0_c:y1, x0_c:x1]
                dlaa_out_tile = torch.clamp(net(tile), 0.0, 1.0)


                th, tw = dlaa_out_tile.shape[2], dlaa_out_tile.shape[3]
                w_map  = torch.ones(1, 1, th, tw, device=x.device)
                ramp   = min(overlap, th // 4, tw // 4)

                if ramp > 0:
                    for k in range(ramp):
                        v = (k + 1) / (ramp + 1)
                        w_map[:, :, k, :]      = torch.clamp(w_map[:, :, k, :],      max=v)
                        w_map[:, :, th-1-k, :] = torch.clamp(w_map[:, :, th-1-k, :], max=v)
                        w_map[:, :, :, k]      = torch.clamp(w_map[:, :, :, k],      max=v)
                        w_map[:, :, :, tw-1-k] = torch.clamp(w_map[:, :, :, tw-1-k], max=v)

                out[:, :, y0_c:y1, x0_c:x1]    += dlaa_out_tile * w_map
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
        off = net.jitter_offsets[idx % net.jitter_offsets.shape[0]]
        B, C, H, W = x.shape
        theta = torch.eye(2, 3, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        theta[:, 0, 2] = off[0] * scale / W
        theta[:, 1, 2] = off[1] * scale / H
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    def _edge_aa(self, x, thr, blur_radius, net):
        if blur_radius <= 0:
            return x

        gray = 0.2126*x[:,0:1] + 0.7152*x[:,1:2] + 0.0722*x[:,2:3]
        sx   = F.conv2d(gray, net.sobel_x, padding=1)
        sy   = F.conv2d(gray, net.sobel_y, padding=1)
        edge = torch.sqrt(sx*sx + sy*sy + 1e-6)

        mask = torch.sigmoid((edge - thr) * 14.0)

        blurred = F.avg_pool2d(
            F.pad(x, [blur_radius]*4, mode="reflect"),
            blur_radius*2+1,
            stride=1
        )

        return x*(1.0 - mask) + blurred*mask
        
    def _temporal_refine(self, current, previous, strength=0.35, motion_threshold=0.08):
        if previous is None:
            return current

        if previous.shape != current.shape:
            return current

        curr_luma = (
            0.2126 * current[:, 0:1] +
            0.7152 * current[:, 1:2] +
            0.0722 * current[:, 2:3]
        )

        prev_luma = (
            0.2126 * previous[:, 0:1] +
            0.7152 * previous[:, 1:2] +
            0.0722 * previous[:, 2:3]
        )

        motion = torch.abs(curr_luma - prev_luma)
        motion = torch.clamp(motion / motion_threshold, 0.0, 1.0)

        blend_mask = (1.0 - motion) * strength

        refined = current * (1.0 - blend_mask) + previous * blend_mask
        return refined.clamp(0.0, 1.0)

    def execute(self, images, preset):
    
        # legacy preset names
        if preset == "Sharp":
            preset = "Detail"
        elif preset == "Cinematic":
            preset = "Smooth"
            
        device = mm.get_torch_device() if mm else \
                 torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        preset_jitter_scale = self.jitter_scale
        
        if preset == "Balanced":
            detail_boost = 1.00
            edge_boost = 1.00
            temporal_strength = 0.35
            micro_limit = 0.04
            motion_threshold = 0.08
            taa_strength = 0.45
            dlaa_strength = 0.65
            tone_strength = 0.12
            edge_sharp_strength = 0.14
            motion_sensitivity = 0.08
            preset_jitter_scale = 0.20

        elif preset == "Detail":
            detail_boost = 1.18
            edge_boost = 1.25
            temporal_strength = 0.22
            motion_threshold = 0.06
            micro_limit = 0.05
            taa_strength = 0.35
            dlaa_strength = 0.95
            tone_strength = 0.12
            edge_sharp_strength = 0.18
            motion_sensitivity = 0.07
            preset_jitter_scale = 0.15

        elif preset == "Smooth":
            detail_boost = 0.85
            edge_boost = 0.75
            temporal_strength = 0.45
            micro_limit = 0.025
            motion_threshold = 0.10
            taa_strength = 0.65
            dlaa_strength = 0.65
            tone_strength = 0.16
            edge_sharp_strength = 0.08
            motion_sensitivity = 0.10
            preset_jitter_scale = 0.25

        else:
            detail_boost = 1.00
            edge_boost = 1.00
            temporal_strength = 0.35
            motion_threshold = 0.08
            taa_strength = 0.45
            dlaa_strength = 0.65
            tone_strength = 0.12
            edge_sharp_strength = 0.14
            motion_sensitivity = 0.08
            preset_jitter_scale = 0.20
        
        B, H, W, C = images.shape
        is_single_image = (B == 1)

        if is_single_image:
            blur_radius = 0
            reset_history = True
        else:
            blur_radius   = 1
            reset_history = False
            
        if reset_history:
            self.taa.reset()
            self._prev_dlaa_output = None

        net        = self._net(device)
        out_list   = []

        # tile size
        if mm is not None:
            try:
                vram_mb = torch.cuda.get_device_properties(device).total_memory // (1024 * 1024)
            except:
                vram_mb = 8192
        else:
            vram_mb = 8192

        if   vram_mb <= 8192:  tile_size = 512
        elif vram_mb <= 16384: tile_size = 1024
        else:                  tile_size = 1024

        # tile count
        if H > tile_size or W > tile_size:
            tile_step = tile_size - 64
            tile_count = ((H + tile_step - 1) // tile_step) * ((W + tile_step - 1) // tile_step)
            logger.debug(f"[DLAA] Tiled inference: {tile_count} tiles")
        
        with torch.inference_mode():
            for i in range(B):
                img = images[i:i+1].to(device).permute(0, 3, 1, 2).float()
                rgb = img[:, :3]
                
                motion_gate = torch.tensor(0.0, device=device)
                
                if preset == "Auto":
                    if self.taa.history is not None and self.taa.history.shape == rgb.shape:
                        scene_motion = torch.abs(rgb - self.taa.history).mean().item()
                    else:
                        scene_motion = 0.02

                    if scene_motion < 0.015:
                        detail_boost = 1.12
                        edge_boost = 1.15
                        temporal_strength = 0.28
                        motion_threshold = 0.07
                        taa_strength = 0.35
                        dlaa_strength = 0.70
                        tone_strength = 0.10
                        edge_sharp_strength = 0.18
                        motion_sensitivity = 0.07
                        preset_jitter_scale = 0.15

                    elif scene_motion < 0.045:
                        detail_boost = 1.00
                        edge_boost = 1.00
                        temporal_strength = 0.35
                        motion_threshold = 0.08
                        taa_strength = 0.45
                        dlaa_strength = 0.65
                        tone_strength = 0.12
                        edge_sharp_strength = 0.14
                        motion_sensitivity = 0.08
                        preset_jitter_scale = 0.20

                    else:
                        detail_boost = 0.85
                        edge_boost = 0.75
                        temporal_strength = 0.42
                        motion_threshold = 0.10
                        taa_strength = 0.60
                        dlaa_strength = 0.60
                        tone_strength = 0.16
                        edge_sharp_strength = 0.08
                        motion_sensitivity = 0.10
                        preset_jitter_scale = 0.25
                
                # jitter
                fid = self.taa.frame_id
                self.taa.frame_id = (fid + 1) % net.jitter_offsets.shape[0]

                adaptive_jitter_scale = preset_jitter_scale

                if self.taa.history is not None and self.taa.history.shape == rgb.shape:
                    motion_estimate = torch.abs(rgb - self.taa.history).mean()
                    motion_gate = torch.clamp(motion_estimate * 12.0, 0.0, 1.0)
                    
                    jitter_damping = (1.0 - motion_estimate * 8.0).clamp(0.45, 1.0)
                    adaptive_jitter_scale = preset_jitter_scale * jitter_damping

                rgb = self._jitter(rgb, fid, adaptive_jitter_scale, net)

                # edge cleanup
                rgb = self._edge_aa(rgb, self.edge_threshold, blur_radius, net)

                # TAA
                taa_out = self.taa.update(rgb, self.taa_alpha, motion_sensitivity)
                rgb     = torch.lerp(rgb, taa_out, taa_strength)

                # DLAA
                if dlaa_strength > 0:

                    try:
                        dlaa_out = self._tiled_forward(
                            net,
                            rgb,
                            tile_size=tile_size,
                            overlap=32
                        )
                    except torch.OutOfMemoryError:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        logger.warning("[DLAA] OOM detected, retrying with smaller tiles")

                        retry_tile = max(256, tile_size // 2)

                        try:
                            dlaa_out = self._tiled_forward(
                                net,
                                rgb,
                                tile_size=retry_tile,
                                overlap=32
                            )
                        except torch.OutOfMemoryError:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            retry_tile = 256
                            dlaa_out = self._tiled_forward(
                                net,
                                rgb,
                                tile_size=retry_tile,
                                overlap=32
                            )
                    
                    
                    # luminance match
                    dlaa_mean = dlaa_out.mean(dim=(1,2,3), keepdim=True)
                    rgb_mean  = rgb.mean(dim=(1,2,3), keepdim=True)

                    # highlights

                    dlaa_out = dlaa_out - dlaa_mean + rgb_mean
                    
                    # detail
                    luma = 0.299 * dlaa_out[:, 0:1] + 0.587 * dlaa_out[:, 1:2] + 0.114 * dlaa_out[:, 2:3]
                    highlight_mask = torch.sigmoid((luma - self.highlight_threshold) * self.highlight_slope)

                    dlaa_out = torch.lerp(dlaa_out, rgb, highlight_mask * self.highlight_pre_blend)
                    dlaa_out = torch.clamp(dlaa_out, 0.0, 1.0)
                    
                    # motion detail suppression
                    detail_boost *= (1.0 - motion_gate * 0.20)
                    edge_boost *= (1.0 - motion_gate * 0.15)
                    micro_limit *= (1.0 - motion_gate * 0.20)
                    
                    # gain
                    local_avg_rgb = F.avg_pool2d(dlaa_out, 3, stride=1, padding=1)
                    fine_detail_rgb = dlaa_out - local_avg_rgb

                    luma = 0.299 * dlaa_out[:, 0:1] + 0.587 * dlaa_out[:, 1:2] + 0.114 * dlaa_out[:, 2:3]
                    fine_detail = 0.299 * fine_detail_rgb[:, 0:1] + 0.587 * fine_detail_rgb[:, 1:2] + 0.114 * fine_detail_rgb[:, 2:3]

                    # clamp detail spikes
                    detail_strength = fine_detail.abs().mean(dim=(1,2,3), keepdim=True)
                    detail_scale = (self.detail_base_scale + (self.detail_ref_scale / (detail_strength + 1e-6))).clamp(self.detail_min_scale, self.detail_max_scale)

                    fine_detail = fine_detail * torch.sigmoid(fine_detail * detail_scale)
                    
                    # micro-detail
                    fine_detail = fine_detail.clamp(-0.08, 0.08)

                    local_detail = F.avg_pool2d(fine_detail.abs(), 7, stride=1, padding=3)
                    global_detail = fine_detail.abs().mean(dim=(1,2,3), keepdim=True)

                    edge_for_detail = torch.sqrt(
                        F.conv2d(luma, net.sobel_x, padding=1) ** 2 +
                        F.conv2d(luma, net.sobel_y, padding=1) ** 2 +
                        1e-6
                    )

                    edge_detail_weight = torch.sigmoid((edge_for_detail - self.edge_sharp_threshold) * self.edge_sharp_slope)

                    detail_gain = (global_detail / (local_detail + 1e-6)).clamp(self.detail_min_gain, self.detail_max_gain)
                    detail_gain = detail_gain * (1.0 - local_detail.clamp(0.0, 0.5))
                    detail_gain = detail_gain * (1.0 + edge_detail_weight * self.detail_edge_boost)
                    detail_gain = detail_gain * (1.0 - highlight_mask * self.detail_highlight_suppression)
                    
                    micro_detail = fine_detail_rgb * detail_gain * detail_boost
                    micro_detail = micro_detail.clamp(-micro_limit, micro_limit)
                    dlaa_out = dlaa_out + micro_detail
                    
                    edge_mask = torch.sigmoid((edge_for_detail - self.edge_sharp_threshold) * self.edge_sharp_slope)
                    edge_detail = fine_detail_rgb * edge_mask * (1.0 - highlight_mask)

                    edge_boosted = edge_detail * edge_sharp_strength * edge_boost
                    edge_boosted = edge_boosted.clamp(-micro_limit * 0.7, micro_limit * 0.7)
                    dlaa_out = dlaa_out + edge_boosted
                    
                    # highlights
                    tone_mapped = dlaa_out / (dlaa_out + self.tone_curve_bias)
                    dlaa_out = torch.lerp(dlaa_out, tone_mapped, highlight_mask * tone_strength)
                    
                    dlaa_out = torch.lerp(dlaa_out, rgb, highlight_mask * self.highlight_post_blend)
                    dlaa_out = torch.clamp(dlaa_out, 0.0, 1.0)

                    if self._prev_dlaa_output is not None:
                        if self._prev_dlaa_output.shape != dlaa_out.shape:
                            self._prev_dlaa_output = None

                    dlaa_out = self._temporal_refine(
                        dlaa_out,
                        self._prev_dlaa_output,
                        strength=temporal_strength,
                        motion_threshold=motion_threshold
                    )

                    self._prev_dlaa_output = dlaa_out.detach()

                    dlaa_out = torch.clamp(dlaa_out, 0.0, 1.0)
                    


                    # final blend
                    blend_weight = dlaa_strength * self.dlaa_blend_scale
                    rgb = torch.lerp(rgb, dlaa_out, blend_weight)
                
                rgb = torch.clamp(rgb, 0.0, 1.0)

                out_list.append(rgb.permute(0, 2, 3, 1).cpu())

                if mm is not None and i > 0 and i % 50 == 0:
                    mm.soft_empty_cache()

        return (torch.cat(out_list, dim=0),)


NODE_CLASS_MAPPINGS        = {"VideoTAADLAA": VideoTAADLAA}
NODE_DISPLAY_NAME_MAPPINGS = {"VideoTAADLAA": "🎮 Video Anti-Aliasing (TAA + DLAA)"}